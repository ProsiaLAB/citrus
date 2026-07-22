use std::fs;
use std::path::Path;

use anyhow::Result;
use anyhow::bail;
use codata as cc;
use ndarray::array;
use planetes_ext::types::Vec3;
use planetes_ext::types::{RMatrix, RVector};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Deserialize;
use serde::Deserializer;

use crate::collparts::MolData;
use crate::constants;
use crate::constants::N_DIMS;
use crate::lines::Spec;
use crate::models::Model;

/// This struct contains all basic settings such as number of grid points,
/// model radius, input and output filenames, etc. Some of these parameters
/// always need to be set by the user, while others are optional with preset
/// default values. There is an exception to this rule, namely when restarting
/// citrus with previously calculated populations. In that case, none of the
/// non-optional parameters are required.
#[derive(Deserialize, Debug)]
#[serde(default, rename_all = "snake_case")]
pub struct ParameterInput {
    /// This value sets the outer radius of the computational domain.
    ///
    /// ### Note
    /// It should be set large enough to cover the entire spatial extend of the
    /// model. In particular, if a cylindrical input model is used (e.g., the
    /// input file for the RATRAN code) one should not use the radius of the
    /// cylinder but rather the distance from the centre to the corner of the
    /// (r,z)-plane.
    #[serde(deserialize_with = "from_au")]
    pub radius: f64,

    /// The smallest spatial scale sampled by the code. Structures smaller
    /// than minScale will not be sampled properly. If one uses spherical
    /// sampling (see below) this number can also be thought of as the inner
    /// edge of the grid. This number should not be set smaller than needed,
    /// because that will cause an undesirably large number of grid points to end up near the
    /// centre of the model.
    #[serde(deserialize_with = "from_au")]
    pub min_scale: f64,

    /// Temperature of the cosmic microwave background in Kelvin.
    ///
    /// It defaults to 2.725 K which is the value at zero redshift
    /// (i.e., the solar neighborhood). One should make sure to set this
    /// parameter properly when calculating models at a redshift larger
    /// than zero since
    /// $$
    /// T_{\text{CMB}} = T_{\text{CMB},0} (1 + z) \text{K}
    /// $$
    /// It should be noted that even though citrus can in this way take
    /// the change n CMB temperature with increasing $z$ into account,
    /// it does not (yet) take cosmological effects into account when
    /// ray tracing (such as stretching of frequencies when using Jansky
    /// as units).
    // TODO: Implement cosmological effects in ray tracing
    pub cmb_temp: f64,

    /// The sinkPoints are grid points that are distributed randomly at
    /// [`radius`][Parameters::radius] forming the surface of the model. As a photon from within
    /// the model reaches a sink point it is said to escape and is not tracked
    /// any longer. The number of sink points is a user-defined quantity since
    /// the exact number may affect the resulting image as well as the running
    /// time of the code. One should choose a number that gives a surface
    /// density large enough not to cause artifacts in the image and low enough
    /// not to slow down the gridding too much. Since this is model dependent, a
    /// global best value cannot be given, but a useful range is between a few
    /// thousands and about ten thousand.
    pub n_sink_points: usize,

    /// This number is the number of model grid points. The more grid points
    /// that are used, the longer the code will take to run. Too few points
    /// however, will cause the model to be under-sampled with the risk of
    /// getting wrong results. Useful numbers are between a few thousands up to
    /// about one hundred thousand.
    pub p_intensity: usize,

    /// If true, line blending is taken into account, however,
    /// only if there are any overlapping lines among the transitions
    /// found in the molecular data files. Enabling line blending
    /// will slow down the calculations considerably, in particular
    /// if there are multiple molecular data files.
    ///
    /// By default, line blending is disabled.
    pub enable_line_blending: bool,

    /// This parameter specifies the algorithm used by citrus to
    /// solve the radiative-transfer equations during ray-tracing.
    /// The default value of zero invokes the algorithm used in
    /// older versions a value of 1 invokes a new algorithm which is
    /// much more time-consuming but which produces much smoother
    /// images, free from step-artifacts.
    ///
    /// If none of the four density-linked parameters are provided,
    /// citrus will attempt to guess the information, in a manner as
    /// close as possible to the way it was done in version 1.5 and
    /// earlier. This is safe enough when a single density value is
    /// returned, and only H2 provided as collision partner in the
    /// moldata file(s), but more complicated situations can very
    /// easily result in the code guessing wrongly. For this reason
    /// we encourage users to make use of these four parameters,
    /// although in order to preserve backward compatibility with
    /// old model.c files, we have not (yet) made them mandatory.
    ///
    /// ### Note
    /// There have been additional modifications to the raytracing
    /// algorithm which have significant effects on the output
    /// images since citrus-1.5. Image-plane interpolation is now
    /// employed in areas of the image where the grid point
    /// spacing is larger than the image pixel spacing. This
    /// leads both to a smoother image and a shorter processing
    /// time.
    pub ray_trace_algorithm: RayTraceAlgorithm,

    /// The sampling algorithm used for the model grid.
    ///
    /// By default, the [`UniformExact`][SamplingAlgorithm::UniformExact] is used
    /// which corresponds to default algorithm in newer versions of citrus.
    pub sampling_algorithm: SamplingAlgorithm,

    /// If non-zero, citrus performs a direct LTE calculation rather than solving
    /// for the populations iteratively. This facility is useful for quick checks. The
    /// default is false, i.e., full non-LTE calculation.
    pub lte_only: bool,

    /// If true, citrus solves for the level populations as usual, but
    /// LTE values are used to initialize the populations at the start
    /// instead of using values at T = 0.
    pub init_lte: bool,

    /// If true, citrus will calculate the polarized continuum emission.
    /// This parameter only has effect for continuum images. The resulting cube
    /// will have three channels containing the Stokes I, Q, and U of the continuum emission
    /// as according to the theory, the V component is zero. In order for polarization
    /// to work, a magnetic field must be defined.
    ///
    /// When polarization is enabled, the implementation follows the
    /// `DustPol` code (Padovani et al. 2012), except that the expression
    /// used in Padovani et al. (2012) given for $\sigma^2$ has shown by
    /// Adele et al. (2015) to be small by a factor of 2. This correction
    /// has been implemented in citrus.
    pub polarization: bool,

    /// This defines the number of solution iterations citrus
    /// should perform when solving non-LTE level populations.
    /// The default is currently 17. Note that it is now possible
    /// to run citrus in an incremental fashion. If the results
    /// of solving the RTE through `N` iterations are stored in
    /// a grid file via setting
    ///  [`Parameters::grid_out_files`],
    /// then a second run of citrus, reading the grid file
    /// via [`Parameters::grid_in_file`],
    /// [`Parameters::n_solve_iters`]` = M>N`, will continue the
    /// RTE iterations starting at iteration `N`.
    /// (If you do this, your results will be slightly
    /// different, in a random way, than if you go to `M`
    /// iterations in one go, because the random seeds
    /// will be different.)
    pub n_solve_iters: usize,

    /// This is the file name of the output file that contains the level
    /// populations. If this parameter is not set, citrus will not output the
    /// populations. There is no default value.
    pub output_file: Option<String>,

    /// This is the file name of the output file that contains the grid,
    /// populations, and molecular data in binary format. This file is used to
    /// restart citrus with previously calculated populations. Once the
    /// populations have been calculated and the binoutputfile has been written,
    /// citrus can re-raytrace for a different set of image parameters without
    /// re-calculating the populations. There is no default value.
    pub binoutput_file: Option<String>,

    /// This is the file name of the output file that contains the grid. If this
    /// parameter is not set, citrus will not output the grid. The grid file is
    /// written out as a VTK file. This is a formatted ascii file that can be
    /// read with a number of 3D visualizing tools (Visualization Tool Kit,
    /// Paraview, and others). There is no default value.
    pub grid_file: Option<String>,

    /// Path to the dust opacity file.
    ///
    /// This must be prpvided if any continuum images are requested.
    /// Optional if only line images are requested.
    ///
    /// ### File Format
    /// This table should be a two column ASCII
    /// file with wavelength in the first column and opacity in the second
    /// column. Currently citrus uses the same tables as RATRAN from Ossenkopf and
    /// Henning (1994), and so the wavelength should be given in microns (1e-6
    /// meters) and the opacity in cm^2/g. This is the only place in citrus where
    /// SI units are not used. There is
    /// no default value. A future version of citrus may allow spatial variance
    /// of the dust opacities, so that opacities can be given as function of x,
    /// y, and z.
    pub dust_file: Option<String>,
    pub grid_in_file: Option<String>,

    /// If this is set non-zero, LIME will use the same random
    /// number seeds at the start of each solution iteration.
    /// This has the effect of choosing the same photon directions
    /// and frequencies for each iteration (although the directions
    /// and frequencies change randomly from one grid point to
    /// the next). This has the effect of decoupling any
    /// oscillation or wandering of the level populations as
    /// they relax towards convergence from the intrinsic
    /// Monte Carlo noise of the discrete solution algorithm.
    /// Best practice might involve alternating episodes
    /// with `par->resetRNG` =0 and 1, storing the intermediate
    /// populations via the :ref:`I/O interface <grid-io>`.
    /// Very little experience has been accumulated as yet
    /// with this facility.
    pub reset_rng: bool,

    /// It is now possible to run LIME in two sessions: the
    /// first to solve the RTE and save the results to file,
    /// the second to read the file and create raytraced images
    /// from it. For a session of the first type you should set
    /// the number of images you specify via the
    /// :ref:`img <images>` parameter to zero, and give a
    /// value for one of the elements of
    /// :ref:`par->gridOutFiles <grid-io>`; for one of
    /// the second type you set
    /// :ref:`par->gridInFile <grid-io>` to the name of
    /// the file you just wrote, and include >0 image
    /// specifications in :ref:`img <images>`. There is
    /// a problem however for sessions of the first
    /// type: if you eventually want full-spectrum cubes
    /// then you will need some way to tell LIME to solve
    /// the RTE. In the past LIME has figured out if you
    /// want this from the presence of spectrum-type images
    /// in your :ref:`img <images>` list. To replace this
    /// capability we have added the present parameter.
    /// Thus, for first-stage sessions (supposing you choose
    /// to run LIME in that way rather than in the previous
    /// single-pass style) when you know that you will
    /// eventually want spectral cubes, you should set
    /// the present parameter. For all other cases it may
    /// be ignored.
    pub do_solve_rte: bool,

    /// We calculate the number density of each of its radiating
    /// species, at each grid point, by multiplying the abundance
    /// of the species (returned via the function of that name)
    /// by a weighted sum of the density values. This parameter
    /// allows the user to specify the weights in that sum.
    pub n_mol_weights: Vec<f64>,

    /// Controls how the dust mass density and hence opacity is calculated.
    /// The calculation of dust mass density in older versions made use of
    /// a hard-wired average gas density value of 2.4, appropriate
    /// to a mix of 90% molecular hydrogen and 10% helium. This
    /// older formula will be used if none of the current four
    /// parameters are set.
    pub dust_weights: Vec<f64>,

    pub collisional_partner_mol_weights: Vec<f64>,

    /// The integer values are the codes given [here](http://home.strw.leidenuniv.nl/~moldata/molformat.html).
    /// Currently recognized values range from 1 to 7 inclusive. E.g if the only colliding
    /// species of interest in your model is H2, your density function should return a
    /// single value, namely the density of molecular hydrogen, and (if you supply a
    /// collPartIds value at all) you should set collPartIds[0] = 1 (the LAMDA code for
    /// H2). However, if you use collisional partners that are not one of LAMDA
    /// partners, it is fine to use any of the values between 1 and 7 to match
    /// the density function with collisional information in the datafiles.  Some of
    /// the messages in citrus will refer to the default LAMDA partner molecules, but
    /// this does not affect the calculations. In future we will introduce a better mechanism to allow the user to specify non-LAMDA collision partners.
    ///
    /// This list acts as a link between the `N` density
    /// function returns (I'm using here `N` as shorthand for `num_densities`) and the `M`
    /// collision partner ID integers found in the moldatfiles. This allows us to
    /// associate density functions with the collision partner transition rates provided
    /// in the moldatfiles.
    // TODO: Perhaps implement this on type-system level using enums?
    pub collisional_partner_ids: Vec<usize>,
    pub g_ir_data_files: Option<Vec<String>>,

    /// This contains paths to molecular data files.
    /// It must be provided if any line images are specified
    /// (or [`do_solve_rte`][Parameters::do_solve_rte] is true).
    /// It is not read if only continuum images are requested.
    ///
    /// Molecular data files contain the
    /// energy states, Einstein coefficients, and collisional rates which are
    /// needed by citrus to solve the excitation. These files must conform to
    /// the standard of the [LAMDA database](http://www.strw.leidenuniv.nl/~moldata).
    /// If a data file name is give that
    /// cannot be found locally, citrus will try and download the file instead.
    /// When downloading data files, the filename can be give both with and
    /// without the extension .dat (i.e., `CO` or `CO.dat`). moldatfile is an
    /// array, so multiple data files can be used for a single citrus run. There is
    /// no default value.
    pub mol_data_files: Vec<String>,

    /// Essentially this has only cosmetic importance
    /// since it has no effect on the functioning of citrus, only on the names of the
    /// collision partners which are printed to stdout. Its main purpose is to reassure
    /// the user who has provided transition rates for a non-LAMDA collision species in
    /// their moldatfile that they are actually getting these values and not some
    /// mysterious reversion to LAMDA.
    pub collisional_partner_names: Vec<String>,

    pub(crate) grid_out_files: Vec<String>,
    pub(crate) collisional_partner_user_set_flags: isize,
    pub(crate) radius_squ: f64,
    pub(crate) min_scale_squ: f64,
    pub(crate) taylor_cutoff: f64,
    pub(crate) grid_density_global_max: f64,
    pub(crate) ncell: usize,
    pub(crate) n_species: usize,
    pub(crate) n_densities: usize,
    pub(crate) n_line_images: usize,
    pub(crate) n_cont_images: usize,
    pub(crate) n_solve_iters_done: usize,
    pub(crate) do_interpolate_vels: bool,
    pub(crate) use_abun: bool,
    pub(crate) do_mol_calcs: bool,
    pub(crate) use_vel_func_in_raytrace: bool,
    pub(crate) edge_vels_available: bool,
    pub(crate) write_grid_at_stage: Vec<bool>,
    pub(crate) do_pre_grid: bool,
}

impl Default for ParameterInput {
    fn default() -> Self {
        ParameterInput {
            radius: 0.0,
            min_scale: 0.0,
            cmb_temp: constants::LOCAL_CMB_TEMP_SI,
            n_sink_points: 0,
            p_intensity: 0,
            enable_line_blending: false,
            ray_trace_algorithm: RayTraceAlgorithm::default(),
            sampling_algorithm: SamplingAlgorithm::default(),
            lte_only: false,
            init_lte: false,
            polarization: false,
            n_solve_iters: 0,
            output_file: None,
            binoutput_file: None,
            grid_file: None,
            dust_file: None,
            grid_in_file: None,
            reset_rng: false,
            do_solve_rte: true,
            n_mol_weights: Vec::new(),
            dust_weights: Vec::new(),
            collisional_partner_mol_weights: Vec::new(),
            collisional_partner_ids: Vec::new(),
            g_ir_data_files: None,
            mol_data_files: Vec::new(),
            collisional_partner_names: Vec::new(),
            grid_out_files: Vec::new(),
            collisional_partner_user_set_flags: 0,
            radius_squ: 0.0,
            min_scale_squ: 0.0,
            taylor_cutoff: 0.0,
            grid_density_global_max: 0.0,
            ncell: 0,
            n_species: 0,
            n_densities: 0,
            n_line_images: 0,
            n_cont_images: 0,
            n_solve_iters_done: 0,
            do_interpolate_vels: false,
            use_abun: false,
            do_mol_calcs: false,
            use_vel_func_in_raytrace: false,
            edge_vels_available: false,
            write_grid_at_stage: vec![false; constants::NUM_OF_GRID_STAGES],
            do_pre_grid: false,
        }
    }
}

pub struct Domain {
    pub radius: f64,
    pub min_scale: f64,
}

pub struct Outputs {
    pub populations: Option<String>,
    pub binary: Option<String>,
    pub grid: Option<String>,
    pub grid_out_files: Vec<String>,
}

pub struct MolecularConfig {
    pub mol_data_files: Vec<String>,
    pub g_ir_files: Option<Vec<String>>,
    pub partner_names: Vec<String>,
    pub partner_ids: Vec<usize>,
    pub mol_weights: Vec<f64>,
    pub partner_weights: Vec<f64>,
}

pub enum DustConfig {
    None,
    Table { path: String, weights: Vec<f64> },
}

pub struct SolverConfig {
    pub n_iters: usize,
    pub lte_only: bool,
    pub init_lte: bool,
    pub solve_rte: bool,
    pub reset_rng: bool,
}

pub struct GridConfig {
    pub n_points: usize,
    pub n_sink_points: usize,
    pub sampling: SamplingAlgorithm,
}

pub struct FreshRun {
    pub grid_in_file: Option<String>,
}

pub struct RestartRun {
    pub grid_in_file: String,
}

pub enum RunMode {
    Fresh(FreshRun),
    Restart(RestartRun),
}

pub struct Parameters {
    pub domain: Domain,
    pub cmb_temp: f64,
    pub grid: GridConfig,
    pub solver: SolverConfig,
    pub outputs: Outputs,
    pub molecular: MolecularConfig,
    pub dust: DustConfig,
    pub raytrace: RayTraceAlgorithm,
    pub polarization: bool,
    pub line_blending: bool,
    pub mode: RunMode,
}

impl TryFrom<ParameterInput> for Parameters {
    type Error = ParamError;

    fn try_from(raw: ParameterInput) -> Result<Self, Self::Error> {
        if raw.radius <= 0.0 || raw.min_scale <= 0.0 {
            return Err(ParamError::InvalidDomain);
        }
        if raw.min_scale >= raw.radius {
            return Err(ParamError::MinScaleTooLarge);
        }

        let domain = Domain {
            radius: raw.radius,
            min_scale: raw.min_scale,
        };

        let outputs = Outputs {
            populations: raw.output_file,
            binary: raw.binoutput_file,
            grid: raw.grid_file,
            grid_out_files: raw.grid_out_files,
        };

        let molecular = MolecularConfig {
            mol_data_files: raw.mol_data_files,
            g_ir_files: raw.g_ir_data_files,
            partner_names: raw.collisional_partner_names,
            partner_ids: raw.collisional_partner_ids,
            mol_weights: raw.n_mol_weights,
            partner_weights: raw.collisional_partner_mol_weights,
        };

        let dust = match raw.dust_file {
            Some(path) => DustConfig::Table {
                path,
                weights: raw.dust_weights,
            },
            None => DustConfig::None,
        };

        let solver = SolverConfig {
            n_iters: raw.n_solve_iters,
            lte_only: raw.lte_only,
            init_lte: raw.init_lte,
            solve_rte: raw.do_solve_rte,
            reset_rng: raw.reset_rng,
        };

        let grid = GridConfig {
            n_points: raw.p_intensity,
            n_sink_points: raw.n_sink_points,
            sampling: raw.sampling_algorithm,
        };

        let mode = match raw.grid_in_file {
            Some(file) if !file.is_empty() => RunMode::Restart(RestartRun { grid_in_file: file }),
            _ => RunMode::Fresh(FreshRun { grid_in_file: None }),
        };

        Ok(Self {
            domain,
            cmb_temp: raw.cmb_temp,
            grid,
            solver,
            outputs,
            molecular,
            dust,
            raytrace: raw.ray_trace_algorithm,
            polarization: raw.polarization,
            line_blending: raw.enable_line_blending,
            mode,
        })
    }
}

/// Sampling algorithms to generate grid points for modeling.
#[derive(Deserialize, Debug, Default)]
pub enum SamplingAlgorithm {
    /// This algorithm can be used uniform sampling in Log(radius)
    /// which is useful for models with a central condensation
    /// (i.e., envelopes, disks)
    Uniform,

    /// This algorithm generates grid points with exact
    /// spherical rotation symmetry.
    #[default]
    UniformExact,

    /// This generates grid points with a uniform-biased
    /// sampling in cartesian coordinates.
    ///
    /// This is useful for models with no central condensation
    /// (molecular clouds, galaxies, slab geometries, etc.)
    UniformBiased,

    /// This is a newer algorithm which can quickly generate points with a
    /// distribution which accurately follows any feasible density function.
    /// including with sharp step-changes.
    ///
    /// The contained value is a vector of [`density maxima`][GridDensityMaxima] which
    /// the user can provide to help guide the point distribution.
    ///
    /// This algorithm also incorporates a quasi-random choice of
    /// point candidates which avoids the requirement for the relatively
    /// time-consuming post-gridding smoothing phase. Therefore,
    /// supplying your own density function will give full control
    /// over the distribution of points. Currently, [`qhull`] is
    /// being used to triangulate the point distribution.
    // TODO: Think about getting rid of qhull dependency and implementing a pure Rust solution for Delaunay triangulation.
    Modern(Vec<GridDensityMaxima>),
}

impl SamplingAlgorithm {
    #[must_use]
    pub fn is_legacy(&self) -> bool {
        !matches!(self, SamplingAlgorithm::Modern(_))
    }
}

/// Represents a grid density maximum for use with
/// the [`Modern`][SamplingAlgorithm::Modern] sampling algorithm.
#[derive(Deserialize, Debug)]
pub struct GridDensityMaxima {
    /// Location of the density maximum in 3-D space.
    pub location: Vec3,

    /// Value of the density maximum at the specified location.
    pub value: f64,
}

#[derive(Deserialize, Debug, Default, PartialEq)]
pub enum RayTraceAlgorithm {
    Legacy,
    #[default]
    Modern,
}

/// The [`Image`] struct represents the configuration for an image to be generated.
#[derive(Deserialize, Debug)]
#[serde(default, rename_all = "snake_case")]
pub struct ImageInput {
    pub nchan: usize,
    pub trans: Option<usize>,
    pub mol_i: usize,
    pub pixel: Vec<Spec>,
    pub vel_res: f64,
    pub img_res: f64,
    pub pxls: usize,
    pub unit: Unit,
    pub freq: f64,
    pub bandwidth: f64,
    pub filename: String,
    pub source_velocity: f64,
    pub theta: f64,
    pub phi: f64,
    pub inclination: f64,
    pub position_angle: f64,
    pub azimuth: f64,
    #[serde(deserialize_with = "from_parsec")]
    pub distance: f64,
    // TODO: Separate config params from data params
    pub rotation_matrix: RMatrix,
    pub do_interpolate_vels: bool,
    pub do_line: bool,
}

impl Default for ImageInput {
    fn default() -> Self {
        ImageInput {
            nchan: 0,
            trans: None,
            mol_i: 0,
            vel_res: 0.0,
            img_res: 0.0,
            pxls: 0,
            unit: Unit::JanskyPerPixel,
            freq: 0.0,
            bandwidth: 0.0,
            source_velocity: 0.0,
            theta: 0.0,
            phi: 0.0,
            inclination: 0.0,
            position_angle: 0.0,
            azimuth: 0.0,
            distance: 0.0,
            do_interpolate_vels: false,
            do_line: false,
            pixel: Vec::new(),
            filename: String::new(),
            rotation_matrix: RMatrix::eye(N_DIMS),
        }
    }
}

pub struct Spatial {
    pub pxls: usize,
    pub img_res: f64,
    pub distance: f64,
}

pub struct Orientation {
    pub theta: f64,
    pub phi: f64,
    pub inclination: f64,
    pub position_angle: f64,
    pub azimuth: f64,
    // pub rotation_matrix: RMatrix,
}

pub struct Image {
    pub spatial: Spatial,
    pub orientation: Orientation,
    pub filename: String,
    pub unit: Unit,
    pub source_velocity: f64,
    pub interpolate_vels: bool,
    pub pixels: Vec<Spec>,
}

pub enum ImageKind {
    Continuum(ContinuumImage),
    Line(LineImage),
}

pub struct ContinuumImage {
    pub image: Image,
    pub freq: f64,
}

pub struct LineImage {
    pub image: Image,
    pub freq: LineFreq,
    pub spectral: SpectralAxis,
}

pub enum LineFreq {
    Transition { trans: usize, mol_i: usize },
    Explicit(f64),
}

pub enum SpectralAxis {
    BandwidthVelRes { bandwidth: f64, vel_res: f64 },
    BandwidthNchan { bandwidth: f64, nchan: usize },
    VelResNchan { vel_res: f64, nchan: usize },
}

/// Unit of the data in the image.
#[derive(Deserialize, Debug, Default)]
#[serde(rename_all = "lowercase")]
pub enum Unit {
    Kelvin,
    #[default]
    JanskyPerPixel,
    SI,
    LsunPerPixel,
    OpticalDepth,
}

impl TryFrom<ImageInput> for ImageKind {
    type Error = ImageError;

    fn try_from(raw: ImageInput) -> Result<Self, Self::Error> {
        let image = Image {
            spatial: Spatial {
                pxls: raw.pxls,
                img_res: raw.img_res,
                distance: raw.distance,
            },
            orientation: Orientation {
                theta: raw.theta,
                phi: raw.phi,
                inclination: raw.inclination,
                position_angle: raw.position_angle,
                azimuth: raw.azimuth,
                // rotation_matrix: raw.rotation_matrix,
            },
            filename: raw.filename,
            unit: raw.unit,
            source_velocity: raw.source_velocity,
            interpolate_vels: raw.do_interpolate_vels,
            pixels: raw.pixel,
        };

        if raw.do_line {
            let freq = if let Some(trans) = raw.trans {
                LineFreq::Transition {
                    trans,
                    mol_i: raw.mol_i,
                }
            } else {
                LineFreq::Explicit(raw.freq)
            };

            let spectral = match (raw.bandwidth, raw.vel_res, raw.nchan) {
                (b, v, 0) if b > 0.0 && v > 0.0 => SpectralAxis::BandwidthVelRes {
                    bandwidth: b,
                    vel_res: v,
                },

                (b, 0.0, n) if b > 0.0 && n > 0 => SpectralAxis::BandwidthNchan {
                    bandwidth: b,
                    nchan: n,
                },

                (0.0, v, n) if v > 0.0 && n > 0 => SpectralAxis::VelResNchan {
                    vel_res: v,
                    nchan: n,
                },

                _ => return Err(ImageError::InvalidSpectralDefinition),
            };

            Ok(ImageKind::Line(LineImage {
                image,
                freq,
                spectral,
            }))
        } else {
            if raw.freq <= 0.0 {
                return Err(ImageError::ContinuumNeedsFreq);
            }

            Ok(ImageKind::Continuum(ContinuumImage {
                image,
                freq: raw.freq,
            }))
        }
    }
}

pub enum ImageError {
    InvalidSpectralDefinition,
    ContinuumNeedsFreq,
}

pub enum ParamError {
    InvalidDomain,
    MinScaleTooLarge,
}

#[derive(Debug, Deserialize)]
pub struct Config {
    pub parameters: ParameterInput,
    pub images: Vec<ImageInput>,
}

impl Config {
    /// Load the configuration from a file
    ///
    /// # Errors
    /// This function may fail if the file cannot be read or if the configuration is invalid.
    pub fn from_path(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?; // Propagate file read errors
        let config: Config = toml::from_str(&content)?; // Propagate config parsing errors
        Ok(config)
    }

    /// Parse the configuration for correctness and consistency
    ///
    /// # Errors
    /// This function may fail if the configuration is invalid.
    ///
    /// # Panics
    /// This function may panic if the model is invalid.
    pub fn parse<M: Model>(&mut self, model: &M) -> Result<()> {
        let mut r = Vec3::zero();

        self.parameters.ncell = self.parameters.p_intensity + self.parameters.n_sink_points;
        self.parameters.radius_squ = self.parameters.radius * self.parameters.radius;
        self.parameters.min_scale_squ = self.parameters.min_scale * self.parameters.min_scale;
        self.parameters.n_solve_iters_done = 0;
        self.parameters.use_abun = true;

        if self.parameters.grid_in_file.is_none() {
            assert!(
                self.parameters.radius > 0.0,
                "Radius must be positive and non-zero."
            );
            assert!(
                self.parameters.min_scale > 0.0,
                "Minimum scale must be positive and non-zero."
            );
            assert!(
                self.parameters.p_intensity > 0,
                "Number of intensity points must be positive and non-zero."
            );
            assert!(
                self.parameters.n_sink_points > 0,
                "Number of sink points must be positive and non-zero."
            );
        }

        self.parameters.grid_density_global_max = 1.0;
        self.parameters.n_densities = 0;

        if self.parameters.grid_in_file.is_none() {
            // Read the grid file in FITS format
            // TODO: Currently not implemented
            todo!()
        }

        if self.parameters.n_densities == 0 {
            // This means the FITS file read above did not return any density values,
            // In this case the default density function will not suffice.
            // It is not straightforward to check if the user type that will implement
            // the [`Model`] trait is using the default density function or did they
            // override it with their own due to lack of runtime reflection in Rust.
            // Therefore for now my plan is to leave this part as a TODO.
            todo!()
        }

        // Not sure this is needed?
        self.parameters.grid_density_global_max = 1.0;

        let mut temp_point_density = model.grid_density(&self.parameters, &r);

        if temp_point_density.is_infinite() || temp_point_density.is_nan() {
            eprintln!("There is a singularity in the grid density function at the origin.");
        } else if temp_point_density <= 0.0 {
            eprintln!("The grid density function is zero at the origin.");
        } else if temp_point_density > self.parameters.grid_density_global_max {
            self.parameters.grid_density_global_max = temp_point_density;
        }

        if temp_point_density.is_infinite()
            || temp_point_density.is_nan()
            || temp_point_density <= 0.0
        {
            r = self.parameters.min_scale * Vec3::one();
            temp_point_density = model.grid_density(&self.parameters, &r);

            if !temp_point_density.is_infinite()
                && !temp_point_density.is_nan()
                && temp_point_density > 0.0
            {
                self.parameters.grid_density_global_max = temp_point_density;
            } else {
                // Hmm ok, let's try a spread of random locations
                let mut rand_gen = if true {
                    // Use fixed seed for reproducibility
                    // Note: SeedableRng::seed_from_u64 takes a u64 seed
                    StdRng::seed_from_u64(140_978)
                } else {
                    let mut thread_rng = rand::rng();
                    // Seed from the system's entropy source for non-reproducible randomness
                    // StdRng::from_entropy is a good way to get a random seed
                    StdRng::from_rng(&mut thread_rng)
                };
                println!("Random number generator initialized.");
                let mut found_good_value = false;
                for _ in 0..100 {
                    // TODO: Better way to do this?
                    r.x = rand_gen.random_range(-self.parameters.radius..self.parameters.radius);
                    r.y = rand_gen.random_range(-self.parameters.radius..self.parameters.radius);
                    r.z = rand_gen.random_range(-self.parameters.radius..self.parameters.radius);

                    temp_point_density = model.grid_density(&self.parameters, &r);
                    if !temp_point_density.is_infinite()
                        && !temp_point_density.is_nan()
                        && temp_point_density > 0.0
                    {
                        found_good_value = true;
                        break;
                    }
                }

                if found_good_value {
                    if temp_point_density > self.parameters.grid_density_global_max {
                        self.parameters.grid_density_global_max = temp_point_density;
                    }
                } else if let SamplingAlgorithm::Modern(grid_density_maxima) =
                    &self.parameters.sampling_algorithm
                {
                    assert!(
                        !grid_density_maxima.is_empty(),
                        "Could not find a non-pathological grid density value, and no grid density maxima were provided to guide the sampling algorithm."
                    );

                    for m in grid_density_maxima {
                        if m.value > self.parameters.grid_density_global_max {
                            self.parameters.grid_density_global_max = m.value;
                        }
                    }
                } else {
                    panic!(
                        "Grid density maxima only supported for Modern sampling algorithm. Could not find a non-pathological grid density value."
                    );
                }
            }
        }

        for i in 0..constants::NUM_OF_GRID_STAGES {
            if !self.parameters.grid_out_files[i].is_empty() {
                self.parameters.write_grid_at_stage[i] = true;
            }
        }

        /*
        Now we need to calculate the cutoff value used in `calc_source_fn()`. The issue is
        to decide between

          y = (1 - exp[-x])/x

        or the approximation using the Taylor expansion of exp(-x), which to 3rd order
        is

          y ~ 1 - x/2 + x^2/6.

        The cutoff will be the value of abs(x) for which the error in the exact
        expression equals the next unused Taylor term, which is x^3/24. This error can
        be shown to be given for small |x| by epsilon/|x|, where epsilon is the
        floating-point precision of the computer. Hence the cutoff evaluates to

          |x|_cutoff = (24*epsilon)^{1/4}.

        */
        self.parameters.taylor_cutoff = (24.0 * f64::EPSILON).powf(0.25);

        // Allocate pixel space and parse image information
        for (i, image) in self.images.iter_mut().enumerate() {
            if image.nchan == 0 && image.vel_res < 0.0 {
                // User has set neither `nchan` nor `vel_res`
                // One of the two are required for a line image
                // Therefore, we assume continuum image
                if self.parameters.polarization {
                    image.nchan = 3;
                } else {
                    image.nchan = 1;
                }
                assert!(
                    image.freq != 0.0,
                    "You must set a frequency for continuum image."
                );
                if image.trans.is_some() || image.bandwidth > 0.0 {
                    eprintln!(
                        "Transition number and bandwidth are not relevant for a continuum image. They will be ignored."
                    );
                    image.do_line = false;
                }
            } else {
                // User has set either `nchan` or `vel_res`
                // Therefore, we assume line image

                /*
                For a valid line image, the user must set one of the following pairs:
                bandwidth, velres (if they also set nchan, this is overwritten)
                bandwidth, nchan (if they also set velres, this is overwritten)
                velres, nchan (if they also set bandwidth, this is overwritten)

                The presence of one of these combinations at least is checked here, although the
                actual calculation is done in raytrace(), because it depends on moldata info
                which we have not yet got.
                */
                if image.bandwidth > 0.0 && image.vel_res > 0.0 {
                    if image.nchan > 0 {
                        eprintln!(
                            "As image had both bandwidth and velres set, nchan will be overwritten."
                        );
                    }
                } else if image.nchan == 0 || image.bandwidth <= 0.0 && image.vel_res <= 0.0 {
                    panic!("You must set either nchan, bandwidth, or velres for a line image.");
                }
                // Check that we have keywords which allow us to calculate the image
                // frequency (if necessary) after reading in the moldata file:
                if image.trans.is_some() {
                    // User has set of `trans`, posssibly also `freq`
                    if image.freq > 0.0 {
                        eprintln!(
                            "Ignoring user-set frequency since transition number is set. The frequency will be calculated from the molecular data file."
                        );
                    }
                    if self.parameters.n_species > 1 {
                        let msg = format!(
                            "WARNING: Image {i} did not have ``mol_i`` set, \
                                Therefore, first molecule will be used.",
                        );
                        eprintln!("{msg}");
                        image.mol_i = 0;
                    }
                } else if image.freq < 0.0 {
                    // User has not set `trans`, nor `freq`
                    bail!(
                        "You must either set `trans` or `freq` for a line image (and optionally the `mol_i`"
                    );
                } // else user has set `freq`
                image.do_line = true;
            } // End of check for line or continuum image
            if image.img_res < 0.0 {
                bail!("You must set image resolution.");
            }
            if image.pxls == 0 {
                bail!("You must set number of pixels.");
            }
            if image.distance <= 0.0 {
                bail!("You must set distance to source.");
            }
            image.img_res = (image.img_res / 3600.0).to_radians();
            image.pixel = {
                let mut v = Vec::with_capacity(image.pxls * image.pxls);
                for _ in 0..(image.pxls * image.pxls) {
                    v.push(Spec::default());
                }
                v
            };
            for spec in &mut image.pixel {
                spec.intense = RVector::zeros(image.nchan);
                spec.tau = RVector::zeros(image.nchan);
            }

            // Calculate the rotation matrix
            /*
            The image rotation matrix is used within traceray() to transform the coordinates
            of a vector (actually two vectors - the ray direction and its starting point) as
            initially specified in the observer-anchored frame into the coordinate frame of
            the model. In linear algebra terms, the model-frame vector v_mod is related to
            the vector v_obs as expressed in observer- (or image-) frame coordinates via the
            image rotation matrix R by

                v_mod = R * v_obs,				1

            the multiplication being the usual matrix-vector style. Note that the ith row of
            R is the ith axis of the model frame with coordinate values expressed in terms
            of the observer frame.

            The matrix R can be broken into a sequence of several (3 at least are needed for
            full degrees of freedom) simpler rotations. Since these constituent rotations
            are usually easier to conceive in terms of rotations of the model in the
            observer framework, it is convenient to invert equation (1) to give

                v_obs = R^T * v_mod,			2

            where ^T here denotes transpose. Supposing now we rotate the model in a sequence
            R_3^T followed by R_2^T followed by R_1^T, equation (2) can be expanded to give

                v_obs = R_1^T * R_2^T * R_3^T * v_mod.	3

            Inverting everything to return to the format of equation (1), which is what we
            need, we find

                v_mod = R_3 * R_2 * R_1 * v_obs.		4

            citrus provides two different schemes of {R_1, R_2, R_3}: {PA, phi, theta} and
            {PA, inclination, azimuth}. As an example, consider phi, which is a rotation of
            the model from the observer Z axis towards the X. The matching obs->mod rotation
            matrix is therefore

                        ( cos(ph)  0  -sin(ph) )
                        (                      )
                R_phi = (    0     0     1     ).
                        (                      )
                        ( sin(ph)  0   cos(ph) )

                */
            let do_theta_phi = image.inclination < -900.0
                || image.azimuth < -900.0
                || image.position_angle < -900.0;
            if do_theta_phi {
                // For the present position angle is not implemented
                // for the theta/phi scheme, so we will just load the
                // the identity matrix
                image.rotation_matrix = RMatrix::eye(3);
            } else {
                // Load position angle matrix
                let cos_pa = image.position_angle.cos();
                let sin_pa = image.position_angle.sin();
                image.rotation_matrix = array![
                    [cos_pa, -sin_pa, 0.0],
                    [sin_pa, cos_pa, 0.0],
                    [0.0, 0.0, 1.0]
                ];
            }
            let aux_rotation_matrix = if do_theta_phi {
                // Load phi rotation matrix R_phi
                let cos_phi = image.phi.cos();
                let sin_phi = image.phi.sin();
                array![
                    [cos_phi, 0.0, -sin_phi],
                    [0.0, 1.0, 0.0],
                    [sin_phi, 0.0, cos_phi]
                ]
            } else {
                // Load inclination matrix R_incl
                let cos_incl = (image.inclination + std::f64::consts::PI).cos();
                let sin_incl = (image.inclination + std::f64::consts::PI).sin();
                array![
                    [cos_incl, 0.0, -sin_incl],
                    [0.0, 1.0, 0.0],
                    [sin_incl, 0.0, cos_incl]
                ]
            };
            // Multiply the two matrices
            let _temp_matrix = image.rotation_matrix.dot(&aux_rotation_matrix);
        }

        self.parameters.n_line_images = 0;
        self.parameters.n_cont_images = 0;
        self.parameters.do_interpolate_vels = false;

        for image in &mut self.images {
            if image.do_line {
                self.parameters.n_line_images += 1;
            } else {
                self.parameters.n_cont_images += 1;
            }
            if image.do_interpolate_vels {
                self.parameters.do_interpolate_vels = true;
            }
        }

        if self.parameters.n_cont_images > 0 {
            match &self.parameters.dust_file {
                Some(dust) if !dust.is_empty() => {
                    // Open the dust file and check if it exists and if it is empty
                    let path = Path::new(dust);
                    if path.exists() {
                        match fs::metadata(path) {
                            Ok(metadata) => {
                                if metadata.len() == 0 {
                                    eprintln!("File {dust} is empty.");
                                }
                            }
                            Err(err) => {
                                eprintln!("Error accessing file {dust}: {err}");
                            }
                        }
                    }
                }
                _ => {
                    bail!("You must set dust parameters for continuum images.");
                }
            }
        }

        self.parameters.use_vel_func_in_raytrace = self.parameters.n_line_images > 0
            && self.parameters.ray_trace_algorithm == RayTraceAlgorithm::Legacy
            && !self.parameters.do_pre_grid;

        self.parameters.edge_vels_available = false;

        if self.parameters.lte_only {
            if self.parameters.n_solve_iters > 0 {
                let msg = "Requesting `n_solve_iters > 0` in LTE only mode \
            will have no effect";
                eprintln!("{msg}");
            } else if self.parameters.n_solve_iters <= self.parameters.n_solve_iters_done {
                let msg = "Requesting `n_solve_iters <= n_solve_iters_done` in LTE only mode \
            will have no effect";
                eprintln!("{msg}");
            }
        }

        let _mol_data: Option<MolData> = if self.parameters.n_species > 0 {
            // defaults::mol_data(self.parameters.n_species)
            todo!()
        } else {
            None
        };

        let _density_power = if self.parameters.sampling_algorithm.is_legacy() {
            constants::DENSITY_EXP
        } else {
            constants::TREE_EXP
        };

        Ok(())
    }
}

/// Deserialize a value in astronomical units (AU) to SI units (meters).
fn from_au<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let val = f64::deserialize(deserializer)?;
    Ok(val * cc::astro::AU_SI)
}

/// Deserialize a value in parsecs (pc) to SI units (meters).
fn from_parsec<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let val = f64::deserialize(deserializer)?;
    Ok(val * cc::astro::PC_SI)
}
