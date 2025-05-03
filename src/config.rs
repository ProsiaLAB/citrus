use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use ndarray::array;
use rgsl::rng::algorithms as GSLRngAlgorithms;
use rgsl::Rng as GSLRng;
use serde::Deserialize;

use crate::collparts::MolData;
use crate::constants as cc;
use crate::defaults::{self, N_DIMS};
use crate::lines::Spec;
use crate::types::{RMatrix, RVector};

/// A container for all the images in the configuration file
type Images = HashMap<String, Image>;
type MolDataVec = Vec<MolData>;

#[derive(Deserialize, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub struct Parameters {
    #[serde(default)]
    pub radius: f64,
    #[serde(default)]
    pub min_scale: f64,
    #[serde(default = "defaults::cmb_temp")]
    pub cmb_temp: f64,
    #[serde(default)]
    pub sink_points: usize,
    #[serde(default)]
    pub p_intensity: usize,
    #[serde(default)]
    pub blend: isize,
    #[serde(default)]
    pub ray_trace_algorithm: isize,
    #[serde(default)]
    pub sampling_algorithm: isize,
    #[serde(default)]
    pub sampling: Sampling,
    #[serde(default)]
    pub lte_only: bool,
    #[serde(default)]
    pub init_lte: bool,
    #[serde(default)]
    pub polarization: bool,
    #[serde(default = "defaults::nthreads")]
    pub nthreads: usize,
    #[serde(default)]
    pub nsolve_iters: usize,
    #[serde(default)]
    pub output_file: String,
    #[serde(default)]
    pub binoutput_file: String,
    #[serde(default)]
    pub grid_file: String,
    #[serde(default)]
    pub pre_grid: String,
    #[serde(default)]
    pub restart: bool,
    #[serde(default)]
    pub dust: Option<String>,
    #[serde(default)]
    pub grid_in_file: String,
    #[serde(default)]
    pub reset_rng: bool,
    #[serde(default)]
    pub do_solve_rte: bool,
    #[serde(default = "defaults::nmol_weights")]
    pub nmol_weights: Vec<f64>,
    #[serde(default = "defaults::dust_weights")]
    pub dust_weights: Vec<f64>,
    #[serde(default = "defaults::grid_density_max_locations")]
    pub grid_density_max_locations: Vec<[f64; 3]>,
    #[serde(default = "defaults::grid_density_max_values")]
    pub grid_density_max_values: Vec<f64>,
    #[serde(default = "defaults::collisional_partner_mol_weights")]
    pub collisional_partner_mol_weights: Vec<f64>,
    /// This list acts as a link between the `N` density
    /// function returns (I'm using here `N` as shorthand for `num_densities`) and the `M`
    /// collision partner ID integers found in the moldatfiles. This allows us to
    /// associate density functions with the collision partner transition rates provided
    /// in the moldatfiles.
    #[serde(default = "defaults::collisional_partner_ids")]
    pub collisional_partner_ids: Vec<usize>,
    #[serde(default = "defaults::grid_data_files")]
    pub grid_data_files: Option<Vec<String>>,
    #[serde(default = "defaults::mol_data_files")]
    pub mol_data_files: Vec<String>,
    /// Essentially this has only cosmetic importance
    /// since it has no effect on the functioning of LIME, only on the names of the
    /// collision partners which are printed to stdout. Its main purpose is to reassure
    /// the user who has provided transition rates for a non-LAMDA collision species in
    /// their moldatfile that they are actually getting these values and not some
    /// mysterious reversion to LAMDA.
    #[serde(default = "defaults::collisional_partner_names")]
    pub collisional_partner_names: Vec<String>,
    #[serde(default = "defaults::grid_out_files")]
    pub grid_out_files: Vec<String>,
    #[serde(default)]
    pub collisional_partner_user_set_flags: isize,
    #[serde(default)]
    pub radius_squ: f64,
    #[serde(default)]
    pub min_scale_squ: f64,
    #[serde(default)]
    pub taylor_cutoff: f64,
    #[serde(default)]
    pub grid_density_global_max: f64,
    #[serde(default)]
    pub ncell: usize,
    #[serde(default)]
    pub n_images: usize,
    #[serde(default)]
    pub n_species: usize,
    #[serde(default)]
    pub num_densities: usize,
    #[serde(default)]
    pub do_pregrid: bool,
    #[serde(default)]
    pub num_grid_density_maxima: i32,
    #[serde(default)]
    pub num_dims: usize,
    #[serde(default)]
    pub n_line_images: usize,
    #[serde(default)]
    pub n_cont_images: usize,
    #[serde(default)]
    pub data_flags: isize,
    #[serde(default)]
    pub n_solve_iters_done: usize,
    #[serde(default)]
    pub do_interpolate_vels: bool,
    #[serde(default)]
    pub use_abun: bool,
    #[serde(default)]
    pub do_mol_calcs: bool,
    #[serde(default)]
    pub use_vel_func_in_raytrace: bool,
    #[serde(default)]
    pub edge_vels_available: bool,
    #[serde(default)]
    pub write_grid_at_stage: [bool; defaults::NUM_OF_GRID_STAGES],
}

#[derive(Deserialize, Debug, Default)]
pub enum Sampling {
    Uniform,
    #[default]
    UniformExact,
    UniformBiased,
}

/// The `Image` struct represents the configuration for an image to be generated.
/// For the field `units`, we have the following mapping:
/// - "Kelvin"
/// - "Jansky per pixel"
/// - SI units
/// - "Lsun per pixel"
/// - Optical depth
#[derive(Deserialize, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub struct Image {
    #[serde(default)]
    pub nchan: usize,
    #[serde(default = "defaults::image_value_i64")]
    pub trans: i64,
    #[serde(default)]
    pub mol_i: usize,
    #[serde(default)]
    pub pixel: Vec<Spec>,
    #[serde(default = "defaults::image_value_f64")]
    pub vel_res: f64,
    #[serde(default = "defaults::image_value_f64")]
    pub img_res: f64,
    #[serde(default = "defaults::image_value_i64")]
    pub pxls: i64,
    #[serde(default)]
    pub unit: Unit,
    #[serde(default)]
    pub img_units: Vec<Unit>,
    #[serde(default)]
    pub num_units: i64,
    #[serde(default = "defaults::image_value_f64")]
    pub freq: f64,
    #[serde(default = "defaults::image_value_f64")]
    pub bandwidth: f64,
    #[serde(default)]
    pub filename: String,
    #[serde(default)]
    pub source_velocity: f64,
    #[serde(default)]
    pub theta: f64,
    #[serde(default)]
    pub phi: f64,
    #[serde(default = "defaults::image_angle")]
    pub inclination: f64,
    #[serde(default = "defaults::image_angle")]
    pub position_angle: f64,
    #[serde(default = "defaults::image_angle")]
    pub azimuth: f64,
    #[serde(default = "defaults::image_value_f64")]
    pub distance: f64,
    #[serde(default)]
    pub rotation_matrix: RMatrix,
    #[serde(default)]
    pub do_interpolate_vels: bool,
    #[serde(default)]
    pub do_line: bool,
    #[serde(default)]
    pub incl: f64,
    #[serde(default)]
    pub posang: f64,
}

#[derive(Deserialize, Debug, Default)]
#[serde(rename_all = "lowercase")]
pub enum Unit {
    Kelvin,
    #[default]
    JskyPerPixel,
    SI,
    LsunPerPixel,
    OpticalDepth,
}

#[derive(Debug, Default)]
pub struct Config {
    pub parameters: Parameters,
    pub images: HashMap<String, Image>,
}

fn to_config(toml_content: &str) -> Result<(Parameters, HashMap<String, Image>)> {
    let config_value: toml::Value =
        toml::from_str(toml_content).context("Failed to parse TOML content")?; // Use context for parsing error

    let config_table = config_value
        .as_table()
        .context("TOML root is not a table")?; // Use context for type error

    let params_value = config_table
        .get("parameters")
        .context("Missing [parameters] section")?; // Use context for missing section

    let parameters: Parameters = params_value
        .clone()
        .try_into()
        .context("Failed to deserialize [parameters] section")?; // Use context for deserialization error

    let mut images: HashMap<String, Image> = HashMap::new();
    for (key, value) in config_table {
        // Skip the parameters section as it's already processed
        if key == "parameters" {
            continue;
        }

        if key.starts_with("image-") {
            let image_params: Image = value
                .clone()
                .try_into()
                .with_context(|| format!("Failed to deserialize image section '{}'", key))?; // Use with_context for dynamic message
            images.insert(key.clone(), image_params);
        } else {
            // Optional: Log or handle unknown top-level keys if necessary
            eprintln!("Warning: Unknown top-level key in config: '{}'", key);
        }
    }

    Ok((parameters, images))
}

pub fn load_config(path: &str) -> Result<Config> {
    let content = fs::read_to_string(path)?; // Propagate file read errors
    let (parameters, images) = to_config(&content)?; // Propagate config parsing errors

    // If both the file read and parsing succeeded, create and return the Config struct
    Ok(Config { parameters, images })
}

/// Parse the configuration file
/// Arguments:
///    - `input_config`: The configuration file
/// # Returns:
///   - A tuple containing the configuration information, image information, and molecular data
///     respectively
///
/// # Errors
///
/// This function will return an error if the number of grid data files is different from the number of species.
///
/// # Panics
///
/// This function will panic if the index is out of bounds for the neighbor array.
pub fn parse_config(input_config: Config) -> Result<(Parameters, Images, Option<MolDataVec>)> {
    // Extract the parameters and images from the parsed TOML file
    let mut pars = input_config.parameters;

    let mut imgs = input_config.images;
    let n_images = imgs.len();

    // Some variables to be used later
    let mut r = [0.0; 3];
    let mut temp_point_density: f64;

    let mut i = 0;
    while i < defaults::MAX_NUM_HIGH && pars.grid_density_max_values[i] >= 0.0 {
        i += 1;
    }
    pars.num_grid_density_maxima = i as i32;
    pars.ncell = pars.p_intensity + pars.sink_points;
    pars.radius_squ = pars.radius * pars.radius;
    pars.min_scale_squ = pars.min_scale * pars.min_scale;
    pars.do_pregrid = !pars.pre_grid.is_empty();
    pars.n_solve_iters_done = 0;
    pars.use_abun = true;
    pars.data_flags = 0;

    if pars.grid_in_file.is_empty() {
        if pars.radius <= 0.0 {
            bail!("Radius must be positive.");
        }
        if pars.min_scale <= 0.0 {
            bail!("Minimum scale must be positive.");
        }
        if pars.p_intensity == 0 {
            bail!("Number of intensity points must be positive.");
        }
        if pars.sink_points == 0 {
            bail!("Number of sink points must be positive.");
        }
    }

    pars.grid_density_global_max = 1.0;
    pars.num_densities = 0;

    let num_func_densities = 1; // Dummy value for now

    if !pars.do_pregrid || pars.restart {
        pars.num_densities = 0;
        if !pars.grid_in_file.is_empty() {
            // Read the grid file in FITS format
            // TODO: Currently not implemented
        }
        if pars.num_densities == 0 {
            // So here is the deal:
            // LIME either asks to supply the number densities (basically from
            // the grid file) or it calculates them a user defined function.
            // These functions had to be written in C by the user.
            // However, I want to do away from this and always ask for a file.
            // Or if that becomes unfeasible, we can ask for parameters
            // from which a function in 3-D space can be generated at runtime.
            // This will be purely written in Rust; with parameters supplied
            // by the user in the configuration file (TOML).

            // Update: 2025-04-30 19:12
            // The Rust implementation of generating a function in 3-D space
            // can be implemented through `macro_rules`. We can just take the parameters
            // as inputs. For example:
            //  ```
            // macro_rules! density {
            //     ($x:expr, $y:expr, $z:expr) => {
            //         $x*$x + $y*$y + $z*$z
            //     }
            // }
            // ```
            // For now, we will just set `num_densities` to 1.
            pars.num_densities = num_func_densities;

            if pars.num_densities == 0 {
                bail!("No density values returned");
            }
        }
    }

    if !pars.do_pregrid || pars.restart || !pars.grid_in_file.is_empty() {
        // In this case, we will need to calculate grid point locations,
        // thus we will need to call the `grid_density()` function
        // Again this implementation requires more thought.
        // At this phase of development, we will just emulate the C code

        // TODO: Impl `density()` function

        // We need some sort of positive value for
        // par.grid_density_global_max before calling the default `grid_density()`
        pars.grid_density_global_max = 1.0;

        // First try `grid_density()` at the origin, where it is often the highest
        temp_point_density = defaults::grid_density(
            &mut r,
            pars.radius_squ,
            pars.num_densities,
            pars.grid_density_global_max,
        );

        // Some sanity checks
        if temp_point_density.is_infinite() || temp_point_density.is_nan() {
            eprintln!("There is a singularity in the grid density function.");
        } else if temp_point_density <= 0.0 {
            eprintln!("The grid density function is zero at the origin.");
        } else if temp_point_density >= pars.grid_density_global_max {
            pars.grid_density_global_max = temp_point_density;
        }

        // Make things work somehow
        if temp_point_density.is_infinite()
            || temp_point_density.is_nan()
            || temp_point_density <= 0.0
        {
            r.iter_mut().take(N_DIMS).for_each(|x| *x = pars.min_scale);
            temp_point_density = defaults::grid_density(
                &mut r,
                pars.radius_squ,
                pars.num_densities,
                pars.grid_density_global_max,
            );

            if !temp_point_density.is_infinite()
                && !temp_point_density.is_nan()
                && temp_point_density > 0.0
            {
                pars.grid_density_global_max = temp_point_density;
            } else {
                // Hmm ok, let's try a spread of random locations
                let rand_gen_opt = GSLRng::new(GSLRngAlgorithms::ranlxs2());

                match rand_gen_opt {
                    Some(mut rand_gen) => {
                        if defaults::FIX_RANDOM_SEEDS {
                            rand_gen.set(140978);
                        } else {
                            rand_gen.set(
                                SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .expect("Time went backwards")
                                    .as_secs() as usize,
                            );
                        }
                        println!("Random number generator initialized.");
                        let mut found_good_value = false;
                        for _ in 0..defaults::NUM_RANDOM_DENS {
                            r.iter_mut().take(N_DIMS).for_each(|x| {
                                *x = pars.radius * (2.0 * GSLRng::uniform(&mut rand_gen) - 1.0)
                            });
                            temp_point_density = defaults::grid_density(
                                &mut r,
                                pars.radius_squ,
                                pars.num_densities,
                                pars.grid_density_global_max,
                            );
                            if !temp_point_density.is_infinite()
                                && !temp_point_density.is_nan()
                                && temp_point_density > 0.0
                            {
                                found_good_value = true;
                                break;
                            }
                        }
                        if found_good_value {
                            if temp_point_density > pars.grid_density_global_max {
                                pars.grid_density_global_max = temp_point_density;
                            }
                        } else if pars.num_grid_density_maxima > 0 {
                            // Test any maxima that user might have provided
                            pars.grid_density_global_max = pars.grid_density_max_values[0];
                            for i in 1..pars.num_grid_density_maxima as usize {
                                if pars.grid_density_max_values[i] > pars.grid_density_global_max {
                                    pars.grid_density_global_max = pars.grid_density_max_values[i];
                                }
                            }
                        } else {
                            bail!("Could not find a non-pathological grid density value.");
                        }
                    }

                    None => {
                        bail!("Could not initialize random number generator.");
                    }
                }
            }
        }
    }

    for i in 0..defaults::NUM_OF_GRID_STAGES {
        if !pars.grid_out_files[i].is_empty() {
            pars.write_grid_at_stage[i] = true;
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

    pars.taylor_cutoff = (24.0 * f64::EPSILON).powf(0.25);
    pars.n_images = n_images;
    pars.num_dims = N_DIMS;

    // Allocate pixel space and parse image information
    for (key, img) in imgs.iter_mut() {
        // if img.unit {
        //     img.num_units = 1; // 1 is Jy/pixel

        //     // Need to allocate space for the pixel data
        //     img.img_units.push(inimgs[key].unit);
        // } else {
        //     /* Otherwise parse image units, populate imgunits array with appropriate
        //      * image identifiers and track number of units requested */
        //     todo!()
        // }

        if img.nchan == 0 && img.vel_res < 0.0 {
            // User has set neither `nchan` nor `vel_res`
            // One of the two are required for a line image
            // Therefore, we assume continuum image
            if pars.polarization {
                img.nchan = 3;
            } else {
                img.nchan = 1;
            }
            if img.freq < 0.0 {
                bail!("You must set a frequency for continuum image.");
            }
            if img.trans > -1 || img.bandwidth > -1.0 {
                let msg = format!(
                    "WARNING: Image {} is a continuum image, but has line parameters set. \
                        These will be ignored.",
                    key
                );
                eprintln!("{}", msg);
                img.do_line = false;
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
            if img.bandwidth > 0.0 && img.vel_res > 0.0 {
                if img.nchan > 0 {
                    let msg = format!(
                        "WARNING: Image {} has both bandwidth and velres set. \
                            nchan will be overwritten.",
                        key
                    );
                    eprintln!("{}", msg);
                }
            } else if img.nchan == 0 || img.bandwidth <= 0.0 && img.vel_res <= 0.0 {
                bail!("You must set either nchan, bandwidth, or velres for a line image.");
            }
            // Check that we have keywords which allow us to calculate the image
            // frequency (if necessary) after reading in the moldata file:
            if img.trans > -1 {
                // User has set of `trans`, posssibly also `freq`
                if img.freq > 0.0 {
                    let msg = format!(
                        "WARNING: Image {} has `trans` set, `freq` will be ignored.",
                        key
                    );
                    eprintln!("{}", msg);
                }
                if pars.n_species > 1 {
                    let msg = format!(
                        "WARNING: Image {} did not have ``mol_i`` set, \
                                Therefore, first molecule will be used.",
                        key
                    );
                    eprintln!("{}", msg);
                    img.mol_i = 0;
                }
            } else if img.freq < 0.0 {
                // User has not set `trans`, nor `freq`
                bail!("You must either set `trans` or `freq` for a line image (and optionally the `mol_i`");
            } // else user has set `freq`
            img.do_line = true;
        } // End of check for line or continuum image
        if img.img_res < 0.0 {
            bail!("You must set image resolution.");
        }
        if img.pxls <= 0 {
            bail!("You must set number of pixels.");
        }
        if img.distance <= 0.0 {
            bail!("You must set distance to source.");
        }
        img.img_res *= cc::ARCSEC_TO_RAD;
        img.pixel = {
            let mut v = Vec::with_capacity((img.pxls * img.pxls) as usize);
            for _ in 0..(img.pxls * img.pxls) {
                v.push(Spec::default());
            }
            v
        };
        for spec in &mut img.pixel {
            spec.intense = RVector::zeros(img.nchan);
            spec.tau = RVector::zeros(img.nchan);
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

        LIME provides two different schemes of {R_1, R_2, R_3}: {PA, phi, theta} and
        {PA, inclination, azimuth}. As an example, consider phi, which is a rotation of
        the model from the observer Z axis towards the X. The matching obs->mod rotation
        matrix is therefore

                    ( cos(ph)  0  -sin(ph) )
                    (                      )
            R_phi = (    0     0     1     ).
                    (                      )
                    ( sin(ph)  0   cos(ph) )

            */

        let do_theta_phi = img.incl < -900.0 || img.azimuth < -900.0 || img.posang < -900.0;
        if do_theta_phi {
            // For the present position angle is not implemented
            // for the theta/phi scheme, so we will just load the
            // the identity matrix
            img.rotation_matrix = RMatrix::eye(3);
        } else {
            // Load position angle matrix
            let cos_pa = img.posang.cos();
            let sin_pa = img.posang.sin();
            img.rotation_matrix = array![
                [cos_pa, -sin_pa, 0.0],
                [sin_pa, cos_pa, 0.0],
                [0.0, 0.0, 1.0]
            ];
        }
        let aux_rotation_matrix = if do_theta_phi {
            // Load phi rotation matrix R_phi
            let cos_phi = img.phi.cos();
            let sin_phi = img.phi.sin();
            array![
                [cos_phi, 0.0, -sin_phi],
                [0.0, 1.0, 0.0],
                [sin_phi, 0.0, cos_phi]
            ]
        } else {
            // Load inclination matrix R_incl
            let cos_incl = (img.incl + std::f64::consts::PI).cos();
            let sin_incl = (img.incl + std::f64::consts::PI).sin();
            array![
                [cos_incl, 0.0, -sin_incl],
                [0.0, 1.0, 0.0],
                [sin_incl, 0.0, cos_incl]
            ]
        };
        // Multiply the two matrices
        let _temp_matrix = img.rotation_matrix.dot(&aux_rotation_matrix);
    }

    pars.n_line_images = 0;
    pars.n_cont_images = 0;
    pars.do_interpolate_vels = false;

    for (_, img) in imgs.iter_mut() {
        if img.do_line {
            pars.n_line_images += 1;
        } else {
            pars.n_cont_images += 1;
        }
        if img.do_interpolate_vels {
            pars.do_interpolate_vels = true;
        }
    }

    if pars.n_cont_images > 0 {
        match &pars.dust {
            Some(dust) if !dust.is_empty() => {
                // Open the dust file and check if it exists and if it is empty
                let path = Path::new(dust);
                if path.exists() {
                    match fs::metadata(path) {
                        Ok(metadata) => {
                            if metadata.len() == 0 {
                                eprintln!("File {} is empty.", dust);
                            }
                        }
                        Err(err) => {
                            eprintln!("Error accessing file {}: {}", dust, err);
                        }
                    }
                }
            }
            _ => {
                bail!("You must set dust parameters for continuum images.");
            }
        }
    }

    pars.use_vel_func_in_raytrace =
        pars.n_line_images > 0 && pars.ray_trace_algorithm == 0 && !pars.do_pregrid;

    pars.edge_vels_available = false;

    if pars.lte_only {
        if pars.nsolve_iters > 0 {
            let msg = "Requesting `nsolve_iters > 0` in LTE only mode \
            will have no effect";
            eprintln!("{}", msg);
        } else if pars.nsolve_iters <= pars.n_solve_iters_done {
            let msg = "Requesting `nsolve_iters <= n_solve_iters_done` in LTE only mode \
            will have no effect";
            eprintln!("{}", msg);
        }
    }

    let mol_data = if pars.n_species > 0 {
        defaults::mol_data(pars.n_species)
    } else {
        None
    };

    // let mut default_density_power: f64;

    // if par.sampling_algorithm == 0 {
    //     default_density_power = defaults::DENSITY_EXP;
    // } else {
    //     default_density_power = defaults::TREE_EXP;
    // }

    Ok((pars, imgs, mol_data))
}
