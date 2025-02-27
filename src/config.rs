use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::bail;
use anyhow::Result;
use rgsl::rng::algorithms as GSLRngAlgorithms;
use rgsl::Rng as GSLRng;

use crate::collparts::MolData;
use crate::constants as cc;
use crate::defaults;
use crate::io::Config;
use crate::lines::Spec;

type Images = HashMap<String, ImageInfo>;
type MolDataVec = Vec<MolData>;

#[derive(Debug, Default)]
pub struct ConfigInfo {
    pub radius: f64,
    pub min_scale: f64,
    pub cmb_temp: f64,
    pub sink_points: usize,
    pub p_intensity: usize,
    pub blend: i32,
    pub ray_trace_algorithm: i32,
    pub sampling_algorithm: i32,
    pub sampling: i32,
    pub lte_only: bool,
    pub init_lte: bool,
    pub anti_alias: i32,
    pub polarization: bool,
    pub nthreads: i32,
    pub nsolve_iters: i32,
    pub collisional_partner_user_set_flags: i32,
    pub output_file: String,
    pub binoutput_file: String,
    pub grid_file: String,
    pub pre_grid: String,
    pub restart: bool,
    pub dust: String,
    pub grid_in_file: String,
    pub reset_rng: bool,
    pub do_solve_rte: bool,
    pub radius_squ: f64,
    pub min_scale_squ: f64,
    pub taylor_cutoff: f64,
    pub grid_density_global_max: f64,
    pub ncell: usize,
    pub n_images: u32,
    pub n_species: usize,
    pub num_densities: usize,
    pub do_pregrid: bool,
    pub num_grid_density_maxima: i32,
    pub num_dims: usize,
    pub n_line_images: i32,
    pub n_cont_images: i32,
    pub data_flags: i32,
    pub n_solve_iters_done: i32,
    pub do_interpolate_vels: bool,
    pub use_abun: bool,
    pub do_mol_calcs: bool,
    pub use_vel_func_in_raytrace: bool,
    pub edge_vels_available: bool,
    pub nmol_weights: Vec<f64>,
    pub grid_density_max_locations: Vec<[f64; 3]>,
    pub grid_density_max_values: Vec<f64>,
    pub collisional_partner_mol_weights: Vec<f64>,
    pub collisional_partner_ids: Vec<i64>,
    pub grid_data_file: Vec<f64>,
    pub mol_data_file: Vec<String>,
    pub collisional_partner_names: Vec<String>,
    pub grid_out_files: Vec<String>,
    pub write_grid_at_stage: [bool; defaults::NUM_OF_GRID_STAGES],
}

#[derive(Debug, Default)]
pub struct ImageInfo {
    pub do_line: bool,
    pub nchan: i64,
    pub trans: i64,
    pub mol_i: i64,
    pub pixel: Vec<Spec>,
    pub vel_res: f64,
    pub img_res: f64,
    pub pxls: i64,
    pub units: String,
    pub img_units: Vec<i32>,
    pub num_units: i64,
    pub freq: f64,
    pub bandwidth: f64,
    pub filename: String,
    pub source_velocity: f64,
    pub theta: f64,
    pub phi: f64,
    pub incl: f64,
    pub posang: f64,
    pub azimuth: f64,
    pub distance: f64,
    pub rotation_matrix: [[f64; 3]; 3],
    pub do_interpolate_vels: bool,
}

// impl Default for ImageInfo {
//     fn default() -> Self {
//         ImageInfo {
//             do_line: 0,
//             nchan: 0,
//             ..ImageInfo::default()
//         }
//     }
// }

pub fn load_config(path: &str) -> Result<Config> {
    let input_config = Config::from_toml_file(path)?;
    Ok(input_config)
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
pub fn parse_config(input_config: Config) -> Result<(ConfigInfo, Images, Option<MolDataVec>)> {
    // Extract the parameters and images from the parsed TOML file
    let inpars = input_config.parameters;

    let inimgs = input_config.images;
    let n_images = inimgs.len() as u32;

    // Some variables to be used later
    let mut r: Vec<f64> = vec![0.0; 3];
    let mut temp_point_density: f64;
    let mut aux_rotation_matrix: [[f64; 3]; 3];

    let mut par = ConfigInfo::default();
    let mut imgs: Images = HashMap::new();

    for key in inimgs.keys() {
        imgs.insert(key.clone(), ImageInfo::default());
    }

    // Map pars to config
    par.radius = inpars.radius;
    par.min_scale = inpars.min_scale;
    par.p_intensity = inpars.p_intensity;
    par.sink_points = inpars.sink_points;
    par.sampling_algorithm = inpars.sampling_algorithm;
    par.sampling = inpars.sampling;
    par.lte_only = inpars.lte_only;
    par.init_lte = inpars.init_lte;
    par.cmb_temp = inpars.cmb_temp;
    par.blend = inpars.blend;
    par.anti_alias = inpars.anti_alias;
    par.polarization = inpars.polarization;
    par.nthreads = inpars.nthreads;
    par.nsolve_iters = inpars.nsolve_iters;
    par.ray_trace_algorithm = inpars.ray_trace_algorithm;
    par.reset_rng = inpars.reset_rng;
    par.do_solve_rte = inpars.do_solve_rte;
    par.dust = inpars.dust;
    par.output_file = inpars.output_file;
    par.binoutput_file = inpars.binoutput_file;
    par.restart = inpars.restart;
    par.grid_file = inpars.grid_file;
    par.pre_grid = inpars.pre_grid;
    par.grid_in_file = inpars.grid_in_file;

    par.grid_out_files = inpars
        .grid_out_files
        .iter()
        .take(defaults::NUM_OF_GRID_STAGES)
        .cloned()
        .collect();
    par.write_grid_at_stage.fill(false);

    if par.pre_grid.is_empty() && par.restart {
        par.n_species = 0;
        par.grid_data_file.clear();
    } else {
        par.n_species = inpars
            .mol_data_file
            .iter()
            .take(defaults::MAX_NUM_OF_SPECIES)
            .filter(|s| !s.is_empty())
            .count();

        let num_grid_data_files = inpars
            .grid_data_file
            .iter()
            .take(defaults::MAX_NUM_OF_SPECIES)
            .filter(|s| !s.is_empty())
            .count();

        if num_grid_data_files == 0 {
            par.grid_data_file.clear();
        } else if num_grid_data_files != par.n_species {
            bail!("Number of grid data files different from number of species.");
        } else {
            par.grid_data_file = inpars
                .grid_data_file
                .iter()
                .take(par.n_species)
                .flat_map(|s| s.parse::<f64>())
                .collect();
        }
    }

    if par.n_species == 0 {
        par.mol_data_file.clear();
    } else {
        par.mol_data_file = inpars
            .mol_data_file
            .iter()
            .take(par.n_species)
            .filter(|filename| !filename.is_empty())
            .cloned()
            .collect();

        for filename in &par.mol_data_file {
            let path = Path::new(filename);
            if path.exists() {
                match fs::metadata(path) {
                    Ok(metadata) => {
                        if metadata.len() == 0 {
                            eprintln!("File {} is empty.", filename);
                        }
                    }
                    Err(err) => {
                        eprintln!("Error accessing file {}: {}", filename, err);
                    }
                }
            }
        }
    }

    par.collisional_partner_ids = inpars
        .collisional_partner_ids
        .iter()
        .take(defaults::MAX_NUM_OF_COLLISIONAL_PARTNERS)
        .cloned()
        .collect();

    par.nmol_weights = inpars
        .nmol_weights
        .iter()
        .take(defaults::MAX_NUM_OF_COLLISIONAL_PARTNERS)
        .cloned()
        .collect();

    par.collisional_partner_names = inpars
        .collisional_partner_names
        .iter()
        .take(defaults::MAX_NUM_OF_COLLISIONAL_PARTNERS)
        .cloned()
        .collect();

    par.collisional_partner_mol_weights = inpars
        .collisional_partner_mol_weights
        .iter()
        .take(defaults::MAX_NUM_OF_COLLISIONAL_PARTNERS)
        .cloned()
        .collect();

    par.grid_density_max_values = inpars
        .grid_density_max_values
        .iter()
        .take(defaults::MAX_NUM_HIGH)
        .cloned()
        .collect();

    par.grid_density_max_locations = inpars
        .grid_density_max_locations
        .iter()
        .take(defaults::MAX_NUM_HIGH)
        .cloned()
        .collect::<Vec<[f64; 3]>>();

    let mut i = 0;
    while i < defaults::MAX_NUM_HIGH && par.grid_density_max_values[i] >= 0.0 {
        i += 1;
    }
    par.num_grid_density_maxima = i as i32;
    par.ncell = par.p_intensity + par.sink_points;
    par.radius_squ = par.radius * par.radius;
    par.min_scale_squ = par.min_scale * par.min_scale;
    par.do_pregrid = !par.pre_grid.is_empty();
    par.n_solve_iters_done = 0;
    par.use_abun = true;
    par.data_flags = 0;

    if par.grid_in_file.is_empty() {
        if par.radius <= 0.0 {
            bail!("Radius must be positive.");
        }
        if par.min_scale <= 0.0 {
            bail!("Minimum scale must be positive.");
        }
        if par.p_intensity == 0 {
            bail!("Number of intensity points must be positive.");
        }
        if par.sink_points == 0 {
            bail!("Number of sink points must be positive.");
        }
    }

    par.grid_density_global_max = 1.0;
    par.num_densities = 0;

    let num_func_densities = 1; // Dummy value for now

    if !par.do_pregrid || par.restart {
        par.num_densities = 0;
        if !par.grid_in_file.is_empty() {
            // Read the grid file in FITS format
            // TODO: Currently not implemented
        }
        if par.num_densities == 0 {
            // So here is the deal:
            // LIME either asks to supply the number densities (basically from
            // the grid file) or it calculates them a user defined function.
            // These functions had to be written in C by the user.
            // However, I want to do away from this and always ask for a file.
            // Or if that becomes unfeasible, we can ask for parameters
            // from which a function in 3-D space can be generated at runtime.
            // This will be purely written in Rust; with parameters supplied
            // by the user in the configuration file (TOML).

            // For now, we will just set `num_densities` to 1.
            par.num_densities = num_func_densities;

            if par.num_densities == 0 {
                bail!("No density values returned");
            }
        }
    }

    if !par.do_pregrid || par.restart || !par.grid_in_file.is_empty() {
        // In this case, we will need to calculate grid point locations,
        // thus we will need to call the `grid_density()` function
        // Again this implementation requires more thought.
        // At this phase of development, we will just emulate the C code

        // TODO: Impl `density()` function

        // We need some sort of positive value for
        // par.grid_density_global_max before calling the default `grid_density()`
        par.grid_density_global_max = 1.0;

        // First try `grid_density()` at the origin, where it is often the highest
        temp_point_density = defaults::grid_density(
            &mut r,
            par.radius_squ,
            par.num_densities,
            par.grid_density_global_max,
        );

        // Some sanity checks
        if temp_point_density.is_infinite() || temp_point_density.is_nan() {
            eprintln!("There is a singularity in the grid density function.");
        } else if temp_point_density <= 0.0 {
            eprintln!("The grid density function is zero at the origin.");
        } else if temp_point_density >= par.grid_density_global_max {
            par.grid_density_global_max = temp_point_density;
        }

        // Make things work somehow
        if temp_point_density.is_infinite()
            || temp_point_density.is_nan()
            || temp_point_density <= 0.0
        {
            r.iter_mut()
                .take(defaults::N_DIMS)
                .for_each(|x| *x = par.min_scale);
            temp_point_density = defaults::grid_density(
                &mut r,
                par.radius_squ,
                par.num_densities,
                par.grid_density_global_max,
            );

            if !temp_point_density.is_infinite()
                && !temp_point_density.is_nan()
                && temp_point_density > 0.0
            {
                par.grid_density_global_max = temp_point_density;
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
                            r.iter_mut().take(defaults::N_DIMS).for_each(|x| {
                                *x = par.radius * (2.0 * GSLRng::uniform(&mut rand_gen) - 1.0)
                            });
                            temp_point_density = defaults::grid_density(
                                &mut r,
                                par.radius_squ,
                                par.num_densities,
                                par.grid_density_global_max,
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
                            if temp_point_density > par.grid_density_global_max {
                                par.grid_density_global_max = temp_point_density;
                            }
                        } else if par.num_grid_density_maxima > 0 {
                            // Test any maxima that user might have provided
                            par.grid_density_global_max = par.grid_density_max_values[0];
                            for i in 1..par.num_grid_density_maxima as usize {
                                if par.grid_density_max_values[i] > par.grid_density_global_max {
                                    par.grid_density_global_max = par.grid_density_max_values[i];
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
        if !par.grid_out_files[i].is_empty() {
            par.write_grid_at_stage[i] = true;
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

    par.taylor_cutoff = (24.0 * f64::EPSILON).powf(0.25);
    par.n_images = n_images;
    par.num_dims = defaults::N_DIMS;

    // Copy over user set image parameters
    if n_images > 0 {
        for key in inimgs.keys() {
            if let Some(img) = imgs.get_mut(key) {
                img.nchan = inimgs[key].nchan;
                img.trans = inimgs[key].trans;
                img.mol_i = inimgs[key].mol_i;
                img.vel_res = inimgs[key].vel_res;
                img.img_res = inimgs[key].img_res;
                img.pxls = inimgs[key].pxls;
                img.units = inimgs[key].units.clone();
                img.freq = inimgs[key].freq;
                img.bandwidth = inimgs[key].bandwidth;
                img.filename = inimgs[key].filename.clone();
                img.source_velocity = inimgs[key].source_velocity;
                img.theta = inimgs[key].theta;
                img.phi = inimgs[key].phi;
                img.incl = inimgs[key].inclination;
                img.posang = inimgs[key].position_angle;
                img.azimuth = inimgs[key].azimuth;
                img.distance = inimgs[key].distance;
                img.do_interpolate_vels = inimgs[key].do_interpolate_vels;
            }
        }
    }

    // Allocate pixel space and parse image information
    for key in inimgs.keys() {
        if let Some(img) = imgs.get_mut(key) {
            if img.units.is_empty() {
                img.num_units = 1; // 1 is Jy/pixel

                // Need to allocate space for the pixel data
                img.img_units.push(inimgs[key].unit);
            } else {
                /* Otherwise parse image units, populate imgunits array with appropriate
                 * image identifiers and track number of units requested */
                let separator = " ,:_";
                // Check if `units` exist
                let units_str = img.units.clone();
                let tokens: Vec<&str> = units_str.split(separator).collect();
                img.img_units.clear();
                img.num_units = 0;

                for token in tokens.iter() {
                    // Try parsing each token as an integer
                    match token.trim().parse::<i32>() {
                        Ok(unit) => {
                            img.img_units.push(unit);
                        }
                        Err(_) => {
                            bail!("Could not parse image units.");
                        }
                    }
                }
                img.num_units = img.img_units.len() as i64;
            }

            if img.nchan == 0 && img.vel_res < 0.0 {
                // User has set neither `nchan` nor `vel_res`
                // One of the two are required for a line image
                // Therefore, we assume continuum image
                if par.polarization {
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
                } else if img.nchan <= 0 || img.bandwidth <= 0.0 && img.vel_res <= 0.0 {
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
                    if img.mol_i < 0 && par.n_species > 1 {
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
            img.pixel = vec![Spec::default(); (img.pxls * img.pxls) as usize];
            for spec in &mut img.pixel {
                spec.intense = vec![0.0; img.nchan as usize];
                spec.tau = vec![0.0; img.nchan as usize];
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
                img.rotation_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
            } else {
                // Load position angle matrix
                let cos_pa = img.posang.cos();
                let sin_pa = img.posang.sin();
                img.rotation_matrix = [
                    [cos_pa, -sin_pa, 0.0],
                    [sin_pa, cos_pa, 0.0],
                    [0.0, 0.0, 1.0],
                ];
            }
            if do_theta_phi {
                // Load phi rotation matrix R_phi
                let cos_phi = img.phi.cos();
                let sin_phi = img.phi.sin();
                aux_rotation_matrix = [
                    [cos_phi, 0.0, -sin_phi],
                    [0.0, 1.0, 0.0],
                    [sin_phi, 0.0, cos_phi],
                ];
            } else {
                // Load inclination matrix R_incl
                let cos_incl = (img.incl + std::f64::consts::PI).cos();
                let sin_incl = (img.incl + std::f64::consts::PI).sin();
                aux_rotation_matrix = [
                    [cos_incl, 0.0, -sin_incl],
                    [0.0, 1.0, 0.0],
                    [sin_incl, 0.0, cos_incl],
                ];
            }
            // Multiply the two matrices
            let mut temp_matrix = [[0.0; 3]; 3];
            temp_matrix.iter_mut().enumerate().for_each(|(i, row)| {
                row.iter_mut().enumerate().for_each(|(j, cell)| {
                    *cell = img.rotation_matrix[i]
                        .iter()
                        .zip(aux_rotation_matrix.iter().map(|row| row[j]))
                        .map(|(a, b)| a * b)
                        .sum();
                });
            });
        }
    }

    par.n_line_images = 0;
    par.n_cont_images = 0;
    par.do_interpolate_vels = false;

    for key in inimgs.keys() {
        if let Some(img) = imgs.get_mut(key) {
            if img.do_line {
                par.n_line_images += 1;
            } else {
                par.n_cont_images += 1;
            }
            if img.do_interpolate_vels {
                par.do_interpolate_vels = true;
            }
        }
    }

    if par.n_cont_images > 0 {
        if par.dust.is_empty() {
            bail!("You must set dust parameters for continuum images.");
        } else {
            // Open the dust file and check if it exists if it does if it is empty
            let path = Path::new(&par.dust);
            if path.exists() {
                match fs::metadata(path) {
                    Ok(metadata) => {
                        if metadata.len() == 0 {
                            eprintln!("File {} is empty.", par.dust);
                        }
                    }
                    Err(err) => {
                        eprintln!("Error accessing file {}: {}", par.dust, err);
                    }
                }
            }
        }
    }

    par.use_vel_func_in_raytrace =
        par.n_line_images > 0 && par.ray_trace_algorithm == 0 && !par.do_pregrid;

    par.edge_vels_available = false;

    if par.lte_only {
        if par.nsolve_iters > 0 {
            let msg = "Requesting `nsolve_iters > 0` in LTE only mode \
            will have no effect";
            eprintln!("{}", msg);
        } else if par.nsolve_iters <= par.n_solve_iters_done {
            let msg = "Requesting `nsolve_iters <= n_solve_iters_done` in LTE only mode \
            will have no effect";
            eprintln!("{}", msg);
        }
    }

    let mol_data = if par.n_species > 0 {
        defaults::mol_data(par.n_species)
    } else {
        None
    };

    // let mut default_density_power: f64;

    // if par.sampling_algorithm == 0 {
    //     default_density_power = defaults::DENSITY_EXP;
    // } else {
    //     default_density_power = defaults::TREE_EXP;
    // }

    Ok((par, imgs, mol_data))
}
