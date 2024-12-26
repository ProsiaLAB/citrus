use rgsl::rng::algorithms as GSLRngAlgorithms;
use rgsl::Rng as GSLRng;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{error::Error, path::Path};
pub mod config;
pub mod constants;
pub mod defaults;
pub mod dims;
pub mod grid;
pub mod interface;
pub mod io;
pub mod messages;
pub mod raytrace;
pub mod solver;
pub mod source;
pub mod tree;
pub mod utils;

use crate::config::ConfigInfo;
use crate::defaults::grid_density;
use crate::io::Config;

pub const MAX_NUM_OF_SPECIES: usize = 100;
pub const MAX_NUM_OF_IMAGES: usize = 100;
pub const NUM_OF_GRID_STAGES: usize = 5;
pub const MAX_NUM_OF_COLLISIONAL_PARTNERS: usize = 20;
pub const TYPICAL_ISM_DENSITY: f64 = 1e3;
pub const DENSITY_POWER: f64 = 0.2;
pub const MAX_NUM_HIGH: usize = 10; // ??? What this bro?

pub const FIX_RANDOM_SEEDS: bool = false;

pub const NUM_RANDOM_DENS: usize = 100;

pub struct CollisionalPartnerData {
    pub down: Vec<f64>,
    pub temp: Vec<f64>,
    pub collisional_partner_id: i64,
    pub ntemp: i64,
    pub ntrans: i64,
    pub lcl: Vec<i64>,
    pub lcu: Vec<i64>,
    pub density_index: i64,
    pub name: String,
}

pub struct MolData {
    pub nlev: i64,
    pub nline: i64,
    pub npart: i64,
    pub lal: Vec<i64>,
    pub lau: Vec<i64>,
    pub aeinst: Vec<f64>,
    pub freq: Vec<f64>,
    pub beinstu: Vec<f64>,
    pub beinstl: Vec<f64>,
    pub eterm: Vec<f64>,
    pub gstat: Vec<f64>,
    pub gir: Vec<f64>,
    pub cmb: Vec<f64>,
    pub amass: f64,
    pub part: CollisionalPartnerData,
    pub mol_name: String,
}

pub struct Point {
    pub x: [f64; dims::N_DIMS],
    pub xn: [f64; dims::N_DIMS],
}

pub struct Rates {
    pub t_binlow: i64,
    pub interp_coeff: f64,
}

pub struct ContinuumLine {
    pub dust: f64,
    pub knu: f64,
}

pub struct Populations {
    pub pops: Vec<f64>,
    pub spec_num_dens: Vec<f64>,
    pub dopb: f64,
    pub binv: f64,
    pub nmol: f64,
    pub abun: f64,
    pub partner: Vec<Rates>,
    pub cont: Vec<ContinuumLine>,
}
pub struct Grid {
    pub id: i64,
    pub x: [f64; dims::N_DIMS],
    pub vel: [f64; dims::N_DIMS],
    pub mag_field: [f64; 3], // Magnetic field can only be 3D
    pub v1: Vec<f64>,
    pub v2: Vec<f64>,
    pub v3: Vec<f64>,
    pub num_neigh: i64,
    pub dir: Vec<Point>,
    pub neigh: Vec<Vec<Grid>>,
    pub w: Vec<f64>,
    pub sink: i64,
    pub nphot: i64,
    pub conv: i64,
    pub dens: Vec<f64>,
    pub t: [f64; 2],
    pub dopb_turb: f64,
    pub ds: Vec<f64>,
    pub mol: Vec<Populations>,
    pub cont: Vec<ContinuumLine>,
}

#[derive(Debug)]
pub struct Spec {
    pub intense: Vec<f64>,
    pub tau: Vec<f64>,
    pub stokes: [f64; 3],
    pub num_rays: i64,
}

#[derive(Debug, Default)]
pub struct ImageInfo {
    pub do_line: i64,
    pub nchan: i64,
    pub trans: i64,
    pub mol_i: i64,
    pub pixel: Vec<Spec>,
    pub vel_res: f64,
    pub img_res: f64,
    pub pxls: i64,
    pub units: String,
    pub img_units: Vec<i64>,
    pub num_units: i64,
    pub freq: f64,
    pub bandwidth: f64,
    pub filename: String,
    pub source_val: f64,
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

pub struct Cell {
    pub vertex: [Option<Box<Grid>>; dims::N_DIMS + 1],
    pub neigh: [Option<Box<Cell>>; dims::N_DIMS * 2],
    pub id: u64,
    pub centre: [f64; dims::N_DIMS],
}

pub fn init() -> (config::ConfigInfo, ImageInfo) {
    println!("===================================================================");
    println!("Welcome to citrus, A verstalie line modelling engine based on LIME.");
    println!("===================================================================");

    // Initialize parameter and image structures with default values
    let par = config::ConfigInfo::default();
    let img = ImageInfo::default();

    return (par, img);
}

pub fn run(
    path: &str,
    par: &mut ConfigInfo,
    _img: &mut ImageInfo,
    n_images: u32,
) -> Result<(), Box<dyn Error>> {
    // Some variables to be used later
    let mut r: Vec<f64> = vec![0.0; 3];
    let mut temp_point_density: f64;

    // Load the configuration file
    let input_data = Config::from_toml_file(path)?;

    // Extract parameters now that we are outside the match block
    let inpars = input_data.parameters;

    // let imgs = input_data.images;

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
        .take(NUM_OF_GRID_STAGES)
        .filter(|filename| !filename.is_empty())
        .cloned()
        .collect();
    par.write_grid_at_stage.fill(0.0);

    if par.pre_grid.is_empty() && par.restart {
        par.n_species = 0;
        par.grid_data_file.clear();
    } else {
        par.n_species = inpars
            .mol_data_file
            .iter()
            .take(MAX_NUM_OF_SPECIES)
            .filter(|s| !s.is_empty())
            .count() as i32;

        let num_grid_data_files = inpars
            .grid_data_file
            .iter()
            .take(MAX_NUM_OF_SPECIES)
            .filter(|s| !s.is_empty())
            .count() as i32;

        if num_grid_data_files == 0 {
            par.grid_data_file.clear();
        } else if num_grid_data_files != par.n_species {
            return Err("Number of grid data files different from number of species.".into());
        } else {
            par.grid_data_file = inpars
                .grid_data_file
                .iter()
                .take(par.n_species as usize)
                .flat_map(|s| s.parse::<f64>())
                .collect();
        }
    }

    if par.n_species <= 0 {
        par.mol_data_file.clear();
    } else {
        par.mol_data_file = inpars
            .mol_data_file
            .iter()
            .take(par.n_species as usize)
            .filter(|filename| !filename.is_empty())
            .cloned()
            .collect();

        for filename in &par.mol_data_file {
            let path = Path::new(filename);
            if path.exists() {
                match fs::metadata(&path) {
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
        .take(MAX_NUM_OF_COLLISIONAL_PARTNERS)
        .cloned()
        .collect();

    par.nmol_weights = inpars
        .nmol_weights
        .iter()
        .take(MAX_NUM_OF_COLLISIONAL_PARTNERS)
        .cloned()
        .collect();

    par.collisional_partner_names = inpars
        .collisional_partner_names
        .iter()
        .take(MAX_NUM_OF_COLLISIONAL_PARTNERS)
        .cloned()
        .collect();

    par.collisional_partner_mol_weights = inpars
        .collisional_partner_mol_weights
        .iter()
        .take(MAX_NUM_OF_COLLISIONAL_PARTNERS)
        .cloned()
        .collect();

    par.grid_density_max_values = inpars
        .grid_density_max_values
        .iter()
        .take(MAX_NUM_HIGH)
        .cloned()
        .collect();

    par.grid_density_max_locations = inpars
        .grid_density_max_locations
        .iter()
        .take(MAX_NUM_HIGH)
        .cloned()
        .collect::<Vec<[f64; 3]>>();

    let mut i = 0;
    while i < MAX_NUM_HIGH && par.grid_density_max_values[i] >= 0.0 {
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
            return Err("Radius must be positive.".into());
        }
        if par.min_scale <= 0.0 {
            return Err("Minimum scale must be positive.".into());
        }
        if par.p_intensity <= 0 {
            return Err("Number of intensity points must be positive.".into());
        }
        if par.sink_points <= 0 {
            return Err("Number of sink points must be positive.".into());
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
        if par.num_densities <= 0 {
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

            if par.num_densities <= 0 {
                return Err("No density values returned".into());
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
        temp_point_density = grid_density(
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
            for i in 0..dims::N_DIMS {
                r[i] = par.min_scale;
            }
            temp_point_density = grid_density(
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
                        if FIX_RANDOM_SEEDS {
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
                        for _ in 0..NUM_RANDOM_DENS as usize {
                            for i in 0..dims::N_DIMS {
                                r[i] = par.radius * (2.0 * GSLRng::uniform(&mut rand_gen) - 1.0);
                            }
                            temp_point_density = grid_density(
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
                            return Err(
                                "Could not find a non-pathological grid density value.".into()
                            );
                        }
                    }

                    None => {
                        eprintln!("Could not initialize random number generator.");
                    }
                }
            }
        }
    }

    for i in 0..NUM_OF_GRID_STAGES {
        if !par.grid_out_files[i].is_empty() {
            par.write_grid_at_stage[i] = 1.0;
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

    let taylor_cutoff = (24.0 * f64::EPSILON).powf(0.25);
    par.n_images = n_images;
    par.num_dims = dims::N_DIMS;

    // Copy over user set image parameters

    Ok(())
}

// Define a struct to hold the configuration
