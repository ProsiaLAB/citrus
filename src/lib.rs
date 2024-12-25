use std::fs;
use std::process::exit;
use std::{error::Error, path::Path};
pub mod config;
pub mod constants;
pub mod dims;
pub mod grid;
pub mod io;
pub mod messages;
pub mod raytrace;
pub mod solver;
pub mod source;
pub mod tree;
pub mod utils;

use crate::config::ConfigInfo;
use crate::io::Config;

pub const MAX_NUM_OF_SPECIES: usize = 100;
pub const MAX_NUM_OF_IMAGES: usize = 100;
pub const NUM_OF_GRID_STAGES: usize = 5;
pub const MAX_NUM_OF_COLLISIONAL_PARTNERS: usize = 20;
pub const TYPICAL_ISM_DENSITY: f64 = 1e3;
pub const DENSITY_POWER: f64 = 0.2;
pub const MAX_NUM_HIGH: usize = 10; // ??? What this bro?

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

pub struct Spec {
    pub intense: Vec<f64>,
    pub tau: Vec<f64>,
    pub stokes: [f64; 3],
    pub num_rays: i64,
}

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

pub struct Cell {
    pub vertex: [Option<Box<Grid>>; dims::N_DIMS + 1],
    pub neigh: [Option<Box<Cell>>; dims::N_DIMS * 2],
    pub id: u64,
    pub centre: [f64; dims::N_DIMS],
}

pub fn run(path: &str) -> Result<(), Box<dyn Error>> {
    println!("Welcome to citrus, A verstalie line modelling engine based on LIME.");
    // Load the configuration file
    println!("Loading configuration file: {}", path);
    let input_data = Config::from_toml_file(path)?;

    // Extract parameters now that we are outside the match block
    let inpars = input_data.parameters;
    let mut par = config::ConfigInfo::default();

    let imgs = input_data.images;

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
                            println!("File {} is empty.", filename);
                        } else {
                            println!("File {} exists and is not empty.", filename);
                        }
                    }
                    Err(err) => {
                        println!("Error accessing file {}: {}", filename, err);
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

    println!("{:?}", par.grid_density_max_values); // TODO: Set default values
                                                   // totally forgot ::facepalm::

    let mut i = 0;
    while i < MAX_NUM_HIGH && par.grid_density_max_values[i] >= 0.0 {
        i += 1;
    }
    par.num_grid_density_maxima = i as i32;

    par.ncell = par.p_intensity + par.sink_points;
    par.radius_squ = par.radius * par.radius;
    par.min_scale_squ = par.min_scale * par.min_scale;
    par.n_solve_iters_done = 0;
    par.use_abun = true;
    par.data_flags = 0;

    par.grid_density_global_max = 1.;
    par.num_densities = 0;

    // if !config.do_pregrid || config.restart {
    //     config.num_densities = 0;
    //     if let Some(grid_file) = &config.grid_file {
    //         if let Err(e) = count_density_columns();
    //     }
    // }
    Ok(())
}

// Define a struct to hold the configuration
