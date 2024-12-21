use io::Config;

pub mod config;
pub mod constants;
pub mod grid;
pub mod io;
pub mod messages;
pub mod raytrace;
pub mod solver;
pub mod source;
pub mod utils;

pub const MAX_NUM_OF_SPECIES: usize = 100;
pub const MAX_NUM_OF_IMAGES: usize = 100;
pub const NUM_OF_GRID_STAGES: usize = 5;
pub const MAX_NUM_OF_COLLISIONAL_PARTNERS: usize = 20;
pub const TYPICAL_ISM_DENSITY: f64 = 1e3;
pub const DENSITY_POWER: f64 = 0.2;
pub const MAX_NUM_HIGH: usize = 10; // ??? What this bro?

pub fn run(path: &str) {
    println!("Welcome to citrus, A verstalie line modelling engine based on LIME.");
    // Load the configuration file
    let input_data = match Config::from_toml_file(path) {
        Ok(data) => data, // Bind the successfully parsed data to `input_data`
        Err(e) => {
            eprintln!("Error reading the input file: {}", e);
            return; // Exit early if there’s an error
        }
    };

    // Extract parameters now that we are outside the match block
    let pars = input_data.parameters;
    let mut config = ConfigParams::default();

    let imgs = input_data.images;
    println!("Images: {:?}", imgs);

    // // Map pars to config
    // config.radius = pars.radius;
    // config.min_scale = pars.min_scale;
    // config.p_intensity = pars.p_intensity;
    // config.sink_points = pars.sink_points;
    // config.sampling_algorithm = pars.sampling_algorithm;
    // config.sampling = pars.sampling;
    // config.lte_only = pars.lte_only;
    // config.init_lte = pars.init_lte;
    // config.cmb_temp = pars.cmb_temp;
    // config.blend = pars.blend;
    // config.anti_alias = pars.anti_alias;
    // config.polarization = pars.polarization;
    // config.nthreads = pars.nthreads;
    // config.nsolve_iters = pars.nsolve_iters;
    // config.ray_trace_algorithm = pars.ray_trace_algorithm;
    // config.reset_rng = pars.reset_rng;
    // config.do_solve_rte = pars.do_solve_rte;
    // config.dust = pars.dust;
    // config.output_file = pars.output_file;
    // config.binoutput_file = pars.binoutput_file;
    // config.restart = pars.restart;
    // config.grid_file = pars.grid_file;
    // config.pre_grid = pars.pre_grid;
    // config.grid_in_file = pars.grid_in_file;

    // config.ncell = config.p_intensity + config.sink_points;
    // config.radius_squ = config.radius * config.radius;
    // config.min_scale_squ = config.min_scale * config.min_scale;
    // config.n_solve_iters_done = 0;
    // config.use_abun = true;
    // config.data_flags = 0;

    // config.grid_density_global_max = 1.;
    // config.num_densities = 0;

    // if !config.do_pregrid || config.restart {
    //     config.num_densities = 0;
    //     if let Some(grid_file) = &config.grid_file {
    //         if let Err(e) = count_density_columns();
    //     }
    // }
}

// Define a struct to hold the configuration

#[derive(Default, Debug)]
pub struct ConfigParams {
    pub radius: f64,
    pub min_scale: f64,
    pub cmb_temp: f64,
    pub sink_points: i32,
    pub p_intensity: i32,
    pub blend: i32,
    pub ray_trace_algorithm: i32,
    pub sampling_algorithm: i32,
    pub sampling: i32,
    pub lte_only: i32,
    pub init_lte: i32,
    pub anti_alias: i32,
    pub polarization: i32,
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
    pub ncell: i32,
    pub n_images: i32,
    pub n_species: i32,
    pub num_densities: i32,
    pub do_pregrid: bool,
    pub num_grid_density_maxima: i32,
    pub num_dims: i32,
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
    pub grid_density_max_values: Vec<[f64; 3]>,
    pub collisional_partner_mol_weights: Vec<f64>,
    pub collisional_partner_ids: Vec<f64>,
    pub grid_data_file: Vec<f64>,
    pub mol_data_file: Vec<f64>,
    pub collisional_partner_names: Vec<f64>,
    pub grid_out_files: Vec<f64>,
    pub write_grid_at_stage: Vec<f64>,
}

#[derive(Default)]
pub struct CollisionalData {
    pub down: f64,
    pub temp: f64,
    pub partner_id: i32,
    pub ntemp: i32,
    pub ntrans: i32,
    pub lcl: i32,
    pub lcu: i32,
    pub density_index: i32,
    pub name: String,
}

// Create an instance of Image::default() and update any fields, e.g. inclination or position_angle, if desired.

#[derive(Default)]
pub struct MoleculeData {
    pub nlev: i32,
    pub nline: i32,
    pub npart: i32,
    pub lal: i32,
    pub lau: i32,
    pub aeinst: f64,
    pub freq: f64,
    pub beinstu: f64,
    pub beinstl: f64,
    pub eterm: f64,
    pub gstat: f64,
    pub girr: f64,
    pub cmb: f64,
    pub amass: f64,
    pub part: CollisionalData,
    pub mol_name: String,
}

#[derive(Default)]
pub struct Rates {
    pub t_binlow: i32,
    pub interp_coeff: f64,
}

#[derive(Default)]
pub struct ContinuumLine {
    pub dust: f64,
    pub knu: f64,
}

#[derive(Default)]
pub struct Populations {
    pub pops: f64,
    pub spec_num_dens: f64,
    pub dopb: f64,
    pub binv: f64,
    pub nmol: f64,
    pub abun: f64,
    pub partner: Rates,
    pub cont: ContinuumLine,
}

#[derive(Default)]
pub struct LineData {
    pub nlev: i32,
    pub nline: i32,
    pub npart: i32,
    pub lal: i32,
    pub lau: i32,
    pub aeinst: f64,
    pub freq: f64,
    pub beinstu: f64,
    pub beinstl: f64,
    pub eterm: f64,
    pub gstat: f64,
    pub girr: f64,
    pub cmb: f64,
    pub amass: f64,
    pub part: CollisionalData,
    pub mol_name: String,
    pub pops: Populations,
}
