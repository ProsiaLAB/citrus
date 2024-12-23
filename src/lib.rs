use io::Config;

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
    let config = config::ConfigParams::default();

    let imgs = input_data.images;
    println!("Images: {:?}", imgs);
    println!("Parameters: {:?}", pars);
    println!("Config: {:?}", config);

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
