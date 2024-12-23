use serde_derive::Deserialize;
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Default, Deserialize)]
#[serde(default)] // This ensures `Default::default()` is used for missing fields.
pub struct InputParams {
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
    pub output_file: String,
    pub binoutput_file: String,
    pub grid_file: String,
    pub pre_grid: String,
    pub restart: bool,
    pub dust: String,
    pub grid_in_file: String,
    pub reset_rng: bool,
    pub do_solve_rte: bool,
    pub nmol_weights: Vec<f64>,
    pub grid_density_max_locations: Vec<[f64; 3]>,
    pub grid_density_max_values: Vec<[f64; 3]>,
    pub collisional_partner_mol_weights: Vec<f64>,
    pub collisional_partner_ids: Vec<f64>,
    pub grid_data_file: Vec<f64>,
    pub mol_data_file: Vec<String>,
    pub collisional_partner_names: Vec<String>,
    pub grid_out_files: Vec<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)] // This ensures `Default::default()` is used for missing fields.
pub struct Image {
    pub nchan: i32,
    pub trans: i32,
    pub mol_i: i32,
    pub vel_res: f64,
    pub img_res: f64,
    pub pixels: i32,
    pub unit: i32,
    pub units: String,
    pub freq: f64,
    pub bandwidth: f64,
    pub filename: String,
    pub source_velocity: f64,
    pub theta: f64,
    pub phi: f64,
    pub inclination: f64,
    pub position_angle: f64,
    pub azimuth: f64,
    pub distance: f64,
    pub do_interpolate_vels: bool,
}

// Implementing a constant for default_angle, as Rust does not support inline static expressions in structs.
impl Image {
    pub const DEFAULT_ANGLE: f64 = -999.0;
}

#[derive(Debug, Deserialize)]
pub struct Config {
    pub parameters: InputParams,
    #[serde(flatten)]
    pub images: HashMap<String, Image>,
}

impl Config {
    /// Reads a TOML file and parses it into the Config struct
    pub fn from_toml_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}
