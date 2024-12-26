use crate::constants::LOCAL_CMB_TEMP_SI;
use crate::{
    MAX_NUM_HIGH, MAX_NUM_OF_COLLISIONAL_PARTNERS, MAX_NUM_OF_SPECIES, NUM_OF_GRID_STAGES,
};
use serde_derive::Deserialize;
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Deserialize)]
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
    pub dust_weights: Vec<f64>,
    pub grid_density_max_locations: Vec<[f64; 3]>,
    pub grid_density_max_values: Vec<f64>,
    pub collisional_partner_mol_weights: Vec<f64>,
    pub collisional_partner_ids: Vec<i64>,
    pub grid_data_file: Vec<String>,
    pub mol_data_file: Vec<String>,
    pub collisional_partner_names: Vec<String>,
    pub grid_out_files: Vec<String>,
}

impl Default for InputParams {
    fn default() -> Self {
        InputParams {
            radius: 0.0,
            min_scale: 0.0,
            p_intensity: 0,
            sink_points: 0,
            dust: String::new(),
            output_file: String::new(),
            binoutput_file: String::new(),
            grid_file: String::new(),
            pre_grid: String::new(),
            restart: false,
            grid_in_file: String::new(),
            collisional_partner_ids: vec![0; MAX_NUM_OF_COLLISIONAL_PARTNERS],
            nmol_weights: vec![-1.0; MAX_NUM_OF_COLLISIONAL_PARTNERS],
            dust_weights: vec![-1.0; MAX_NUM_OF_COLLISIONAL_PARTNERS],
            collisional_partner_mol_weights: vec![-1.0; MAX_NUM_OF_COLLISIONAL_PARTNERS],
            grid_density_max_values: vec![-1.0; MAX_NUM_HIGH],
            grid_density_max_locations: vec![[0.0; 3]; MAX_NUM_HIGH],
            cmb_temp: LOCAL_CMB_TEMP_SI,
            lte_only: 0,
            init_lte: 0,
            sampling_algorithm: 0,
            sampling: 2,
            blend: 0,
            anti_alias: 1,
            polarization: 0,
            nthreads: 1,
            nsolve_iters: 0,
            ray_trace_algorithm: 0,
            reset_rng: false,
            do_solve_rte: false,
            grid_out_files: vec![String::new(); NUM_OF_GRID_STAGES],
            mol_data_file: vec![String::new(); MAX_NUM_OF_SPECIES],
            grid_data_file: vec![String::new(); MAX_NUM_OF_SPECIES],
            collisional_partner_names: vec![String::new(); MAX_NUM_OF_COLLISIONAL_PARTNERS],
        }
    }
}

#[derive(Debug, Deserialize)]
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

impl Default for Image {
    fn default() -> Self {
        Image {
            nchan: 0,
            trans: -1,
            mol_i: -1,
            vel_res: -1.0,
            img_res: -1.0,
            pixels: -1,
            unit: 0,
            freq: -1.0,
            bandwidth: -1.0,
            filename: String::new(),
            units: String::new(),
            source_velocity: 0.0,
            theta: 0.0,
            phi: 0.0,
            inclination: Image::DEFAULT_ANGLE,
            position_angle: Image::DEFAULT_ANGLE,
            azimuth: Image::DEFAULT_ANGLE,
            distance: -1.0,
            do_interpolate_vels: false,
        }
    }
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
