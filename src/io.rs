use std::collections::HashMap;
use std::fs;

use anyhow::Result;
use serde::Deserialize;

use crate::constants::LOCAL_CMB_TEMP_SI;
use crate::defaults;
use crate::types::{RVector, UVector};

#[derive(Debug, Deserialize)]
#[serde(default)] // This ensures `Default::default()` is used for missing fields.
pub struct InputParams {
    pub radius: f64,
    pub min_scale: f64,
    pub cmb_temp: f64,
    pub sink_points: usize,
    pub p_intensity: usize,
    pub blend: isize,
    pub ray_trace_algorithm: isize,
    pub sampling_algorithm: isize,
    pub sampling: isize,
    pub lte_only: bool,
    pub init_lte: bool,
    pub anti_alias: isize,
    pub polarization: bool,
    pub nthreads: isize,
    pub nsolve_iters: usize,
    pub output_file: String,
    pub binoutput_file: String,
    pub grid_file: String,
    pub pre_grid: String,
    pub restart: bool,
    pub dust: String,
    pub grid_in_file: String,
    pub reset_rng: bool,
    pub do_solve_rte: bool,
    pub nmol_weights: RVector,
    pub dust_weights: RVector,
    pub grid_density_max_locations: Vec<[f64; 3]>,
    pub grid_density_max_values: RVector,
    pub collisional_partner_mol_weights: RVector,
    pub collisional_partner_ids: UVector,
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
            collisional_partner_ids: UVector::zeros(defaults::MAX_NUM_OF_COLLISIONAL_PARTNERS),
            nmol_weights: -1.0 * RVector::ones(defaults::MAX_NUM_OF_COLLISIONAL_PARTNERS),
            dust_weights: -1.0 * RVector::ones(defaults::MAX_NUM_OF_COLLISIONAL_PARTNERS),
            collisional_partner_mol_weights: -1.0
                * RVector::ones(defaults::MAX_NUM_OF_COLLISIONAL_PARTNERS),
            grid_density_max_values: -1.0 * RVector::ones(defaults::MAX_NUM_HIGH),
            grid_density_max_locations: vec![[0.0; 3]; defaults::MAX_NUM_HIGH],
            cmb_temp: LOCAL_CMB_TEMP_SI,
            lte_only: false,
            init_lte: false,
            sampling_algorithm: 0,
            sampling: 2,
            blend: 0,
            anti_alias: 1,
            polarization: false,
            nthreads: 1,
            nsolve_iters: 0,
            ray_trace_algorithm: 0,
            reset_rng: false,
            do_solve_rte: false,
            grid_out_files: vec![String::new(); defaults::NUM_OF_GRID_STAGES],
            mol_data_file: vec![String::new(); defaults::MAX_NUM_OF_SPECIES],
            grid_data_file: vec![String::new(); defaults::MAX_NUM_OF_SPECIES],
            collisional_partner_names: vec![
                String::new();
                defaults::MAX_NUM_OF_COLLISIONAL_PARTNERS
            ],
        }
    }
}

/// The `Image` struct represents the configuration for an image to be generated.
/// For the field `units`, we have the following mapping:
/// 0: "Kelvin"
/// 1: "Jansky per pixel"
/// 2: SI units
/// 3: "Lsun per pixel"
/// 4: Optical depth
#[derive(Debug, Deserialize)]
#[serde(default)] // This ensures `Default::default()` is used for missing fields.
pub struct Image {
    pub nchan: usize,
    pub trans: i64,
    pub mol_i: i64,
    pub vel_res: f64,
    pub img_res: f64,
    pub pxls: i64,
    pub unit: isize,
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
            pxls: -1,
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
    pub fn from_toml_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}
