//! Default functions for `citrus` interface

use prosia_extensions::types::RVector;

use crate::collparts::MolData;
use crate::constants as cc;
use crate::interface;

pub const N_DIMS: usize = 3;

pub const MAX_NUM_OF_SPECIES: usize = 100;
const MAX_NUM_OF_IMAGES: usize = 100;
pub const NUM_OF_GRID_STAGES: usize = 5;
pub const MAX_NUM_OF_COLLISIONAL_PARTNERS: usize = 20;
const TYPICAL_ISM_DENSITY: f64 = 1e3;
pub const MAX_NUM_HIGH: usize = 10; // ??? What this bro?

pub const FIX_RANDOM_SEEDS: bool = false;

pub const NUM_RANDOM_DENS: usize = 100;

pub const DENSITY_EXP: f64 = 0.2;
const TREE_EXP: f64 = 2.0;

pub const RAYS_PER_POINT: i64 = 200;

const DEFAULT_ANGLE: f64 = -999.0;

pub fn collisional_partner_ids() -> Vec<usize> {
    vec![0; MAX_NUM_OF_COLLISIONAL_PARTNERS]
}

pub fn nmol_weights() -> Vec<f64> {
    vec![-1.0; MAX_NUM_OF_COLLISIONAL_PARTNERS]
}

pub fn dust_weights() -> Vec<f64> {
    vec![-1.0; MAX_NUM_OF_COLLISIONAL_PARTNERS]
}

pub fn collisional_partner_mol_weights() -> Vec<f64> {
    vec![-1.0; MAX_NUM_OF_COLLISIONAL_PARTNERS]
}

pub fn grid_density_max_values() -> Vec<f64> {
    vec![-1.0; MAX_NUM_HIGH]
}

pub fn grid_density_max_locations() -> Vec<[f64; 3]> {
    vec![[0.0; 3]; MAX_NUM_HIGH]
}

pub fn cmb_temp() -> f64 {
    cc::LOCAL_CMB_TEMP_SI
}

pub fn nthreads() -> usize {
    1
}

pub fn grid_out_files() -> Vec<String> {
    vec![String::new(); NUM_OF_GRID_STAGES]
}

pub fn mol_data_files() -> Vec<String> {
    vec![String::new(); MAX_NUM_OF_SPECIES]
}

pub fn grid_data_files() -> Option<Vec<String>> {
    Some(vec![String::new(); MAX_NUM_OF_SPECIES])
}

pub fn collisional_partner_names() -> Vec<String> {
    vec![String::new(); MAX_NUM_OF_COLLISIONAL_PARTNERS]
}

pub fn image_value_i64() -> i64 {
    -1
}

pub fn image_value_f64() -> f64 {
    -1.0
}

pub fn image_angle() -> f64 {
    DEFAULT_ANGLE
}

pub fn grid_density(
    r: &mut [f64; 3],
    radius_squ: f64,
    num_densities: usize,
    grid_dens_global_max: f64,
) -> f64 {
    let mut val = RVector::zeros(99);

    let r_squared = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

    if r_squared >= radius_squ {
        return 0.0;
    }

    val[0] = interface::density(r[0], r[1], r[2]);

    let total_density: f64 = val.iter().take(num_densities).sum();

    total_density.powf(DENSITY_EXP) / grid_dens_global_max
}

pub fn mol_data(n_species: usize) -> Option<Vec<MolData>> {
    let mut mol_data: Vec<MolData> = Vec::new();
    for _ in 0..n_species {
        mol_data.push(MolData::new());
    }
    Some(mol_data)
}
