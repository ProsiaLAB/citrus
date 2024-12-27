// Default functions for `citrus` interface

use crate::{interface, MolData};

pub const DENSITY_EXP: f64 = 0.2;
pub const TREE_EXP: f64 = 2.0;

pub fn grid_density(
    r: &mut Vec<f64>,
    radius_squ: f64,
    num_densities: i32,
    grid_dens_global_max: f64,
) -> f64 {
    let mut val: Vec<f64> = vec![0.0; 99];
    let mut total_density = 0.0;
    let r_squared = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

    if r_squared >= radius_squ {
        return 0.0;
    }

    val[0] = interface::density(r[0], r[1], r[2]);

    for i in 0..num_densities as usize {
        total_density += val[i];
    }
    let frac_density = total_density.powf(DENSITY_EXP) / grid_dens_global_max;

    return frac_density;
}

pub fn mol_data(n_species: i32) -> Option<Vec<MolData>> {
    let mut mol_data: Vec<MolData> = Vec::new();
    for _ in 0..n_species {
        mol_data.push(MolData::default());
    }
    Some(mol_data)
}
