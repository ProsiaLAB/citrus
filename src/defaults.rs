// Default functions for `citrus` interface
use crate::collparts::MolData;
use crate::interface;

pub const N_DIMS: usize = 3;

pub const MAX_NUM_OF_SPECIES: usize = 100;
pub const MAX_NUM_OF_IMAGES: usize = 100;
pub const NUM_OF_GRID_STAGES: usize = 5;
pub const MAX_NUM_OF_COLLISIONAL_PARTNERS: usize = 20;
pub const TYPICAL_ISM_DENSITY: f64 = 1e3;
pub const MAX_NUM_HIGH: usize = 10; // ??? What this bro?

pub const FIX_RANDOM_SEEDS: bool = false;

pub const NUM_RANDOM_DENS: usize = 100;

pub const DENSITY_EXP: f64 = 0.2;
pub const TREE_EXP: f64 = 2.0;

pub const RAYS_PER_POINT: i64 = 200;

pub fn grid_density(
    r: &mut [f64],
    radius_squ: f64,
    num_densities: usize,
    grid_dens_global_max: f64,
) -> f64 {
    let mut val: Vec<f64> = vec![0.0; 99];

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
        mol_data.push(MolData::default());
    }
    Some(mol_data)
}
