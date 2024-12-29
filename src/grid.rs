use rgsl::rng::algorithms as GSLRngAlgorithms;
use rgsl::Rng as GSLRng;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{config::ConfigInfo, Grid, Populations, FIX_RANDOM_SEEDS};

pub struct LinkType {
    pub id: u32,
    pub gis: [u32; 2],
    pub vels: Vec<f64>,
}

pub struct MolInfoType {
    pub mol_name: String,
    pub n_levels: u64,
    pub n_lines: u64,
}

pub struct GridInfoType {
    pub n_internal_points: u32,
    pub n_sink_points: u32,
    pub n_links: u32,
    pub n_nn_indices: u32,
    pub n_dims: u16,
    pub n_species: u16,
    pub n_densities: u16,
    pub n_link_vels: u16,
    pub mols: Vec<MolInfoType>,
}

pub struct KeywordType {
    pub datatype: i64,
    pub keyname: String,
    pub comment: String,
    pub char_value: String,
    pub int_value: i64,
    pub float_value: f64,
    pub double_value: f64,
    pub bool_value: bool,
}

pub fn set_default_grid(num_points: usize, num_species: usize) -> Vec<Grid> {
    let mut gp = Vec::new();
    for _ in 0..num_points {
        let mut g = Grid::default();
        if num_species > 0 {
            g.mol
                .get_or_insert_with(Vec::new)
                .push(Populations::default());
        } else {
            g.mol = None;
        }
        gp.push(g);
    }

    gp
}

pub fn pre_define(par: &mut ConfigInfo, gp: &mut Vec<Grid>) {
    let ran_opt = GSLRng::new(GSLRngAlgorithms::ranlxs2());
    match ran_opt {
        Some(mut ran) => {
            if FIX_RANDOM_SEEDS {
                ran.set(6611304);
            } else {
                ran.set(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards")
                        .as_secs() as usize,
                );
            }
        }
        None => {
            panic!("Failed to create random number generator");
        }
    }

    par.num_densities = 1;

    for i in 0..par.ncell {
        gp[i].dens = vec![0.0; par.num_densities];
    }
}
