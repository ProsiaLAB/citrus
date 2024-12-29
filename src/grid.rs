use std::time::{SystemTime, UNIX_EPOCH};

use rgsl::rng::algorithms as GSLRngAlgorithms;
use rgsl::Rng as GSLRng;

use crate::config::ConfigInfo;
use crate::defaults;
use crate::lines::ContinuumLine;
use crate::pops::Populations;

pub struct Point {
    pub x: [f64; defaults::N_DIMS],
    pub xn: [f64; defaults::N_DIMS],
}

pub struct Grid {
    pub id: i64,
    pub x: [f64; defaults::N_DIMS],
    pub vel: [f64; defaults::N_DIMS],
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
    pub mol: Option<Vec<Populations>>,
    pub cont: Vec<ContinuumLine>,
}

impl Default for Grid {
    fn default() -> Self {
        Grid {
            v1: Vec::new(),
            v2: Vec::new(),
            v3: Vec::new(),
            dir: Vec::new(),
            neigh: Vec::new(),
            w: Vec::new(),
            ds: Vec::new(),
            dens: Vec::new(),
            t: [-1.0; 2],
            mag_field: [0.0; 3],
            conv: 0,
            cont: Vec::new(),
            dopb_turb: 0.0,
            sink: 0,
            nphot: 0,
            num_neigh: 0,
            id: -1,
            mol: None,
            x: [0.0; defaults::N_DIMS],
            vel: [0.0; defaults::N_DIMS],
        }
    }
}

pub struct Cell {
    pub vertex: [Option<Box<Grid>>; defaults::N_DIMS + 1],
    pub neigh: [Option<Box<Cell>>; defaults::N_DIMS * 2],
    pub id: u64,
    pub centre: [f64; defaults::N_DIMS],
}

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
            if defaults::FIX_RANDOM_SEEDS {
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
