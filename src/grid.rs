use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::{SystemTime, UNIX_EPOCH};

use qhull::sys;
use qhull::Qh;
use rgsl::rng::algorithms as GSLRngAlgorithms;
use rgsl::Rng as GSLRng;

use crate::config::ConfigInfo;
use crate::constants as cc;
use crate::lines::ContinuumLine;
use crate::pops::Populations;
use crate::{defaults, utils};

pub struct Point {
    pub x: [f64; defaults::N_DIMS],
    pub xn: [f64; defaults::N_DIMS],
}

pub struct Grid {
    pub id: i32,
    pub x: [f64; defaults::N_DIMS],
    pub vel: [f64; defaults::N_DIMS],
    pub mag_field: [f64; 3], // Magnetic field can only be 3D
    pub v1: Vec<f64>,
    pub v2: Vec<f64>,
    pub v3: Vec<f64>,
    pub num_neigh: usize,
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

pub fn pre_define(par: &mut ConfigInfo, gp: &mut Vec<Grid>) -> Result<(), Box<dyn Error>> {
    let rand_gen_opt = GSLRng::new(GSLRngAlgorithms::ranlxs2());

    match rand_gen_opt {
        Some(mut rand_gen) => {
            if defaults::FIX_RANDOM_SEEDS {
                rand_gen.set(6611304);
            } else {
                rand_gen.set(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards")
                        .as_secs() as usize,
                );
            }
            par.num_densities = 1;

            for i in 0..par.ncell {
                gp[i].dens = vec![0.0; par.num_densities];
            }

            // Open grid file
            let file = File::open(&par.pre_grid)?;
            let reader = BufReader::new(file);

            for (i, line) in reader.lines().enumerate().take(par.p_intensity) {
                let line = line?;
                let values: Vec<f64> = line
                    .split_whitespace()
                    .map(|v| {
                        v.parse::<f64>()
                            .unwrap_or_else(|_| panic!("Failed to parse {}", v))
                    })
                    .collect();

                if values.len() != 9 {
                    panic!("Error: Expected 9 values, got {}", values.len());
                }

                let id = values[0] as i32;
                if id >= par.p_intensity as i32 {
                    panic!("Error: Invalid grid point ID: {}", id);
                }

                gp[i].id = id;
                gp[i].x[0] = values[1];
                gp[i].x[1] = values[2];
                gp[i].x[2] = values[3];

                gp[i].dens[0] = values[4];

                gp[i].t[0] = values[5];
                gp[i].t[1] = values[5];

                gp[i].vel[0] = values[6];
                gp[i].vel[1] = values[7];
                gp[i].vel[2] = values[8];

                gp[i].dopb_turb = 200.0;

                if let Some(ref mut molecule) = gp[i].mol {
                    molecule[0].abun = 1e-9;
                } else {
                    panic!("Error: No molecular data found");
                }

                gp[i].sink = 0;
                gp[i].dir = Vec::new();
                gp[i].ds = Vec::new();
                gp[i].neigh = Vec::new();
                gp[i].mag_field = [0.0; 3];

                utils::progress_bar(i as f64 / par.p_intensity as f64, 50);
            }

            // check grid densities
            if par.do_mol_calcs {
                check_grid_densities(gp, par);
            }
            // if random generator was set assign densities for the rest of the grid
            let mut i = par.p_intensity; // Initialize `i` manually since we want to control it
            while i < par.ncell {
                let x = 2.0 * GSLRng::uniform(&mut rand_gen) - 1.0;
                let y = 2.0 * GSLRng::uniform(&mut rand_gen) - 1.0;
                let z = 2.0 * GSLRng::uniform(&mut rand_gen) - 1.0;

                if (x * x + y * y + z * z) < 1.0 {
                    let scale = par.radius * (1.0 / (x * x + y * y + z * z)).sqrt();

                    gp[i].id = i as i32;
                    gp[i].x[0] = x * scale;
                    gp[i].x[1] = y * scale;
                    gp[i].x[2] = z * scale;
                    gp[i].sink = 1;

                    // Update the molecule data if it's available
                    if let Some(molecule) = &mut gp[i].mol {
                        molecule[0].abun = 0.0;
                        molecule[0].nmol = 0.0;
                    } else {
                        panic!("Error: No molecular data found");
                    }

                    gp[i].dens[0] = cc::CITRUS_EPS; // Assuming CITRUS_EPS is defined
                    gp[i].t[0] = par.cmb_temp;
                    gp[i].t[1] = par.cmb_temp;
                    gp[i].mag_field = [0.0; 3];
                    gp[i].dopb_turb = 0.0;

                    // Initialize velocity array to 0.0
                    for j in 0..defaults::N_DIMS {
                        gp[i].vel[j] = 0.0;
                    }

                    // Continue to next iteration
                    i += 1;
                } else {
                    // If the condition is not met, retry this iteration
                    // Do not increment `i` when the condition fails
                }
            }
        }
        None => {
            panic!("Failed to create random number generator");
        }
    }

    // call Delaunay triangulation
    delaunay(
        defaults::N_DIMS,
        gp,
        par.ncell as i32,
        false,
        true,
        &Vec::new(),
        0,
    )?;

    Ok(())
}

fn check_grid_densities(gp: &Vec<Grid>, par: &ConfigInfo) {
    let mut warning_already_issued = false;

    for (i, _) in gp.iter().enumerate().take(par.p_intensity) {
        if !warning_already_issued && gp[i].dens[0] < cc::TYPICAL_ISM_DENS {
            warning_already_issued = true;
            let msg = "WARNING: You have a grid point with a density lower than the typical ISM density. \
            This may cause numerical issues. Please check your input file.";
            eprintln!("{}", msg);
        }
    }
}

/// The principal purpose of this function is to perform a Delaunay triangulation
/// for the set of points defined by the input argument 'g'. This is achieved via
/// routines in the 3rd-party package qhull.
///
/// A note about qhull nomenclature: a vertex is what you think it is - i.e., a
/// point; but a facet means in this context a Delaunay triangle (in 2D) or
/// tetrahedron (in 3D). This nomenclature arises because the Delaunay cells are
/// indeed facets (or rather projections of facets) of the convex hull constructed
/// in 1 higher dimension.
///
/// Required elements of structs:
///         struct grid *gp:
///                 .id
///                 .x
///
/// Elements of structs are set as follows:
///         struct grid *gp:
///                 .sink
///                 .numNeigh
///                 .neigh (this is malloc'd too large and at present not
/// realloc'd.)
///
///         cellType *dc (if getCells>0):
///                 .id
///                 .neigh
///                 .vertx
fn delaunay(
    _num_dims: usize,
    gp: &mut Vec<Grid>,
    _num_points: i32,
    _get_cells: bool,
    check_sink: bool,
    _dc: &Vec<Cell>,
    _num_cells: u64,
) -> Result<(), Box<dyn Error>> {
    // pt_array  contains the grid point locations in the format required by qhull.
    let pt_array: Vec<Vec<f64>> = gp.iter().map(|point| point.x.to_vec()).collect();

    let qh = Qh::new_delaunay(pt_array)?;

    let mut indices: Vec<usize> = Vec::new();

    if check_sink {
        for face in qh.all_facets() {
            if !face.upper_delaunay() {
                let neighbors = face.neighbors().ok_or("Failed to get neighbors")?;
                for neighbor in neighbors.iter() {
                    if neighbor.upper_delaunay() {
                        let vertices = neighbor.vertices().ok_or("Failed to get vertices")?;
                        for vertex in vertices.iter() {
                            let ppi = vertex.point_id(&qh);
                            match ppi {
                                Ok(ppi) => {
                                    indices.push(ppi as usize);
                                }
                                Err(_) => {
                                    panic!("Failed to get point ID");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for &index in indices.iter() {
        gp[index].sink = 1;
    }

    for face in qh.all_facets() {
        let vertices = face.vertices().ok_or("Failed to get vertices")?;
        for vertex in vertices.iter() {
            let id = vertex.id() as usize;
            gp[id].num_neigh = vertices.size(&qh);
        }
    }

    Ok(())
}
