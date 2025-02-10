use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::mem::swap;
use std::num::ParseFloatError;
use std::time::{SystemTime, UNIX_EPOCH};

use qhull::helpers::{prepare_delaunay_points, CollectedCoords};
use qhull::QhBuilder;
use rgsl::rng::algorithms as GSLRngAlgorithms;
use rgsl::Rng as GSLRng;

use crate::config::ConfigInfo;
use crate::constants as cc;
use crate::lines::ContinuumLine;
use crate::pops::Populations;
use crate::{defaults, utils};

#[derive(Clone, Default)]
pub struct Point {
    pub x: [f64; defaults::N_DIMS],
    pub xn: [f64; defaults::N_DIMS],
}

#[derive(Clone)]
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
    pub neigh: Vec<Option<Box<Grid>>>,
    pub w: Vec<f64>,
    pub sink: bool,
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
            sink: false,
            nphot: 0,
            num_neigh: 0,
            id: -1,
            mol: None,
            x: [0.0; defaults::N_DIMS],
            vel: [0.0; defaults::N_DIMS],
        }
    }
}

#[derive(Clone)]
pub struct Cell {
    pub vertex: [Option<Box<Grid>>; defaults::N_DIMS + 1],
    pub neigh: [Option<Box<Cell>>; defaults::N_DIMS * 2],
    pub id: u32,
    pub centre: [f64; defaults::N_DIMS],
}

pub struct Link {
    pub id: u32,
    pub gis: [u32; 2],
    pub vels: Vec<f64>,
}

pub struct MoleculeInfo {
    pub mol_name: String,
    pub n_levels: u64,
    pub n_lines: u64,
}

pub struct GridInfo {
    pub n_internal_points: u32,
    pub n_sink_points: u32,
    pub n_links: u32,
    pub n_nn_indices: u32,
    pub n_dims: u16,
    pub n_species: u16,
    pub n_densities: u16,
    pub n_link_vels: u16,
    pub mols: Vec<MoleculeInfo>,
}

#[derive(Default, Debug, Clone)]
pub struct Keyword {
    pub datatype: i64,
    pub keyname: String,
    pub comment: String,
    pub char_value: String,
    pub int_value: i32,
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
                        v.parse::<f64>().map_err(|e| {
                            eprintln!("Failed to parse {}: {}", v, e);
                            e
                        })
                    })
                    .collect::<Result<Vec<f64>, ParseFloatError>>()?;

                if values.len() != 9 {
                    return Err("Error: Expected 9 values".into());
                }

                let id = values[0] as i32;
                if id >= par.p_intensity as i32 {
                    return Err(format!("Error: Invalid grid point ID: {}", id).into());
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
                    return Err("Error: No molecular data found".into());
                }

                gp[i].sink = false;
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
                    gp[i].sink = true;

                    // Update the molecule data if it's available
                    if let Some(molecule) = &mut gp[i].mol {
                        molecule[0].abun = 0.0;
                        molecule[0].nmol = 0.0;
                    } else {
                        return Err("Error: No molecular data found".into());
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
            return Err("Failed to create random number generator".into());
        }
    }

    // call Delaunay triangulation
    delaunay(gp, par.ncell, false, true)?;

    /* We just asked delaunay() to flag any grid points with IDs lower than
     * par->pIntensity (which means their distances from model centre are less
     * than the model radius) but which are nevertheless found to be sink points
     * by virtue of the geometry of the mesh of Delaunay cells. Now we need to
     * reshuffle the list of grid points, then reset par->pIntensity, such that
     * all the non-sink points still have IDs lower than par->pIntensity.
     */

    let n_extra_sinks = reorder_grid(gp, par.ncell)?;
    par.p_intensity -= n_extra_sinks as usize;
    par.p_intensity += n_extra_sinks as usize;

    dist_calc(gp, par.ncell);

    if !par.grid_file.is_empty() {
        write_vtk_unstructured_points(gp, par)?;
    }

    Ok(())
}

pub fn read_or_build_grid(par: &mut ConfigInfo) -> Result<Vec<Grid>, Box<dyn Error>> {
    par.data_flags = 0;
    if !par.grid_in_file.is_empty() {
        read_grid_init(par);
    }

    Ok(Vec::new())
}

fn read_grid_init(par: &mut ConfigInfo) {
    let num_desired_kwds = 3;
    let mut desired_kwds = vec![Keyword::default(); num_desired_kwds];

    desired_kwds[0].datatype = 3; // LIME DOUBLE
    desired_kwds[0].keyname = "RADIUS  ".to_string();

    desired_kwds[1].datatype = 3; // LIME DOUBLE
    desired_kwds[1].keyname = "MINSCALE".to_string();

    desired_kwds[2].datatype = 1; // LIME INT
    desired_kwds[2].keyname = "NSOLITER".to_string();

    read_grid(); // TODO: Implement this function

    par.radius = desired_kwds[0].double_value;
    par.min_scale = desired_kwds[1].double_value;
    par.n_solve_iters_done = desired_kwds[2].int_value;

    par.radius_squ = par.radius * par.radius;
    par.min_scale_squ = par.min_scale * par.min_scale;
}

/// This is designed to be a generic function to read the grid data from file. It is
/// assumed that the data will be stored in several tables of different size,
/// corresponding to the different dimensionalities of the elements of the 'grid'
/// struct. See 'writeGrid' for a description.
///
/// Some sanity checks are performed here and also in the deeper functions, but any
/// check of the validity of the state of completeness of the grid data (as encoded
/// in the returned argument dataFlags) is left to the calling routine.
fn read_grid() {
    unimplemented!()
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
    gp: &mut Vec<Grid>,
    num_points: usize,
    get_cells: bool,
    check_sink: bool,
) -> Result<(), Box<dyn Error>> {
    // pt_array  contains the grid point locations in the format required by qhull.
    let pt_array: Vec<Vec<f64>> = gp.iter().map(|point| point.x.to_vec()).collect();

    let CollectedCoords {
        coords,
        count: _,
        dim,
    } = prepare_delaunay_points(pt_array);
    let qh = QhBuilder::default()
        .delaunay(true)
        .scale_last(true)
        .triangulate(true)
        .build_managed(dim, coords)?;

    let mut indices: Vec<usize> = Vec::new();

    if check_sink {
        for facet in qh.all_facets() {
            if !facet.upper_delaunay() {
                let neighbors = facet.neighbors().ok_or("Failed to get neighbors")?;
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
                                    return Err("Failed to get point ID".into());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for &index in indices.iter() {
        gp[index].sink = true;
    }

    for vertex in qh.vertices() {
        let id = vertex.point_id(&qh);
        match id {
            Ok(id) => {
                let neighbors = vertex.neighbors().ok_or("Failed to get neighbors")?;
                gp[id as usize].num_neigh = neighbors.size(&qh);
                if gp[id as usize].num_neigh <= 0 {
                    return Err(
                        "Error: `qhull` failed silently. A smoother `grid_density` might help."
                            .into(),
                    );
                }
                gp[id as usize].neigh = vec![None; gp[id as usize].num_neigh];
            }
            Err(_) => {
                return Err("Failed to get point ID".into());
            }
        }
    }

    let mut point_ids_this_facet: [i32; defaults::N_DIMS + 1] = [0; defaults::N_DIMS + 1];
    let mut num_cells: u64 = 0;
    for facet in qh.all_facets() {
        if !facet.upper_delaunay() {
            let mut j: usize = 0;
            let vertices = facet.vertices().ok_or("Failed to get vertices")?;
            for vertex in vertices.iter() {
                let point_id_this = vertex.point_id(&qh);
                match point_id_this {
                    Ok(point_id_this) => {
                        point_ids_this_facet[j] = point_id_this;
                        j += 1;
                    }
                    Err(_) => {
                        return Err("Failed to get point ID".into());
                    }
                }
            }
            for i in 0..defaults::N_DIMS + 1 {
                let id_i = point_ids_this_facet[i];
                for j in 0..defaults::N_DIMS + 1 {
                    let id_j = point_ids_this_facet[j];
                    if i != j {
                        let mut k: usize = 0;
                        while gp[id_i as usize].neigh[k].is_some() {
                            match gp[id_i as usize].neigh[k] {
                                Some(ref neigh_grid) => {
                                    let grid_id_j = gp[id_j as usize].id;
                                    if neigh_grid.id != grid_id_j {
                                        k += 1;
                                    } else {
                                        break;
                                    }
                                }
                                None => {
                                    return Err("Error: `qhull` failed silently. A smoother `grid_density` might help.".into());
                                }
                            }
                        }
                    }
                }
            }
            num_cells += 1;
        }
    }

    for ppi in 0..num_points {
        let mut j = 0;
        for k in 0..gp[ppi].num_neigh {
            if gp[ppi].neigh[k].is_some() {
                j += 1;
            }
        }
        gp[ppi].num_neigh = j;
    }

    if get_cells {
        let mut dc: Vec<Box<Cell>> = Vec::with_capacity(num_cells as usize);
        let mut fi: usize = 0;
        for facet in qh.all_facets() {
            if !facet.upper_delaunay() {
                dc[fi].id = facet.id();
                fi += 1;
            }
        }
        let mut fi: usize = 0;
        for facet in qh.all_facets() {
            if !facet.upper_delaunay() {
                let mut i: usize = 0;
                let neighbors = facet.neighbors().ok_or("Failed to get neighbors")?;
                for neighbor in neighbors.iter() {
                    if neighbor.upper_delaunay() {
                        dc[fi].neigh[i] = None;
                    } else {
                        let mut ffi: usize = 0;
                        let mut neighbor_not_found = true;
                        while ffi < num_cells as usize && neighbor_not_found {
                            if dc[ffi].id == neighbor.id() {
                                dc[fi].neigh[i] = Some(dc[ffi].clone());
                                neighbor_not_found = false;
                            }
                            ffi += 1;
                        }
                        if ffi >= num_cells as usize && neighbor_not_found {
                            return Err(
                                "Error: Something went wrong with the Delaunay triangulation"
                                    .into(),
                            );
                        }
                    }
                    i += 1;
                }
                fi += 1;
            }
        }
    }

    Ok(())
}

/// The algorithm works its way up the list of points with one index and down with
/// another. The 'up' travel stops at the 1st sink point it finds, the 'down' at the
/// 1st non-sink point. If at that point the 'up' index is lower in value than the
/// 'down', the points are swapped. This is just a tiny bit tricky because we also
/// need to make sure all the neigh pointers are swapped. That's why we make an
/// ordered list of indices and perform the swaps on that as well.
fn reorder_grid(gp: &mut Vec<Grid>, num_points: usize) -> Result<i32, Box<dyn Error>> {
    let mut n_extra_sinks = 0;
    let mut indices: Vec<usize> = vec![0; num_points];

    for up_i in 0..num_points {
        indices[up_i] = up_i;
    }

    let mut up_i = 0;
    let mut down_i = num_points - 1;
    loop {
        while up_i < num_points && !gp[up_i].sink {
            up_i += 1;
        }
        while down_i > 0 && gp[down_i].sink {
            down_i -= 1;
        }
        if up_i >= down_i {
            break;
        }
        n_extra_sinks += 1;

        let i = indices[down_i];
        indices[down_i] = indices[up_i];
        indices[up_i] = i;

        // do the swap
        let (before, after) = gp.as_mut_slice().split_at_mut(down_i + 1);
        swap(&mut before[down_i], &mut after[up_i - down_i]);

        // retain the id values as sequential
        gp[down_i].id = down_i as i32;
        gp[up_i].id = up_i as i32;
    }

    /*
    Now we sort out the .neigh values. An example of how this should work is as
    follows. Suppose we swapped points 30 and 41. We have fixed up the .id values,
    but the swap is still shown in the 'indices' array. Thus we will have

            gp[30].id == 30 (but all the other data is from 41)
            gp[41].id == 41 (but all the other data is from 30)
            indices[30] == 41
            indices[41] == 30

    Suppose further that the old value of gp[i].neigh[j] is &gp[30]. We detect that
    we need to fix it (change it to &gp[41]) because ngi=&gp[30].id=30 !=
    indices[ngi=30]=41. gp[i].neigh[j] is then reset to &gp[indices[30]] = &gp[41],
    i.e. to point to the same data as used to be in location 30.
      */

    for i in 0..num_points {
        for j in 0..gp[i].num_neigh {
            if let Some(ref mut neigh_grid) = gp[i].neigh[j] {
                let ng_i = neigh_grid.id;
                if ng_i != indices[ng_i as usize] as i32 {
                    let index = indices[ng_i as usize] as usize;
                    let new_neigh = gp[index].clone();
                    gp[i].neigh[j] = Some(Box::new(new_neigh));
                }
            }
        }
    }

    Ok(n_extra_sinks)
}

/// Calculate the distance between grid points
/// The distance between two points is calculated as the Euclidean distance
/// between the two points.
fn dist_calc(gp: &mut Vec<Grid>, num_points: usize) {
    for i in 0..num_points {
        gp[i].dir = vec![Point::default(); gp[i].num_neigh];
        gp[i].ds = vec![0.0; gp[i].num_neigh];
        for k in 0..gp[i].num_neigh {
            for l in 0..3 {
                if let Some(ref neigh) = gp[i].neigh[k] {
                    gp[i].dir[k].x[l] = neigh.x[l] - gp[i].x[l];
                }
            }
            gp[i].ds[k] = (gp[i].dir[k].x[0] * gp[i].dir[k].x[0]
                + gp[i].dir[k].x[1] * gp[i].dir[k].x[1]
                + gp[i].dir[k].x[2] * gp[i].dir[k].x[2])
                .sqrt();

            for l in 0..3 {
                gp[i].dir[k].xn[l] = gp[i].dir[k].x[l] / gp[i].ds[k];
            }
        }
        gp[i].nphot = defaults::RAYS_PER_POINT;
    }
}

/// Write the grid points to a VTK file
/// The VTK file is written in the unstructured points format.
fn write_vtk_unstructured_points(gp: &Vec<Grid>, par: &ConfigInfo) -> Result<(), Box<dyn Error>> {
    let pt_array: Vec<Vec<f64>> = gp.iter().map(|point| point.x.to_vec()).collect();

    let mut file = File::create(&par.grid_file)?;

    // Write the VTK header
    writeln!(file, "# vtk DataFile Version 3.0")?;
    writeln!(file, "citrus grid points")?;
    writeln!(file, "ASCII")?;
    writeln!(file, "DATASET UNSTRUCTURED_GRID")?;
    writeln!(file, "POINTS {} double", par.ncell)?;

    for i in 0..par.ncell {
        writeln!(
            file,
            "{:.6e} {:.6e} {:.6e}",
            gp[i].x[0], gp[i].x[1], gp[i].x[2]
        )?;
    }
    let CollectedCoords {
        coords,
        count: _,
        dim,
    } = prepare_delaunay_points(pt_array);
    let qh = QhBuilder::default()
        .delaunay(true)
        .scale_last(true)
        .is_tracing(0)
        .build_managed(dim, coords)?;

    let mut l: usize = 0;

    for facet in qh.all_facets() {
        if !facet.upper_delaunay() {
            l += 1;
        }
    }

    writeln!(file, "CELLS {} {}", l, 5 * l)?;
    for facet in qh.all_facets() {
        if !facet.upper_delaunay() {
            writeln!(file, "4 ")?;
            let vertices = facet.vertices().ok_or("Failed to get vertices")?;
            for vertex in vertices.iter() {
                let point_id = vertex.point_id(&qh);
                match point_id {
                    Ok(point_id) => {
                        writeln!(file, "{} ", point_id)?;
                    }
                    Err(_) => {
                        return Err("Failed to get point ID".into());
                    }
                }
            }
        }
    }

    writeln!(file, "CELL_TYPES {}", l)?;
    for _ in 0..l {
        writeln!(file, "10")?;
    }
    writeln!(file, "POINT_DATA {}", par.ncell)?;
    writeln!(file, "SCALARS H2_density float 1")?;
    writeln!(file, "LOOKUP_TABLE default")?;
    for i in 0..par.ncell {
        writeln!(file, "{:.6e}", gp[i].dens[0])?;
    }
    writeln!(file, "SCALARS Mol_density float 1")?;
    writeln!(file, "LOOKUP_TABLE default")?;
    if par.n_species > 0 {
        for i in 0..par.ncell {
            let mol_pop = gp[i].mol.as_ref().ok_or("Failed to get molecular data")?;
            writeln!(file, "{:.6e}", mol_pop[0].abun * gp[i].dens[0])?;
        }
    } else {
        for _ in 0..par.ncell {
            writeln!(file, "{:.6e}", 0.0)?;
        }
    }
    writeln!(file, "SCALARS Gas_temperature float 1")?;
    writeln!(file, "LOOKUP_TABLE default")?;
    for i in 0..par.ncell {
        writeln!(file, "{:.6e}", gp[i].t[0])?;
    }
    writeln!(file, "SCALARS velocity float 1")?;
    for i in 0..par.ncell {
        let length = (gp[i].vel[0] * gp[i].vel[0]
            + gp[i].vel[1] * gp[i].vel[1]
            + gp[i].vel[2] * gp[i].vel[2])
            .sqrt();
        if length > 0.0 {
            writeln!(
                file,
                "{:.6e} {:.6e} {:.6e}",
                gp[i].vel[0] / length,
                gp[i].vel[1] / length,
                gp[i].vel[2] / length
            )?;
        } else {
            writeln!(
                file,
                "{:.6e} {:.6e} {:.6e}",
                gp[i].vel[0], gp[i].vel[1], gp[i].vel[2]
            )?;
        }
    }

    Ok(())
}
