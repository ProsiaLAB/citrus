use std::cell::RefCell;
use std::f64::consts::PI;
use std::rc::Rc;
use std::vec;

use anyhow::Result;
use ndarray_linalg::Solve;
use ndarray_linalg::SVD;

use crate::collparts::MolData;
use crate::config::RayTraceAlgorithm;
use crate::config::{Image, Parameters};
use crate::constants as cc;
use crate::defaults::N_DIMS;
use crate::grid::delaunay;
use crate::grid::{Cell, Grid};
use crate::interface::gas_to_dust_ratio;
use crate::lines::ContinuumLine;
use crate::pops::Populations;
use crate::types::{RMatrix, RVector, UVector};
use crate::utils::{calc_dust_data, get_dtg, get_dust_temp, interpolate_kappa, planck_fn};

// Define error types for the raytrace module
#[derive(Debug)]
pub enum RayThroughCellsError {
    SVDFail,
    NonSpan,
    SolverFail,
    TooManyEntry,
    UnknownError,
    NotFound,
}

#[derive(Debug)]
pub enum RayTraceError {
    EmptyGrid,
    Other(String),
}

impl From<anyhow::Error> for RayTraceError {
    fn from(err: anyhow::Error) -> Self {
        RayTraceError::Other(err.to_string())
    }
}

impl std::fmt::Display for RayTraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RayTraceError::EmptyGrid => write!(f, "Grid is empty"),
            RayTraceError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for RayTraceError {}

/// NOTE: it is assumed that `vertex[i]` is opposite the face that abuts with
/// * `neigh[i]` for all `i`.
#[derive(Debug, Default)]
pub struct Simplex {
    pub id: usize,
    pub vertex: UVector,
    pub centres: RVector,
    pub neigh: Vec<Option<usize>>, // use index, not reference
}

/// This struct is meant to record all relevant information about the
/// intersection between a ray (defined by a direction unit vector 'dir' and a
/// starting position 'r') and a face of a simplex.
#[derive(Debug, Default)]
pub struct Intersect {
    /// The index (in the range {0...N}) of the face (and thus of the opposite
    /// vertex, i.e. the one 'missing' from the bary[] list of this face).
    pub fi: i32,
    /// `> 0` means the ray exits, `< 0` means it enters, `== 0` means the
    /// face is parallel to the ray.
    pub orientation: Orientation,
    pub bary: RVector,
    /// `dist` is defined via `r_int = r + dist*dir`.
    pub dist: f64,
    /// `coll_par` is a measure of how close to any edge of the face `r_int`
    /// lies.
    pub coll_par: f64,
}

#[derive(Debug, Default)]
pub enum Orientation {
    Exit,
    Entry,
    #[default]
    Parallel,
}

impl Intersect {
    fn new() -> Self {
        Intersect {
            fi: -1,
            ..Default::default()
        }
    }
}

#[derive(Debug, Default)]
pub struct Face {
    /// `r` is a list of the the `N` vertices of the face, each of which has `N`
    /// cartesian components.
    pub r: Vec<RVector>,
    /// `simplex_centres` is a convenience pointer array which gives
    /// the location of the geometric centres of the simplexes.
    pub simplex_centres: RVector,
}

#[derive(Debug, Default)]
pub struct FaceBasis {
    pub axes: Vec<RVector>,
    ///  `r` expresses the location of the N vertices of a simplicial polytope face
    /// in N-space, in terms of components along the N-1 orthogonal axes in the
    /// sub-plane of the face. Thus you should malloc r as r[N][N-1].
    pub r: Vec<RVector>,
    pub origin: RVector,
}

impl FaceBasis {
    /// Constructor for `FaceBasis`, initializes vectors dynamically based on [`N_DIMS`].
    fn new() -> Self {
        let n_dims = N_DIMS;
        FaceBasis {
            axes: vec![RVector::zeros(n_dims); n_dims - 1],
            r: vec![RVector::zeros(n_dims - 1); n_dims],
            origin: RVector::zeros(n_dims),
        }
    }

    /// Set a specific axis value.
    fn set_axis(&mut self, axis_index: usize, component_index: usize, value: f64) {
        if axis_index < N_DIMS - 1 && component_index < N_DIMS {
            self.axes[axis_index][component_index] = value;
        } else {
            todo!()
        }
    }

    /// Set a specific vertex value in `r`.
    fn set_vertex(&mut self, vertex_index: usize, component_index: usize, value: f64) {
        if vertex_index < N_DIMS && component_index < N_DIMS - 1 {
            self.r[vertex_index][component_index] = value;
        } else {
            todo!()
        }
    }

    /// Set the origin.
    fn set_origin(&mut self, values: RVector) {
        if values.len() == N_DIMS {
            self.origin = values;
        } else {
            todo!()
        }
    }
}

#[derive(Debug)]
pub struct FaceList {
    /// A collection of faces.
    pub faces: Vec<Rc<RefCell<Face>>>,
    /// A collection of optional references to the faces, up to `N_DIMS + 1`.
    pub face_ptrs: Vec<Option<Rc<RefCell<Face>>>>,
}

impl FaceList {
    /// Constructor for `FaceList`, initializes the struct with empty vectors.
    fn new(num_faces: usize) -> Self {
        FaceList {
            faces: Vec::with_capacity(num_faces),
            face_ptrs: vec![None; N_DIMS + 1],
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct RayData {
    pub x: f64,
    pub y: f64,
    pub intensity: RVector,
    pub tau: RVector,
    pub ppi: usize,
    pub is_inside_image: bool,
}

#[derive(Debug, Default)]
pub struct BaryVelocityBuffer {
    pub num_vertices: usize,
    pub num_edges: usize,
    pub edge_vertex_indices: Vec<[usize; 2]>,
    pub vertex_velocities: Vec<RVector>,
    pub edge_velocities: Vec<RVector>,
    pub entry_cell_bary: RVector,
    pub mid_cell_bary: RVector,
    pub exit_cell_bary: RVector,
    pub shape_fns: RVector,
}

impl BaryVelocityBuffer {
    fn new() -> Self {
        let num_vertices = N_DIMS + 1;
        let num_edges = num_vertices * (num_vertices - 1) / 2;
        let mut vb = BaryVelocityBuffer {
            num_vertices,
            num_edges,
            entry_cell_bary: RVector::zeros(num_vertices),
            mid_cell_bary: RVector::zeros(num_vertices),
            exit_cell_bary: RVector::zeros(num_vertices),
            vertex_velocities: vec![RVector::zeros(num_vertices); N_DIMS],
            edge_vertex_indices: vec![[0; 2]; num_edges],
            edge_velocities: vec![RVector::zeros(num_edges); N_DIMS],
            shape_fns: RVector::zeros(num_vertices + num_edges),
        };
        let mut ei = 0;
        for i0 in 0..vb.num_vertices - 1 {
            for i1 in i0 + 1..vb.num_vertices {
                vb.edge_vertex_indices[ei][0] = i0;
                vb.edge_vertex_indices[ei][1] = i1;
                ei += 1;
            }
        }
        vb
    }
}

#[derive(Debug, Default)]
pub struct GridInterp {
    pub x: [f64; N_DIMS],
    pub magnetic_field: [f64; N_DIMS],
    pub x_component_ray: f64,
    pub mol: Vec<Populations>,
    pub cont: ContinuumLine,
}

pub struct RayTracePreparation {
    pub simplices: Vec<Simplex>,
    pub vel_buffer: Option<BaryVelocityBuffer>,
    pub vertex_coords: RVector,
}

/// Given a simplex dc and the face index (in the range {0...numDims}) fi,
/// this returns the desired information about that face. Note that the ordering of
/// the elements of face.r[] is the same as the ordering of the vertices of the
/// simplex, dc[].vertx[]; just the vertex fi is omitted.
///
/// Note that the element 'centre' of the faceType struct is mean to contain the
/// spatial coordinates of the centre of the simplex, not of the face. This is
/// designed to facilitate orientation of the face and thus to help determine
/// whether rays which cross it are entering or exiting the simplex.
fn extract_face(fi: usize, dc: &[Simplex], dci: usize, vertex_coords: RVector) -> Face {
    let num_faces = N_DIMS + 1;
    let mut face = Face::default();
    let mut vvi = 0;
    for vi in 0..num_faces {
        if vi != fi {
            let gi = dc[dci].vertex[vi];
            for di in 0..N_DIMS {
                face.r[vvi][di] = vertex_coords[N_DIMS * gi + di];
            }
            vvi += 1;
        }
    }
    for di in 0..N_DIMS {
        face.simplex_centres[di] = dc[dci].centres[di];
    }
    face
}

fn get_new_entry_face_index(new_cell: &Simplex, dci: usize) -> Result<isize, RayThroughCellsError> {
    let num_faces = N_DIMS + 1;
    new_cell
        .neigh
        .iter()
        .take(num_faces)
        .enumerate()
        .find_map(|(i, &neigh_idx)| {
            if neigh_idx == Some(dci) {
                Some(i as isize)
            } else {
                None
            }
        })
        .ok_or(RayThroughCellsError::NotFound)
}

fn calc_face_in_nminus(nvertices: usize, face: &Face) -> FaceBasis {
    let mut vs = vec![RVector::zeros(nvertices); nvertices - 1];
    let mut facebasis = FaceBasis::new();
    for di in 0..N_DIMS {
        facebasis.origin[di] = face.r[0][di];
    }
    for (vi, vsi) in vs.iter_mut().enumerate().take(nvertices - 1) {
        for di in 0..N_DIMS {
            vsi[di] = face.r[vi + 1][di] - face.r[0][di];
        }
    }
    for (i, _) in vs.iter().enumerate().take(N_DIMS - 1) {
        for di in 0..N_DIMS {
            facebasis.axes[i][di] = vs[i][di];
        }
        for j in 0..i {
            let mut dotval = 0.0;
            for di in 0..N_DIMS {
                dotval += facebasis.axes[i][di] * facebasis.axes[j][di];
            }
            for di in 0..N_DIMS {
                facebasis.axes[i][di] -= dotval * facebasis.axes[j][di];
            }
        }
        let mut norm = 0.0;
        for di in 0..N_DIMS {
            norm += facebasis.axes[i][di] * facebasis.axes[i][di];
        }
        norm = 1.0 / norm.sqrt();
        for di in 0..N_DIMS {
            facebasis.axes[i][di] *= norm;
        }
    }

    for ddi in 0..N_DIMS - 1 {
        facebasis.r[0][ddi] = 0.0;
    }
    for vi in 1..nvertices {
        for ddi in 0..N_DIMS - 1 {
            let mut dotval = 0.0;
            for di in 0..N_DIMS {
                dotval += vs[vi - 1][di] * facebasis.axes[ddi][di];
            }
            facebasis.r[vi][ddi] = dotval;
        }
    }
    facebasis
}

fn intersect_line_with_face(
    face: Face,
    x: RVector,
    dir: RVector,
    eps: f64,
) -> Result<Intersect, RayThroughCellsError> {
    let eps_inv = 1.0 / eps;
    let mut vs = RMatrix::zeros((N_DIMS - 1, N_DIMS));
    let mut norm = RVector::zeros(N_DIMS);
    let mut px_in_face = RVector::zeros(N_DIMS - 1);
    let mut t_mat = vec![RVector::zeros(N_DIMS - 1); N_DIMS - 1];
    let mut b_vec = RVector::zeros(N_DIMS - 1);
    let mut intersect = Intersect::new();

    for vi in 0..(N_DIMS - 1) {
        for di in 0..N_DIMS {
            vs[[vi, di]] = face.r[vi + 1][di] - face.r[0][di];
        }
    }
    if N_DIMS == 2 {
        norm[0] = -vs[[0, 1]];
        norm[1] = vs[[0, 0]];
    } else if N_DIMS == 3 {
        // Calculate norm via cross product.
        for di in 0..N_DIMS {
            let j = (di + 1) % N_DIMS;
            let k = (di + 2) % N_DIMS;
            norm[di] = vs[[0, j]] * vs[[1, k]] - vs[[0, k]] * vs[[1, j]];
        }
    } else {
        // Calculate norm via SVD
        let svd_res = vs
            .svd(false, true)
            .map_err(|_| RayThroughCellsError::SVDFail)?;
        let svs = svd_res.1;
        let svv = svd_res.2.ok_or(RayThroughCellsError::SVDFail)?;

        let ci = 0;
        let mut ci_of_max = ci;
        let mut max_singular_value = svs[ci];
        for ci in 1..N_DIMS {
            if svs[ci] > max_singular_value {
                ci_of_max = ci;
                max_singular_value = svs[ci];
            }
        }
        let mut ci_of_min: isize = -1;
        for ci in 0..N_DIMS {
            if ci == ci_of_max {
                continue;
            }
            let singular_value = svs[ci];
            if singular_value * eps_inv < max_singular_value {
                if ci_of_min >= 0 {
                    return Err(RayThroughCellsError::NonSpan);
                }
                ci_of_min = ci as isize;
            }
        }
        for di in 0..N_DIMS {
            norm[di] = svv[[di, ci_of_min as usize]];
        }
    }

    let mut test_sum_for_clockwise = 0.0;
    for di in 0..N_DIMS {
        test_sum_for_clockwise += norm[di] * (face.r[0][di] - face.simplex_centres[di]);
    }
    if test_sum_for_clockwise < 0.0 {
        norm *= -1.0;
    }

    let norm_dot_dx = (&norm * &dir).sum();
    intersect.orientation = match norm_dot_dx {
        x if x > 0.0 => Orientation::Exit,
        x if x < 0.0 => Orientation::Entry,
        _ => return Ok(intersect),
    };

    let mut numerator = 0.0;
    for di in 0..N_DIMS {
        numerator += norm[di] * (face.r[0][di] - x[di]);
    }
    intersect.dist = numerator / norm_dot_dx;
    let face_plus_basis = calc_face_in_nminus(N_DIMS, &face);
    for i in 0..N_DIMS - 1 {
        px_in_face[i] = 0.0;
        for di in 0..N_DIMS {
            px_in_face[i] += (x[di] + intersect.dist * dir[di] - face_plus_basis.origin[di])
                * face_plus_basis.axes[i][di];
        }
    }

    if N_DIMS == 2 || N_DIMS == 3 {
        for i in 0..N_DIMS - 1 {
            for j in 0..N_DIMS - 1 {
                t_mat[i][j] = face_plus_basis.r[j + 1][i] - face_plus_basis.r[0][i];
                b_vec[i] = px_in_face[i] - face_plus_basis.r[0][i];
            }
        }
        if N_DIMS == 2 {
            intersect.bary[1] = b_vec[0] / t_mat[0][0];
        } else {
            let det = t_mat[0][0] * t_mat[1][1] - t_mat[0][1] * t_mat[1][0];
            intersect.bary[1] = (t_mat[1][1] * b_vec[0] - t_mat[0][1] * b_vec[1]) / det;
            intersect.bary[2] = (-t_mat[1][0] * b_vec[0] + t_mat[0][0] * b_vec[1]) / det;
        }
    } else {
        let mut t = RMatrix::zeros((N_DIMS - 1, N_DIMS - 1));
        let mut b = RVector::zeros(N_DIMS - 1);
        for i in 0..N_DIMS - 1 {
            for j in 0..N_DIMS - 1 {
                t[[i, j]] = face_plus_basis.r[j + 1][i] - face_plus_basis.r[0][i];
            }
            b[i] = px_in_face[i] - face_plus_basis.r[0][i];
        }
        let x = t.solve(&b).map_err(|_| RayThroughCellsError::SolverFail)?;

        for i in 0..N_DIMS - 1 {
            intersect.bary[i + 1] = x[i];
        }
    }

    intersect.bary[0] = 1.0;
    for i in 1..N_DIMS {
        intersect.bary[0] -= intersect.bary[i];
    }
    let di = 0;
    if intersect.bary[di] < 0.5 {
        intersect.coll_par = intersect.bary[di];
    } else {
        intersect.coll_par = 1.0 - intersect.bary[di];
    }
    for di in 1..N_DIMS {
        if intersect.bary[di] < 0.5 {
            if intersect.bary[di] < intersect.coll_par {
                intersect.coll_par = intersect.bary[di];
            }
        } else if 1.0 - intersect.bary[di] < intersect.coll_par {
            intersect.coll_par = 1.0 - intersect.bary[di];
        }
    }

    Ok(intersect)
}

fn build_ray_cell_chain(cell_visited: &mut [bool], dci: usize) {
    let mut following_single_chain = true;
    // while !following_single_chain {
    //     cell_visited[dci] = true;
    //     todo!()
    // }
}

fn follow_ray_through_cells() {
    todo!()
}

fn calc_grid_cont_dust_opacity(
    gp: &mut [Grid],
    par: &Parameters,
    freq: f64,
    lam_kap: &Option<(RVector, RVector)>,
) -> Result<()> {
    let kappa = if par.dust.is_none() {
        RVector::from_elem(1, 0.0)
    } else if let Some((lam, kap)) = lam_kap {
        RVector::from_elem(1, interpolate_kappa(freq, &lam.view(), &kap.view())?)
    } else {
        RVector::from_elem(1, 0.0)
    };

    let mut knus = RVector::from_elem(1, 0.0);
    let mut dusts = RVector::from_elem(1, 0.0);
    let freqs = RVector::from_elem(1, 0.0);

    for gpi in gp.iter_mut().take(par.ncell) {
        let gas_to_dust_ratio = gas_to_dust_ratio();
        let t_kelvin = get_dust_temp(&gpi.t);
        let dtg = get_dtg(par, &gpi.dens.view(), gas_to_dust_ratio);
        calc_dust_data(
            &mut knus,
            &mut dusts,
            &kappa.view(),
            &freqs.view(),
            t_kelvin,
            dtg,
            1,
        );
        gpi.cont.knu = knus[0];
        gpi.cont.dust = dusts[0];
    }

    Ok(())
}

fn calc_line_amp_sample() {
    todo!()
}

fn calc_line_amp_interp() {
    todo!()
}

fn calc_line_amp_erf() {
    todo!()
}

fn line_plane_intersect() {
    todo!()
}

fn trace_ray() {
    todo!()
}

fn do_barycentric_interpolation() {
    todo!()
}

fn do_segment_interpolation() {
    todo!()
}

fn calc_second_order_shape_functions() {
    todo!()
}

fn do_barycentric_interpolation_vel() {
    todo!()
}

fn do_barycentric_interpolations_vel() {
    todo!()
}

fn do_segment_interpolation_vector() {
    todo!()
}

fn do_segment_interpolation_scalar() {
    todo!()
}

fn trace_ray_smooth() {
    todo!()
}

fn locate_ray_on_image(
    x: &[f64; 2],
    size: f64,
    img_centre_x_pxls: f64,
    img_centre_y_pxls: f64,
    img: &Image,
) -> (bool, usize) {
    let xi = (x[0] / size + img_centre_x_pxls).floor() as i64;
    let yi = (x[1] / size + img_centre_y_pxls).floor() as i64;

    if xi < 0 || xi >= img.pxls || yi < 0 || yi >= img.pxls {
        (false, 0)
    } else {
        (true, (yi * img.pxls + xi) as usize)
    }
}

fn assign_ray_on_image(
    x: &[f64; 2],
    size: f64,
    img_centre_x_pxls: f64,
    img_centre_y_pxls: f64,
    max_num_rays_per_pixel: usize,
    img: &mut Image,
) -> Option<RayData> {
    let (is_inside_image, ppi) =
        locate_ray_on_image(x, size, img_centre_x_pxls, img_centre_y_pxls, img);

    if !is_inside_image {
        return Some(RayData {
            is_inside_image: false,
            ppi: 0,
            x: x[0],
            y: x[1],
            intensity: RVector::zeros(img.nchan),
            tau: RVector::zeros(img.nchan),
        });
    }
    let pixel_data = &mut img.pixel[ppi];
    if max_num_rays_per_pixel > 0 && pixel_data.num_rays >= max_num_rays_per_pixel {
        return None;
    }

    pixel_data.num_rays += 1;

    Some(RayData {
        is_inside_image: true,
        ppi,
        x: x[0],
        y: x[1],
        intensity: RVector::zeros(img.nchan),
        tau: RVector::zeros(img.nchan),
    })
}

fn calc_triangular_barycentric_coords() {
    todo!()
}

fn extract_grid_xs(num_points: usize, gp: &[Grid]) -> RVector {
    let mut xvals = RVector::zeros(num_points * N_DIMS);
    for iul in 0..num_points {
        for ius in 0..N_DIMS {
            xvals[iul * N_DIMS + ius] = gp[iul].x[ius];
        }
    }
    xvals
}

fn convert_cell_to_simplex(cells: &[Cell], par: &Parameters, gp: &[Grid]) -> Vec<Simplex> {
    let mut simplices = Vec::with_capacity(par.ncell);

    // Step 1: Initialize all simplices with geometry and vertex data
    for cell in cells.iter().take(par.ncell) {
        let mut simplex = Simplex {
            id: cell.id,
            neigh: vec![None; N_DIMS + 1],
            ..Default::default()
        };

        for vi in 0..N_DIMS + 1 {
            simplex.vertex[vi] = cell.vertex[vi].as_ref().map(|grid| grid.id).unwrap_or(0) as usize;

            let gi = simplex.vertex[vi];
            for di in 0..N_DIMS {
                simplex.centres[di] += gp[gi].x[di];
            }
        }

        for di in 0..N_DIMS {
            simplex.centres[di] *= 1.0 / (N_DIMS + 1) as f64;
        }

        simplices.push(simplex);
    }

    // Step 2: Assign neighbor indices
    for (icell, cell) in cells.iter().enumerate().take(par.ncell) {
        for vi in 0..N_DIMS + 1 {
            if let Some(neigh_cell) = cell.neigh[vi].as_ref() {
                let neigh_id = neigh_cell.id;
                simplices[icell].neigh[vi] = Some(neigh_id);
            } else {
                simplices[icell].neigh[vi] = None;
            }
        }
    }

    simplices
}

fn get_2d_cells() {
    todo!()
}

fn prepare_raytrace(
    gp: &mut [Grid],
    par: &Parameters,
    img: &Image,
) -> Result<Option<RayTracePreparation>> {
    const NUM_FACES: usize = N_DIMS + 1;
    const N_FACES_INV: f64 = 1.0 / (NUM_FACES as f64);
    match par.ray_trace_algorithm {
        RayTraceAlgorithm::Modern => match delaunay(gp, par.ncell, true, false)? {
            DelaunayResult::Cells(mut cells) => {
                for (icell, cell) in cells.iter_mut().enumerate() {
                    for di in 0..N_DIMS {
                        let sum: f64 = (0..NUM_FACES)
                            .map(|vi| cell.vertex[vi].as_ref().map_or(0.0, |grid| grid.x[di]))
                            .sum();
                        cell.centre[di] = sum * N_FACES_INV;
                    }
                    cell.id = icell;
                }

                let vertex_coords = extract_grid_xs(par.ncell, gp);
                let simplices = convert_cell_to_simplex(&cells, par, gp);
                let vel_buffer = if img.do_line && img.do_interpolate_vels {
                    Some(BaryVelocityBuffer::new())
                } else {
                    None
                };

                Ok(Some(RayTracePreparation {
                    simplices,
                    vel_buffer,
                    vertex_coords,
                }))
            }
            DelaunayResult::NoCells => Ok(None),
        },
        _ => Ok(None),
    }
}

pub fn raytrace(
    img: &mut Image,
    par: &Parameters,
    gp: &mut [Grid],
    mol_data: &[MolData],
    lam_kap: &Option<(RVector, RVector)>,
) -> Result<()> {
    const MAX_NUM_RAYS_PER_PIXEL: usize = 20;
    const NUM_FACES: usize = N_DIMS + 1;
    const NUM_INTERP_PTS: usize = 3;
    const NUM_SEGMENTS: usize = 5;
    const MIN_NUM_RAYS_FOR_AVERAGE: usize = 2;
    const N_FACES_INV: f64 = 1.0 / (NUM_FACES as f64);
    const N_SEGMENTS_INV: f64 = 1.0 / (NUM_SEGMENTS as f64);
    const EPS: f64 = 1.0e-6;
    const N_STEPS_THROUGH_CELL: usize = 10;
    const N_STEPS_INV: f64 = 1.0 / (N_STEPS_THROUGH_CELL as f64);

    let cutoff = par.min_scale * 1.0e-7;

    let pixel_size = img.distance * img.img_res;
    let tot_n_img_pxls = (img.pxls * img.pxls) as usize;
    let img_centre_x_pxls = img.pxls as f64 / 2.0;
    let img_centre_y_pxls = img.pxls as f64 / 2.0;

    if img.do_line {
        if img.trans > -1 {
            img.freq = mol_data[img.mol_i].freq[img.trans as usize];
        }
        if img.bandwidth > 0.0 && img.vel_res > 0.0 {
            img.nchan = (img.bandwidth / (img.vel_res / cc::SPEED_OF_LIGHT_SI * img.freq)) as usize;
        } else if img.bandwidth > 0.0 && img.nchan > 0 {
            img.vel_res = img.bandwidth * cc::SPEED_OF_LIGHT_SI / img.freq / img.nchan as f64;
        } else {
            img.bandwidth = img.nchan as f64 * img.vel_res / cc::SPEED_OF_LIGHT_SI * img.freq;
        }
    }

    let (cmb_freq, cmb_mol_i, cmb_line_i): (f64, Option<usize>, Option<i64>) = if img.do_line {
        let (cmb_mol_i, cmb_line_i) = if img.trans >= 0 {
            (img.mol_i, img.trans)
        } else {
            let mut min_freq = (img.freq - mol_data[0].freq[0]).abs();
            let mut cmb_mol_i = 0;
            let mut cmb_line_i = 0i64;
            for (mol_i, mol_data_i) in mol_data.iter().enumerate().take(par.n_species) {
                for line_i in 0..mol_data[mol_i].nline {
                    if mol_i == 0 && line_i == 0 {
                        continue;
                    }
                    let abs_delta_freq = img.freq - mol_data_i.freq[line_i as usize];
                    if abs_delta_freq < min_freq {
                        min_freq = abs_delta_freq.abs();
                        cmb_mol_i = mol_i;
                        cmb_line_i = line_i as i64;
                    }
                }
            }
            (cmb_mol_i, cmb_line_i)
        };
        (
            mol_data[cmb_mol_i].freq[cmb_line_i as usize],
            Some(cmb_mol_i),
            Some(cmb_line_i),
        )
    } else {
        (img.freq, None, None)
    };

    let local_cmb = planck_fn(cmb_freq, cc::LOCAL_CMB_TEMP_SI);
    calc_grid_cont_dust_opacity(gp, par, cmb_freq, lam_kap)?;

    for ppi in 0..tot_n_img_pxls {
        for ichan in 0..img.nchan {
            img.pixel[ppi].intense[ichan] = 0.0;
            img.pixel[ppi].tau[ichan] = 0.0;
        }
    }

    for ppi in 0..tot_n_img_pxls {
        img.pixel[ppi].num_rays = 0;
    }

    let mut num_pts_in_annulus = 0;
    for gpi in gp.iter().take(par.p_intensity) {
        let rsqu = gpi.x[0] * gpi.x[0] + gpi.x[1] * gpi.x[1];
        if rsqu > (4.0 / 9.0) * par.radius_squ {
            num_pts_in_annulus += 1;
        }
    }
    let num_circle_rays = if num_pts_in_annulus > 0 {
        let circle_spacing =
            (1.0 / 6.0) * par.radius * (5.0 * PI / num_pts_in_annulus as f64).sqrt();
        (2.0 * PI * par.radius / circle_spacing) as usize
    } else {
        0
    };

    let mut xs = [0.0; 2];

    let mut rays = Vec::with_capacity(par.p_intensity + num_circle_rays);

    for gpi in gp.iter().take(par.p_intensity) {
        for (i, xsi) in xs.iter_mut().enumerate() {
            *xsi = (0..N_DIMS)
                .map(|d| gpi.x[d] * img.rotation_matrix[[d, i]])
                .sum();
        }
        if let Some(ray) = assign_ray_on_image(
            &xs,
            pixel_size,
            img_centre_x_pxls,
            img_centre_y_pxls,
            MAX_NUM_RAYS_PER_PIXEL,
            img,
        ) {
            rays.push(ray);
        }
    }

    let num_active_rays_internal = rays.len();
    if num_circle_rays > 0 {
        let scale = 2.0 * PI / (num_circle_rays as f64);
        for i in 0..num_circle_rays {
            let angle = i as f64 * scale;
            xs[0] = par.radius * angle.cos();
            xs[1] = par.radius * angle.sin();
            if let Some(ray) = assign_ray_on_image(
                &xs,
                pixel_size,
                img_centre_x_pxls,
                img_centre_y_pxls,
                MAX_NUM_RAYS_PER_PIXEL,
                img,
            ) {
                rays.push(ray);
            }
        }
    }

    let num_active_rays_minus_one_inv = 1.0 / (num_active_rays_internal - 1) as f64;

    if par.ray_trace_algorithm == RayTraceAlgorithm::Modern {
        match delaunay(gp, par.ncell, true, false) {
            Ok(Some(mut cells)) => {
                for (icell, cell) in cells.iter_mut().enumerate() {
                    for di in 0..N_DIMS {
                        let sum = (0..NUM_FACES)
                            .map(|vi| {
                                // Safely access the value inside the Option<Box<Grid>>
                                cell.vertex[vi]
                                    .as_ref()
                                    .map(|grid| grid.x[di]) // Access the x field of Grid if Some(Box<Grid>)
                                    .unwrap_or(0.0) // Provide a default value if None
                            })
                            .sum::<f64>();

                        cell.centre[di] = sum * N_FACES_INV;
                    }
                    cell.id = icell;
                }
                let vertex_coords = extract_grid_xs(par.ncell, gp);
                let simplices = convert_cell_to_simplex(cells.as_mut_slice(), par, gp);
                if img.do_line && img.do_interpolate_vels {
                    let mut vel_buffer = BaryVelocityBuffer::default();
                    vel_buffer.num_vertices = N_DIMS + 1;
                    vel_buffer.num_edges =
                        vel_buffer.num_vertices * (vel_buffer.num_vertices - 1) / 2;
                    vel_buffer.entry_cell_bary = RVector::zeros(vel_buffer.num_vertices);
                    vel_buffer.mid_cell_bary = RVector::zeros(vel_buffer.num_vertices);
                    vel_buffer.exit_cell_bary = RVector::zeros(vel_buffer.num_vertices);
                    vel_buffer.vertex_velocities =
                        vec![RVector::zeros(vel_buffer.num_vertices); N_DIMS];
                    vel_buffer.edge_vertex_indices = vec![[0; 2]; vel_buffer.num_edges];
                    vel_buffer.edge_velocities = vec![RVector::zeros(vel_buffer.num_edges); N_DIMS];
                    vel_buffer.shape_fns =
                        RVector::zeros(vel_buffer.num_vertices + vel_buffer.num_edges);
                    let mut ei = 0;
                    for i0 in 0..vel_buffer.num_vertices - 1 {
                        for i1 in i0 + 1..vel_buffer.num_vertices {
                            vel_buffer.edge_vertex_indices[ei][0] = i0;
                            vel_buffer.edge_vertex_indices[ei][1] = i1;
                            ei += 1;
                        }
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("Error in delaunay: {:?}", e);
                return Err(e);
            }
        }
    }

    // todo!();

    Ok(())
}
