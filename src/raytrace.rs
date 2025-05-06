use std::cell::RefCell;
use std::f64::consts::PI;
use std::mem;
use std::rc::Rc;
use std::vec;

use anyhow::Result;
use ndarray_linalg::Solve;
use ndarray_linalg::SVD;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;

use crate::collparts::MolData;
use crate::config::RayTraceAlgorithm;
use crate::config::{Image, Parameters};
use crate::constants as cc;
use crate::defaults::N_DIMS;
use crate::grid::delaunay;
use crate::grid::DelaunayResult;
use crate::grid::{Cell, Grid};
use crate::interface::gas_to_dust_ratio;
use crate::interface::velocity;
use crate::lines::ContinuumLine;
use crate::pops::Populations;
use crate::source::source_fn_cont;
use crate::source::source_fn_line;
use crate::source::source_fn_polarized;
use crate::types::{RMatrix, RVector, UVector};
use crate::utils::calc_source_fn;
use crate::utils::gauss_line;
use crate::utils::{calc_dust_data, get_dtg, get_dust_temp, interpolate_kappa, planck_fn};

#[derive(Debug)]
pub enum RTCResult {
    NoEntryFaces {
        entry: Intersect,
    },
    Success {
        entry: Intersect,
        chain_cell_ids: Vec<usize>,
        exit_intersects: Vec<Intersect>,
        len_chain_ptrs: usize,
    },
    FailedToBuildChain {
        entry: Intersect,
    },
}

#[derive(Debug)]
pub enum RTCError {
    SVDFail,
    NonSpan,
    SolverFail,
    TooManyEntries,
    UnknownError,
    MultipleCandidates,
    NotFound,
    EmptyGrid,
    Other(String),
}

impl From<anyhow::Error> for RTCError {
    fn from(err: anyhow::Error) -> Self {
        RTCError::Other(err.to_string())
    }
}

impl std::fmt::Display for RTCError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RTCError::SVDFail => write!(f, "SVD decomposition failed"),
            RTCError::NonSpan => write!(f, "Non-spanning simplex"),
            RTCError::SolverFail => write!(f, "Solver failed"),
            RTCError::TooManyEntries => write!(f, "Too many entries"),
            RTCError::UnknownError => write!(f, "Unknown error"),
            RTCError::MultipleCandidates => write!(f, "Multiple candidates"),
            RTCError::NotFound => write!(f, "Not found"),
            RTCError::EmptyGrid => write!(f, "Grid is empty"),
            RTCError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for RTCError {}

#[derive(Debug)]
pub struct ChainContext {
    cell_visited: Vec<bool>,
    isimplex: usize,
    entry_face_index: usize,
    ncells_in_chain: usize,
    len_chain_ptrs: usize,
    chain_of_cell_ids: Vec<usize>,
    cell_exit_intersects: Vec<Intersect>,
}

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
    pub fi: usize,
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

#[derive(Debug, Default, PartialEq)]
pub enum Orientation {
    Exit,
    Entry,
    #[default]
    Parallel,
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

pub struct RTPreparation {
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
fn extract_face(fi: usize, dc: &[Simplex], dci: usize, vertex_coords: &RVector) -> Face {
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

fn get_new_entry_face_index(new_cell: &Simplex, dci: usize) -> Result<usize, RTCError> {
    let num_faces = N_DIMS + 1;
    new_cell
        .neigh
        .iter()
        .take(num_faces)
        .enumerate()
        .find_map(|(i, &neigh_idx)| {
            if neigh_idx == Some(dci) {
                Some(i)
            } else {
                None
            }
        })
        .ok_or(RTCError::NotFound)
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
    face: &Face,
    x: &RVector,
    dir: &RVector,
    eps: f64,
) -> Result<Intersect, RTCError> {
    let eps_inv = 1.0 / eps;
    let mut vs = RMatrix::zeros((N_DIMS - 1, N_DIMS));
    let mut norm = RVector::zeros(N_DIMS);
    let mut px_in_face = RVector::zeros(N_DIMS - 1);
    let mut t_mat = vec![RVector::zeros(N_DIMS - 1); N_DIMS - 1];
    let mut b_vec = RVector::zeros(N_DIMS - 1);
    let mut intersect = Intersect::default();

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
        let svd_res = vs.svd(false, true).map_err(|_| RTCError::SVDFail)?;
        let svs = svd_res.1;
        let svv = svd_res.2.ok_or(RTCError::SVDFail)?;

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
                    return Err(RTCError::NonSpan);
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

    let norm_dot_dx = (&norm * dir).sum();
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
        let x = t.solve(&b).map_err(|_| RTCError::SolverFail)?;

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

fn build_ray_cell_chain(
    cntxt: &mut ChainContext,
    x: &RVector,
    dir: &RVector,
    rtp: &RTPreparation,
    simplices: &[Simplex],
) -> Result<i32, RTCError> {
    const EPS: f64 = 1.0e-6;
    const NUM_FACES: usize = N_DIMS + 1;

    let buffer_size = 1024;
    let mut intersects = Vec::with_capacity(NUM_FACES);
    let mut good_exit_fis = Vec::new();
    let mut marginal_exit_fis = Vec::new();

    let mut following_single_chain = true;
    while following_single_chain {
        cntxt.cell_visited[cntxt.isimplex] = true;
        if cntxt.ncells_in_chain >= cntxt.len_chain_ptrs {
            cntxt.len_chain_ptrs += buffer_size;
            cntxt.chain_of_cell_ids.resize(cntxt.len_chain_ptrs, 0);
            cntxt
                .cell_exit_intersects
                .resize_with(cntxt.len_chain_ptrs, Intersect::default);
        }
        (cntxt.chain_of_cell_ids)[cntxt.ncells_in_chain] = cntxt.isimplex;
        for fi in 0..NUM_FACES {
            if fi != cntxt.entry_face_index
                && (simplices[cntxt.isimplex].neigh[fi].is_none()
                    || simplices[cntxt.isimplex].neigh[fi]
                        .is_some_and(|n| !(cntxt.cell_visited)[n]))
            {
                let face = extract_face(
                    fi,
                    rtp.simplices.as_slice(),
                    cntxt.isimplex,
                    &rtp.vertex_coords,
                );
                let mut intersect = intersect_line_with_face(&face, x, dir, EPS)?;
                intersect.fi = fi;
                if intersect.orientation == Orientation::Exit {
                    if intersect.coll_par - EPS > 0.0 {
                        good_exit_fis.push(fi);
                    } else if intersect.coll_par + EPS > 0.0 {
                        marginal_exit_fis.push(fi);
                    }
                }
                intersects.push(intersect);
            }
        }
        if good_exit_fis.len() > 1 {
            return Err(RTCError::MultipleCandidates);
        } else if good_exit_fis.len() == 1 || marginal_exit_fis.len() == 1 {
            let exit_fi = if good_exit_fis.len() == 1 {
                good_exit_fis[0]
            } else {
                marginal_exit_fis[0]
            };
            (cntxt.cell_exit_intersects)[cntxt.ncells_in_chain] =
                mem::take(&mut intersects[exit_fi]);
            cntxt.ncells_in_chain += 1;
            if simplices[cntxt.isimplex].neigh[exit_fi].is_none() {
                cntxt.chain_of_cell_ids.resize(cntxt.ncells_in_chain, 0);
                cntxt
                    .cell_exit_intersects
                    .resize_with(cntxt.ncells_in_chain, Intersect::default);
                cntxt.len_chain_ptrs = cntxt.ncells_in_chain;
                return Ok(0);
            } else if let Some(neigh_idx) = simplices[cntxt.isimplex].neigh[exit_fi] {
                cntxt.entry_face_index =
                    get_new_entry_face_index(&simplices[neigh_idx], cntxt.isimplex)?;
                cntxt.isimplex = neigh_idx;
            } else {
                return Err(RTCError::NotFound);
            };
        } else {
            following_single_chain = false;
        }
    }

    if marginal_exit_fis.is_empty() {
        return Ok(3);
    }

    let mut i = 0;
    let mut status = 4;
    while i < marginal_exit_fis.len() && status > 0 {
        let exit_fi = marginal_exit_fis[i];
        (cntxt.cell_exit_intersects)[cntxt.ncells_in_chain] = mem::take(&mut intersects[exit_fi]);
        if simplices[cntxt.isimplex].neigh[exit_fi].is_none() {
            cntxt.chain_of_cell_ids.resize(cntxt.ncells_in_chain, 0);
            cntxt
                .cell_exit_intersects
                .resize_with(cntxt.ncells_in_chain, Intersect::default);
            cntxt.len_chain_ptrs = cntxt.ncells_in_chain;
            return Ok(0);
        } else if let Some(neigh_idx) = simplices[cntxt.isimplex].neigh[exit_fi] {
            cntxt.entry_face_index =
                get_new_entry_face_index(&simplices[neigh_idx], cntxt.isimplex)?;
            cntxt.isimplex = neigh_idx;
            status = build_ray_cell_chain(cntxt, x, dir, rtp, simplices)?;
        } else {
            return Err(RTCError::NotFound);
        };

        i += 1;
    }

    Ok(status)
}

fn follow_ray_through_cells(
    x: &RVector,
    dir: &RVector,
    rtp: &RTPreparation,
) -> Result<RTCResult, RTCError> {
    const NUM_FACES: usize = N_DIMS + 1;
    const MAX_NUM_ENTRY_FACES: usize = 100;
    const EPS: f64 = 1.0e-6;

    let mut num_entry_faces = 0;
    let mut entry_simplices = Vec::new();
    let mut entry_fis = Vec::new();
    let mut entry_intersects = Vec::new();

    for (isimplex, simplex) in rtp.simplices.iter().enumerate() {
        for fi in 0..NUM_FACES {
            if simplex.neigh[fi].is_none() {
                let face = extract_face(fi, rtp.simplices.as_slice(), isimplex, &rtp.vertex_coords);
                let mut intersect = intersect_line_with_face(&face, x, dir, EPS)?;
                intersect.fi = fi;
                if intersect.orientation == Orientation::Entry && intersect.coll_par + EPS > 0.0 {
                    if num_entry_faces > MAX_NUM_ENTRY_FACES {
                        return Err(RTCError::TooManyEntries);
                    }
                    entry_simplices.push(isimplex);
                    entry_fis.push(fi);
                    entry_intersects.push(intersect);
                    num_entry_faces += 1;
                }
            }
        }
    }

    if num_entry_faces == 0 {
        return Ok(RTCResult::NoEntryFaces {
            entry: Intersect::default(),
        });
    }

    let mut i = 0;
    let mut status = 1;
    let mut cntxt = ChainContext {
        cell_visited: vec![false; rtp.simplices.len()],
        isimplex: entry_simplices[i],
        entry_face_index: entry_fis[i],
        ncells_in_chain: 0,
        len_chain_ptrs: 0,
        chain_of_cell_ids: Vec::with_capacity(1024),
        cell_exit_intersects: Vec::with_capacity(1024),
    };
    while i < num_entry_faces && status > 0 {
        status = build_ray_cell_chain(&mut cntxt, x, dir, rtp, rtp.simplices.as_slice())?;
        i += 1;
    }

    if status == 0 {
        Ok(RTCResult::Success {
            entry: mem::take(&mut entry_intersects[i - 1]),
            chain_cell_ids: cntxt.chain_of_cell_ids,
            exit_intersects: cntxt.cell_exit_intersects,
            len_chain_ptrs: cntxt.len_chain_ptrs,
        })
    } else {
        Ok(RTCResult::FailedToBuildChain {
            entry: Intersect::default(),
        })
    }
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

/// The bulk velocity of the model material can vary significantly with position,
/// thus so can the value of the line-shape function at a given frequency and
/// direction. The present function calculates 'vfac', an approximate average of the
/// line-shape function along a path of length ds in the direction of the line of
/// sight.
///
/// # Note
/// This is called from within the multi-threaded block.
fn calc_line_amp_sample(
    vfac: &mut f64,
    projection_velocities: &RVector,
    delta_v: f64,
    binv: f64,
    nsteps: usize,
    nsteps_inv: f64,
) -> f64 {
    for i in 0..nsteps {
        let v = delta_v - projection_velocities[i];
        let val = v.abs() * binv;
        if val <= 2500.0 {
            *vfac += -(val * val).exp();
        }
    }
    *vfac *= nsteps_inv;
    *vfac
}

/// The bulk velocity of the model material can vary significantly with position,
/// thus so can the value of the line-shape function at a given frequency and
/// direction. The present function calculates 'vfac', an approximate average of the
/// line-shape function along a path of length ds in the direction of the line of
/// sight.
///
/// # Note
/// This is called from within the multi-threaded block.
fn calc_line_amp_interp() {
    todo!()
}

/// Approximates the average line-shape function along a path segment.
///
/// The bulk velocity of the model material can vary significantly with position,
/// which affects the value of the line-shape function at a given frequency and
/// direction. This function computes `vfac`, an approximate average of the
/// line-shape function along a path of length `ds` in the direction of the
/// line of sight.
///
/// # Note
/// This function is invoked within a multi-threaded context.
fn calc_line_amp_erf() {
    todo!()
}

/// Computes the distance to the next Voronoi face in a given direction.
///
/// Returns `ds`, the (always positive) distance from the current position `x`
/// to the next Voronoi face in the direction of vector `dx`, and `nposn`,
/// the ID of the neighboring grid cell adjacent to that face.
///
/// # Note
/// This function is invoked within a multi-threaded context.
fn line_plane_intersect(
    grid: &Grid,
    x: &RVector,
    dx: &RVector,
    ds: f64,
    cutoff: f64,
) -> (f64, usize) {
    for i in 0..grid.num_neigh {
        let numerator = (grid.x[0] + grid.dir[i].x[0] / 2.0 - x[0]) * grid.dir[i].x[0]
            + (grid.x[1] + grid.dir[i].x[1] / 2.0 - x[1]) * grid.dir[i].x[1]
            + (grid.x[2] + grid.dir[i].x[2] / 2.0 - x[2]) * grid.dir[i].x[2];
        let denominator =
            dx[0] * grid.dir[i].x[0] + dx[1] * grid.dir[i].x[1] + dx[2] * grid.dir[i].x[2];
        if denominator.abs() > 0.0 {
            let newdist = numerator / denominator;
            if newdist < ds && newdist > cutoff {
                if let Some(neigh) = &grid.neigh[i] {
                    return (newdist, neigh.id as usize);
                }
            }
        }
    }
    (ds, 0)
}

/// Evaluates the light intensity along a line of sight for a given image pixel.
///
/// For a specified pixel position, this function computes the total light
/// emitted or absorbed along the corresponding line of sight through the
/// (potentially rotated) model. The computation is performed across multiple
/// frequencies—one per channel in the output image.
///
/// The algorithm follows a similar approach to `calculateJBar()`, which determines
/// the average radiant flux impinging on a grid cell. A notional photon is
/// initiated at the model side nearest to the observer and propagated in the
/// receding direction until it reaches the far side.
///
/// While this approach is conceptually non-physical, it simplifies the computation.
///
/// # Note
/// This function is invoked within a multi-threaded context.
fn trace_ray(
    ray: &mut RayData,
    gp: &[Grid],
    img: &Image,
    par: &Parameters,
    mol_data: &[MolData],
) -> Result<(), RTCError> {
    const N_STEPS_THROUGH_CELL: usize = 10;
    const N_STEPS_INV: f64 = 1.0 / (N_STEPS_THROUGH_CELL as f64);
    let mut projection_velocities = RVector::zeros(N_STEPS_THROUGH_CELL);

    let cutoff = par.min_scale * 1.0e-7;
    let xp = ray.x;
    let yp = ray.y;
    if xp * xp + yp * yp > par.radius_squ {
        return Ok(());
    }

    let mut x = RVector::zeros(N_DIMS);
    let mut dx = RVector::zeros(N_DIMS);

    let zp = -(par.radius_squ - (xp * xp + yp * yp)).sqrt();
    for di in 0..N_DIMS {
        x[di] = xp * img.rotation_matrix[[di, 0]]
            + yp * img.rotation_matrix[[di, 1]]
            + zp * img.rotation_matrix[[di, 2]];
        dx[di] = img.rotation_matrix[[di, 2]];
    }

    let mut dist2 = (x[0] - gp[0].x[0]) * (x[0] - gp[0].x[0])
        + (x[1] - gp[0].x[1]) * (x[1] - gp[0].x[1])
        + (x[2] - gp[0].x[2]) * (x[2] - gp[0].x[2]);
    let mut posn = 0;
    for (i, gpi) in gp.iter().enumerate().take(par.ncell).skip(1) {
        let ndist2 = (x[0] - gpi.x[0]) * (x[0] - gpi.x[0])
            + (x[1] - gpi.x[1]) * (x[1] - gpi.x[1])
            + (x[2] - gpi.x[2]) * (x[2] - gpi.x[2]);
        if ndist2 < dist2 {
            posn = i;
            dist2 = ndist2;
        }
    }
    let mut col = 0.0;
    while col < 2.0 * zp.abs() {
        let (ds, nposn) = line_plane_intersect(&gp[posn], &x, &dx, -2.0 * zp - col, cutoff);
        if par.polarization {
            let (snu_pol, alpha) =
                source_fn_polarized(&gp[posn].mag_field, &gp[posn].cont, &img.rotation_matrix)?;
            let dtau = alpha * ds;
            let (mut remnant_snu, _) = calc_source_fn(dtau, par.taylor_cutoff);
            remnant_snu *= ds;
            for (stokesi, snu_poli) in snu_pol.iter().enumerate().take(img.nchan) {
                let brightness_increment = (-ray.tau[stokesi]).exp() * remnant_snu * snu_poli;
                ray.intensity[stokesi] += brightness_increment;
                ray.tau[stokesi] += dtau;
            }
        } else if img.do_line && par.use_vel_func_in_raytrace {
            for i in 0..N_STEPS_THROUGH_CELL {
                let d = i as f64 * ds * N_STEPS_INV;
                let vel = velocity(x[0] + (dx[0] * d), x[1] + (dx[1] * d), x[2] + (dx[2] * d));
                projection_velocities[i] = dx.dot(&vel);
            }
            let mut cont_jnu = 0.0;
            let mut cont_alpha = 0.0;
            (cont_jnu, cont_alpha) = source_fn_cont(cont_jnu, cont_alpha, &gp[posn].cont);
            for ichan in 0..img.nchan {
                let mut jnu = cont_jnu;
                let mut alpha = cont_alpha;
                let v_this_chan = (ichan as f64 - (img.nchan - 1) as f64 * 0.5) * img.vel_res;
                if img.do_line {
                    for moli in 0..par.n_species {
                        for linei in 0..mol_data[moli].nline {
                            if mol_data[moli].freq[linei] > img.freq - img.bandwidth * 0.5
                                && mol_data[moli].freq[linei] < img.freq + img.bandwidth * 0.5
                            {
                                let line_red_shift = if img.trans > -1 {
                                    (mol_data[moli].freq[img.trans as usize]
                                        - mol_data[moli].freq[linei])
                                        / mol_data[moli].freq[img.trans as usize]
                                        * cc::SPEED_OF_LIGHT_SI
                                } else {
                                    (img.freq - mol_data[moli].freq[linei])
                                        / mol_data[moli].freq[linei]
                                        * cc::SPEED_OF_LIGHT_SI
                                };
                                let delta_v = v_this_chan - img.source_velocity - line_red_shift;
                                let binv = if let Some(mols) = &gp[posn].mol {
                                    mols[moli].binv
                                } else {
                                    // This should never happen
                                    return Err(RTCError::EmptyGrid);
                                };
                                let vfac = if par.use_vel_func_in_raytrace {
                                    calc_line_amp_sample(
                                        &mut 0f64,
                                        &projection_velocities,
                                        delta_v,
                                        binv,
                                        N_STEPS_THROUGH_CELL,
                                        N_STEPS_INV,
                                    )
                                } else {
                                    gauss_line(delta_v - dx.dot(&gp[posn].vel), binv)
                                };
                                if let Some(mols) = &gp[posn].mol {
                                    (jnu, alpha) = source_fn_line(
                                        &mols[moli],
                                        &mol_data[moli],
                                        vfac,
                                        linei,
                                        jnu,
                                        alpha,
                                    );
                                } else {
                                    return Err(RTCError::EmptyGrid);
                                };
                            }
                        }
                    }
                }
                let dtau = alpha * ds;
                let (mut remnant_snu, _) = calc_source_fn(dtau, par.taylor_cutoff);
                remnant_snu *= jnu * ds;
                let brightness_increment = (-ray.tau[ichan]).exp() * remnant_snu;
                ray.intensity[ichan] += brightness_increment;
                ray.tau[ichan] += dtau;
            }
        }

        for di in 0..N_DIMS {
            x[di] += ds * dx[di];
        }
        col += ds;
        posn = nposn;
    }
    Ok(())
}

/// Performs linear interpolation over a simplex (triangle) using barycentric coordinates.
///
/// This function takes:
/// 1. `N` values `V_i` at the vertices of a simplex (triangle),
/// 2. The barycentric coordinates of a point within the simplex,
///    and returns the interpolated value at that point.
///
/// This approach is equivalent to using linear shape functions in Finite Element Analysis.
/// The interpolation uses shape functions `Q_i`, where each shape function:
/// - Is zero at all vertices except the `i`-th,
/// - Has value one at the `i`-th vertex.
///
/// The interpolated value at point `r_` is computed as:
///
/// ```
///           N
///           ┌
/// f(r_) =   │   V_i * Q_i(r_)
///           └
///          i=1
/// ```
///
/// For linear interpolation, each shape function `Q_i(r_)` is equal to the `i`-th barycentric coordinate `B_i` of `r_`.
///
/// In this case, `N == 3`; the simplex is a triangular face of a Delaunay cell,
/// and the interpolation point is the intersection of a ray with that face.
/// This technique is used to interpolate several grid-based quantities.
///
/// For further reference, see the Wikipedia article on [Barycentric coordinates](https://en.wikipedia.org/wiki/Barycentric_coordinate_system).
///
/// # Note
/// This is called from within the multi-threaded block.
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

/// For a given image pixel position, this function evaluates the intensity of the
/// total light emitted/absorbed along that line of sight through the (possibly
/// rotated) model. The calculation is performed for several frequencies, one per
/// channel of the output image.
///
/// Note that the algorithm employed here to solve the RTE is similar to that
/// employed in the function calculateJBar() which calculates the average radiant
/// flux impinging on a grid cell: namely the notional photon is started at the side
/// of the model near the observer and 'propagated' in the receding direction until
/// it 'reaches' the far side. This is rather non-physical in conception but it
/// makes the calculation easier.
///
/// This version of traceray implements a new algorithm in which the population
/// values are interpolated linearly from those at the vertices of the Delaunay cell
/// which the working point falls within.
///
/// A note about the object 'gips': this is an array with 3 elements, each one a
/// struct of type 'gridInterp'. This struct is meant to store as many of the
/// grid-point quantities (interpolated from the appropriate values at actual grid
/// locations) as are necessary for solving the radiative transfer equations along
/// the ray path. The first 2 entries give the values for the entry and exit points
/// to a Delaunay cell, but which is which can change, and is indicated via the
/// variables entryI and exitI (this is a convenience to avoid copying the values,
/// since the values for the exit point of one cell are obviously just those for
/// entry point of the next). The third entry stores values interpolated along the
/// ray path within a cell.
///
/// # Note
/// This is called from within the multi-threaded block.
fn trace_ray_smooth(
    ray: &mut RayData,
    img: &Image,
    par: &Parameters,
    gp: &[Grid],
    mol_data: &[MolData],
    gips: Vec<GridInterp>,
    rtp: &RTPreparation,
) -> Result<(), RTCError> {
    const NUM_SEGMENTS: usize = 5;
    const N_SEGMENTS_INV: f64 = 1.0 / (NUM_SEGMENTS as f64);
    const EPS: f64 = 1.0e-6;

    const NUM_FACES: usize = N_DIMS + 1;
    const N_VERT_PER_FACE: usize = 3;
    const NUM_RAY_INTERP_SAMPLE: usize = 3;

    for ichan in 0..img.nchan {
        ray.tau[ichan] = 0.0;
        ray.intensity[ichan] = 0.0;
    }

    let xp = ray.x;
    let yp = ray.y;

    if (xp * xp + yp * yp) > par.radius_squ {
        return Ok(());
    }

    let mut x = RVector::zeros(N_DIMS);
    let mut dir = RVector::zeros(N_DIMS);

    let zp = -(par.radius_squ - (xp * xp + yp * yp)).sqrt();
    for di in 0..N_DIMS {
        x[di] = xp * img.rotation_matrix[[di, 0]]
            + yp * img.rotation_matrix[[di, 1]]
            + zp * img.rotation_matrix[[di, 2]];
        dir[di] = img.rotation_matrix[[di, 2]];
    }

    let status = follow_ray_through_cells(&x, &dir, rtp)?;
    let (entry, chain_cell_ids, exit_intersects, len_chain_ptrs) = match status {
        RTCResult::Success {
            entry,
            chain_cell_ids,
            exit_intersects,
            len_chain_ptrs,
        } => (entry, chain_cell_ids, exit_intersects, len_chain_ptrs),
        _ => return Ok(()),
    };

    Ok(())
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
    for (icell, cell) in cells.iter().enumerate() {
        let mut simplex = Simplex {
            id: icell,
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
    for (icell, cell) in cells.iter().enumerate() {
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
) -> Result<Option<RTPreparation>> {
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

                Ok(Some(RTPreparation {
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
) -> Result<(), RTCError> {
    const MAX_NUM_RAYS_PER_PIXEL: usize = 20;
    const NUM_INTERP_PTS: usize = 3;

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

    let rtp = prepare_raytrace(gp, par, img)?;

    rays.par_iter_mut()
        .take(num_active_rays_internal)
        .try_for_each(|ray| {
            // Thread-local variables
            let mut gips = Vec::with_capacity(NUM_INTERP_PTS);
            for _ in 0..NUM_INTERP_PTS {
                gips.push(GridInterp::default());
            }
            match par.ray_trace_algorithm {
                RayTraceAlgorithm::Legacy => {
                    trace_ray(ray, gp, img, par, mol_data)?;
                }
                RayTraceAlgorithm::Modern => {
                    let rtp = rtp
                        .as_ref()
                        .expect("RayTracePreparation missing for Modern algorithm");
                    trace_ray_smooth(ray, img, par, gp, mol_data, gips, rtp)?;
                }
            }
            Ok::<(), RTCError>(())
        })?;

    Ok(())
}
