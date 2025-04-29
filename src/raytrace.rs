use std::cell::RefCell;
use std::rc::Rc;
use std::vec;

use anyhow::Result;
use ndarray_linalg::SVD;

use crate::defaults::N_DIMS;
use crate::lines::ContinuumLine;
use crate::pops::Populations;
use crate::types::{RMatrix, RVector, UVector};

// Define error types for the raytrace module
pub enum RayThroughCellsError {
    SVDFail,
    NonSpan,
    LUDecompFail,
    LUSolveFail,
    TooManyEntry,
    UnknownError,
    NotFound,
}

/// NOTE: it is assumed that `vertex[i]` is opposite the face that abuts with
/// * `neigh[i]` for all `i`.
#[derive(Debug, Default)]
pub struct Simplex<'a> {
    pub id: usize,
    pub vertex: UVector,
    pub centres: RVector,
    pub neigh: Vec<Option<&'a Simplex<'a>>>,
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
    pub orientation: i32,
    pub bary: RVector,
    /// `dist` is defined via `r_int = r + dist*dir`.
    pub dist: f64,
    /// `coll_par` is a measure of how close to any edge of the face `r_int`
    /// lies.
    pub coll_par: f64,
}

impl Intersect {
    pub fn new() -> Self {
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
    pub fn new() -> Self {
        let n_dims = N_DIMS;
        FaceBasis {
            axes: vec![RVector::zeros(n_dims); n_dims - 1],
            r: vec![RVector::zeros(n_dims - 1); n_dims],
            origin: RVector::zeros(n_dims),
        }
    }

    /// Set a specific axis value.
    pub fn set_axis(&mut self, axis_index: usize, component_index: usize, value: f64) {
        if axis_index < N_DIMS - 1 && component_index < N_DIMS {
            self.axes[axis_index][component_index] = value;
        } else {
            todo!()
        }
    }

    /// Set a specific vertex value in `r`.
    pub fn set_vertex(&mut self, vertex_index: usize, component_index: usize, value: f64) {
        if vertex_index < N_DIMS && component_index < N_DIMS - 1 {
            self.r[vertex_index][component_index] = value;
        } else {
            todo!()
        }
    }

    /// Set the origin.
    pub fn set_origin(&mut self, values: RVector) {
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
    pub fn new(num_faces: usize) -> Self {
        FaceList {
            faces: Vec::with_capacity(num_faces),
            face_ptrs: vec![None; N_DIMS + 1],
        }
    }
}

pub struct RayData {
    pub x: f64,
    pub y: f64,
    pub intensity: RVector,
    pub tau: RVector,
    pub ppi: u64,
    pub is_inside_image: bool,
}

pub struct BaryVelocityBuffer {
    pub num_vertices: usize,
    pub num_edges: usize,
    pub edge_vertex_indices: Vec<[usize; 2]>,
    pub vertex_velocities: Vec<RVector>,
    pub edge_velocities: Vec<RVector>,
    pub entry_cell_bary: RVector,
    pub mid_cell_bary: RVector,
    pub exit_cell_bary: RVector,
    pub shape_fns: Vec<RVector>,
}

pub struct GridInterp {
    pub x: [f64; N_DIMS],
    pub magnetic_field: [f64; N_DIMS],
    pub x_component_ray: f64,
    pub mol: Vec<Populations>,
    pub cont: ContinuumLine,
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
pub fn extract_face(fi: usize, dc: &[Simplex], dci: usize, vertex_coords: RVector) -> Face {
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

pub fn get_new_entry_face_index(
    new_cell: Simplex,
    dci: usize,
) -> Result<isize, RayThroughCellsError> {
    let num_faces = N_DIMS + 1;
    new_cell
        .neigh
        .iter()
        .take(num_faces)
        .enumerate()
        .find_map(|(i, neigh)| neigh.filter(|n| n.id == dci).map(|_| i as isize))
        .ok_or(RayThroughCellsError::NotFound)
}

pub fn calc_face_in_nminus(nvertices: usize, face: &Face) -> FaceBasis {
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

fn intersect_line_with_face(face: Face, eps: f64) -> Result<()> {
    let eps_inv = 1.0 / eps;
    let mut vs = RMatrix::zeros((N_DIMS - 1, N_DIMS));
    let mut norm = RVector::zeros(N_DIMS);
    let px_in_face = RVector::zeros(N_DIMS - 1);
    let t_mat = vec![RVector::zeros(N_DIMS - 1); N_DIMS - 1];
    let b_vec = RVector::zeros(N_DIMS - 1);
    let test_sum_for_cw = 0.0;

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
        let (u, s, vt) = vs
            .svd(false, true)
            .map_err(|_| RayThroughCellsError::SVDFail)?;
    }
    Ok(())
}
