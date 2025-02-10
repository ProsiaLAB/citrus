use std::cell::RefCell;
use std::rc::Rc;

use crate::defaults;
use crate::lines::ContinuumLine;
use crate::pops::Populations;

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
#[derive(Debug)]
pub struct Simplex<'a> {
    pub id: u64,
    pub vertex: Vec<u64>,
    pub centres: Vec<f64>,
    pub neigh: Vec<Option<&'a Simplex<'a>>>,
}

impl<'a> Simplex<'a> {
    /// Constructor for Simplex
    pub fn new(id: u64) -> Self {
        Simplex {
            id,
            vertex: vec![0; defaults::N_DIMS + 1],
            centres: vec![0.0; defaults::N_DIMS],
            neigh: vec![None; defaults::N_DIMS + 1],
        }
    }

    /// Set a specific neighbor
    pub fn set_neighbor(&mut self, index: usize, neighbor: Option<&'a Simplex<'a>>) {
        if index < self.neigh.len() {
            self.neigh[index] = neighbor;
        } else {
            todo!()
        }
    }
}

/// This struct is meant to record all relevant information about the
/// intersection between a ray (defined by a direction unit vector 'dir' and a
/// starting position 'r') and a face of a simplex.
#[derive(Debug)]
pub struct Intersect {
    /// The index (in the range {0...N}) of the face (and thus of the opposite
    /// vertex, i.e. the one 'missing' from the bary[] list of this face).
    pub fi: i32,
    /// `> 0` means the ray exits, `< 0` means it enters, `== 0` means the
    /// face is parallel to the ray.
    pub orientation: i32,
    pub bary: Vec<f64>,
    /// `dist` is defined via `r_int = r + dist*dir`.
    pub dist: f64,
    /// `coll_par` is a measure of how close to any edge of the face `r_int`
    /// lies.
    pub coll_par: f64,
}

impl Intersect {
    /// Constructor for `Intersect`
    pub fn new(fi: i32, orientation: i32, dist: f64, coll_par: f64) -> Self {
        Intersect {
            fi,
            orientation,
            bary: vec![0.0; defaults::N_DIMS],
            dist,
            coll_par,
        }
    }

    /// Set barycentric coordinates.
    pub fn set_bary(&mut self, values: Vec<f64>) {
        if values.len() == defaults::N_DIMS {
            self.bary = values;
        } else {
            todo!()
        }
    }
}

#[derive(Debug)]
pub struct Face {
    /// `r` is a list of the the `N` vertices of the face, each of which has `N`
    /// cartesian components.
    pub r: Vec<Vec<f64>>,
    /// `simplex_centres` is a convenience pointer array which gives
    /// the location of the geometric centres of the simplexes.
    pub simplex_centres: Vec<f64>,
}

#[derive(Debug)]
pub struct FaceBasis {
    pub axes: Vec<Vec<f64>>,
    ///  `r` expresses the location of the N vertices of a simplicial polytope face
    /// in N-space, in terms of components along the N-1 orthogonal axes in the
    /// sub-plane of the face. Thus you should malloc r as r[N][N-1].
    pub r: Vec<Vec<f64>>,
    pub origin: Vec<f64>,
}

impl FaceBasis {
    /// Constructor for `FaceBasis`, initializes vectors dynamically based on `defaults::N_DIMS`.
    pub fn new() -> Self {
        let n_dims = defaults::N_DIMS;
        FaceBasis {
            axes: vec![vec![0.0; n_dims]; n_dims - 1],
            r: vec![vec![0.0; n_dims - 1]; n_dims],
            origin: vec![0.0; n_dims],
        }
    }

    /// Set a specific axis value.
    pub fn set_axis(&mut self, axis_index: usize, component_index: usize, value: f64) {
        if axis_index < defaults::N_DIMS - 1 && component_index < defaults::N_DIMS {
            self.axes[axis_index][component_index] = value;
        } else {
            todo!()
        }
    }

    /// Set a specific vertex value in `r`.
    pub fn set_vertex(&mut self, vertex_index: usize, component_index: usize, value: f64) {
        if vertex_index < defaults::N_DIMS && component_index < defaults::N_DIMS - 1 {
            self.r[vertex_index][component_index] = value;
        } else {
            todo!()
        }
    }

    /// Set the origin.
    pub fn set_origin(&mut self, values: Vec<f64>) {
        if values.len() == defaults::N_DIMS {
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
            face_ptrs: vec![None; defaults::N_DIMS + 1],
        }
    }

    /// Adds a new face to the face list.
    pub fn add_face(&mut self, face: Face) -> Rc<RefCell<Face>> {
        let face_rc = Rc::new(RefCell::new(face));
        self.faces.push(face_rc.clone());
        face_rc
    }

    /// Sets a reference to a face in the `face_ptrs` array.
    pub fn set_face_ptr(&mut self, index: usize, face: Option<Rc<RefCell<Face>>>) {
        if index < defaults::N_DIMS + 1 {
            self.face_ptrs[index] = face;
        } else {
            todo!()
        }
    }

    /// Gets a face reference at a given index in the `face_ptrs` array.
    pub fn get_face_ptr(&self, index: usize) -> Option<Rc<RefCell<Face>>> {
        if index < self.face_ptrs.len() {
            self.face_ptrs[index].clone()
        } else {
            None
        }
    }
}

pub struct RayData {
    pub x: f64,
    pub y: f64,
    pub intensity: Vec<f64>,
    pub tau: Vec<f64>,
    pub ppi: u64,
    pub is_inside_image: bool,
}

pub struct BaryVelocityBuffer {
    pub num_vertices: usize,
    pub num_edges: usize,
    pub edge_vertex_indices: Vec<[usize; 2]>,
    pub vertex_velocities: Vec<Vec<f64>>,
    pub edge_velocities: Vec<Vec<f64>>,
    pub entry_cell_bary: Vec<f64>,
    pub mid_cell_bary: Vec<f64>,
    pub exit_cell_bary: Vec<f64>,
    pub shape_fns: Vec<Vec<f64>>,
}

pub struct GridInterp {
    pub x: [f64; defaults::N_DIMS],
    pub magnetic_field: [f64; defaults::N_DIMS],
    pub x_component_ray: f64,
    pub mol: Vec<Populations>,
    pub cont: ContinuumLine,
}
