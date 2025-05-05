use ndarray::{Array1, Array2, Array3, ArrayView1};

/// n-dimensional real vector (1D array).
pub type RVector = Array1<f64>;

/// n-dimensional real vector view (1D view).
pub type RVecView<'a> = ArrayView1<'a, f64>;

/// A real matrix (2-dimensional ndarray).
pub type RMatrix = Array2<f64>;

/// A real tensor (3-dimensional ndarray).
pub type RTensor = Array3<f64>;

/// 1-dimensional unsigned-integer vector.
pub type UVector = Array1<usize>;

/// 1-dimensional signed-integer vector.
pub type IVector = Array1<isize>;

/// 1-dimensional boolean vector.
pub type BVector = Array1<bool>;
