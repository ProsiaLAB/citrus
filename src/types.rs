use ndarray::ArrayView1;
use ndarray::{Array1, Array2, Array3};

/// 1-dimensional real vector.
pub type RVector = Array1<f64>;

/// 1-dimensional real vector view.
pub type RVecView<'a> = ArrayView1<'a, f64>;

/// A real matrix (2-dimensional real array).
pub type RMatrix = Array2<f64>;

/// A real tensor (3-dimensional real array).
pub type RTensor = Array3<f64>;

/// A 1-dimensional unsigned-integer vector.
pub type UVector = Array1<usize>;

/// A 1-dimensional signed-integer vector.
pub type IVector = Array1<isize>;

/// A 1-dimensional boolean vector.
pub type BVector = Array1<bool>;
