use rand::rngs::StdRng;

use crate::config::Parameters;
use crate::defaults::N_DIMS;
use crate::utils::qrng::Halton;

pub const N_RANDOMS: usize = 10000;
pub const MAX_RECURSION: usize = 100;

type MonitorFn = dyn Fn(
    i32,                // num_dims
    i32,                // cell_i
    [f64; N_DIMS],      // field_origin
    [f64; N_DIMS],      // field_width
    u32,                // desired_num_points
    Vec<[f64; N_DIMS]>, // out_random_locs
    u32,                // first_point_i
    u32,                // actual_num_points
);

pub struct TreeRandomConstantType {
    pub par: Parameters,
    pub random_gen_type: StdRng,
    pub random_seed: u64,
    pub quasi_random_gen_type: Halton,
    pub num_dims: i64,
    pub num_in_randoms: i64,
    pub verbosity: i64,
    pub total_num_high_points: i64,
    pub max_recursion: i64,
    pub max_num_trials: i64,
    pub leaf_buf_len_i: i64,
    pub in_random_buffer_len_i: i64,
    pub abst_and_frac: f64,
    pub dither: f64,
    pub whole_field_origin: [f64; N_DIMS],
    pub whole_field_width: [f64; N_DIMS],
    pub all_high_point_locs: Vec<[f64; N_DIMS]>,
    pub all_high_point_densities: Vec<f64>,
    pub desired_num_points: u32,
    pub do_shuffle: bool,
    pub do_quasi_random: bool,
    pub monitor_fn: Option<Box<MonitorFn>>,
}

/// Fields of this struct are constant but
/// are set at runtime.
pub struct TreeRandomInternalType {
    pub num_sub_fields: i64,
    pub max_num_trials: f64,
    pub in_random_locs: Vec<[f64; N_DIMS]>,
    /// Random number generator - should be the value
    /// returned by gsl_rng_alloc()
    pub random_gen: StdRng,
    /// Quasi-random number generator
    pub quasi_random_gen: Halton,
}

pub struct SubCellType {
    pub num_high_points: i64,
    pub axis_indices: [i64; N_DIMS],
    pub field_origin: [f64; N_DIMS],
    pub field_width: [f64; N_DIMS],
    pub axis_signs: [f64; N_DIMS],
    pub abs_random_acceptable_range: [f64; N_DIMS],
    pub expected_desired_num_points: f64,
    pub sum_density: f64,
    pub max_density: f64,
    pub density_integral: f64,
    pub high_point_locations: Vec<[f64; N_DIMS]>,
    pub high_point_densities: Vec<f64>,
}

pub struct TreeType {
    pub leaves: Vec<SubCellType>,
    pub last_leaf_index: i64,
    pub max_leaf_index: i64,
}
