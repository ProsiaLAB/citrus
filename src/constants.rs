/// Local mean Cosmic Microwave Background (CMB) temperature in kelvin
/// Temperature value from Fixsen (2009): 2.72548 K
pub const LOCAL_CMB_TEMP_SI: f64 = 2.72548;

/// Typical Interstellar Medium (ISM) density in particles per cubic centimeter
/// A typical value for the ISM is 1000 particles/cm³
pub const TYPICAL_ISM_DENS: f64 = 1000.0;

/// A small number to avoid division by zero
/// Used to prevent numerical errors such as division by zero
pub const CITRUS_GLOBAL_EPS: f64 = 1e-30;
pub const CITRUS_RT_EPS: f64 = 1e-6;

pub const N_DIMS: usize = 3;
pub const NUM_OF_GRID_STAGES: usize = 5;

pub const TREE_EXP: f64 = 2.0;
pub const DENSITY_EXP: f64 = 0.2;
