use serde::{Deserialize, Serialize};

use crate::types::RVector;

#[derive(Debug, Default)]
pub struct ContinuumLine {
    pub dust: f64,
    pub knu: f64,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Spec {
    pub intense: RVector,
    pub tau: RVector,
    pub stokes: [f64; 3],
    pub num_rays: usize,
}
