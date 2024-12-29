pub struct ContinuumLine {
    pub dust: f64,
    pub knu: f64,
}

#[derive(Debug, Default, Clone)]
pub struct Spec {
    pub intense: Vec<f64>,
    pub tau: Vec<f64>,
    pub stokes: [f64; 3],
    pub num_rays: i64,
}
