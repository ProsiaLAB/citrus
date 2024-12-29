use crate::collparts::Rates;
use crate::lines::ContinuumLine;

pub struct Populations {
    pub pops: Vec<f64>,
    pub spec_num_dens: Vec<f64>,
    pub dopb: f64,
    pub binv: f64,
    pub nmol: f64,
    pub abun: f64,
    pub partner: Vec<Rates>,
    pub cont: Vec<ContinuumLine>,
}

impl Default for Populations {
    fn default() -> Self {
        Populations {
            pops: Vec::new(),
            spec_num_dens: Vec::new(),
            partner: Vec::new(),
            cont: Vec::new(),
            dopb: 0.0,
            binv: 0.0,
            nmol: 0.0,
            abun: 0.0,
        }
    }
}
