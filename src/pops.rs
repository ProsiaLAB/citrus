use extensions::types::RVector;

use crate::collparts::Rates;
use crate::lines::ContinuumLine;

#[derive(Debug, Default)]
pub struct Populations {
    pub pops: RVector,
    pub spec_num_dens: RVector,
    pub dopb: f64,
    pub binv: f64,
    pub nmol: f64,
    pub abun: f64,
    pub partner: Vec<Rates>,
    pub cont: Vec<ContinuumLine>,
}

pub fn popsin() {
    println!("popsin");
}
