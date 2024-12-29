#[derive(Debug, Default)]
pub struct CollisionalPartnerData {
    pub down: Vec<f64>,
    pub temp: Vec<f64>,
    pub collisional_partner_id: i64,
    pub ntemp: i64,
    pub ntrans: i64,
    pub lcl: Vec<i64>,
    pub lcu: Vec<i64>,
    pub density_index: i64,
    pub name: String,
}

#[derive(Debug)]
pub struct MolData {
    pub nlev: i64,
    pub nline: i64,
    pub npart: i64,
    pub lal: Vec<i64>,
    pub lau: Vec<i64>,
    pub aeinst: Vec<f64>,
    pub freq: Vec<f64>,
    pub beinstu: Vec<f64>,
    pub beinstl: Vec<f64>,
    pub eterm: Vec<f64>,
    pub gstat: Vec<f64>,
    pub gir: Vec<f64>,
    pub cmb: Vec<f64>,
    pub amass: f64,
    pub part: CollisionalPartnerData,
    pub mol_name: String,
}

impl Default for MolData {
    fn default() -> Self {
        MolData {
            nlev: -1,
            nline: -1,
            npart: -1,
            amass: -1.0,
            part: CollisionalPartnerData::default(),
            lal: Vec::new(),
            lau: Vec::new(),
            aeinst: Vec::new(),
            freq: Vec::new(),
            beinstu: Vec::new(),
            beinstl: Vec::new(),
            eterm: Vec::new(),
            gstat: Vec::new(),
            cmb: Vec::new(),
            gir: Vec::new(),
            mol_name: String::new(),
        }
    }
}

pub struct Rates {
    pub t_binlow: i64,
    pub interp_coeff: f64,
}
