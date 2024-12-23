pub struct GridPointData {
    pub jbar: Vec<f64>,
    pub phot: Vec<f64>,
    pub vfac: Vec<f64>,
    pub vfac_loc: Vec<f64>,
}
pub struct Blend {
    pub mol_j: i64,
    pub line_j: i64,
    pub delta_v: f64,
}

pub struct LineWithBlends {
    pub line_i: i64,
    pub num_blends: i64,
    pub blends: Vec<Blend>,
}

pub struct MolWithBlends {
    pub mol_i: i64,
    pub num_lines_with_blends: i64,
    pub lines_with_blends: Vec<LineWithBlends>,
}

pub struct BlendInfo {
    pub num_mols_with_blends: i64,
    pub mols: Vec<MolWithBlends>,
}
