pub struct LinkType {
    pub id: u32,
    pub gis: [u32; 2],
    pub vels: Vec<f64>,
}

pub struct MolInfoType {
    pub mol_name: String,
    pub n_levels: u64,
    pub n_lines: u64,
}

pub struct GridInfoType {
    pub n_internal_points: u32,
    pub n_sink_points: u32,
    pub n_links: u32,
    pub n_nn_indices: u32,
    pub n_dims: u16,
    pub n_species: u16,
    pub n_densities: u16,
    pub n_link_vels: u16,
    pub mols: Vec<MolInfoType>,
}

pub struct KeywordType {
    pub datatype: i64,
    pub keyname: String,
    pub comment: String,
    pub char_value: String,
    pub int_value: i64,
    pub float_value: f64,
    pub double_value: f64,
    pub bool_value: bool,
}
