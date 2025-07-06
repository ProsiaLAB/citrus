use super::defaults::DefaultModel;
use super::python::PythonModel;
use super::traits::PhysicsModel;

pub struct CompositeModel {
    pub default: DefaultModel,
    pub user: Option<PythonModel>,
}

impl CompositeModel {
    pub fn new(user: Option<PythonModel>) -> Self {
        Self {
            default: DefaultModel,
            user,
        }
    }
}

impl PhysicsModel for CompositeModel {
    fn density(&self, x: f64, y: f64, z: f64) -> f64 {
        self.user
            .as_ref()
            .map(|u| u.density(x, y, z))
            .unwrap_or_else(|| self.default.density(x, y, z))
    }

    fn temperature(&self, x: f64, y: f64, z: f64) -> f64 {
        self.user
            .as_ref()
            .map(|u| u.temperature(x, y, z))
            .unwrap_or_else(|| self.default.temperature(x, y, z))
    }

    fn abundance(&self, x: f64, y: f64, z: f64) -> f64 {
        self.user
            .as_ref()
            .map(|u| u.abundance(x, y, z))
            .unwrap_or_else(|| self.default.abundance(x, y, z))
    }

    fn mol_num_density(&self, x: f64, y: f64, z: f64) -> f64 {
        self.user
            .as_ref()
            .map(|u| u.mol_num_density(x, y, z))
            .unwrap_or_else(|| self.default.mol_num_density(x, y, z))
    }

    fn doppler(&self, x: f64, y: f64, z: f64) -> f64 {
        self.user
            .as_ref()
            .map(|u| u.doppler(x, y, z))
            .unwrap_or_else(|| self.default.doppler(x, y, z))
    }

    fn velocity(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        self.user
            .as_ref()
            .map(|u| u.velocity(x, y, z))
            .unwrap_or_else(|| self.default.velocity(x, y, z))
    }

    fn magfield(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        self.user
            .as_ref()
            .map(|u| u.magfield(x, y, z))
            .unwrap_or_else(|| self.default.magfield(x, y, z))
    }

    fn gas_to_dust(&self, x: f64, y: f64, z: f64) -> f64 {
        self.user
            .as_ref()
            .map(|u| u.gas_to_dust(x, y, z))
            .unwrap_or_else(|| self.default.gas_to_dust(x, y, z))
    }

    fn grid_density(&self, r: &[f64; 3]) -> f64 {
        self.user
            .as_ref()
            .map(|u| u.grid_density(r))
            .unwrap_or_else(|| self.default.grid_density(r))
    }
}
