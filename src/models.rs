use prosia_extensions::types::Vec3;

use crate::config::{ParameterInput, SamplingAlgorithm};

pub trait Model {
    fn density(&self, _point: &Vec3) -> f64 {
        0.0
    }

    fn temperature(&self, _point: &Vec3) -> f64 {
        0.0
    }

    fn abundance(&self, _point: &Vec3) -> f64 {
        0.0
    }

    fn mol_num_density(&self, _point: &Vec3) -> f64 {
        0.0
    }

    fn doppler(&self, _point: &Vec3) -> f64 {
        0.0
    }

    fn velocity(&self, _point: &Vec3) -> Vec3 {
        Vec3::zero()
    }

    fn magnetic_field(&self, _point: &Vec3) -> Vec3 {
        Vec3::zero()
    }

    fn gas_to_dust_ratio(&self, _point: &Vec3) -> f64 {
        100.0
    }

    fn grid_density(&self, par: &ParameterInput, r: &Vec3) -> f64 {
        let r_squared = r.dot(r);

        if r_squared >= par.radius_squ {
            return 0.0;
        }

        let mut total_density = 0.0;

        for _ in 0..par.n_densities {
            let density = self.density(r);
            total_density += density;
        }

        let default_density_power = if matches!(par.sampling_algorithm, SamplingAlgorithm::Uniform)
        {
            cc::DENSITY_POWER
        } else {
            cc::TREE_POWER
        };

        total_density.powf(default_density_power) / par.grid_density_global_max
    }
}
