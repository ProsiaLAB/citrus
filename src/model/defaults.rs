use super::traits::PhysicsModel;

pub struct DefaultModel;

impl PhysicsModel for DefaultModel {
    fn density(&self, x: f64, y: f64, z: f64) -> f64 {
        // Simplified fallback
        let r2 = x * x + y * y + z * z;
        let power = 0.2;
        1.0 / (1.0 + r2).powf(power)
    }

    fn temperature(&self, _x: f64, _y: f64, _z: f64) -> f64 {
        20.0
    }

    fn abundance(&self, _x: f64, _y: f64, _z: f64) -> f64 {
        1e-4
    }

    fn mol_num_density(&self, _x: f64, _y: f64, _z: f64) -> f64 {
        0.0
    }

    fn doppler(&self, _x: f64, _y: f64, _z: f64) -> f64 {
        1e5 // cm/s
    }

    fn velocity(&self, _x: f64, _y: f64, _z: f64) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }

    fn magfield(&self, _x: f64, _y: f64, _z: f64) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }

    fn gas_to_dust(&self, _x: f64, _y: f64, _z: f64) -> f64 {
        100.0
    }
}