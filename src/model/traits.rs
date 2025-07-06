pub trait PhysicsModel: Send + Sync {
    fn density(&self, x: f64, y: f64, z: f64) -> f64;
    fn temperature(&self, x: f64, y: f64, z: f64) -> f64;
    fn abundance(&self, x: f64, y: f64, z: f64) -> f64;
    fn mol_num_density(&self, x: f64, y: f64, z: f64) -> f64;
    fn doppler(&self, x: f64, y: f64, z: f64) -> f64;
    fn velocity(&self, x: f64, y: f64, z: f64) -> [f64; 3];
    fn magfield(&self, x: f64, y: f64, z: f64) -> [f64; 3];
    fn gas_to_dust(&self, x: f64, y: f64, z: f64) -> f64;

    fn grid_density(&self, r: &[f64; 3]) -> f64 {
        self.density(r[0], r[1], r[2])
    }
}