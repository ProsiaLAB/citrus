use prosia_extensions::types::Vec3;

pub trait Model {
    fn density(&self, point: &Vec3) -> f64;
    fn temperature(&self, point: &Vec3) -> f64;
    fn abundance(&self, point: &Vec3) -> f64;
    fn doppler(&self, point: &Vec3) -> f64;
    fn velocity(&self, point: &Vec3) -> Vec3;
}
