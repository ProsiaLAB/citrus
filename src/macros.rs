use std::vec;

/// Macro to generate a density function based on the Lynden-Bell & Pringle (1974) disk model.
///
/// # Arguments
/// - `$Sc`: Surface density normalization constant.
/// - `$rc`: Characteristic radius of the disk.
/// - `$gamma`: Power-law exponent of the surface density profile.
/// - `$hc`: Scale height at the characteristic radius.
/// - `$psi`: Flaring index determining how scale height changes with radius.
macro_rules! density_function {
    ($Sc:expr, $rc:expr, $gamma:expr, $hc:expr, $psi:expr) => {
        fn rho(r: f64, z: f64) -> f64 {
            let h = $hc * (r / $rc).powf($psi);
            let s = $Sc * (r / $rc).powf(-$gamma) * (-((r / $rc).powf(2.0 - $gamma))).exp();
            s / ((2.0 * std::f64::consts::PI).sqrt() * h) * (-0.5 * (z / h).powi(2)).exp()
        }
    };
}

fn _main() {
    density_function!(1.0, 1.0, 1.0, 1.0, 1.0);
    let r = 2.0;
    let z = 3.0;
    println!("rho({}, {}) = {}", r, z, rho(r, z));
    vec![1, 2, 3];
}
