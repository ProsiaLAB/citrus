//! Temporary module to define user-facing functions
//! which serve as the interface to the `citrus` engine.

use extensions::types::RVector;

use crate::constants as cc;

use bitflags::bitflags;

bitflags! {
    pub struct UserFuncFlags: u32 {
        const DENSITY         = 1 << 0;
        const TEMPERATURE     = 1 << 1;
        const ABUNDANCE       = 1 << 2;
        const MOL_NUM_DENSITY = 1 << 3;
        const DOPPLER         = 1 << 4;
        const VELOCITY        = 1 << 5;
        const MAGFIELD        = 1 << 6;
        const GASII_DUST      = 1 << 7;
        const GRID_DENSITY    = 1 << 8;
    }
}

#[deprecated = "use `python`-defined function instead"]
pub fn density(x: f64, y: f64, z: f64) -> f64 {
    let r_min = 0.7 * cc::AU_SI;

    // Calculate radial distance from origin
    let r = x * x + y * y + z * z;

    // Calculate a spherical power-law density profile
    let r_to_use = if r > r_min { r } else { r_min };

    // Get density after converting to SI units
    1.5e6 * (r_to_use / (300.0 * cc::AU_SI)).powf(-1.5) * 1e6
}

#[deprecated = "use `python`-defined function instead"]
fn temperature(x: f64, y: f64, z: f64) -> f64 {
    let mut x0: usize = 0;

    let temp = [
        [
            2.0e13, 5.0e13, 8.0e13, 1.1e14, 1.4e14, 1.7e14, 2.0e14, 2.3e14, 2.6e14, 2.9e14,
        ],
        [
            44.777, 31.037, 25.718, 22.642, 20.560, 19.023, 17.826, 16.857, 16.050, 15.364,
        ],
    ];

    // Calculate radial distance from origin
    let r = (x * x + y * y + z * z).sqrt();

    // Linear interpolation for temperature input
    if r > temp[0][0] && r < temp[0][9] {
        for i in 0..9 {
            if r > temp[0][i] && r < temp[0][i + 1] {
                x0 = i;
            }
        }
    }
    if r < temp[0][0] {
        temp[1][0]
    } else if r > temp[0][9] {
        temp[1][9]
    } else {
        temp[1][x0]
            + (r - temp[0][x0]) * (temp[1][x0 + 1] - temp[1][x0]) / (temp[0][x0 + 1] - temp[0][x0])
    }
}

#[deprecated = "use `python`-defined function instead"]
pub const fn abundance() -> f64 {
    1e-9
}

#[deprecated = "use `python`-defined function instead"]
pub const fn doppler() -> f64 {
    200.0
}

#[deprecated = "use `python`-defined function instead"]
pub const fn gas_to_dust_ratio() -> f64 {
    100.0
}

#[deprecated = "use `python`-defined function instead"]
pub fn velocity(x: f64, y: f64, z: f64) -> RVector {
    let r_min = 0.1 * cc::AU_SI;

    // Calculate radial distance from origin
    let r = (x * x + y * y + z * z).sqrt();

    let r_to_use = if r > r_min { r } else { r_min };

    // Free-fall velocity in the radial direction onto a central mass of 1 solar mass
    let free_fall_velocity = (2.0 * cc::GRAVITATIONAL_CONST_SI * 1.989e30 / r_to_use).sqrt();
    RVector::from_vec(vec![
        -x * free_fall_velocity / r_to_use,
        -y * free_fall_velocity / r_to_use,
        -z * free_fall_velocity / r_to_use,
    ])
}
