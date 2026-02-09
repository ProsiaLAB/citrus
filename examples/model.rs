use std::env;

use anyhow::Result;
use anyhow::bail;

use citrus::config::Config;
use citrus::constants as cc;
use citrus::engine;
use citrus::models::Model;
use prosia_extensions::types::Vec3;

pub struct Lime;

impl Model for Lime {
    fn density(&self, point: &Vec3) -> f64 {
        let rmin = 0.7 * cc::AU_SI;
        let r = point.norm();

        let r = if r > rmin { r } else { rmin };

        1.5e6 * (r / (300.0 * cc::AU_SI)).powf(-1.5) * 1e6
    }

    fn temperature(&self, point: &Vec3) -> f64 {
        const TEMP: [[f64; 10]; 2] = [
            [
                2.0e13, 5.0e13, 8.0e13, 1.1e14, 1.4e14, 1.7e14, 2.0e14, 2.3e14, 2.6e14, 2.9e14,
            ],
            [
                44.777, 31.037, 25.718, 22.642, 20.560, 19.023, 17.826, 16.857, 16.050, 15.364,
            ],
        ];

        let r = point.norm();

        let x0 = if r < TEMP[0][0] || r > TEMP[0][9] {
            0
        } else {
            TEMP[0].iter().position(|&x| x > r).unwrap_or(9)
        };

        if r < TEMP[0][0] || r > TEMP[0][9] {
            TEMP[1][0]
        } else if r > TEMP[0][9] {
            TEMP[1][9]
        } else {
            let x1 = TEMP[0][x0];
            let x2 = TEMP[0][x0 - 1];
            let y1 = TEMP[1][x0];
            let y2 = TEMP[1][x0 - 1];

            y2 + (y1 - y2) * (r - x2) / (x1 - x2)
        }
    }

    fn abundance(&self, _point: &Vec3) -> f64 {
        1e-9
    }

    fn doppler(&self, _point: &Vec3) -> f64 {
        200.0
    }

    fn velocity(&self, point: &Vec3) -> Vec3 {
        let rmin = 0.1 * cc::AU_SI;

        let r = point.norm();

        let r = if r > rmin { r } else { rmin };

        let free_fall_velocity = (2.0 * cc::GRAVITATIONAL_CONST_SI * 1.989e30 / r).sqrt();

        Vec3::new(
            -point.x * free_fall_velocity / r,
            -point.y * free_fall_velocity / r,
            -point.z * free_fall_velocity / r,
        )
    }
}

fn main() -> Result<()> {
    // Collect command line arguments
    let args: Vec<String> = env::args().skip(1).collect();

    // Ensure that the user has provided a path to a TOML file
    if args.len() != 1 {
        bail!("Usage: citrus <path-to-input-file>");
    }

    let path = &args[0];

    // Load the TOML file
    let input_config = Config::from_path(path).expect("Failed to load config");

    dbg!("Loaded config: {:?}", &input_config);

    engine::run(
        Lime,
        &mut input_config.parameters,
        &mut input_config.images,
        &input_config.molecular_data,
    )?;
    Ok(())
}
