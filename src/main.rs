use std::env;
use std::fs;

use anyhow::Result;
use anyhow::{anyhow, bail};

use citrus::config::Config;
use citrus::config::{load_config, parse_config};
use citrus::engine;

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

    // Parse the loaded `Config` struct
    let (mut par, mut imgs, mol_data) = parse_config(input_config)?;

    engine::run(&mut par, &mut imgs, &mol_data)?;
    Ok(())
}
