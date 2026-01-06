use std::env;
use std::fs;

use anyhow::Result;
use anyhow::{anyhow, bail};

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

    // Try to get the absolute path and handle errors properly
    let path = match fs::canonicalize(path) {
        Ok(p) => p, // Successfully resolved to an absolute path
        Err(e) => {
            return Err(e.into());
        }
    };

    // Ensure the path exists after resolving it
    if !path.exists() {
        bail!("The path does not exist.");
    }

    // Load the TOML file
    let input_config = load_config(
        path.to_str()
            .ok_or_else(|| anyhow!("Error: The canonicalized path is not valid UTF-8."))?,
    )?;

    dbg!("Loaded config: {:?}", &input_config);

    // Parse the loaded `Config` struct
    let (mut par, mut imgs, mol_data) = parse_config(input_config)?;

    engine::run(&mut par, &mut imgs, &mol_data)?;
    Ok(())
}
