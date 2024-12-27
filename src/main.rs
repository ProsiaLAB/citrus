use citrus::load_config;
use citrus::messages;
use citrus::parse_config;
use citrus::run;
use std::env;
use std::error::Error;
use std::fs;
use std::process;

fn main() -> Result<(), Box<dyn Error>> {
    messages::greetings();
    // messages::description();

    // Collect command line arguments
    let args: Vec<String> = env::args().skip(1).collect();

    // Ensure that the user has provided a path to a TOML file
    if args.len() != 1 {
        eprintln!("Usage: citrus <path-to-input-file>");
        std::process::exit(1);
    }

    let path = &args[0];

    // Try to get the absolute path and handle errors properly
    let path = match fs::canonicalize(path) {
        Ok(p) => p, // Successfully resolved to an absolute path
        Err(e) => {
            eprintln!("Error resolving the path '{}': {}", path, e);
            process::exit(1); // Exit with failure code
        }
    };

    // Ensure the path exists after resolving it
    if !path.exists() {
        eprintln!("The path '{}' does not exist.", path.display());
        process::exit(1); // Exit with failure code
    }

    // Load the TOML file
    let input_config = load_config(path.to_str().unwrap_or_else(|| {
        eprintln!("Error: The canonicalized path is not valid UTF-8.");
        process::exit(1);
    }))?;

    // Parse the loaded `Config` struct
    let (mut par, mut img) = parse_config(input_config)?;

    run(&mut par, &mut img)?;
    Ok(())
}
