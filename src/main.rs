use citrus::{run, source};
use std::env;

fn main() {
    // Collect command line arguments
    let args: Vec<String> = env::args().skip(1).collect();

    // Ensure that the user has provided a path to a TOML file
    if args.len() != 1 {
        eprintln!("Usage: citrus <path-to-input-file>");
        std::process::exit(1);
    }

    let path = &args[0];
    let trig = source::stokes_angles(&mut [0.2; 3], [[1.3; 3]; 3], &mut [0.9; 3]);
    println!("{:?}", trig);
    // run(path);
}
