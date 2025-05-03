use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::Result;

use crate::types::RVector;

pub fn read_dust_file(filename: &str) -> Result<(RVector, RVector)> {
    let path = Path::new(filename);
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut lam = Vec::new();
    let mut kap = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() == 2 {
            lam.push(cols[0].parse::<f64>().unwrap_or_default());
            kap.push(cols[1].parse::<f64>().unwrap_or_default());
        }
    }

    Ok((RVector::from_vec(lam), RVector::from_vec(kap)))
}
