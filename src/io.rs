use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

use anyhow::Result;
use planetes_ext::types::RVector;

use crate::engine::LamKap;

pub fn read_dust_file(filename: &str) -> Result<LamKap> {
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

    let lam_kap = LamKap {
        lam: RVector::from_vec(lam),
        kap: RVector::from_vec(kap),
    };

    Ok(lam_kap)
}
