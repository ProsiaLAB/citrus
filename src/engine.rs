use std::collections::HashMap;

use anyhow::Result;
use anyhow::bail;
use prosia_extensions::types::RVector;
use serde::{Deserialize, Serialize};

use crate::collparts::MolData;
use crate::collparts::check_user_density_weights;
use crate::config::{Image, Parameters};
use crate::grid;
use crate::io::read_dust_file;
use crate::pops::popsin;
use crate::raytrace::raytrace;

pub fn run(
    par: &mut Parameters,
    imgs: &mut Vec<Image>,
    mol_data: &Option<Vec<MolData>>,
) -> Result<()> {
    if par.restart {
        par.do_solve_rte = false;
        par.do_mol_calcs = par.n_line_images > 0;
    } else {
        if par.nsolve_iters > par.n_solve_iters_done || par.lte_only {
            par.do_solve_rte = true;
        }
        par.do_mol_calcs = par.do_solve_rte || par.n_line_images > 0;
        if par.do_mol_calcs && par.mol_data_files.is_empty() {
            bail!("You must set the molecular data file.");
        }
    }

    if !par.do_mol_calcs && par.init_lte {
        let msg = "WARNING: Your choice of `init_lte` will have no effect \
        as no molecular calculations are requested.";
        eprintln!("{}", msg);
    }

    if par.n_species > 0 && !par.do_mol_calcs {
        bail!(
            "If you only want to do continuum calculations, \
        you must supply no molecular data files."
        );
    }

    if par.nthreads > 1 {
        println!("Running with {} threads.", par.nthreads);
    }

    let mut gp = if par.do_pregrid {
        let mut gp = grid::set_default_grid(par.ncell, par.n_species);
        grid::pre_define(par, &mut gp)?; // sets `par.num_densities`
        check_user_density_weights(par)?;
        gp
    } else if par.restart {
        popsin(); // TODO: Implement this function
        todo!()
    } else {
        check_user_density_weights(par)?;
        grid::read_or_build_grid(par)?
    };

    let lam_kap: Option<(RVector, RVector)> = match &par.dust {
        Some(dust) if !dust.is_empty() => Some(read_dust_file(dust)?),
        _ => {
            eprintln!("No dust file provided.");
            None
        }
    };

    let mol_slice = mol_data.as_ref().expect("mol_data is None").as_slice();

    if par.n_cont_images > 0 {
        for (_, img) in imgs.iter_mut().enumerate() {
            raytrace(img, gp.as_mut_slice(), par, mol_slice, &lam_kap)?;
        }
    }

    Ok(())
}
