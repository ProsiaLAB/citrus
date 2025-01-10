use std::collections::HashMap;
use std::error::Error;

use crate::collparts::{check_user_density_weights, MolData};
use crate::config::{ConfigInfo, ImageInfo};
use crate::grid;
use crate::pops::popsin;

pub fn run(
    par: &mut ConfigInfo,
    _img: &mut HashMap<String, ImageInfo>,
    _mol_data: &mut Option<Vec<MolData>>,
) -> Result<(), Box<dyn Error>> {
    if par.restart {
        par.do_solve_rte = false;
        par.do_mol_calcs = par.n_line_images > 0;
    } else {
        if par.nsolve_iters > par.n_solve_iters_done || par.lte_only {
            par.do_solve_rte = true;
        }
        par.do_mol_calcs = par.do_solve_rte || par.n_line_images > 0;
        if par.do_mol_calcs && par.mol_data_file.is_empty() {
            return Err("You must set the molecular data file.".into());
        }
    }

    if !par.do_mol_calcs && par.init_lte {
        let msg = "WARNING: Your choice of `init_lte` will have no effect \
        as no molecular calculations are requested.";
        eprintln!("{}", msg);
    }

    if par.n_species > 0 && !par.do_mol_calcs {
        return Err("If you only want to do continuum calculations, \
        you must supply no molecular data files."
            .into());
    }

    if par.nthreads > 1 {
        println!("Running with {} threads.", par.nthreads);
    }

    if par.do_pregrid {
        let mut gp = grid::set_default_grid(par.ncell, par.n_species);
        grid::pre_define(par, &mut gp)?; // sets `par.num_densities`
        check_user_density_weights(par)?;
    } else if par.restart {
        popsin(); // TODO: Implement this function
    } else {
        check_user_density_weights(par)?;
        let _gp = grid::read_or_build_grid(par)?;
    }

    Ok(())
}
