use std::collections::HashMap;

use anyhow::bail;
use anyhow::Result;
use bitflags::bitflags;
use prosia_extensions::types::RVector;
use serde::{Deserialize, Serialize};

use crate::collparts::check_user_density_weights;
use crate::collparts::MolData;
use crate::config::{Image, Parameters};
use crate::grid;
use crate::io::read_dust_file;
use crate::pops::popsin;
use crate::raytrace::raytrace;

bitflags! {
    /// Which “stages” of data are present in a grid.
    #[derive(Serialize, Deserialize, Debug, Default)]
    pub struct DataStage: u32 {
        const X              = 1 << 0;  // id, x, sink
        const NEIGHBOURS     = 1 << 1;  // neigh, dir, ds, numNeigh
        const VELOCITY       = 1 << 2;  // vel
        const DENSITY        = 1 << 3;  // dens
        const ABUNDANCE      = 1 << 4;  // abun, nmol
        const TURB_DOPPLER   = 1 << 5;  // dopb
        const TEMPERATURES   = 1 << 6;  // t
        const MAGFIELD       = 1 << 7;  // B
        const ACOEFF         = 1 << 8;  // a0–a4
        const POPULATIONS    = 1 << 9;  // mol

        // composite masks
        const MASK_NEIGHBOURS = Self::NEIGHBOURS.bits()  | Self::X.bits();
        const MASK_VELOCITY   = Self::VELOCITY.bits()    | Self::X.bits();
        const MASK_DENSITY    = Self::DENSITY.bits()     | Self::X.bits();
        const MASK_ABUNDANCE  = Self::ABUNDANCE.bits()   | Self::X.bits();
        const MASK_TURB       = Self::TURB_DOPPLER.bits()| Self::X.bits();
        const MASK_TEMPS      = Self::TEMPERATURES.bits()| Self::X.bits();
        const MASK_ACOEFF     = Self::ACOEFF.bits()      | Self::MASK_NEIGHBOURS.bits() | Self::MASK_VELOCITY.bits();

        const MASK_POPULATIONS = Self::POPULATIONS.bits()      | Self::MASK_ACOEFF.bits()
                               | Self::DENSITY.bits()          | Self::TEMPERATURES.bits()
                               | Self::ABUNDANCE.bits()        | Self::TURB_DOPPLER.bits();
        const MASK_ALL         = Self::MASK_POPULATIONS.bits() | Self::MAGFIELD.bits();
        const MASK_ALL_BUT_MAG = Self::MASK_ALL.bits() & !Self::MAGFIELD.bits();
    }
}

pub fn run(
    par: &mut Parameters,
    imgs: &mut HashMap<String, Image>,
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
        for (_, img) in imgs.iter_mut() {
            raytrace(img, gp.as_mut_slice(), par, mol_slice, &lam_kap)?;
        }
    }

    Ok(())
}
