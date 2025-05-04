use anyhow::bail;
use anyhow::Result;

use crate::defaults;

use crate::config::Parameters;
use crate::types::{IVector, RVector};

#[derive(Debug, Default)]
pub struct CollisionalPartnerData {
    pub down: RVector,
    pub temp: RVector,
    pub collisional_partner_id: isize,
    pub ntemp: isize,
    pub ntrans: isize,
    pub lcl: IVector,
    pub lcu: IVector,
    pub density_index: isize,
    pub name: String,
}

#[derive(Debug, Default)]
pub struct MolData {
    pub nlev: isize,
    pub nline: isize,
    pub npart: isize,
    pub lal: IVector,
    pub lau: IVector,
    pub aeinst: RVector,
    pub freq: RVector,
    pub beinstu: RVector,
    pub beinstl: RVector,
    pub eterm: RVector,
    pub gstat: RVector,
    pub gir: RVector,
    pub cmb: RVector,
    pub amass: f64,
    pub part: CollisionalPartnerData,
    pub mol_name: String,
}

impl MolData {
    pub fn new() -> Self {
        MolData {
            nlev: -1,
            nline: -1,
            npart: -1,
            amass: -1.0,
            ..Default::default()
        }
    }
}

#[derive(Debug)]
pub struct Rates {
    pub t_binlow: isize,
    pub interp_coeff: isize,
}

/// This deals with four user-settable fields of [`ConfigInfo`] which relate
/// to collision partners and their number densities: `collisional_partner_ids`, `nmol_weights`,
/// `collisional_partner_mol_weights` and `collisional_partner_names`. We have to see if these
/// (optional) parameters were set, do some basic checks on them, and if they were
/// set make sure they match the number of density values, which by this time should
/// be stored in `num_densities` field of [`ConfigInfo`].
///
///
///
/// The user can specify either, none, or both of these two parameters, with
/// the following effects:
///
///  Ids | Names | Effect
///  --- | ----- | ------------
///  0   | 0      | LAMDA collision partners are assumed and the association between the density functions and the moldatfiles is essentia
///  0   | 1      | par->collPartIds is constructed to contain
/// integers in a sequence from 1 to N. Naturally the user should write matching
/// collision partner ID integers in their moldatfiles.
///
///                 1   0   LAMDA collision partners are assumed.
///
///                 1   1   User will get what they ask for.
///                 ----------------------
///
///         * par->collPartMolWeights: this MUST be present if par->collPartNames
/// has been supplied, and it MUST then have the same number and order of elements
/// as all the other collision-partner lists. If this parameter is supplied but
/// par->collPartNames not, it will be ignored.
///
///         * par->nMolWeights: this list gives the weights to be applied to the N
/// density values when calculating molecular densities from abundances.
pub fn check_user_density_weights(par: &mut Parameters) -> Result<()> {
    par.collisional_partner_user_set_flags = 0;

    // Get the numbers of elements set by the user for each of the 4 parameters:
    let mut i = 0;
    while i < defaults::MAX_NUM_OF_COLLISIONAL_PARTNERS
        && i < par.collisional_partner_ids.len()
        && par.collisional_partner_ids[i] > 0
    {
        i += 1;
    }
    let mut num_user_set_coll_part_ids = i;
    if i > 0 {
        par.collisional_partner_user_set_flags = 1;
    }

    let mut i = 0;
    while i < defaults::MAX_NUM_OF_COLLISIONAL_PARTNERS
        && i < par.nmol_weights.len()
        && par.nmol_weights[i] >= 0.0
    {
        i += 1;
    }
    let mut num_user_set_nmol_weights = i;
    if i > 0 {
        par.collisional_partner_user_set_flags = 2;
    }

    let mut i = 0;
    while i < defaults::MAX_NUM_OF_COLLISIONAL_PARTNERS
        && i < par.collisional_partner_names.len()
        && !par.collisional_partner_names[i].is_empty()
    {
        i += 1;
    }
    let mut num_user_set_coll_part_names = i;
    if i > 0 {
        par.collisional_partner_user_set_flags = 3;
    }

    let mut i = 0;
    while i < defaults::MAX_NUM_OF_COLLISIONAL_PARTNERS
        && i < par.collisional_partner_mol_weights.len()
        && par.collisional_partner_mol_weights[i] >= 0.0
    {
        i += 1;
    }
    let num_user_set_coll_part_mol_weights = i;
    if i > 0 {
        par.collisional_partner_user_set_flags = 4;
    }

    /* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .*/
    /* Perform checks on the numbers.
     */
    /* Check that we have either 0 par->collPartIds or the same number as the
     * number of density values. If not, the par->collPartIds values the user set
     * will be thrown away, the pointer will be reallocated, and new values will
     * be written to it in setUpDensityAux(), taken from the values in the
     * moldatfiles.
     */

    if num_user_set_coll_part_ids != par.num_densities {
        /* Note that in the present case we will (for a line-emission image) look
         * for the collision partners listed in the moldatfiles and set
         * par->collPartIds from them. For that to happen, we require the number of
         * collision partners found in the files to equal par->numDensities. */

        /* numUserSetCPIds==0 is ok, this just means the user has not set the
         * parameter at all, but for other values we should issue some warnings,
         * because if the user sets any at all, they should set the same number as
         * there are returns from density():
         */
        if num_user_set_coll_part_ids > 0 {
            eprintln!(
                "WARNING: The number of user-set collision partner IDs does not match the number of density values. The user-set values will be ignored."
            );
        }
        num_user_set_coll_part_ids = 0;
    } else {
        // Resize the vector to the number of densities
        par.collisional_partner_ids = vec![0; par.num_densities];
    }

    if !par.use_abun && num_user_set_coll_part_mol_weights > 0 {
        eprintln!(
            "You only need to set par.nmol_weights if you have provided an  \
            abundance function."
        );
    }

    /* Check if we have either 0 par->nMolWeights or the same number as the number
     * of density values.
     */
    if par.use_abun && num_user_set_nmol_weights != par.num_densities {
        /* Note that in the present case we will (for a line-emission image) look
         * for the collision partners listed in the moldatfiles and set
         * par->nMolWeights from them. */

        /* numUserSetNMWs==0 is ok, this just means the user has not set the
         * parameter at all, but for other values we should issue some warnings,
         * because if the user sets any at all, they should set the same number as
         * there are returns from density():
         */
        if num_user_set_coll_part_names > 0 {
            eprintln!(
                "WARNING: The number of user-set molecular weights does not match the number of density values. The user-set values will be ignored."
            );
        }
        num_user_set_nmol_weights = 0;
    } else {
        // Resize the vector to the number of densities
        par.nmol_weights = vec![0.0; par.num_densities];
    }

    /* Re the interaction between par->collPartIds and par->collPartNames: the
     * possible scenarios are given in the function header.
     */
    if num_user_set_coll_part_names != par.num_densities {
        /* numUserSetCPNames==0 is ok, this just means the user has not set the
         * parameter at all, but for other values we should issue some warnings,
         * because if the user sets any at all, they should set the same number as
         * there are returns from density():
         */
        if num_user_set_coll_part_names > 0 {
            eprintln!(
                "WARNING: The number of user-set collision partner names does not match the number of density values. The user-set values will be ignored."
            );
        }
        num_user_set_coll_part_names = 0;
    } else {
        // Resize the vector to the number of densities
        par.collisional_partner_names
            .resize(par.num_densities, String::new());
        if num_user_set_coll_part_ids == 0 {
            for i in 0..par.num_densities {
                par.collisional_partner_ids[i] = i + 1;
            }
            num_user_set_coll_part_ids = par.num_densities;
        }
    }

    /* The constraints on the list of CP molecular weights are similar, but NULL +
     * warn that they will be ignored if there are no CP names.
     */
    if num_user_set_coll_part_mol_weights != par.num_densities || num_user_set_coll_part_names == 0
    {
        if num_user_set_coll_part_mol_weights > 0 {
            if num_user_set_coll_part_names == 0 {
                eprintln!(
                    "WARNING: The user-set molecular weights will be ignored because no collision partner names were set."
                );
            } else {
                eprintln!(
                    "WARNING: The number of user-set molecular weights does not match the number of density values. The user-set values will be ignored."
                );
            }
            // num_user_set_coll_part_mol_weights = 0;
        }
    } else {
        // Resize the vector to the number of densities
        par.collisional_partner_mol_weights = vec![0.0; par.num_densities];
    }

    /* Now we do some sanity checks.
     */
    if num_user_set_coll_part_ids > 0 {
        // Check that they are unique
        let mut uniuqe_coll_part_ids = vec![0; num_user_set_coll_part_ids];
        for i in 0..num_user_set_coll_part_ids {
            for partner_id in &uniuqe_coll_part_ids {
                if par.collisional_partner_ids[i] == *partner_id {
                    bail!("The user-set collision partner IDs must be unique.");
                }
            }
            uniuqe_coll_part_ids[i] = par.collisional_partner_ids[i];
        }
    }

    if par.use_abun && num_user_set_nmol_weights > 0 {
        let mut sum = 0.0;
        for i in 0..num_user_set_nmol_weights {
            sum += par.nmol_weights[i];
        }
        if sum <= 0.0 {
            bail!("The user-set molecular weights must be positive.");
        }
    }
    Ok(())
}
