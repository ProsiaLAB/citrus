use anyhow::Result;
use anyhow::bail;
use prosia_extensions::types::{RVector, UVector};

use crate::config::Parameters;

#[derive(Debug, Default)]
pub struct CollisionalPartnerData {
    pub down: RVector,
    pub temp: RVector,
    pub collisional_partner_id: isize,
    pub ntemp: isize,
    pub ntrans: isize,
    pub lcl: UVector,
    pub lcu: UVector,
    pub density_index: isize,
    pub name: String,
}

#[derive(Debug, Default)]
pub struct MolData {
    pub nlev: usize,
    pub nline: usize,
    pub npart: usize,
    pub lal: UVector,
    pub lau: UVector,
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

/// This deals with four user-settable fields of [`Parameters`] which relate
/// to collision partners and their number densities: `collisional_partner_ids`, `nmol_weights`,
/// `collisional_partner_mol_weights` and `collisional_partner_names`. We have to see if these
/// (optional) parameters were set, do some basic checks on them, and if they were
/// set make sure they match the number of density values, which by this time should
/// be stored in `num_densities` field of [`Parameters`].
///
///
///
/// The user can specify either, none, or both of these two parameters, with
/// the following effects:
///
///  Ids | Names  | Effect
///  --- | ------ | ------------
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
    todo!()
}
