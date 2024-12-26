// Some auxiliary functions

use crate::constants as cc;
// use rgsl::InterpAccel as GSLInterpAccel;

/// The source function `S` is defined as `j_nu/alpha`, which is clearly not
/// defined for `alpha==0`. However `S` is used in the algorithm only in the
/// term `(1-exp[-alpha*ds])*S`, which is defined for all values of alpha.
/// The present function calculates this term and returns it in the
/// argument remnantSnu. For values of `abs(alpha*ds)` less than a pre-
/// calculated cutoff supplied in `ConfigInfo`, a Taylor approximation is
/// used.
/// Note that the same cutoff condition holds for replacement of
/// `exp(-dTau)` by its Taylor expansion to 3rd order.
pub fn calc_source_fn(dtau: f64, taylor_cutoff: f64) -> (f64, f64) {
    let remant_snu: f64;
    let exp_dtau: f64;
    if dtau.abs() < taylor_cutoff {
        remant_snu = 1.0 - dtau * (1.0 - dtau * (1.0 / 3.0)) * (1.0 / 2.0);
        exp_dtau = 1.0 - dtau * (remant_snu);
    } else {
        exp_dtau = (-dtau).exp();
        remant_snu = (1.0 - exp_dtau) / dtau;
    }

    return (remant_snu, exp_dtau);
}

pub fn planck_fn(freq: f64, t_kelvin: f64) -> f64 {
    let bb: f64;

    if t_kelvin < cc::CITRUS_EPS {
        bb = 0.0;
    } else {
        let wn = freq / cc::SPEED_OF_LIGHT_SI;
        if cc::PLANCK_SI * freq > 100.0 * cc::BOLTZMANN_SI * t_kelvin {
            bb = 2.0
                * cc::PLANCK_SI
                * wn
                * wn
                * freq
                * (-cc::PLANCK_SI * freq / cc::BOLTZMANN_SI / t_kelvin).exp();
        } else {
            bb = 2.0 * cc::PLANCK_SI * wn * wn * freq
                / ((cc::PLANCK_SI * freq / cc::BOLTZMANN_SI / t_kelvin - 1.0).exp() - 1.0);
        }
    }

    return bb;
}

pub fn gauss_line(v: f64, one_on_sigma: f64) -> f64 {
    return v * v * one_on_sigma * one_on_sigma;
}

pub fn dot_product_3d(a: Vec<f64>, b: Vec<f64>) -> f64 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

pub fn interpolate_kappa() -> f64 {
    return 0.0;
}

pub fn calc_dust_data() -> f64 {
    return 0.0;
}
