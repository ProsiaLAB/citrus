// Some auxiliary functions
use std::io::{self, Write}; // Import the Write trait

use anyhow::Result;

use crate::config::Parameters;
use crate::constants as cc;
use crate::types::{RVecView, RVector};

use self::interp::{CubicSpline, SplineError};

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

    (remant_snu, exp_dtau)
}

pub fn planck_fn(freq: f64, t_kelvin: f64) -> f64 {
    if t_kelvin < cc::CITRUS_EPS {
        0.0
    } else {
        let wn = freq / cc::SPEED_OF_LIGHT_SI;
        if cc::PLANCK_SI * freq > 100.0 * cc::BOLTZMANN_SI * t_kelvin {
            2.0 * cc::PLANCK_SI
                * wn
                * wn
                * freq
                * (-cc::PLANCK_SI * freq / cc::BOLTZMANN_SI / t_kelvin).exp()
        } else {
            2.0 * cc::PLANCK_SI * wn * wn * freq
                / ((cc::PLANCK_SI * freq / cc::BOLTZMANN_SI / t_kelvin - 1.0).exp() - 1.0)
        }
    }
}

pub fn gauss_line(v: f64, sigma_inv: f64) -> f64 {
    v * v * sigma_inv * sigma_inv
}

/// This function:
/// 1. Converts frequency to wavelength in log10 space
/// 2. Handles extrapolation outside the table range
/// 3. Converts results from cm²/g to m²/kg (0.1 factor)
///
/// Parameters:
/// - freq: Frequency
/// - lamtab: Array of log10 wavelength values
/// - kaptab: Array of log10 opacity values
/// - n_entries: Number of entries in the arrays
pub fn interpolate_kappa(freq: f64, lam: &RVecView, kap: &RVecView) -> Result<f64, SplineError> {
    if lam.len() != kap.len() {
        return Err(SplineError::DifferentLengths);
    }

    // Create the spline
    let spline = CubicSpline::new(lam, kap)?;

    // Calculate log10 of wavelength (SPEED_OF_LIGHT_SI / freq)
    let loglam = (cc::SPEED_OF_LIGHT_SI / freq).log10();

    let n_entries = lam.len();
    let kappa_log10: f64;

    if loglam < lam[0] {
        // Below range: linear extrapolation in log space
        kappa_log10 = kap[0] + (loglam - lam[0]) * (kap[1] - kap[0]) / (lam[1] - lam[0]);
    } else if loglam > lam[n_entries - 1] {
        // Above range: linear extrapolation in log space
        kappa_log10 = kap[n_entries - 2]
            + (loglam - lam[n_entries - 2]) * (kap[n_entries - 1] - kap[n_entries - 2])
                / (lam[n_entries - 1] - lam[n_entries - 2]);
    } else {
        // In range: use the spline
        kappa_log10 = spline.interpolate(loglam)?;
    }

    // Convert from log10 opacity and apply unit conversion factor (cm²/g to m²/kg)
    let kappa = 0.1 * 10f64.powf(kappa_log10);

    Ok(kappa)
}

pub fn get_dust_temp(ts_kelvin: &[f64; 2]) -> f64 {
    if ts_kelvin[1] <= 0.0 {
        ts_kelvin[0]
    } else {
        ts_kelvin[1]
    }
}

pub fn get_dtg(par: &Parameters, dens: &RVecView, gtd: f64) -> f64 {
    if par.collisional_partner_user_set_flags == 0 {
        cc::AMU_SI * 2.4 * dens[0] / gtd
    } else {
        let mut gas_mass_density_amus = 0.0;
        for i in 0..par.num_densities {
            gas_mass_density_amus += dens[i] * par.collisional_partner_mol_weights[i];
        }
        cc::AMU_SI * gas_mass_density_amus / gtd
    }
}

pub fn calc_dust_data(
    knus: &mut RVector,
    dusts: &mut RVector,
    kap: &RVecView,
    freqs: &RVecView,
    t_kelvin: f64,
    dtg: f64,
    nlines: usize,
) {
    for iline in 0..nlines {
        knus[iline] = kap[iline] * dtg;
        dusts[iline] = planck_fn(freqs[iline], t_kelvin);
    }
}

pub fn progress_bar(progress: f64, width: usize) {
    let completed = (progress * width as f64).round() as usize;
    let remaining = width - completed;

    let bar: String = format!(
        "[{}{}] {:.2}%",
        "█".repeat(completed),
        "░".repeat(remaining),
        progress * 100.0
    );

    print!("\r{}", bar);
    io::stdout().flush().unwrap(); // Flush stdout to update the terminal
}

pub mod interp {
    use anyhow::Result;

    use crate::types::{RVecView, RVector};

    pub struct CubicSpline {
        x: RVector,
        y: RVector,
        // Second derivatives at each point
        y2: RVector,
    }

    impl CubicSpline {
        /// Create a new cubic spline interpolator
        pub fn new(x: &RVecView, y: &RVecView) -> Result<Self, SplineError> {
            if x.len() != y.len() {
                return Err(SplineError::DifferentLengths);
            }
            if x.len() < 2 {
                return Err(SplineError::InsufficientPoints);
            }

            // Check that x is sorted in ascending order
            for i in 1..x.len() {
                if x[i] <= x[i - 1] {
                    return Err(SplineError::UnsortedPoints);
                }
            }

            let n = x.len();
            let mut y2 = RVector::zeros(n);

            // This computes the second derivatives needed for the spline
            // Natural spline conditions: y''(x[0]) = y''(x[n-1]) = 0
            let mut u = vec![0.0; n - 1];

            // Decomposition loop of the tridiagonal algorithm
            for i in 1..n - 1 {
                let sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
                let p = sig * y2[i - 1] + 2.0;
                // Check for numerical stability
                if p.abs() < f64::EPSILON {
                    return Err(SplineError::NumericalError);
                }
                y2[i] = (sig - 1.0) / p;
                u[i] =
                    (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
                u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
            }

            // Back-substitution loop of the tridiagonal algorithm
            for i in (0..n - 1).rev() {
                y2[i] = y2[i] * y2[i + 1] + u[i];
            }

            Ok(CubicSpline {
                x: x.to_owned(),
                y: y.to_owned(),
                y2,
            })
        }

        /// Interpolate at the given x value
        pub fn interpolate(&self, x: f64) -> Result<f64, SplineError> {
            // Find the right interval using binary search
            let n = self.x.len();

            // Handle out-of-bounds
            if x < self.x[0] || x > self.x[n - 1] {
                return Err(SplineError::OutOfBounds);
            }

            // Binary search to find the right interval
            let mut low = 0;
            let mut high = n - 1;

            while high - low > 1 {
                let mid = (high + low) / 2;
                if self.x[mid] > x {
                    high = mid;
                } else {
                    low = mid;
                }
            }

            // Now x is between self.x[low] and self.x[high]
            let h = self.x[high] - self.x[low];
            if h == 0.0 {
                return Err(SplineError::PointsTooClose);
            }

            // Cubic spline polynomial evaluation
            let a = (self.x[high] - x) / h;
            let b = (x - self.x[low]) / h;

            // Evaluate the spline polynomial
            let y = a * self.y[low]
                + b * self.y[high]
                + ((a * a * a - a) * self.y2[low] + (b * b * b - b) * self.y2[high]) * (h * h)
                    / 6.0;

            Ok(y)
        }

        /// Evaluate the spline with extrapolation outside bounds
        /// This is a special method to handle cases where x is outside the range
        pub fn eval_with_extrapolation(&self, x: f64) -> Result<f64, SplineError> {
            let n = self.x.len();

            // Handle in-bounds case
            if x >= self.x[0] && x <= self.x[n - 1] {
                return self.interpolate(x);
            }

            // If we're out of bounds, use linear extrapolation
            if x < self.x[0] {
                // Linear extrapolation below range using the first two points
                let slope = (self.y[1] - self.y[0]) / (self.x[1] - self.x[0]);
                Ok(self.y[0] + slope * (x - self.x[0]))
            } else {
                // Linear extrapolation above range using the last two points
                let slope = (self.y[n - 1] - self.y[n - 2]) / (self.x[n - 1] - self.x[n - 2]);
                Ok(self.y[n - 1] + slope * (x - self.x[n - 1]))
            }
        }
    }
    #[derive(Debug, Clone, PartialEq)]
    pub enum SplineError {
        /// Input arrays have different lengths
        DifferentLengths,
        /// Not enough points provided for interpolation
        InsufficientPoints,
        /// Input x values must be in strictly ascending order
        UnsortedPoints,
        /// Interpolation requested outside range of input data
        OutOfBounds,
        /// Points in the input data are too close together
        PointsTooClose,
        /// Numerical instability detected during calculation
        NumericalError,
    }

    impl std::fmt::Display for SplineError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                SplineError::DifferentLengths => {
                    write!(f, "Input arrays must have the same length")
                }
                SplineError::InsufficientPoints => {
                    write!(f, "Need at least two points for interpolation")
                }
                SplineError::UnsortedPoints => {
                    write!(f, "Input x values must be in strictly ascending order")
                }
                SplineError::OutOfBounds => write!(f, "Interpolation x value out of bounds"),
                SplineError::PointsTooClose => write!(f, "Interpolation points too close"),
                SplineError::NumericalError => write!(f, "Numerical error during computation"),
            }
        }
    }

    impl std::error::Error for SplineError {}

    /// Create a new cubic spline from slices (for easier interop with C-like code)
    pub fn cubic_spline_from_slices(x: &[f64], y: &[f64]) -> Result<CubicSpline, SplineError> {
        let x_view = RVecView::from(x);
        let y_view = RVecView::from(y);
        CubicSpline::new(&x_view, &y_view)
    }
}
