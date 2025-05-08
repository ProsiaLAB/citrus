// Some auxiliary functions
use std::io::{self, Write}; // Import the Write trait

use anyhow::Result;

use crate::config::Parameters;
use crate::constants as cc;
use crate::types::{RVecView, RVector};

use self::erf::erf;
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
    if t_kelvin < cc::CITRUS_GLOBAL_EPS {
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

pub fn get_erf(x0: f64, x1: f64) -> f64 {
    let val0 = erf(x0);
    let val1 = erf(x1);
    let diff = val1 - val0;
    let delta_x = x1 - x0;
    (diff / delta_x).abs()
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

pub mod qrng {
    //! A simple quasi-random number generator based on the Halton sequence.
    //! The Halton sequence is a sequence of numbers that are generated by
    //! taking the fractional part of a number raised to a power. The Halton
    //! sequence is defined for any positive integer `n`, and the power is
    //! defined as `2^n`.

    /// Halton State
    pub struct Halton {
        index: u64,
        dimension: usize,
    }

    #[derive(Debug)]
    pub enum HaltonError {
        DimensionTooLarge,
    }

    impl std::fmt::Display for HaltonError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                HaltonError::DimensionTooLarge => {
                    write!(f, "Halton sequence cannot have dimension greater than 1229")
                }
            }
        }
    }

    impl std::error::Error for HaltonError {}

    impl Halton {
        pub fn new(dimension: usize) -> Result<Self, HaltonError> {
            if dimension == 0 || dimension > PRIMES.len() {
                return Err(HaltonError::DimensionTooLarge);
            }
            Ok(Halton {
                index: 0,
                dimension,
            })
        }

        pub fn next_point(&mut self) -> Vec<f64> {
            let mut point = Vec::with_capacity(self.dimension);
            for &base in &PRIMES[..self.dimension] {
                point.push(halton(self.index, base));
            }
            self.index += 1;
            point
        }
    }

    fn halton(mut index: u64, base: u64) -> f64 {
        let mut result = 0.0;
        let mut f = 1.0 / base as f64;
        while index > 0 {
            result += f * (index % base) as f64;
            index /= base;
            f /= base as f64;
        }
        result
    }

    static PRIMES: &[u64; 1229] = &[
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
        191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
        283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
        401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
        509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619,
        631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743,
        751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
        877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997,
        1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093,
        1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213,
        1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303,
        1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439,
        1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543,
        1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627,
        1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753,
        1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877,
        1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999,
        2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111,
        2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239,
        2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347,
        2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447,
        2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593,
        2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699,
        2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801,
        2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927,
        2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061,
        3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203,
        3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323,
        3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457,
        3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557,
        3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673,
        3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797,
        3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919,
        3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049,
        4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159,
        4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283,
        4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441,
        4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561,
        4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679,
        4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813,
        4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957,
        4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077,
        5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209,
        5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351,
        5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471, 5477,
        5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591,
        5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717,
        5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849,
        5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987,
        6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121,
        6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257,
        6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359,
        6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521,
        6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659,
        6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781,
        6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907,
        6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019,
        7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187,
        7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321,
        7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487,
        7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583,
        7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703,
        7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867,
        7873, 7877, 7879, 7883, 7901, 7907, 7919, 7927, 7933, 7937, 7949, 7951, 7963, 7993, 8009,
        8011, 8017, 8039, 8053, 8059, 8069, 8081, 8087, 8089, 8093, 8101, 8111, 8117, 8123, 8147,
        8161, 8167, 8171, 8179, 8191, 8209, 8219, 8221, 8231, 8233, 8237, 8243, 8263, 8269, 8273,
        8287, 8291, 8293, 8297, 8311, 8317, 8329, 8353, 8363, 8369, 8377, 8387, 8389, 8419, 8423,
        8429, 8431, 8443, 8447, 8461, 8467, 8501, 8513, 8521, 8527, 8537, 8539, 8543, 8563, 8573,
        8581, 8597, 8599, 8609, 8623, 8627, 8629, 8641, 8647, 8663, 8669, 8677, 8681, 8689, 8693,
        8699, 8707, 8713, 8719, 8731, 8737, 8741, 8747, 8753, 8761, 8779, 8783, 8803, 8807, 8819,
        8821, 8831, 8837, 8839, 8849, 8861, 8863, 8867, 8887, 8893, 8923, 8929, 8933, 8941, 8951,
        8963, 8969, 8971, 8999, 9001, 9007, 9011, 9013, 9029, 9041, 9043, 9049, 9059, 9067, 9091,
        9103, 9109, 9127, 9133, 9137, 9151, 9157, 9161, 9173, 9181, 9187, 9199, 9203, 9209, 9221,
        9227, 9239, 9241, 9257, 9277, 9281, 9283, 9293, 9311, 9319, 9323, 9337, 9341, 9343, 9349,
        9371, 9377, 9391, 9397, 9403, 9413, 9419, 9421, 9431, 9433, 9437, 9439, 9461, 9463, 9467,
        9473, 9479, 9491, 9497, 9511, 9521, 9533, 9539, 9547, 9551, 9587, 9601, 9613, 9619, 9623,
        9629, 9631, 9643, 9649, 9661, 9677, 9679, 9689, 9697, 9719, 9721, 9733, 9739, 9743, 9749,
        9767, 9769, 9781, 9787, 9791, 9803, 9811, 9817, 9829, 9833, 9839, 9851, 9857, 9859, 9871,
        9883, 9887, 9901, 9907, 9923, 9929, 9931, 9941, 9949, 9967, 9973,
    ];
}

pub mod erf {
    //! Provides the [error](https://en.wikipedia.org/wiki/Error_function) and
    //! related functions
    //!
    //! This module is directly copied from the `statrs` crate.
    //! The original code is licensed under the MIT license.
    //! It was purely done to avoid a dependency on the `statrs` crate.
    //!
    //! The original code can be found here:
    //! <https://github.com/statrs-dev/statrs/blob/master/src/function/erf.rs>
    //!

    #![allow(clippy::excessive_precision)]
    use std::f64;

    use super::aux;

    /// `erf` calculates the error function at `x`.
    pub fn erf(x: f64) -> f64 {
        if x.is_nan() {
            f64::NAN
        } else if x >= 0.0 && x.is_infinite() {
            1.0
        } else if x <= 0.0 && x.is_infinite() {
            -1.0
        } else if x == 0.0 {
            0.0
        } else {
            erf_impl(x, false)
        }
    }

    /// `erf_inv` calculates the inverse error function
    /// at `x`.
    pub fn erf_inv(x: f64) -> f64 {
        if x == 0.0 {
            0.0
        } else if x >= 1.0 {
            f64::INFINITY
        } else if x <= -1.0 {
            f64::NEG_INFINITY
        } else if x < 0.0 {
            erf_inv_impl(-x, 1.0 + x, -1.0)
        } else {
            erf_inv_impl(x, 1.0 - x, 1.0)
        }
    }

    /// `erfc` calculates the complementary error function
    /// at `x`.
    pub fn erfc(x: f64) -> f64 {
        if x.is_nan() {
            f64::NAN
        } else if x == f64::INFINITY {
            0.0
        } else if x == f64::NEG_INFINITY {
            2.0
        } else {
            erf_impl(x, true)
        }
    }

    /// `erfc_inv` calculates the complementary inverse
    /// error function at `x`.
    pub fn erfc_inv(x: f64) -> f64 {
        if x <= 0.0 {
            f64::INFINITY
        } else if x >= 2.0 {
            f64::NEG_INFINITY
        } else if x > 1.0 {
            erf_inv_impl(-1.0 + x, 2.0 - x, -1.0)
        } else {
            erf_inv_impl(1.0 - x, x, 1.0)
        }
    }

    // **********************************************************
    // ********** Coefficients for erf_impl polynomial **********
    // **********************************************************

    /// Polynomial coefficients for a numerator of `erf_impl`
    /// in the interval [1e-10, 0.5].
    const ERF_IMPL_AN: &[f64] = &[
        0.00337916709551257388990745,
        -0.00073695653048167948530905,
        -0.374732337392919607868241,
        0.0817442448733587196071743,
        -0.0421089319936548595203468,
        0.0070165709512095756344528,
        -0.00495091255982435110337458,
        0.000871646599037922480317225,
    ];

    /// Polynomial coefficients for a denominator of `erf_impl`
    /// in the interval [1e-10, 0.5]
    const ERF_IMPL_AD: &[f64] = &[
        1.0,
        -0.218088218087924645390535,
        0.412542972725442099083918,
        -0.0841891147873106755410271,
        0.0655338856400241519690695,
        -0.0120019604454941768171266,
        0.00408165558926174048329689,
        -0.000615900721557769691924509,
    ];

    /// Polynomial coefficients for a numerator in `erf_impl`
    /// in the interval [0.5, 0.75].
    const ERF_IMPL_BN: &[f64] = &[
        -0.0361790390718262471360258,
        0.292251883444882683221149,
        0.281447041797604512774415,
        0.125610208862766947294894,
        0.0274135028268930549240776,
        0.00250839672168065762786937,
    ];

    /// Polynomial coefficients for a denominator in `erf_impl`
    /// in the interval [0.5, 0.75].
    const ERF_IMPL_BD: &[f64] = &[
        1.0,
        1.8545005897903486499845,
        1.43575803037831418074962,
        0.582827658753036572454135,
        0.124810476932949746447682,
        0.0113724176546353285778481,
    ];

    /// Polynomial coefficients for a numerator in `erf_impl`
    /// in the interval [0.75, 1.25].
    const ERF_IMPL_CN: &[f64] = &[
        -0.0397876892611136856954425,
        0.153165212467878293257683,
        0.191260295600936245503129,
        0.10276327061989304213645,
        0.029637090615738836726027,
        0.0046093486780275489468812,
        0.000307607820348680180548455,
    ];

    /// Polynomial coefficients for a denominator in `erf_impl`
    /// in the interval [0.75, 1.25].
    const ERF_IMPL_CD: &[f64] = &[
        1.0,
        1.95520072987627704987886,
        1.64762317199384860109595,
        0.768238607022126250082483,
        0.209793185936509782784315,
        0.0319569316899913392596356,
        0.00213363160895785378615014,
    ];

    /// Polynomial coefficients for a numerator in `erf_impl`
    /// in the interval [1.25, 2.25].
    const ERF_IMPL_DN: &[f64] = &[
        -0.0300838560557949717328341,
        0.0538578829844454508530552,
        0.0726211541651914182692959,
        0.0367628469888049348429018,
        0.00964629015572527529605267,
        0.00133453480075291076745275,
        0.778087599782504251917881e-4,
    ];

    /// Polynomial coefficients for a denominator in `erf_impl`
    /// in the interval [1.25, 2.25].
    const ERF_IMPL_DD: &[f64] = &[
        1.0,
        1.75967098147167528287343,
        1.32883571437961120556307,
        0.552528596508757581287907,
        0.133793056941332861912279,
        0.0179509645176280768640766,
        0.00104712440019937356634038,
        -0.106640381820357337177643e-7,
    ];

    ///  Polynomial coefficients for a numerator in `erf_impl`
    /// in the interval [2.25, 3.5].
    const ERF_IMPL_EN: &[f64] = &[
        -0.0117907570137227847827732,
        0.014262132090538809896674,
        0.0202234435902960820020765,
        0.00930668299990432009042239,
        0.00213357802422065994322516,
        0.00025022987386460102395382,
        0.120534912219588189822126e-4,
    ];

    /// Polynomial coefficients for a denominator in `erf_impl`
    /// in the interval [2.25, 3.5].
    const ERF_IMPL_ED: &[f64] = &[
        1.0,
        1.50376225203620482047419,
        0.965397786204462896346934,
        0.339265230476796681555511,
        0.0689740649541569716897427,
        0.00771060262491768307365526,
        0.000371421101531069302990367,
    ];

    /// Polynomial coefficients for a numerator in `erf_impl`
    /// in the interval [3.5, 5.25].
    const ERF_IMPL_FN: &[f64] = &[
        -0.00546954795538729307482955,
        0.00404190278731707110245394,
        0.0054963369553161170521356,
        0.00212616472603945399437862,
        0.000394984014495083900689956,
        0.365565477064442377259271e-4,
        0.135485897109932323253786e-5,
    ];

    /// Polynomial coefficients for a denominator in `erf_impl`
    /// in the interval [3.5, 5.25].
    const ERF_IMPL_FD: &[f64] = &[
        1.0,
        1.21019697773630784832251,
        0.620914668221143886601045,
        0.173038430661142762569515,
        0.0276550813773432047594539,
        0.00240625974424309709745382,
        0.891811817251336577241006e-4,
        -0.465528836283382684461025e-11,
    ];

    /// Polynomial coefficients for a numerator in `erf_impl`
    /// in the interval [5.25, 8].
    const ERF_IMPL_GN: &[f64] = &[
        -0.00270722535905778347999196,
        0.0013187563425029400461378,
        0.00119925933261002333923989,
        0.00027849619811344664248235,
        0.267822988218331849989363e-4,
        0.923043672315028197865066e-6,
    ];

    /// Polynomial coefficients for a denominator in `erf_impl`
    /// in the interval [5.25, 8].
    const ERF_IMPL_GD: &[f64] = &[
        1.0,
        0.814632808543141591118279,
        0.268901665856299542168425,
        0.0449877216103041118694989,
        0.00381759663320248459168994,
        0.000131571897888596914350697,
        0.404815359675764138445257e-11,
    ];

    /// Polynomial coefficients for a numerator in `erf_impl`
    /// in the interval [8, 11.5].
    const ERF_IMPL_HN: &[f64] = &[
        -0.00109946720691742196814323,
        0.000406425442750422675169153,
        0.000274499489416900707787024,
        0.465293770646659383436343e-4,
        0.320955425395767463401993e-5,
        0.778286018145020892261936e-7,
    ];

    /// Polynomial coefficients for a denominator in `erf_impl`
    /// in the interval [8, 11.5].
    const ERF_IMPL_HD: &[f64] = &[
        1.0,
        0.588173710611846046373373,
        0.139363331289409746077541,
        0.0166329340417083678763028,
        0.00100023921310234908642639,
        0.24254837521587225125068e-4,
    ];

    /// Polynomial coefficients for a numerator in `erf_impl`
    /// in the interval [11.5, 17].
    const ERF_IMPL_IN: &[f64] = &[
        -0.00056907993601094962855594,
        0.000169498540373762264416984,
        0.518472354581100890120501e-4,
        0.382819312231928859704678e-5,
        0.824989931281894431781794e-7,
    ];

    /// Polynomial coefficients for a denominator in `erf_impl`
    /// in the interval [11.5, 17].
    const ERF_IMPL_ID: &[f64] = &[
        1.0,
        0.339637250051139347430323,
        0.043472647870310663055044,
        0.00248549335224637114641629,
        0.535633305337152900549536e-4,
        -0.117490944405459578783846e-12,
    ];

    /// Polynomial coefficients for a numerator in `erf_impl`
    /// in the interval [17, 24].
    const ERF_IMPL_JN: &[f64] = &[
        -0.000241313599483991337479091,
        0.574224975202501512365975e-4,
        0.115998962927383778460557e-4,
        0.581762134402593739370875e-6,
        0.853971555085673614607418e-8,
    ];

    /// Polynomial coefficients for a denominator in `erf_impl`
    /// in the interval [17, 24].
    const ERF_IMPL_JD: &[f64] = &[
        1.0,
        0.233044138299687841018015,
        0.0204186940546440312625597,
        0.000797185647564398289151125,
        0.117019281670172327758019e-4,
    ];

    /// Polynomial coefficients for a numerator in `erf_impl`
    /// in the interval [24, 38].
    const ERF_IMPL_KN: &[f64] = &[
        -0.000146674699277760365803642,
        0.162666552112280519955647e-4,
        0.269116248509165239294897e-5,
        0.979584479468091935086972e-7,
        0.101994647625723465722285e-8,
    ];

    /// Polynomial coefficients for a denominator in `erf_impl`
    /// in the interval [24, 38].
    const ERF_IMPL_KD: &[f64] = &[
        1.0,
        0.165907812944847226546036,
        0.0103361716191505884359634,
        0.000286593026373868366935721,
        0.298401570840900340874568e-5,
    ];

    /// Polynomial coefficients for a numerator in `erf_impl`
    /// in the interval [38, 60].
    const ERF_IMPL_LN: &[f64] = &[
        -0.583905797629771786720406e-4,
        0.412510325105496173512992e-5,
        0.431790922420250949096906e-6,
        0.993365155590013193345569e-8,
        0.653480510020104699270084e-10,
    ];

    /// Polynomial coefficients for a denominator in `erf_impl`
    /// in the interval [38, 60].
    const ERF_IMPL_LD: &[f64] = &[
        1.0,
        0.105077086072039915406159,
        0.00414278428675475620830226,
        0.726338754644523769144108e-4,
        0.477818471047398785369849e-6,
    ];

    /// Polynomial coefficients for a numerator in `erf_impl`
    /// in the interval [60, 85].
    const ERF_IMPL_MN: &[f64] = &[
        -0.196457797609229579459841e-4,
        0.157243887666800692441195e-5,
        0.543902511192700878690335e-7,
        0.317472492369117710852685e-9,
    ];

    /// Polynomial coefficients for a denominator in `erf_impl`
    /// in the interval [60, 85].
    const ERF_IMPL_MD: &[f64] = &[
        1.0,
        0.052803989240957632204885,
        0.000926876069151753290378112,
        0.541011723226630257077328e-5,
        0.535093845803642394908747e-15,
    ];

    /// Polynomial coefficients for a numerator in `erf_impl`
    /// in the interval [85, 110].
    const ERF_IMPL_NN: &[f64] = &[
        -0.789224703978722689089794e-5,
        0.622088451660986955124162e-6,
        0.145728445676882396797184e-7,
        0.603715505542715364529243e-10,
    ];

    /// Polynomial coefficients for a denominator in `erf_impl`
    /// in the interval [85, 110].
    const ERF_IMPL_ND: &[f64] = &[
        1.0,
        0.0375328846356293715248719,
        0.000467919535974625308126054,
        0.193847039275845656900547e-5,
    ];

    // **********************************************************
    // ********** Coefficients for erf_inv_impl polynomial ******
    // **********************************************************

    /// Polynomial coefficients for a numerator of `erf_inv_impl`
    /// in the interval [0, 0.5].
    const ERF_INV_IMPL_AN: &[f64] = &[
        -0.000508781949658280665617,
        -0.00836874819741736770379,
        0.0334806625409744615033,
        -0.0126926147662974029034,
        -0.0365637971411762664006,
        0.0219878681111168899165,
        0.00822687874676915743155,
        -0.00538772965071242932965,
    ];

    /// Polynomial coefficients for a denominator of `erf_inv_impl`
    /// in the interval [0, 0.5].
    const ERF_INV_IMPL_AD: &[f64] = &[
        1.0,
        -0.970005043303290640362,
        -1.56574558234175846809,
        1.56221558398423026363,
        0.662328840472002992063,
        -0.71228902341542847553,
        -0.0527396382340099713954,
        0.0795283687341571680018,
        -0.00233393759374190016776,
        0.000886216390456424707504,
    ];

    /// Polynomial coefficients for a numerator of `erf_inv_impl`
    /// in the interval [0.5, 0.75].
    const ERF_INV_IMPL_BN: &[f64] = &[
        -0.202433508355938759655,
        0.105264680699391713268,
        8.37050328343119927838,
        17.6447298408374015486,
        -18.8510648058714251895,
        -44.6382324441786960818,
        17.445385985570866523,
        21.1294655448340526258,
        -3.67192254707729348546,
    ];

    /// Polynomial coefficients for a denominator of `erf_inv_impl`
    /// in the interval [0.5, 0.75].
    const ERF_INV_IMPL_BD: &[f64] = &[
        1.0,
        6.24264124854247537712,
        3.9713437953343869095,
        -28.6608180499800029974,
        -20.1432634680485188801,
        48.5609213108739935468,
        10.8268667355460159008,
        -22.6436933413139721736,
        1.72114765761200282724,
    ];

    /// Polynomial coefficients for a numerator of `erf_inv_impl`
    /// in the interval [0.75, 1] with x less than 3.
    const ERF_INV_IMPL_CN: &[f64] = &[
        -0.131102781679951906451,
        -0.163794047193317060787,
        0.117030156341995252019,
        0.387079738972604337464,
        0.337785538912035898924,
        0.142869534408157156766,
        0.0290157910005329060432,
        0.00214558995388805277169,
        -0.679465575181126350155e-6,
        0.285225331782217055858e-7,
        -0.681149956853776992068e-9,
    ];

    /// Polynomial coefficients for a denominator of `erf_inv_impl`
    /// in the interval [0.75, 1] with x less than 3.
    const ERF_INV_IMPL_CD: &[f64] = &[
        1.0,
        3.46625407242567245975,
        5.38168345707006855425,
        4.77846592945843778382,
        2.59301921623620271374,
        0.848854343457902036425,
        0.152264338295331783612,
        0.01105924229346489121,
    ];

    /// Polynomial coefficients for a numerator of `erf_inv_impl`
    /// in the interval [0.75, 1] with x between 3 and 6.
    const ERF_INV_IMPL_DN: &[f64] = &[
        -0.0350353787183177984712,
        -0.00222426529213447927281,
        0.0185573306514231072324,
        0.00950804701325919603619,
        0.00187123492819559223345,
        0.000157544617424960554631,
        0.460469890584317994083e-5,
        -0.230404776911882601748e-9,
        0.266339227425782031962e-11,
    ];

    /// Polynomial coefficients for a denominator of `erf_inv_impl`
    /// in the interval [0.75, 1] with x between 3 and 6.
    const ERF_INV_IMPL_DD: &[f64] = &[
        1.0,
        1.3653349817554063097,
        0.762059164553623404043,
        0.220091105764131249824,
        0.0341589143670947727934,
        0.00263861676657015992959,
        0.764675292302794483503e-4,
    ];

    /// Polynomial coefficients for a numerator of `erf_inv_impl`
    /// in the interval [0.75, 1] with x between 6 and 18.
    const ERF_INV_IMPL_EN: &[f64] = &[
        -0.0167431005076633737133,
        -0.00112951438745580278863,
        0.00105628862152492910091,
        0.000209386317487588078668,
        0.149624783758342370182e-4,
        0.449696789927706453732e-6,
        0.462596163522878599135e-8,
        -0.281128735628831791805e-13,
        0.99055709973310326855e-16,
    ];

    /// Polynomial coefficients for a denominator of `erf_inv_impl`
    /// in the interval [0.75, 1] with x between 6 and 18.
    const ERF_INV_IMPL_ED: &[f64] = &[
        1.0,
        0.591429344886417493481,
        0.138151865749083321638,
        0.0160746087093676504695,
        0.000964011807005165528527,
        0.275335474764726041141e-4,
        0.282243172016108031869e-6,
    ];

    /// Polynomial coefficients for a numerator of `erf_inv_impl`
    /// in the interval [0.75, 1] with x between 18 and 44.
    const ERF_INV_IMPL_FN: &[f64] = &[
        -0.0024978212791898131227,
        -0.779190719229053954292e-5,
        0.254723037413027451751e-4,
        0.162397777342510920873e-5,
        0.396341011304801168516e-7,
        0.411632831190944208473e-9,
        0.145596286718675035587e-11,
        -0.116765012397184275695e-17,
    ];

    /// Polynomial coefficients for a denominator of `erf_inv_impl`
    /// in the interval [0.75, 1] with x between 18 and 44.
    const ERF_INV_IMPL_FD: &[f64] = &[
        1.0,
        0.207123112214422517181,
        0.0169410838120975906478,
        0.000690538265622684595676,
        0.145007359818232637924e-4,
        0.144437756628144157666e-6,
        0.509761276599778486139e-9,
    ];

    /// Polynomial coefficients for a numerator of `erf_inv_impl`
    /// in the interval [0.75, 1] with x greater than 44.
    const ERF_INV_IMPL_GN: &[f64] = &[
        -0.000539042911019078575891,
        -0.28398759004727721098e-6,
        0.899465114892291446442e-6,
        0.229345859265920864296e-7,
        0.225561444863500149219e-9,
        0.947846627503022684216e-12,
        0.135880130108924861008e-14,
        -0.348890393399948882918e-21,
    ];

    /// Polynomial coefficients for a denominator of `erf_inv_impl`
    /// in the interval [0.75, 1] with x greater than 44.
    const ERF_INV_IMPL_GD: &[f64] = &[
        1.0,
        0.0845746234001899436914,
        0.00282092984726264681981,
        0.468292921940894236786e-4,
        0.399968812193862100054e-6,
        0.161809290887904476097e-8,
        0.231558608310259605225e-11,
    ];

    /// `erf_impl` computes the error function at `z`.
    /// If `inv` is true, `1 - erf` is calculated as opposed to `erf`
    fn erf_impl(z: f64, inv: bool) -> f64 {
        if z < 0.0 {
            if !inv {
                return -erf_impl(-z, false);
            }
            if z < -0.5 {
                return 2.0 - erf_impl(-z, true);
            }
            return 1.0 + erf_impl(-z, false);
        }

        let result = if z < 0.5 {
            if z < 1e-10 {
                z * 1.125 + z * 0.003379167095512573896158903121545171688
            } else {
                z * 1.125 + z * aux::polynomial(z, ERF_IMPL_AN) / aux::polynomial(z, ERF_IMPL_AD)
            }
        } else if z < 110.0 {
            let (r, b) = if z < 0.75 {
                (
                    aux::polynomial(z - 0.5, ERF_IMPL_BN) / aux::polynomial(z - 0.5, ERF_IMPL_BD),
                    0.3440242112,
                )
            } else if z < 1.25 {
                (
                    aux::polynomial(z - 0.75, ERF_IMPL_CN) / aux::polynomial(z - 0.75, ERF_IMPL_CD),
                    0.419990927,
                )
            } else if z < 2.25 {
                (
                    aux::polynomial(z - 1.25, ERF_IMPL_DN) / aux::polynomial(z - 1.25, ERF_IMPL_DD),
                    0.4898625016,
                )
            } else if z < 3.5 {
                (
                    aux::polynomial(z - 2.25, ERF_IMPL_EN) / aux::polynomial(z - 2.25, ERF_IMPL_ED),
                    0.5317370892,
                )
            } else if z < 5.25 {
                (
                    aux::polynomial(z - 3.5, ERF_IMPL_FN) / aux::polynomial(z - 3.5, ERF_IMPL_FD),
                    0.5489973426,
                )
            } else if z < 8.0 {
                (
                    aux::polynomial(z - 5.25, ERF_IMPL_GN) / aux::polynomial(z - 5.25, ERF_IMPL_GD),
                    0.5571740866,
                )
            } else if z < 11.5 {
                (
                    aux::polynomial(z - 8.0, ERF_IMPL_HN) / aux::polynomial(z - 8.0, ERF_IMPL_HD),
                    0.5609807968,
                )
            } else if z < 17.0 {
                (
                    aux::polynomial(z - 11.5, ERF_IMPL_IN) / aux::polynomial(z - 11.5, ERF_IMPL_ID),
                    0.5626493692,
                )
            } else if z < 24.0 {
                (
                    aux::polynomial(z - 17.0, ERF_IMPL_JN) / aux::polynomial(z - 17.0, ERF_IMPL_JD),
                    0.5634598136,
                )
            } else if z < 38.0 {
                (
                    aux::polynomial(z - 24.0, ERF_IMPL_KN) / aux::polynomial(z - 24.0, ERF_IMPL_KD),
                    0.5638477802,
                )
            } else if z < 60.0 {
                (
                    aux::polynomial(z - 38.0, ERF_IMPL_LN) / aux::polynomial(z - 38.0, ERF_IMPL_LD),
                    0.5640528202,
                )
            } else if z < 85.0 {
                (
                    aux::polynomial(z - 60.0, ERF_IMPL_MN) / aux::polynomial(z - 60.0, ERF_IMPL_MD),
                    0.5641309023,
                )
            } else {
                (
                    aux::polynomial(z - 85.0, ERF_IMPL_NN) / aux::polynomial(z - 85.0, ERF_IMPL_ND),
                    0.5641584396,
                )
            };
            let g = (-z * z).exp() / z;
            g * b + g * r
        } else {
            0.0
        };

        if inv && z >= 0.5 {
            result
        } else if z >= 0.5 || inv {
            1.0 - result
        } else {
            result
        }
    }

    // `erf_inv_impl` computes the inverse error function where
    // `p`,`q`, and `s` are the first, second, and third intermediate
    // parameters respectively
    fn erf_inv_impl(p: f64, q: f64, s: f64) -> f64 {
        let result = if p <= 0.5 {
            let y = 0.0891314744949340820313;
            let g = p * (p + 10.0);
            let r = aux::polynomial(p, ERF_INV_IMPL_AN) / aux::polynomial(p, ERF_INV_IMPL_AD);
            g * y + g * r
        } else if q >= 0.25 {
            let y = 2.249481201171875;
            let g = (-2.0 * q.ln()).sqrt();
            let xs = q - 0.25;
            let r = aux::polynomial(xs, ERF_INV_IMPL_BN) / aux::polynomial(xs, ERF_INV_IMPL_BD);
            g / (y + r)
        } else {
            let x = (-q.ln()).sqrt();
            if x < 3.0 {
                let y = 0.807220458984375;
                let xs = x - 1.125;
                let r = aux::polynomial(xs, ERF_INV_IMPL_CN) / aux::polynomial(xs, ERF_INV_IMPL_CD);
                y * x + r * x
            } else if x < 6.0 {
                let y = 0.93995571136474609375;
                let xs = x - 3.0;
                let r = aux::polynomial(xs, ERF_INV_IMPL_DN) / aux::polynomial(xs, ERF_INV_IMPL_DD);
                y * x + r * x
            } else if x < 18.0 {
                let y = 0.98362827301025390625;
                let xs = x - 6.0;
                let r = aux::polynomial(xs, ERF_INV_IMPL_EN) / aux::polynomial(xs, ERF_INV_IMPL_ED);
                y * x + r * x
            } else if x < 44.0 {
                let y = 0.99714565277099609375;
                let xs = x - 18.0;
                let r = aux::polynomial(xs, ERF_INV_IMPL_FN) / aux::polynomial(xs, ERF_INV_IMPL_FD);
                y * x + r * x
            } else {
                let y = 0.99941349029541015625;
                let xs = x - 44.0;
                let r = aux::polynomial(xs, ERF_INV_IMPL_GN) / aux::polynomial(xs, ERF_INV_IMPL_GD);
                y * x + r * x
            }
        };
        s * result
    }
}

pub mod aux {
    //! Auxiliary functions
    //!
    //! This module contains code from elsewhere which may be required by other
    //! functions in this module.
    //!
    //! # Evaluation of polynomials
    //! Provides functions that don't have a numerical solution and must
    //! be solved computationally (e.g. evaluation of a polynomial)
    //!
    //! This module is directly copied from the `statrs` crate.
    //! The original code is licensed under the MIT license.
    //! It was purely done to avoid a dependency on the `statrs` crate.
    //!
    //! The original code can be found here:
    //! <https://github.com/statrs-dev/statrs/blob/master/src/function/evaluate.rs>

    /// evaluates a polynomial at `z` where `coeff` are the coeffecients
    /// to a polynomial of order `k` where `k` is the length of `coeff` and the
    /// coeffecient
    /// to the `k`th power is the `k`th element in coeff. E.g. [3,-1,2] equates to
    /// `2z^2 - z + 3`
    ///
    /// # Remarks
    ///
    /// Returns 0 for a 0 length coefficient slice
    pub fn polynomial(z: f64, coeff: &[f64]) -> f64 {
        let n = coeff.len();
        if n == 0 {
            return 0.0;
        }

        let mut sum = *coeff.last().unwrap();
        for c in coeff[0..n - 1].iter().rev() {
            sum = *c + z * sum;
        }
        sum
    }
}
