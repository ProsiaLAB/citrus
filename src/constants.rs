/// Atomic mass unit (AMU) in kilograms
/// 1 atomic mass unit = 1.66053904 × 10⁻²⁷ kg
pub const AMU_SI: f64 = 1.66053904e-27;

/// Speed of light in vacuum in meters per second
/// 2.99792458 × 10⁸ m/s
pub const SPEED_OF_LIGHT_SI: f64 = 2.99792458e8;

/// Planck constant in joule seconds
/// 6.626070040 × 10⁻³⁴ J·s
pub const PLANCK_SI: f64 = 6.626070040e-34;

/// Boltzmann constant in joules per kelvin
/// 1.38064852 × 10⁻²³ J/K
pub const BOLTZMANN_SI: f64 = 1.38064852e-23;

/// Length of the Julian year in seconds
/// 365.25 days * 24 hours * 3600 seconds
pub const JULIAN_YEAR_SI: f64 = 365.25 * 24.0 * 3600.0;

/// Stefan-Boltzmann constant in watts per square meter per kelvin to the fourth power
/// 5.670367 × 10⁻⁸ W/m²/K⁴
pub const STEFAN_BOLTZMANN_SI: f64 = 5.670367e-8;

/// Gravitational constant in cubic meters per kilogram per second squared
/// 6.67428 × 10⁻¹¹ m³/kg/s²
pub const GRAVITATIONAL_CONST_SI: f64 = 6.67428e-11;

/// Astronomical unit (AU) in meters
/// 1 AU = 1.495978707 × 10¹¹ m
pub const AU_SI: f64 = 1.495978707e11;

/// Local mean Cosmic Microwave Background (CMB) temperature in kelvin
/// Temperature value from Fixsen (2009): 2.72548 K
pub const LOCAL_CMB_TEMP_SI: f64 = 2.72548;

/// Parsecs in meters
/// 1 parsec ≈ 3.08567758 × 10¹⁶ m
pub const PARSEC_SI: f64 = 3.08567758e16;

/// Solar mass in kilograms
/// 1 Solar mass = 1.9891 × 10³⁰ kg
pub const SOLAR_MASS_SI: f64 = 1.9891e30;

/// Solar radius in meters
/// 1 Solar radius = 6.957 × 10⁸ m
pub const SOLAR_RADIUS_SI: f64 = 6.957e8;

// CGS Units (Centimeter-Gram-Second system):

/// Atomic mass unit in grams
/// 1 AMU = 1.66053904 × 10⁻²⁷ kg = 1.66053904 × 10⁻²⁴ g
pub const AMU_CGS: f64 = AMU_SI * 1000.0;

/// Speed of light in vacuum in centimeters per second
/// 1 c = 2.99792458 × 10⁸ m/s = 2.99792458 × 10¹⁰ cm/s
pub const SPEED_OF_LIGHT_CGS: f64 = SPEED_OF_LIGHT_SI * 100.0;

/// Planck constant in erg seconds
/// 1 erg = 1 × 10⁻⁷ joules
pub const PLANCK_CGS: f64 = PLANCK_SI * 1.0e7;

/// Boltzmann constant in ergs per kelvin
/// 1 erg/K = 1.38064852 × 10⁻²³ J/K × 10⁷
pub const BOLTZMANN_CGS: f64 = BOLTZMANN_SI * 1.0e7;

/// Stefan-Boltzmann constant in erg per square centimeter per kelvin to the fourth power per second
/// 5.670367 × 10⁻⁸ W/m²/K⁴ = 5.670367 × 10⁻⁵ erg/cm²/K⁴/s
pub const STEFAN_BOLTZMANN_CGS: f64 = STEFAN_BOLTZMANN_SI * 1000.0;

/// Gravitational constant in cubic centimeters per gram per second squared
/// 6.67428 × 10⁻¹¹ m³/kg/s² = 6.67428 × 10⁻⁸ cm³/g/s²
pub const GRAVITATIONAL_CONST_CGS: f64 = GRAVITATIONAL_CONST_SI * 1000.0;

/// Astronomical unit in centimeters
/// 1 AU = 1.495978707 × 10¹¹ m = 1.495978707 × 10¹³ cm
pub const AU_CGS: f64 = AU_SI * 100.0;

/// Solar mass in grams
/// 1 Solar mass = 1.9891 × 10³⁰ kg = 1.9891 × 10³³ g
pub const SOLAR_MASS_CGS: f64 = SOLAR_MASS_SI * 1000.0;

/// Solar radius in centimeters
/// 1 Solar radius = 6.957 × 10⁸ m = 6.957 × 10¹⁰ cm
pub const SOLAR_RADIUS_CGS: f64 = SOLAR_RADIUS_SI * 100.0;

/// Typical Interstellar Medium (ISM) density in particles per cubic centimeter
/// A typical value for the ISM is 1000 particles/cm³
pub const TYPICAL_ISM_DENS: f64 = 1000.0;

/// Constant: HPLANCK * CLIGHT / (4.0 * PI * SPI)
/// Value: 8.918502221e-27 (in appropriate units)
pub const HPIP: f64 = 8.918_502_221e-27;

/// Constant: 100.0 * HPLANCK * CLIGHT / KBOLTZ
/// Value: 1.43877735 (in appropriate units)
pub const HCKB: f64 = 1.438_777_35;

/// Conversion factor: Arcseconds to radians
/// 1 arcsec = π / 180 / 3600 radians
pub const ARCSEC_TO_RAD: f64 = std::f64::consts::PI / 180.0 / 3600.0;

/// A small number to avoid division by zero
/// Used to prevent numerical errors such as division by zero
pub const CITRUS_GLOBAL_EPS: f64 = 1e-30;
pub const CITRUS_RT_EPS: f64 = 1e-6;

pub const N_DIMS: usize = 3;
