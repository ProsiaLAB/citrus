pub const AMU_SI: f64 = 1.66053904e-27; // atomic mass unit             [kg]		*/
pub const SPEED_OF_LIGHT_SI: f64 = 2.99792458e8; // speed of light in vacuum     [m/s]		*/
pub const PLANCK_SI: f64 = 6.626070040e-34; // Planck constant              [J s]		*/
pub const BOLTZMANN_SI: f64 = 1.38064852e-23; // Boltzmann constant           [J/K]		*/
pub const JULIAN_YEAR_SI: f64 = 365.25 * 24.0 * 3600.0; // Length of the Julian year    [s]		*/
pub const STEFAN_BOLTZMANN_SI: f64 = 5.670367e-8; // Stefan-Boltzmann constant    [W/m^2/K^4]	*/
                                                  // From IAU 2009: */
pub const GRAVITATIONAL_CONST_SI: f64 = 6.67428e-11; // gravitational constant       [m^3/kg/s^2]	*/
pub const AU_SI: f64 = 1.495978707e11; // astronomical unit            [m]		*/
pub const LOCAL_CMB_TEMP_SI: f64 = 2.72548; // local mean CMB temp. from Fixsen (2009) [K]	*/
                                            // Derived: */
pub const PARSEC_SI: f64 = 3.08567758e16; // parsec (~3600*180*AU/PI)     [m]		*/
pub const SOLAR_MASS_SI: f64 = 1.9891e30; // Solar mass                   [kg]		*/
pub const SOLAR_RADIUS_SI: f64 = 6.957e8; // Solar radius                 [m]		*/
                                          // CGS: */
pub const AMU_CGS: f64 = AMU_SI * 1000.0; // atomic mass unit             [g]		*/
pub const SPEED_OF_LIGHT_CGS: f64 = SPEED_OF_LIGHT_SI * 100.0; // speed of light in vacuum     [cm/s]		*/
pub const PLANCK_CGS: f64 = PLANCK_SI * 1.0e7; // Planck constant              [erg s]		*/
pub const BOLTZMANN_CGS: f64 = BOLTZMANN_SI * 1.0e7; // Boltzmann constant           [erg/K]		*/
pub const STEFAN_BOLTZMANN_CGS: f64 = STEFAN_BOLTZMANN_SI * 1000.0; // Stefan-Boltzmann constant    [erg/cm^2/K^4/s]	*/
pub const GRAVITATIONAL_CONST_CGS: f64 = GRAVITATIONAL_CONST_SI * 1000.0; // gravitational constant       [cm^3/g/s^2]	*/
pub const AU_CGS: f64 = AU_SI * 100.0; // astronomical unit            [cm]		*/
pub const SOLAR_MASS_CGS: f64 = SOLAR_MASS_SI * 1000.0; // Solar mass                   [g]		*/
pub const SOLAR_RADIUS_CGS: f64 = SOLAR_RADIUS_SI * 100.0; // Solar radius                 [cm]		*/
pub const TYPICAL_ISM_DENS: f64 = 1000.0; // typical ISM density          [cm^-3]	*/
/// Conversion factors:
pub const ARCSEC_TO_RAD: f64 = std::f64::consts::PI / 180.0 / 3600.0; // arcsec to radian             [rad]		*/
pub const CITRUS_EPS: f64 = 1e-30; // Small number to avoid division by zero	*/
