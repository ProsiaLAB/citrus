AMU_SI = 1.66053904e-27  # atomic mass unit             [kg]		*/
SPEED_OF_LIGHT_SI = 2.99792458e8  # speed of light in vacuum     [m/s]		*/
PLANCK_SI = 6.626070040e-34  # Planck constant              [J s]		*/
BOLTZMANN_SI = 1.38064852e-23  # Boltzmann constant           [J/K]		*/
JULIAN_YEAR_SI = 365.25 * 24.0 * 3600.0  # Length of the Julian year    [s]		*/
STEFAN_BOLTZMANN_SI = 5.670367e-8  # Stefan-Boltzmann constant    [W/m^2/K^4]	*/

# From IAU 2009: */
GRAVITATIONAL_CONST_SI = 6.67428e-11  # gravitational constant       [m^3/kg/s^2]	*/
AU_SI = 1.495978707e11  # astronomical unit            [m]		*/

LOCAL_CMB_TEMP_SI = 2.72548  # local mean CMB temp. from Fixsen (2009) [K]	*/

# Derived: */
PARSEC_SI = 3.08567758e16  # parsec (~3600*180*AU/PI)     [m]		*/
SOLAR_MASS_SI = 1.9891e30  # Solar mass                   [kg]		*/
SOLAR_RADIUS_SI = 6.957e8  # Solar radius                 [m]		*/

# CGS: */
AMU_CGS = AMU_SI * 1000.0  # atomic mass unit             [g]		*/
SPEED_OF_LIGHT_CGS = (
    SPEED_OF_LIGHT_SI * 100.0
)  # speed of light in vacuum     [cm/s]		*/
PLANCK_CGS = PLANCK_SI * 1.0e7  # Planck constant              [erg s]		*/
BOLTZMANN_CGS = BOLTZMANN_SI * 1.0e7  # Boltzmann constant           [erg/K]		*/
STEFAN_BOLTZMANN_CGS = (
    STEFAN_BOLTZMANN_SI * 1000.0
)  # Stefan-Boltzmann constant    [erg/cm^2/K^4/s]	*/
GRAVITATIONAL_CONST_CGS = (
    GRAVITATIONAL_CONST_SI * 1000.0
)  # gravitational constant       [cm^3/g/s^2]	*/
AU_CGS = AU_SI * 100.0  # astronomical unit            [cm]		*/
SOLAR_MASS_CGS = SOLAR_MASS_SI * 1000.0  # Solar mass                   [g]		*/
SOLAR_RADIUS_CGS = (
    SOLAR_RADIUS_SI * 100.0
)  # Solar radius                 [cm]		*/
