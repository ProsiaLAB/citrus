use crate::constants as cc;

/// Function to calculate density with cutoff
pub fn get_density(x: f64, y: f64, z: f64, density: &mut [f64], cutoff: f64) -> &mut [f64] {
    // Calculate r_min from cutoff
    let r_min = cutoff * cc::AU_SI;

    // Calculate r from the 3D coordinates (Euclidean distance)
    let r = (x * x + y * y + z * z).sqrt();

    // Determine the appropriate distance to use based on r_min
    let r_to_use = if r > r_min { r } else { r_min };

    // Update the density array with the calculated value
    density[0] = 1.5e6 * ((r_to_use / (300.0 * cc::AU_SI)).powf(-1.5)) * 1e6;

    density // Return a reference to the density array
}
