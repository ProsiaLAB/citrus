use std::error::Error;

/// This function rotates the B-field vector from the model frame to the observer
/// frame, then calculates and returns some useful values which will in function
/// `source_fn_polarized()` make it easy to obtain the Stokes parameters of polarized
/// submillimetre dust emission. (For an explanation of the reasons for choosing the
/// particular quantities we do, see the comment in that function.)
///
/// Whenever one deals with polarized light, it is important to specify the
/// coordinate systems carefully. In LIME the observer frame is defined such that,
/// when the observer looks at the sky, the frame axes appear as follows:
///
///                ^ Y
///                |
///                |
///                |
///         <------+
///         X
///
/// The Z axis points into the page, away from the observer. Comparing this to
/// normal astronomical coordinates one can see that X is in the direction of +ve
/// right ascension and Y in the direction of +ve declination.
///
/// The IAU-recommended coordinate frame for expressing polarized light however is
///
///                ^ X
///                |
///                |
///                |
///         <------O
///         Y
///
/// with Z now emerging from the page (i.e pointing in the direction of propagation,
/// towards the observer).
///
/// A vector defined in the LIME model basis can be converted to the observer basis
/// by post-multiplying it with the image rotation matrix rotMat. (Another way of
/// putting this is that the rows of rotMat are the unit vectors of the model
/// coordinate frame expressed in the observer basis.) For the B field, this is
/// expressed symbolically as
///
///         Bp^T = B^T * rotation_matrix
///
/// where ^T denotes transpose.
pub fn stokes_angles(
    mag_field: &mut [f64; 3],
    rotation_matrix: [[f64; 3]; 3],
    trig_fncs: &mut [f64],
) -> Result<(), Box<dyn Error>> {
    let ndim = 3;
    let b_p = &mut [0.0; 3];

    // Rotate `mag_field` to the observer's frame
    for i in 0..ndim {
        b_p[i] = 0.0;
        for j in 0..ndim {
            b_p[i] += rotation_matrix[i][j] * mag_field[j];
        }
    }

    // Square of length of B projected into the observer XY plane
    let b_xy_squared = b_p[0] * b_p[0] + b_p[1] * b_p[1];
    if b_xy_squared == 0.0 {
        trig_fncs[0] = 0.;
        trig_fncs[1] = 0.;
        trig_fncs[2] = 0.;
        return Err("B field is zero".into());
    }

    let b_squared = b_xy_squared + b_p[2] * b_p[2];
    trig_fncs[0] = b_xy_squared / b_squared; // cos^2 of the angle which Bp makes with the XY plane

    //cos(2*phi) = cos^2(phi) - sin^2(phi)
    trig_fncs[1] = (b_p[0] * b_p[0] - b_p[1] * b_p[1]) / b_xy_squared;

    //sin(2*phi) = 2*sin(phi)*cos(phi)
    trig_fncs[2] = 2.0 * b_p[0] * b_p[1] / b_xy_squared;

    Ok(())
}
