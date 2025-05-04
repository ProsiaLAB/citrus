use anyhow::bail;
use anyhow::Result;

use crate::types::{RMatrix, RVector};

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
pub fn stokes_angles(mag_field: &RVector, rotation_matrix: &RMatrix) -> Result<()> {
    let b_p = rotation_matrix.t().dot(mag_field);
    let mut trig_fncs = RVector::zeros(3);

    // Square of length of B projected into the observer XY plane
    let b_xy_squared = b_p[0] * b_p[0] + b_p[1] * b_p[1];
    if b_xy_squared == 0.0 {
        bail!("B field is zero");
    }

    let b_squared = b_xy_squared + b_p[2] * b_p[2];
    trig_fncs[0] = b_xy_squared / b_squared; // cos^2 of the angle which Bp makes with the XY plane

    //cos(2*phi) = cos^2(phi) - sin^2(phi)
    trig_fncs[1] = (b_p[0] * b_p[0] - b_p[1] * b_p[1]) / b_xy_squared;

    //sin(2*phi) = 2*sin(phi)*cos(phi)
    trig_fncs[2] = 2.0 * b_p[0] * b_p[1] / b_xy_squared;

    Ok(())
}
