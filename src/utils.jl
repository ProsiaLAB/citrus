

"""
    calc_source_fn(dτ::Float64, taylor_cutoff::Float64) -> (Float64, Float64)
 
The source function S is defined as ``j_ν/α``, which is clearly not
defined for alpha==0. However S is used in the algorithm only in the
term (1-exp[-alpha*ds])*S, which is defined for all values of alpha.
The present function calculates this term and returns it in the
argument remnantSnu. For values of abs(alpha*ds) less than a pre-
calculated cutoff supplied in configInfo, a Taylor approximation is
used.

Note that the same cutoff condition holds for replacement of
exp(-dTau) by its Taylor expansion to 3rd order.

Note that this is called from within the multi-threaded block.
"""
function calc_source_fn(dτ::Float64, taylor_cutoff::Float64)
    """

    """
    remnant_S_ν = 0.0
    exp_dτ = 0.0
    if abs(dτ) < taylor_cutoff
        remnant_S_ν = 1.0 - dτ * (1.0 - dτ * (1.0 / 3.0)) * (1.0 / 2.0)
        exp_dτ = 1.0 - dτ * remnant_S_ν
    else
        exp_dτ = exp(-dτ)
        remnant_S_ν = (1.0 - exp_dτ) / dτ
    end
    return remnant_S_ν, exp_dτ
end

"""
Calculate the Planck function for a given frequency and temperature.
"""
function planck_fn(freq::Float64, temp::Float64)
    bb = 10.0
    if temp < eps
        bb = 0.0
    else
        wn = freq / clight
        if (hplanck * freq > 100 * kboltz * temp)
            bb = 2.0 * hplanck * wn * wn * freq * exp(-hplanck * freq / kboltz / temp)
        else
            bb = 2.0 * hplanck * wn * wn * freq / (exp(hplanck * freq / kboltz / temp) - 1.0)
        end
    end
    return bb
end

function gauss_line(v::Float64, one_on_sigma::Float64)
    val = v * v * one_on_sigma * one_on_sigma
    return exp(-val)
end