mutable struct CollisionalData
    down::Float64
    temp::Float64
    partner_ID::Int32
    ntemp::Int32
    ntrans::Int32
    lcl::Int32 # lower level
    lcu::Int32 # upper level
    density_index::Int32
    name::String
end

mutable struct MoleculeData
    nlev::Int32
    nline::Int32
    npart::Int32
    lal::Int32 # lower level
    lau::Int32 # upper level
    aeinst::Float64 # einstein coefficient A
    freq::Float64 # frequency
    beinstu::Float64 # einstein coefficient B for upper level
    beinstl:: Float64 # einstein coefficient B for lower level
    eterm::Float64 # energy of upper level
    gstat::Float64 # statistical weight of upper level
    girr::Float64 # isotropic g-factor
    cmb::Float64 # cmb temperature
    amass::Float64 # atomic mass
    part::CollisionalData
    mol_name::String
end

mutable struct Point{N}
    x::Vector{Float64}
    xn::Vector{Float64}
end

mutable struct Rates
    t_binlow::Int32
    interp_coeff::Float64
end


"""
This is the mutable struct for the continuum line. It contains the dust and knu
parameters.
"""
mutable struct ContinuumLine
    dust::Float64
    knu::Float64
end

mutable struct Populations
    pops::Float64
    spec_num_dens::Float64
    dopb::Float64
    binv::Float64
    nmol::Float64
    abun::Float64
    partner::Rates
    cont::ContinuumLine
end

mutable struct LineData
    nlev::Int32
    nline::Int32
    npart::Int32
    lal::Int32 # lower level
    lau::Int32 # upper level
    aeinst::Float64 # einstein coefficient A
    freq::Float64 # frequency
    beinstu::Float64 # einstein coefficient B for upper level
    beinstl:: Float64 # einstein coefficient B for lower level
    eterm::Float64 # energy of upper level
    gstat::Float64 # statistical weight of upper level
    girr::Float64 # isotropic g-factor
    cmb::Float64 # cmb temperature
    amass::Float64 # atomic mass
    part::CollisionalData
    mol_name::String
    pops::Populations
end