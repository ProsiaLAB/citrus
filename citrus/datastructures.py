from dataclasses import dataclass, field

import numpy as np

from . import constants as cc

MAX_NUM_OF_SPECIES = 100
MAX_NUM_OF_IMAGES = 100
NUM_OF_GRID_STAGES = 5
MAX_NUM_OF_COLLISIONAL_PARTNERS = 20
TYPICAL_ISM_DENSITY = 1e3
DENSITY_POWER = 0.2
MAX_NUM_HIGH = 10  # ??? What this bro?


@dataclass
class InputParams:
    radius: float = 0.0
    min_scale: float = 0.0
    cmb_temp: float = cc.LOCAL_CMB_TEMP_SI
    sink_points: int = 0
    p_intensity: int = 0
    blend: int = 0
    ray_trace_algorithm: int = 0
    sampling_algorithm: int = 0
    sampling: int = 2
    LTE_only: int = 0
    init_LTE: int = 0
    anti_alias: int = 1
    polarization: int = 0
    nthreads: int = 1
    nsolve_iters: int = 0
    output_file: str = ""
    binoutput_file: str = ""
    grid_file: str = ""
    pre_grid: str = ""
    restart: str = ""
    dust: str = ""
    grid_in_file: str = ""
    reset_RNG: bool = False
    do_solve_rte: bool = False

    nmol_weights: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_NUM_OF_COLLISIONAL_PARTNERS)
    )
    grid_density_max_locations: np.ndarray = field(
        default_factory=lambda: np.zeros((MAX_NUM_HIGH, 3))
    )
    grid_density_max_values: np.ndarray = field(
        default_factory=lambda: np.zeros((MAX_NUM_HIGH, 3))
    )
    collisional_partner_mol_weights: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_NUM_OF_COLLISIONAL_PARTNERS)
    )
    collisional_partner_IDs: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_NUM_OF_COLLISIONAL_PARTNERS)
    )
    grid_data_file: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_NUM_OF_SPECIES)
    )
    mol_data_file: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_NUM_OF_SPECIES)
    )
    collisional_partner_names: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_NUM_OF_COLLISIONAL_PARTNERS)
    )
    grid_out_files: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_NUM_OF_IMAGES)
    )


@dataclass
class ConfigParams:
    radius: float = 0.0
    min_scale: float = 0.0
    cmb_temp: float = 0.0
    sink_points: int = 0
    p_intensity: int = 0
    blend: int = 0
    ray_trace_algorithm: int = 0
    sampling_algorithm: int = 0
    sampling: int = 0
    LTE_only: int = 0
    init_LTE: int = 0
    anti_alias: int = 0
    polarization: int = 0
    nthreads: int = 0
    nsolve_iters: int = 0
    collisional_partner_user_set_flags: int = 0
    output_file: str = ""
    binoutput_file: str = ""
    grid_file: str = ""
    pre_grid: str = ""
    restart: str = ""
    dust: str = ""
    grid_in_file: str = ""
    reset_RNG: bool = False
    do_solve_rte: bool = False
    radius_squ: float = 0.0
    min_scale_squ: float = 0.0
    taylor_cutoff: float = 0.0
    grid_density_global_max: float = 0.0
    ncell: int = 0
    n_images: int = 0
    n_species: int = 0
    num_densities: int = 0
    do_pregrid: int = 0
    num_grid_density_maxima: int = 0
    num_dims: int = 0
    n_line_images: int = 0
    n_cont_images: int = 0
    data_flags: int = 0
    n_solve_iters_done: int = 0
    do_interpolate_vels: bool = False
    use_abun: bool = False
    do_mol_calcs: bool = False
    use_vel_func_in_raytrace: bool = False
    edge_vels_available: bool = False

    nmol_weights: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_NUM_OF_COLLISIONAL_PARTNERS)
    )
    grid_density_max_locations: np.ndarray = field(
        default_factory=lambda: np.zeros((MAX_NUM_HIGH, 3))
    )
    grid_density_max_values: np.ndarray = field(
        default_factory=lambda: np.zeros((MAX_NUM_HIGH, 3))
    )
    collisional_partner_mol_weights: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_NUM_OF_COLLISIONAL_PARTNERS)
    )
    collisional_partner_IDs: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_NUM_OF_COLLISIONAL_PARTNERS)
    )
    grid_data_file: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_NUM_OF_SPECIES)
    )
    mol_data_file: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_NUM_OF_SPECIES)
    )
    collisional_partner_names: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_NUM_OF_COLLISIONAL_PARTNERS)
    )
    grid_out_files: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_NUM_OF_IMAGES)
    )
    write_grid_at_stage: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_OF_GRID_STAGES)
    )


@dataclass
class CollisionalData:
    down: float
    temp: float
    partner_ID: int
    ntemp: int
    ntrans: int
    lcl: int
    lcu: int
    density_index: int
    name: str


@dataclass
class Image:
    default_angle = -999.0
    nchan: int = 0
    trans: int = -1
    mol_I: int = -1
    vel_res: float = -1.0
    img_res: float = -1.0
    pixels: int = -1
    unit: int = 0
    units: str = "Jy/pixel"
    freq: float = -1.0
    bandwidth: float = -1.0
    filename: str = ""
    source_velocity: float = 0.0
    theta: float = 0.0
    phi: float = 0.0
    inclination: float = default_angle
    position_angle: float = default_angle
    azimuth: float = default_angle
    distance: float = -1.0
    do_interpolate_vels: bool = False


@dataclass
class MoleculeData:
    nlev: int
    nline: int
    npart: int
    lal: int
    lau: int
    aeinst: float
    freq: float
    beinstu: float
    beinstl: float
    eterm: float
    gstat: float
    girr: float
    cmb: float
    amass: float
    part: CollisionalData
    mol_name: str


@dataclass
class Rates:
    t_binlow: int
    interp_coeff: float


@dataclass
class ContinuumLine:
    dust: float
    knu: float


@dataclass
class Populations:
    pops: float
    spec_num_dens: float
    dopb: float
    binv: float
    nmol: float
    abun: float
    partner: Rates
    cont: ContinuumLine


@dataclass
class LineData:
    nlev: int
    nline: int
    npart: int
    lal: int
    lau: int
    aeinst: float
    freq: float
    beinstu: float
    beinstl: float
    eterm: float
    gstat: float
    girr: float
    cmb: float
    amass: float
    part: CollisionalData
    mol_name: str
    pops: Populations
