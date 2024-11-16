import numpy as np

MAX_NUM_OF_SPECIES = 100
MAX_NUM_OF_IMAGES = 100
NUM_OF_GRID_STAGES = 5
MAX_NUM_OF_COLLISIONAL_PARTNERS = 20
TYPICAL_ISM_DENSITY = 1e3
DENSITY_POWER = 0.2
MAX_NUM_HIGH = 10  # ??? What this bro?


def init_input_parameters():
    fields = np.dtype(
        [
            ("radius", "f8"),
            ("min_scale", "f8"),
            ("cmb_temp", "f8"),
            ("nmol_weights", "f8", MAX_NUM_OF_COLLISIONAL_PARTNERS),
            ("grid_density_max_locations", "f8", (MAX_NUM_HIGH, 3)),
            ("grid_density_max_values", "f8", (MAX_NUM_HIGH, 3)),
            ("collisional_partner_mol_weights", "f8", MAX_NUM_OF_COLLISIONAL_PARTNERS),
            ("sink_points", "i8"),
            ("p_intensity", "i8"),
            ("blend", "i8"),
            ("collisional_partner_IDs", "i8", MAX_NUM_OF_COLLISIONAL_PARTNERS),
            ("ray_trace_algorithm", "i8"),
            ("sampling_algorithm", "i8"),
            ("sampling", "i8"),
            ("LTE_only", "i8"),
            ("init_LTE", "i8"),
            ("anti_alias", "i8"),
            ("polarization", "i8"),
            ("nthreads", "i8"),
            ("nsolve_iters", "i8"),
            # ("collisional_partner_user_set_flags", "i8"),
            ("grid_data_file", "U128", MAX_NUM_OF_SPECIES),
            ("mol_data_file", "U128", MAX_NUM_OF_SPECIES),
            ("collisional_partner_names", "U128", MAX_NUM_OF_COLLISIONAL_PARTNERS),
            ("output_file", "U128"),
            ("binoutput_file", "U128"),
            ("grid_file", "U128"),
            ("pre_grid", "U128"),
            ("restart", "U128"),
            ("dust", "U128"),
            ("grid_in_file", "U128"),
            ("grid_out_files", "U128", MAX_NUM_OF_IMAGES),
            ("reset_RNG", "bool"),
            ("do_solve_rte", "bool"),
        ],
        align=True,
    )
    return np.zeros((1,), dtype=fields)


def init_collisional_data():
    fields = np.dtype(
        [
            ("down", "f8"),
            ("temp", "f8"),
            ("partner_ID", "i4"),
            ("ntemp", "i4"),
            ("ntrans", "i4"),
            ("lcl", "i4"),
            ("lcu", "i4"),
            ("density_index", "i4"),
            ("name", "U10"),
        ],
        align=True,
    )
    return np.zeros((1,), dtype=fields)


def init_images(n_images):
    if n_images == 1:
        return init_image()
    fields = np.dtype(
        [
            ("nchan", "i8", n_images),
            ("trans", "i8", n_images),
            ("mol_I", "i8", n_images),
            ("vel_res", "f8", n_images),
            ("img_res", "f8", n_images),
            ("pixels", "i8", n_images),
            ("unit", "i8", n_images),
            ("units", "U10", n_images),
            ("freq", "f8", n_images),
            ("bandwidth", "f8", n_images),
            ("filename", "U128", n_images),
            ("source_velocity", "f8", n_images),
            ("theta", "f8", n_images),
            ("phi", "f8", n_images),
            ("inclination", "f8", n_images),
            ("position_angle", "f8", n_images),
            ("azimuth", "f8", n_images),
            ("distance", "f8", n_images),
            ("do_interpolate_vels", "bool", n_images),
        ]
    )
    return np.zeros(n_images, dtype=fields)


def init_image():
    fields = np.dtype(
        [
            ("nchan", "i8"),
            ("trans", "i8"),
            ("mol_I", "i8"),
            ("vel_res", "f8"),
            ("img_resolution", "f8"),
            ("pixels", "i8"),
            ("unit", "i8"),
            ("units", "U10"),
            ("freq", "f8"),
            ("bandwidth", "f8"),
            ("filename", "U128"),
            ("source_velocity", "f8"),
            ("theta", "f8"),
            ("phi", "f8"),
            ("inclination", "f8"),
            ("position_angle", "f8"),
            ("azimuth", "f8"),
            ("distance", "f8"),
            ("do_interpolate_vels", "bool"),
        ]
    )
    return np.zeros((1,), dtype=fields)


def init_molecule_data():
    fields = np.dtype(
        [
            ("nlev", "i4"),
            ("nline", "i4"),
            ("npart", "i4"),
            ("lal", "i4"),
            ("lau", "i4"),
            ("aeinst", "f8"),
            ("freq", "f8"),
            ("beinstu", "f8"),
            ("beinstl", "f8"),
            ("eterm", "f8"),
            ("gstat", "f8"),
            ("girr", "f8"),
            ("cmb", "f8"),
            ("amass", "f8"),
            ("part", init_collisional_data()),
            ("mol_name", "U10"),
        ],
        align=True,
    )
    return np.zeros((1,), dtype=fields)


def init_rates():
    fields = np.dtype(
        [
            ("t_binlow", "i4"),
            ("interp_coeff", "f8"),
        ],
        align=True,
    )
    return np.zeros((1,), dtype=fields)


def init_continuum_line():
    fields = np.dtype(
        [
            ("dust", "f8"),
            ("knu", "f8"),
        ],
        align=True,
    )
    return np.zeros((1,), dtype=fields)


def init_populations():
    fields = np.dtype(
        [
            ("pops", "f8"),
            ("spec_num_dens", "f8"),
            ("dopb", "f8"),
            ("binv", "f8"),
            ("nmol", "f8"),
            ("abun", "f8"),
            ("partner", init_rates()),
            ("cont", init_continuum_line()),
        ],
        align=True,
    )
    return np.zeros((1,), dtype=fields)


def init_line_data():
    fields = np.dtype(
        [
            ("nlev", "i4"),
            ("nline", "i4"),
            ("npart", "i4"),
            ("lal", "i4"),
            ("lau", "i4"),
            ("aeinst", "f8"),
            ("freq", "f8"),
            ("beinstu", "f8"),
            ("beinstl", "f8"),
            ("eterm", "f8"),
            ("gstat", "f8"),
            ("girr", "f8"),
            ("cmb", "f8"),
            ("amass", "f8"),
            ("part", init_collisional_data()),
            ("mol_name", "U10"),
            ("pops", init_populations()),
        ],
        align=True,
    )
    return np.zeros((1,), dtype=fields)
