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
            ("nmol_weights", MAX_NUM_OF_COLLISIONAL_PARTNERS, "f8"),
            ("grid_density_max_locations", (MAX_NUM_HIGH, 3), "f8"),
        ]
    )
    return np.zeros(1, dtype=fields)


# typedef struct {
#   double radius,minScale,tcmb,*nMolWeights,*dustWeights;
#   double (*gridDensMaxLoc)[DIM],*gridDensMaxValues,*collPartMolWeights;
#   int sinkPoints,pIntensity,blend,*collPartIds,traceRayAlgorithm,samplingAlgorithm;
#   int sampling,lte_only,init_lte,antialias,polarization,nThreads,nSolveIters;
#   char **girdatfile,**moldatfile,**collPartNames;
#   char *outputfile,*binoutputfile,*gridfile,*pregrid,*restart,*dust;
#   char *gridInFile,**gridOutFiles;
#   _Bool resetRNG,doSolveRTE;
# } inputPars;


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
    return np.zeros(1, dtype=fields)


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
    return np.zeros(1, dtype=fields)


def init_rates():
    fields = np.dtype(
        [
            ("t_binlow", "i4"),
            ("interp_coeff", "f8"),
        ],
        align=True,
    )
    return np.zeros(1, dtype=fields)


def init_continuum_line():
    fields = np.dtype(
        [
            ("dust", "f8"),
            ("knu", "f8"),
        ],
        align=True,
    )
    return np.zeros(1, dtype=fields)


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
    return np.zeros(1, dtype=fields)


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
    return np.zeros(1, dtype=fields)
