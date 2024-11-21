from typing import List

from .datastructures import ConfigParams, Image, InputParams
from .gridio import count_density_columns


def run(par: InputParams, imgs: List[Image]):
    config = ConfigParams()

    config.radius = par.radius
    config.min_scale = par.min_scale
    config.p_intensity = par.p_intensity
    config.sink_points = par.sink_points
    config.sampling_algorithm = par.sampling_algorithm
    config.sampling = par.sampling
    config.cmb_temp = par.cmb_temp
    config.LTE_only = par.LTE_only
    config.init_LTE = par.init_LTE
    config.blend = par.blend
    config.anti_alias = par.anti_alias
    config.polarization = par.polarization
    config.nthreads = par.nthreads
    config.nsolve_iters = par.nsolve_iters
    config.ray_trace_algorithm = par.ray_trace_algorithm
    config.reset_RNG = par.reset_RNG
    config.do_solve_rte = par.do_solve_rte
    config.dust = par.dust
    config.output_file = par.output_file
    config.binoutput_file = par.binoutput_file
    config.restart = par.restart
    config.grid_file = par.grid_file
    config.pre_grid = par.pre_grid
    config.grid_in_file = par.grid_in_file

    config.ncell = config.p_intensity + config.sink_points
    config.radius_squ = config.radius**2
    config.min_scale_squ = config.min_scale**2
    config.n_solve_iters_done = 0
    config.use_abun = 1
    config.data_flags = 0

    config.grid_density_global_max = 1.0
    config.num_densities = 0

    if not config.do_pregrid or config.restart:
        config.num_densities = 0
        if config.grid_in_file is not None:
            status = count_density_columns(config.grid_in_file, config.num_densities)
            if status:
                exit(1)
        if config.num_densities <= 0:
            # Check if user provided a density() function
            ...
            # config.num_densities = num_func_densities # ? How to handle user defined functions?
