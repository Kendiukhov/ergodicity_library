"""
This file contains the main adjustable configurations for the Ergodicity Library.
These are the most important global variables that can be changed by the user.
The user can change the values of these variables to customize the behavior of the package.
Initially, the parameters are set to enable explanatory command line outputs and conduct
analysis pipelines with parameters optimized for Levy processes and computational power of a personal computer.
"""

default_comments = True # This parameter is used to enable or disable notifications about what is happenning when running the code. They may be informative for some, but annoying for others. Select what you prefer.
use_external_simulators = False # Not recommended to set True. This parameter is used to enable or disable the use of external simulators, such as for example the simulators provided by the stochastic library.
# External simulators are not available for many processes and may be problematic for other processes, but they are typically faster. So use them if you want speed and if you are sure that they work for your process.
verbose = True # This parameter is used to enable or disable the verbose mode. If enabled, the code will print more information about what is happening.
simulate_with_differential = True # This parameter is used to enable or disable the use of the simulation with the stochastic differentials. If enabled, the code will use the stochastic differential equations explicitly to simulate the process. If disabled, the code will use the simulation with probability distributions without the differentials.
# Both types of simulations are typically comparable in terms of speed and precision, but the simulaton with the differential usually allows for explicit representation of the process and is more readable by the user
output_dir_general = 'output_general' # This parameter is used to set the general output directory for the simulations and computation. The output directory is used to store the results of the methods and functions.
pipeline_parameters = {'t': 1, 'timestep': 0.1, 'num_instances': 3, 'time_average_time': 10,
                       'ensemble_average_num_instances': 10, 'save': True, 'plot': True, 'output_dir': 'output',
                       'print_debug': False} # This parameter is used to set the default parameters for the multiprocessing research pipelines. The parameters are used to set the time of the simulation, the timestep, the number of instances, the time for time average, the number of instances for ensemble average, the save and plot options, the output directory and the print debug option.
optimize_with_language = 'c' # This parameter is used to set the language for the optimization of the code.
# This is an experimental feature and not yet operational. We plan to implement the optimization of the code with the use of the C, Rust, and Wolfram language in the future.
lib_plot = False # Controls whether the plots are build automatically for the functions in the process.lib.py submodule.
