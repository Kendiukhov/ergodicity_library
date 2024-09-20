"""
automate Submodule

The `automate` submodule provides tools to automate simulations and visualizations of stochastic processes over various parameter configurations. It enables batch processing of parameter ranges for a given process class, making it useful for large-scale simulations or research that requires exploration of many different process behaviors.

Key Features:

1. **Automated Parameter Exploration**:

   - `automated_live_visualization`: Automates the generation of visualizations for all combinations of parameters in a specified range. It supports 1D, 2D, and 3D visualizations, making it versatile for different types of stochastic processes.

2. **Parameter Combination**:

   - The submodule uses `itertools.product` to generate all possible combinations of parameter values, allowing exhaustive exploration of the parameter space.

3. **Dynamic Visualization**:

   - Based on the dimensionality of the process (1, 2, or 3 dimensions), the function automatically selects the correct visualization method and produces live simulations of the process. Simulations can be saved as videos for further analysis.

4. **Customizable Simulations**:

   - The user can control the total simulation time, time step, number of instances, and speed of the simulation.

   - Simulations can be tailored to different dimensions (1D, 2D, or 3D) and saved for later review.

Typical Use Cases:

- **Research and Development**:

  Useful in exploring how different parameter values affect the behavior of stochastic processes, particularly in areas like finance, physics, and biology.

- **Simulation Studies**:

  Helps automate large-scale simulation studies where multiple parameter combinations need to be analyzed in an efficient manner.

- **Educational Tools**:

  Ideal for demonstrating the dynamics of stochastic processes in classrooms or presentations, with real-time visualizations across different parameter sets.

Example Usage:

param_ranges = {
    'alpha': [0.1, 0.5, 1.0],
    'beta': [0.1, 0.2, 0.3]
}

automated_live_visualization(
    dimensions=2,
    process_class=MyStochasticProcess,
    param_ranges=param_ranges,
    t=10,
    timestep=0.01,
    num_instances=100,
    speed=1.0
)
"""

import ergodicity.process as ep
import numpy as np
from ergodicity.process.default_values import *
# from ergodicity.tools.research import *
import numpy as np
from ergodicity.configurations import *
import itertools
import os
from ergodicity.tools.compute import *

def automated_live_visualization(dimensions: int, process_class, param_ranges: dict,
                                 t: float = 10, timestep: float = 0.01,
                                 num_instances: int = 100, speed: float = 1.0):
    """
    Generate live visualizations for all parameter combinations of a stochastic process.

    :param dimensions: Number of dimensions (1, 2, or 3)
    :type dimensions: int
    :param process_class: The stochastic process class
    :type process_class: class
    :param param_ranges: Dictionary of parameter ranges (key: param name, value: list of values)
    :type param_ranges: dict
    :param t: Total time for the simulation
    :type t: float
    :param timestep: Time step for the simulation
    :type timestep: float
    :param num_instances: Number of instances to simulate
    :type num_instances: int
    :param speed: Speed multiplier for the video
    :type speed: float
    """
    if dimensions not in [1, 2, 3]:
        raise ValueError("Dimensions must be 1, 2, or 3")

    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    param_combinations = list(itertools.product(*param_values))

    for params in param_combinations:
        # Create a dictionary of parameters for this combination
        param_dict = dict(zip(param_names, params))

        # Create an instance of the process
        process = process_class(**param_dict)

        # Call the appropriate visualization method
        if dimensions == 1:
            process.simulate_live(t, timestep, num_instances, speed, save=True)
        elif dimensions == 2:
            process.simulate_live_2d(t, timestep, num_instances, speed, save=True)
        else:  # dimensions == 3
            process.simulate_live_3d(t, timestep, num_instances, speed, save=True)

    print(f"All visualizations completed. Videos saved in 'simulation_videos' directory.")



# Example usage:
'''

param_ranges = {
    'param1': [1, 2, 3],
    'param2': [0.1, 0.2]
}

automated_live_visualization(
    dimensions=2,
    process_class=MyStochasticProcess,
    param_ranges=param_ranges,
    t=10,
    timestep=0.01,
    num_instances=100,
    speed=1.0
)

'''
