"""
tools Module Overview

The **`tools`** module is a comprehensive collection of submodules designed to facilitate the simulation, analysis, and evaluation of stochastic processes and related models. Each submodule serves a specific purpose, providing tools for various aspects of mathematical modeling, statistical analysis, numerical computation, and automation.

The module is structured to support high-efficiency parallel simulations, fitting of complex processes, analysis of preasymptotic behavior, and solving partial stochastic differential equations (PSDEs). The **`tools`** module is flexible, allowing for integration with different types of stochastic processes such as Brownian Motion, Lévy processes, and fractional processes.

Submodules:

1. **automate.py**

    - Provides functions for automating large-scale simulations, including live visualizations and parameter sweeps. Designed for scalability and repetitive tasks in research and experimentation.

2. **compute.py**

    - Contains numerical methods and algorithms to compute key metrics and statistics from simulations. It includes tools for analyzing ergodicity, ensemble averages, and time averages.

3. **evaluate.py**

    - Implements evaluation methods to assess the behavior of stochastic processes. It includes functions for calculating rates of convergence and evaluating results against theoretical expectations.

4. **fit.py**

    - Tools for fitting stochastic processes to observed data. Provides utilities to fit parameters of various distributions, including Lévy stable and geometric Brownian motion, using techniques like maximum likelihood estimation (MLE).

5. **helper.py**

    - A utility module containing helper functions that support other submodules. These include data separation, saving and loading results, and visualization tools.

6. **multiprocessing.py**

    - Manages parallel execution of simulations and tasks. This submodule allows for efficient parallelization using multiprocessing, enabling high-speed simulations of stochastic processes across different parameter ranges.

7. **partial_sde.py**

    - Tools for simulating and solving partial stochastic differential equations (PSDEs). This submodule includes functions for handling boundary conditions, drift and diffusion terms, and creating visualizations of the resulting solutions.

8. **preasymptotics.py**

    - Focuses on preasymptotic behavior, convergence analysis, and the speed of convergence to limiting distributions (e.g., normal or Lévy). Includes tools for analyzing transient fluctuations, time to stationarity, and convergence rates.

9. **research.py**

    - A high-level submodule that organizes pipelines for running massive simulations and computing statistical measures. It provides pre-defined functions for simulating various stochastic processes, including Lévy and Brownian Motion, using parallel computing for large-scale experiments.

10. **solve.py**

    - Includes methods for solving stochastic differential equations (SDEs) and related optimization tasks. Provides tools for applying Ito's Lemma, solving complex SDEs, and performing integration with substitution.

Key Features:

- **Parallel Computing**: The module extensively leverages multiprocessing for efficient simulations, allowing large-scale parameter sweeps and statistical analysis.

- **Live Visualizations**: Real-time 2D and 3D visualizations to help researchers observe the dynamic evolution of stochastic processes.

- **Comprehensive Analysis Tools**: From preasymptotic behavior to statistical validation, the module provides all necessary tools to assess the behavior and properties of simulated processes.

- **Flexible and Scalable**: The structure of the module allows for easy expansion with additional processes, fitting techniques, and evaluation metrics.

Example Usage:

```python

from ergodicity.tools import research, fit, multiprocessing

# Run a Geometric Lévy Process pipeline with a wide range of parameters

results = research.GeometricLevyPipeline_case()

# Fit a Lévy stable distribution to observed data

data = np.random.normal(size=1000)

fitted_distribution = fit.levy_stable_fit(data)

# Run parallel simulations of a process

sim_results = multiprocessing.multi_simulations(MyStochasticProcess, param_ranges)
"""
import os
import glob
import importlib

# Get the current directory
current_dir = os.path.dirname(__file__)

from ergodicity.process.default_values import *
from ..configurations import *

# Get all Python files in the current directory
modules = glob.glob(os.path.join(current_dir, "*.py"))

# Exclude the partial_sde.py file from the list
modules = [module for module in modules if os.path.basename(module) != "partial_sde.py"]

# Import all modules dynamically
for module in modules:
    module_name = os.path.basename(module)[:-3]
    if module_name != "__init__":
        importlib.import_module(f".{module_name}", package="ergodicity.tools")


