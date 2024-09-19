"""
process Module

The `Process Module` serves as the core component of this library, encapsulating a wide variety of stochastic processes, both classic and advanced, for modeling random dynamics in continuous time. The module is highly extensible and organized into submodules, each focusing on specific classes of processes or characteristics, such as memory effects, increments, and custom processes. Whether you're modeling financial markets, physical systems, or biological phenomena, this module provides a robust framework for simulating and analyzing processes with diverse properties.

Key Submodules:

1. **Basic Submodule**:

   - Contains fundamental stochastic processes like **Wiener Process** (Brownian motion), **Levy Stable Process**, and **Fractional Brownian Motion**.

   - These processes form the building blocks for more advanced models and serve as templates for defining custom processes.

2. **Increments Submodule**:

   - Provides functions and classes for generating **increments** for various stochastic processes.

   - Increments are essential for simulating the paths of processes, and this submodule ensures they have variance 1 and are consistent with their process's timestep.

3. **With Memory Submodule**:

   - Focuses on **non-Markovian processes**, where future behavior depends on past increments or states.

   - Includes processes like **BrownianMotionWithMeanMemory**, which adapt their drift or volatility based on historical information, capturing long-term dependencies or "memory" effects.

4. **Multiplicative Submodule**:

   - Contains processes that exhibit **multiplicative dynamics**, where changes are proportional to the current value.

   - Includes processes like **Geometric Brownian Motion**, widely used in financial modeling, and **Geometric Levy Process**, which generalizes this to heavy-tailed distributions.

5. **Constructor Submodule**:

   - Allows users to create custom processes interactively, defining parameters, drift, and stochastic terms.

   - It provides a mechanism for dynamically constructing stochastic processes based on user input, with full flexibility in defining how the process evolves.

6. **Custom Classes Submodule**:

   - Contains specialized stochastic processes, such as the **ConstantElasticityOfVarianceProcess** (CEV), which models processes with state-dependent volatility.

   - These custom classes offer advanced modeling capabilities for scenarios where traditional processes are insufficient.

Key Features of the Process Module:

- **Extensibility**: The module is designed to be extended with new processes, either through direct inheritance from the base classes or by using the `Constructor` submodule for dynamic process creation.

- **Simulation and Analysis**: Every process in the module includes methods for simulating sample paths, generating increments, and calculating statistical properties like variance, mean, or higher moments.

- **Real-World Applications**: The processes implemented here can model diverse phenomena in fields such as:

    - **Finance**: Asset prices, interest rates, and volatility models.

    - **Physics**: Diffusion processes and particle motion.

    - **Biology**: Population dynamics and evolutionary processes.

    - **Machine Learning**: Stochastic optimization algorithms.

Usage:

This module is intended for users who need to simulate and analyze stochastic processes in a flexible and extensible manner. The core functionality is built around the following concepts:

- **Increment Calculation**: Each process has a method to calculate its increment over a given timestep.

- **Customizability**: Many processes allow users to specify custom drift, volatility, and other parameters.

- **Dynamic Process Creation**: Users can define custom processes on the fly, making this module suitable for both researchers and practitioners who need tailored stochastic models.

Future Extensions:

The Process Module is continuously evolving, with plans to incorporate:

- **More memory-based processes**: To simulate processes with complex dependencies on past behavior.

- **Higher-dimensional processes**: Such as multivariate stochastic processes with intricate correlation structures.

- **Hybrid processes**: Combining different types of stochastic dynamics, such as additive and multiplicative effects, within the same model.

The `Process Module` is a versatile and powerful tool for stochastic modeling, designed to meet the needs of both academic researchers and industry professionals.
"""
import os
import glob
import importlib

# Get the current directory
current_dir = os.path.dirname(__file__)

from .default_values import *
from ..configurations import *

# Get all Python files in the current directory
modules = glob.glob(os.path.join(current_dir, "*.py"))

# Remove the increments file
modules.remove(os.path.join(current_dir, "increments.py"))

# Import all modules dynamically
for module in modules:
    module_name = os.path.basename(module)[:-3]
    if module_name != "__init__":
        importlib.import_module(f".{module_name}", package="ergodicity.process")
