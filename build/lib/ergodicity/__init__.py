"""
Ergodicity Library: A Python toolkit for stochastic processes and ergodicity economics.

The Ergodicity Library is a comprehensive Python package designed for analyzing stochastic processes, with a particular focus on ergodicity economics. It provides tools for:

Simulating and visualizing various stochastic processes, including Brownian motion, Lévy processes, and custom processes. It is focused on:

Analyzing time averages and ensemble averages of stochastic processes.

Implementing ergodic transformations and other key concepts from ergodicity economics.

Fitting stochastic models to empirical data and estimating process parameters.

Creating and training artificial agents for decision-making under uncertainty.

Performing multi-core computations for efficient large-scale simulations.

Key features:

Object-oriented design with a flexible class hierarchy for stochastic processes

Integration with popular scientific Python libraries (NumPy, SciPy, SymPy, Matplotlib)

Symbolic computation capabilities for stochastic calculus

Tools for analyzing non-ergodic and heavy-tailed processes

Support for both Itô and non-Itô processes

Customizable configurations and default parameters

The library is suitable for researchers, students, and practitioners in fields such as economics, finance, physics, and applied mathematics. It aims to bridge the gap between theoretical concepts in ergodicity economics and practical computational tools.
For more information, visit: www.ergodicitylibrary.com
Version: 0.3.0
"""

version = "0.3.0"

from .process.default_values import *
from .configurations import *
