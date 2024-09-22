# Ergodicity Library

## Overview

The **Ergodicity Library** is a Python-based open-source library focused on the study of stochastic processes, ergodicity, and time-average behaviors. Initially inspired by the work of Ole Peters and Alexander Adamou in *Ergodicity Economics*, the library extends far beyond economics and can be applied in fields such as biology, physics, finance, and more. It provides tools for simulating and analyzing stochastic processes, with a particular focus on processes with heavy tails, and enables research on non-ergodic dynamics.

## Key Features

- **Stochastic Process Simulation**: Includes a wide variety of stochastic processes, including Brownian motion, Lévy processes, and fractional Brownian motion.
- **Ergodicity Research**: Tools for studying ergodicity and time-average behavior of stochastic processes.
- **Artificial Agents**: Evolutionary algorithms and machine learning models for training agents to make decisions under uncertainty.
- **Heavy-Tailed Processes**: Special focus on processes with heavy-tailed distributions, such as Lévy alpha-stable processes.
- **Symbolic Computations**: Powerful tools for solving stochastic differential equations (SDEs) both numerically and symbolically.
- **Parallel Processing**: Support for multiprocessing to handle large-scale simulations efficiently.
- **Visualization**: Dynamic visualization of stochastic processes, with support for 3D interactive graphs.

## Installation

To install the Ergodicity Library, you can clone the repository from GitHub:

```bash
git clone https://github.com/Kendiukhov/ergodicity_library.git
cd ergodicity-library
pip install -r requirements.txt
```
Or you just install it directly from PyPI:

```bash
pip install ergodicity-library
```

The library relies on widely used Python libraries like `numpy`, `matplotlib`, `scipy`, `tensorflow`, and `keras`.

## Getting Started

### Programming Prerequisites

- Basic knowledge of Python
- Familiarity with scientific libraries: `NumPy`, `SymPy`, `Matplotlib`
- Optional: `TensorFlow` and `Keras` for machine learning components

### Academic Prerequisites

A basic understanding of the following topics is recommended:

- Probability theory and stochastic processes
- Ito calculus and stochastic differential equations (SDEs)
- Statistical estimation and fitting

### Example

Here is a simple example to get started with simulating a Brownian motion process:

```python
from ergodicity.process.basic import BrownianMotion

# Create a Brownian motion process
bm = BrownianMotion()

# Simulate the process
realization = bm.simulate(t=10, timestep=0.1 num_instance=1000, plot=True)
```

## Documentation

Extensive documentation, examples, and tutorials are available at the Ergodicity Library website: [https://ergodicitylibrary.com](https://ergodicitylibrary.com).

The documentation is also available on GitHub: [https://kendiukhov.github.io/ergodicity_library](https://kendiukhov.github.io/ergodicity_library).

## Structure

The library is structured into three main modules:

1. **Processes**: Provides the core stochastic processes such as Brownian motion, Poisson processes, Lévy processes, and custom-defined processes.
   - *Submodules*:
     - `basic.py`: Core stochastic processes.
     - `multiplicative.py`: Processes with multiplicative dynamics.
     - `with_memory.py`: Stochastic processes with memory.
     - `increments.py`: Defines various stochastic differentials.
     - `lib.py`: Additional less common processes.
     - `constructor.py`: Allows creating new processes without programming.
     - `custom_classes.py`: Facilitates creating new process classes.
     - `default_values.py`: Contains default process parameters.
     - `definitions.py`: Core of the library, containing abstract classes and methods.
     - `discrete.py`: Handles discrete stochastic processes.

2. **Tools**: Functions for computation, analysis, and symbolic/numerical solutions for stochastic processes.
   - *Submodules*:
     - `compute.py`: Functions for calculating parameters of stochastic processes.
     - `evaluate.py`: Evaluates time-series data.
     - `fit.py`: Fits data to various distributions.
     - `solve.py`: Symbolic solutions for stochastic analysis problems.
     - `multiprocessing.py`: Supports multi-core computations.
     - `automate.py`: Automates research calls for comparative results.
     - `preasymptotics.py`: Analyzes pre-asymptotic behavior of processes.
     - `research.py`: Automated research pipelines.
     - `partial_sde.py`: Numerical simulations of partial stochastic differential equations.
     - `helper.py`: Auxiliary functions for different parts of the tools module.

3. **Agents**: Tools to create and train artificial agents using evolutionary algorithms and machine learning models.
   - *Submodules*:
     - `agents.py`: Initializes artificial agents and elementary algorithms for training.
     - `agent_pool.py`: Manages ensembles of interacting agents.
     - `evaluation.py`: Tool for analyzing agent behavior and utility functions.
     - `evolutionary_nn.py`: Implements evolutionary training for agent neural networks.
     - `sml.py`: Standard machine learning algorithms for training agents.
     - `portfolio.py`: Simulates portfolios of stochastic processes.
     - `probability_weighting.py`: Implements probability weighting functions for decision making.

Additionally, the library contains the following components:

- `developer_tools`: Tools for developers working on the library.
- `gui`: Placeholder for future graphical user interface implementation.
- `integrations`: Placeholder for integration with other languages and domain-specific libraries.
- `cases.py`: Practical cases demonstrating the library’s functionality.
- `configurations.py`: Global parameters for the library’s behavior.
- `custom_warnings.py`: Custom warnings designed to help users with library-specific issues.

## Example Projects

Here are a few example projects you can explore with the Ergodicity Library:

1. **Simulating Financial Models**: Apply the library’s tools to simulate and analyze geometric Brownian motion and other stochastic processes in financial markets.
2. **Biological Population Dynamics**: Use stochastic population models to simulate growth patterns in ecosystems and analyze stability using ergodic theory.
3. **Training Artificial Agents**: Implement machine learning agents that optimize their decision-making strategies under stochastic environments using evolutionary algorithms.

## Contributions

The library is in active development, and contributions are welcome. Here’s how you can help:

1. **Bug Reports & Feedback**: If you encounter any bugs or have suggestions for improvements, feel free to report them on the GitHub issues page.
2. **New Features & Modules**: You can help by developing new features, improving existing ones, or expanding the library into new scientific domains like finance, biology, or physics.
3. **Testing**: If you are using the library for your own research, we would greatly appreciate your feedback and testing of its features.

To contribute, fork the repository and submit a pull request with your changes.

## Roadmap

Future releases of the Ergodicity Library will include:

- **Graphical User Interface (GUI)**: To enable easier interaction with the library without requiring deep programming knowledge.
- **Expanded Machine Learning Support**: Improved agent training models with integration of reinforcement learning and other neural network architectures.
- **Performance Enhancements**: Integration with C and Wolfram languages to optimize large-scale stochastic process simulations.
- **New Domain-Specific Modules**: Modules for domain-specific analysis in biology, finance, and physics.

## License

The Ergodicity Library is open-source and licensed under the MIT License. See the `LICENSE` file for more details.
