"""
agents Module Overview

The **`agents`** module provides a comprehensive suite of tools for simulating, analyzing, and evaluating agents interacting with stochastic processes. The module is built with a focus on various decision-making frameworks, optimization strategies, evolutionary algorithms, and portfolio management. It integrates methods from utility theory, reinforcement learning, and stochastic modeling, offering flexible ways to simulate agent behaviors and dynamics in complex environments.

Submodules:

1. **`agent_pool.py`**:

   - Defines a pool of agents that interact with stochastic processes.

   - Agents in the pool can share wealth, update wealth based on individual dynamics, and simulate multi-agent environments.

   - Inequality measures such as Gini coefficient, Palma ratio, and mean logarithmic deviation are tracked to monitor wealth distribution across the agent population.

2. **`agents.py`**:

   - Contains core agent models, focusing on decision-making in stochastic environments.

   - Includes base classes for agents interacting with processes, updating wealth, and calculating fitness based on wealth accumulation.

   - Supports agent cloning, mutation, and other operations essential for evolutionary algorithms.

3. **`evaluation.py`**:

   - Focuses on the evaluation of agents' decision-making using utility functions.

   - Supports utility function inference, Bayesian fitting, and the use of utility functions in reinforcement learning (RL) contexts.

   - Provides visualization and regression tools to understand how agents' decisions evolve over time and under different stochastic conditions.

4. **`evolutionary_nn.py`**:

   - Implements evolutionary neural networks for agents to learn optimal strategies.

   - Provides a framework for evolving agent behavior using neural networks, mutation, and crossover.

   - Includes reinforcement-based evolution for optimizing agent performance, with visualization tools for network evolution and performance tracking.

5. **`portfolio.py`**:

   - Simulates and manages portfolios of assets that follow stochastic processes (e.g., Geometric Brownian Motion).

   - Enables agents to allocate resources across multiple processes, dynamically adjusting weights and tracking portfolio wealth over time.

   - Provides visualization of portfolio wealth and weight dynamics.

6. **`probability_weighting.py`**:

   - Applies probability weighting to stochastic processes using Girsanov's theorem.

   - Adjusts drift and volatility based on changes in the probability measure, simulating weighted processes and visualizing their behavior over time.

   - Useful in risk management, derivative pricing, and understanding the effects of probability distortions on process outcomes.

7. **`sml.py`**:

   - Provides methods for Stochastic Maximum Likelihood (SML) estimation, utility optimization, and process simulation.

   - Agents can optimize their decisions based on estimated likelihood functions, updating their strategies to maximize utility or wealth in stochastic environments.

   - Includes utility function definitions and optimization routines for modeling agent preferences.

Applications:
The **`agents`** module is designed for use in multi-agent simulations, financial modeling, evolutionary algorithms, and stochastic decision-making research. It supports applications in:

   - **Financial modeling**: Simulating portfolios, agent wealth dynamics, and stochastic processes like Geometric Brownian Motion.

   - **Multi-agent systems**: Tracking wealth distribution, inequality, and cooperative behaviors across agents in a shared environment.

   - **Reinforcement learning**: Training agents to optimize their strategies using neural networks, evolutionary algorithms, and utility function-based evaluation.

   - **Risk management and probability weighting**: Using Girsanov's theorem to model the effect of probability distortions on stochastic process outcomes.

Example Usage:

```python
from ergodicity.agents import agent_pool, portfolio, probability_weighting

# Example: Simulating a portfolio of Geometric Brownian Motion processes

from ergodicity.process.multiplicative import GeometricBrownianMotion

processes = [GeometricBrownianMotion(drift=0.01, volatility=0.2) for _ in range(10)]

weights = [0.1] * 10

my_portfolio = portfolio.Portfolio(processes, weights)

wealth_history, weight_history = my_portfolio.simulate(timestep=0.1, time_period=1.0, total_time=100)

my_portfolio.visualize()

# Example: Using Girsanov's theorem for probability weighting

weighted_pdf = probability_weighting.gbm_weighting(initial_mu=0.05, sigma=0.2)

probability_weighting.visualize_weighting(weighted_pdf, new_mu=0.03, sigma=0.2, timestep=0.01, num_samples=1000, t=1.0)
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

# Import all modules dynamically
for module in modules:
    module_name = os.path.basename(module)[:-3]
    if module_name != "__init__":
        importlib.import_module(f".{module_name}", package="ergodicity.agents")
