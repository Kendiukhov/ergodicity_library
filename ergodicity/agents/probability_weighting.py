"""
probability_weighting Submodule Overview

The **`probability_weighting`** submodule provides tools for computing and visualizing probability weighting functions in stochastic processes, particularly focusing on how different drifts and diffusions influence stochastic trajectories under changes of measure (e.g., Girsanov's theorem).

Key Features:

1. **Girsanov's Theorem**:

   - Apply Girsanov's theorem to transform a probability measure in Geometric Brownian Motion (GBM) and other martingale processes.

   - Adjusts the drift of the process based on the probability weighting function.

2. **Stochastic Simulation**:

   - Simulate weighted stochastic processes based on the drift and volatility parameters using the adjusted probability density function (PDF).

3. **Visualization**:

   - Provides functions for simulating and plotting stochastic trajectories under different probability weighting schemes.

Example Usage:

from ergodicity.probability_weighting import gbm_weighting, visualize_weighting

# Parameters for Geometric Brownian Motion (GBM)

initial_mu = 0.05  # Initial drift

sigma = 0.2  # Volatility

# Get the weighted PDF using Girsanov's theorem

weighted_pdf = gbm_weighting(initial_mu, sigma)

# Visualize the weighted process

new_mu = 0.03  # New drift for visualization

X = visualize_weighting(weighted_pdf, new_mu, sigma, timestep=0.01, num_samples=1000, t=1.0)
"""

import sympy as sp
from tenacity import wait_exponential_jitter
from ergodicity.configurations import *
from ergodicity.process.default_values import *

from ergodicity.tools.compute import random_variable_from_pdf
from ergodicity.tools.solve import apply_girsanov
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def gbm_weighting(initial_mu, sigma):
    """
    Apply Girsanov's theorem to transform the probability measure of a Geometric Brownian Motion (GBM) process to a new measure.
    The new measure is defined by adjusting the drift of the GBM process.

    :param initial_mu: Initial drift of the GBM process.
    :type initial_mu: float
    :param sigma: Volatility of the GBM process.
    :type sigma: float
    :return: Weighted probability density function (PDF) after applying Girsanov's theorem.
    :rtype: sympy.core.add.Add
    """
    t = sp.symbols('t')
    new_drift = initial_mu - 0.5 * sigma**2
    weighted_pdf = apply_girsanov(initial_drift=initial_mu, new_drift=new_drift, diffusion=sigma, time_horizon=t)
    return weighted_pdf

def martingale_weighting(initial_mu, sigma):
    """
    Apply Girsanov's theorem to transform the probability measure of a martingale process to a new measure.
    The new measure is defined by adjusting the drift of the martingale process.

    :param initial_mu: Initial drift of the martingale process.
    :param initial_mu: float
    :param sigma: Volatility of the martingale process.
    :param sigma: float
    :return: Weighted probability density function (PDF) after applying Girsanov's theorem.
    :rtype: sympy.core.add.Add
    """
    t = sp.symbols('t')
    new_drift = 0
    weighted_pdf = apply_girsanov(initial_drift=initial_mu, new_drift=new_drift, diffusion=sigma, time_horizon=t)
    return weighted_pdf

def visualize_weighting(weighted_pdf, new_mu, sigma, timestep=timestep_default, num_samples=num_instances_default, t=t_default):
    """
    Visualize the weighted stochastic process based on the adjusted drift and volatility parameters.

    :param weighted_pdf
    :type weighted_pdf: sympy.core.add.Add
    :param new_mu: for now, it must be float
    :type new_mu: float
    :param sigma: for now, it must be float
    :type sigma: float
    :param timestep:
    :type timestep: float
    :param num_samples:
    :type num_samples: int
    :param t:
    :type t: float
    :return: X
    :rtype: numpy.ndarray
    """

    dt = timestep
    W_t = sp.symbols('W_t')
    dW_q = random_variable_from_pdf(weighted_pdf, x=W_t, num_samples=num_samples)
    dW_q = np.array(dW_q).astype(float)
    X = np.ones(num_samples)
    for i in range(1, int(t/dt)):
        dX = new_mu * dt + sigma * dW_q
        X += dX
    plt.plot(X)
    plt.show()
    return X




