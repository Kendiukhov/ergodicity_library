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

def gbm_weighting(initial_mu, sigma, time_horizon=1):
    """
    Apply Girsanov's theorem to transform the probability measure of a Geometric Brownian Motion (GBM) process to a new measure.
    The new measure is defined by adjusting the drift of the GBM process.

    :param initial_mu: Initial drift of the GBM process. It corresponds to the intended time average of the process. So you insert the time average to get such probability weighting that taking expected value under the new measure gives the time average.
    :type initial_mu: float
    :param sigma: Volatility of the GBM process.
    :type sigma: float
    :return: Weighted probability density function (PDF) after applying Girsanov's theorem.
    :rtype: sympy.core.add.Add
    """
    # t = sp.symbols('t')
    # new_drift = initial_mu - 0.5 * sigma**2
    # weighted_pdf = apply_girsanov(initial_drift=initial_mu, new_drift=new_drift, diffusion=sigma, time_horizon=t)
    # return weighted_pdf
    x = sp.symbols('x')
    new_drift = (initial_mu + 0.5 * sigma ** 2) * x
    print(f'new_drift: {new_drift}')
    initial_mu = initial_mu*x
    sigma = sigma*x
    weighted_pdf = apply_girsanov(initial_drift=initial_mu, new_drift=new_drift, diffusion=sigma,
                                  time_horizon=time_horizon)
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


def visualize_weighting(weighted_pdf, new_mu, sigma, timestep=0.01, num_samples=1000, t=1.0):
    """
    Visualize the weighted stochastic process based on the adjusted drift and volatility parameters.

    :param weighted_pdf: Weighted probability density function.
    :type weighted_pdf: sympy.core.add.Add
    :param new_mu: New drift parameter.
    :type new_mu: float
    :param sigma: Volatility parameter.
    :type sigma: float
    :param timestep: Time step for simulation.
    :type timestep: float
    :param num_samples: Number of sample paths to simulate.
    :type num_samples: int
    :param t: Total time for simulation.
    :type t: float
    :return: Simulated paths.
    :rtype: numpy.ndarray
    """
    dt = timestep
    num_steps = int(t / dt)
    W_t = sp.symbols('W_t')

    X = np.ones((num_samples, num_steps))

    for i in range(1, num_steps):
        current_x = X[:, i - 1]
        dW_q = dt**0.5 * random_variable_from_pdf(weighted_pdf, x=W_t, num_samples=num_samples, t=1)
        print(dW_q)
        dX = current_x * (new_mu * dt + sigma * dW_q)
        print(dX)
        X[:, i] = X[:, i - 1] + dX

    plt.figure(figsize=(10, 6))
    for path in X:
        plt.plot(np.linspace(0, t, num_steps), path, alpha=0.1)
    plt.title("Simulated Paths of Weighted Process")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

    return X




