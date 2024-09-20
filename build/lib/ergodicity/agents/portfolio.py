"""
portfolio Submodule Overview

The **`portfolio`** submodule allows for the simulation and analysis of portfolios composed of multiple stochastic processes. This is particularly useful for studying portfolio dynamics, wealth growth, and the evolution of asset weights over time in stochastic investment environments.

Key Features:

1. **Portfolio Simulation**:

   - Simulates a portfolio consisting of different stochastic processes (e.g., Geometric Brownian Motion).

   - Dynamically adjusts the weights of each process in the portfolio based on the simulated returns of the processes.

2. **Wealth and Weight Dynamics**:

   - Tracks the evolution of portfolio wealth and the individual weights of each asset in the portfolio over time.

   - Allows visualization of both wealth and weight dynamics to analyze performance and diversification.

3. **Process Integration**:

   - Works with any stochastic process object that provides a `simulate()` method (e.g., GeometricBrownianMotion).

Example Usage:

from ergodicity.portfolio import Portfolio

from ergodicity.process.multiplicative import GeometricBrownianMotion

# Number of processes (e.g., 100 assets)

n = 100

# Create stochastic processes (e.g., GBMs for assets)

processes = []

for i in range(n):

    gbm = GeometricBrownianMotion(drift=0.016, volatility=0.3)

    processes.append(gbm)

# Initialize portfolio with equal weights for all assets

weights = [1/n] * n

portfolio = Portfolio(processes, weights)

# Simulate portfolio over time

wealth_history, weight_history = portfolio.simulate(timestep=0.5, time_period=1, total_time=1000)

# Visualize portfolio wealth and weight dynamics

portfolio.visualize()
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any

class Portfolio:
    """
    Portfolio Class

    This class represents a portfolio of stochastic processes, simulating their combined behavior
    over time. It's designed to model and analyze the dynamics of a diversified investment portfolio
    in a stochastic environment.

    Attributes:
        processes (List[Any]): A list of stochastic process objects representing different assets.
        initial_weights (np.array): The initial allocation weights for each process, normalized to sum to 1.
        current_weights (np.array): The current allocation weights, updated during simulation.
        initial_wealth (float): The initial wealth of the portfolio, set to 1.0.
        current_wealth (float): The current wealth of the portfolio, updated during simulation.
        _current_portfolio (np.array): The current value of each asset in the portfolio.
        wealth_history (List[float]): A record of the portfolio's wealth over time.
        weight_history (List[np.array]): A record of the asset weights over time.

    This class is particularly useful for:

    1. Simulating the behavior of a diversified portfolio over time.

    2. Analyzing how different stochastic processes (assets) interact within a portfolio.

    3. Visualizing the evolution of portfolio wealth and asset allocation.

    4. Studying the effects of various rebalancing strategies (implicitly implemented through weight updates).

    Usage:

        processes = [GeometricBrownianMotion(drift=0.05, volatility=0.2) for _ in range(5)]

        weights = [0.2, 0.2, 0.2, 0.2, 0.2]

        portfolio = Portfolio(processes, weights)

        wealth_history, weight_history = portfolio.simulate(timestep=0.01, time_period=1, total_time=100)

        portfolio.visualize()

    Note:
    The simulation assumes that each process has a 'simulate' method that returns a time series
    of values. The portfolio is rebalanced at each 'time_period' interval, reflecting a dynamic
    asset allocation strategy.
    """
    def __init__(self, processes: List[Any], weights: List[float]):
        """
        Initialize the Portfolio.

        :param processes: List of stochastic process objects
        :type processes: List[Any]
        :param weights: List of initial weights for each process
        :type weights: List[float]
        """
        if len(processes) != len(weights):
            raise ValueError("The number of processes must match the number of weights.")

        self.processes = processes
        self.initial_weights = np.array(weights) / np.sum(weights)  # Normalize weights
        self.current_weights = self.initial_weights.copy()
        self.initial_wealth = 1.0
        self.current_wealth = self.initial_wealth
        self._current_portfolio = self.initial_wealth * self.initial_weights
        self.wealth_history = [self.initial_wealth]
        self.weight_history = [self.current_weights]

    def simulate(self, timestep: float, time_period: float, total_time: float):
        """
        Simulate the portfolio over time.

        :param timestep: Simulation timestep
        :type timestep: float
        :param time_period: Period for recalculating weights and wealth
        :type time_period: float
        :param total_time: Total simulation time
        :type total_time: float
        :return: Tuple of wealth history and weight history
        :rtype: Tuple[List[float], List
        """
        num_steps = int(total_time / timestep)
        num_steps_per_period = int(time_period / timestep)
        num_periods = int(total_time / time_period)

        for period in range(num_periods):

            changes = np.array([
                process.simulate(t=time_period, timestep=timestep, num_instances=1)[1, -1]-1
                for process in self.processes
            ])

            # Update wealth
            self.current_wealth *= 1 + np.sum(self.current_weights * changes)
            self.wealth_history.append(self.current_wealth)

            # Update weights
            self._current_portfolio = self._current_portfolio * (1 + changes)
            self.current_weights = self._current_portfolio / np.sum(self._current_portfolio)

            self.weight_history.append(self.current_weights)

        return self.wealth_history, self.weight_history

    def visualize(self):
        """
        Visualize the wealth and weight dynamics of the portfolio.
        """
        # Plot wealth dynamics
        plt.figure(figsize=(12, 6))
        plt.plot(self.wealth_history)
        plt.title('Portfolio Wealth Dynamics')
        plt.xlabel('Time Step')
        plt.ylabel('Wealth')
        plt.grid(True)
        plt.show()

        # Plot weight dynamics
        weight_history_array = np.array(self.weight_history)
        plt.figure(figsize=(12, 6))
        for i in range(len(self.processes)):
            plt.plot(weight_history_array[:, i], label=f'Process {i + 1}')
        plt.title('Portfolio Weight Dynamics')
        plt.xlabel('Time Period')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage:
if __name__ == "__main__":
    from ergodicity.process.multiplicative import GeometricBrownianMotion
    n = 100
    # Create some example processes
    processes = []
    for i in range(n):
        gbm = GeometricBrownianMotion(drift=0.016, volatility=0.3)
        processes.append(gbm)

    weights = [1/n] * n
    # Create a portfolio
    portfolio = Portfolio(processes, weights)

    # Simulate the portfolio
    wealth_history, weight_history = portfolio.simulate(timestep=0.5, time_period=1, total_time=1000)

    # Visualize the results
    portfolio.visualize()
