"""
agent_pool Submodule Overview

The **`agent_pool`** submodule focuses on simulating and analyzing the wealth dynamics of a pool of agents interacting with stochastic processes. Agents share wealth dynamically based on a specified sharing rate, and the module provides tools to study how wealth distributions and inequality evolve over time.

Key Features:

1. **Agent Pool Simulation**:

   - Simulate the evolution of wealth for a group of agents interacting with a stochastic process (e.g., Geometric Brownian Motion, Lévy processes).

   - Each agent starts with a specified initial wealth, and the evolution of their wealth depends on both the process and the sharing rules.

2. **Dynamic Sharing Rate**:

   - Agents can share wealth dynamically based on their relative wealth or a fixed sharing rate.

   - The simulation can be run with either static or dynamic sharing rates.

3. **Wealth History**:

   - Track the wealth of each agent over time. The full wealth history is recorded and can be saved for further analysis.

4. **Wealth Inequality Measures**:

   - The module provides several key metrics to measure wealth inequality, including:

     - **Mean Logarithmic Deviation (MLD)**: A measure of wealth inequality based on the deviation from the mean.

     - **Gini Coefficient**: A common measure of inequality based on the distribution of wealth.

     - **Coefficient of Variation (CV)**: The ratio of standard deviation to the mean wealth.

     - **Palma Ratio**: The ratio of wealth held by the richest 10% compared to the poorest 40%.

5. **Wealth Distribution Analysis**:

   - Save and visualize the final wealth distribution in both normal and log-log scales.

   - Fit a power law distribution to the wealth data to study the tail behavior.

6. **3D Wealth Visualization**:

   - Create static and interactive 3D plots of total wealth as a function of the number of agents and the sharing rate.

   - Use Matplotlib for static 3D visualization and Plotly for interactive 3D graphs, which can be saved as HTML.

Example Usage:

### Basic Simulation:

from ergodicity.process.multiplicative import GeometricBrownianMotion

from ergodicity.agent_pool import AgentPool

# Initialize process (e.g., GeometricBrownianMotion)

process = GeometricBrownianMotion(drift=0.02, volatility=0.15)

# Initialize agent pool with 100 agents, starting wealth of 100, sharing rate of 0.1, and time horizon of 10

pool = AgentPool(process, n=100, w=100, s=0.1, time=10)

# Simulate the wealth dynamics

pool.simulate(dynamic_s=False)

# Plot wealth evolution

pool.plot()

# Save and plot final wealth distribution

pool.save_and_plot_wealth_distribution('final_wealth')
"""

import numpy as np
import matplotlib.pyplot as plt
from ergodicity.process.definitions import *
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy import stats

class AgentPool:
    """
    AgentPool represents a collection of agents participating in a wealth dynamics simulation.
    This class simulates the evolution of wealth for a group of agents over time, incorporating
    a stochastic process for wealth changes and an optional wealth-sharing mechanism. It provides
    methods for simulation, data analysis, and visualization of wealth dynamics and inequality measures.

    Attributes:

        process: The stochastic process used for simulating wealth changes.

        n (int): Number of agents in the pool.

        initial_wealth (float): Starting wealth for each agent.

        base_s (float): Base sharing rate for wealth redistribution.

        time (float): Total simulation time.

        timestep (float): Time step for wealth updates and sharing.

        simulation_timestep (float): Time step for the underlying stochastic process simulation.

        wealth (numpy.ndarray): Current wealth of each agent.

        history (list): Historical record of wealth for all agents at each time step.

    This class is useful for studying wealth inequality, the effects of different sharing mechanisms,
    and the impact of various stochastic processes on wealth distribution in a population of agents.
    """
    def __init__(self, process, n, w, s, time, simulation_timestep=0.01, timestep=1):
        """
        Initialize an AgentPool with a given stochastic process, number of agents, initial wealth, sharing rate, and time horizon.

        :param process: Stochastic process object (e.g., GeometricBrownianMotion)
        :type process: StochasticProcess
        :param n: Number of agents in the pool
        :type n: int
        :param w: Initial wealth for each agent
        :type w: float
        :param s: Sharing rate (fixed or dynamic)
        :type s: float
        :param time: Total simulation time
        :type time: float
        :param simulation_timestep: Timestep for the stochastic process simulation
        :type simulation_timestep: float
        :param timestep: Timestep for the AgentPool simulation
        """
        self.process = process
        self.n = n
        self.initial_wealth = float(w)
        self.base_s = s
        self.time = time
        self.timestep = timestep
        self.simulation_timestep = simulation_timestep

        self.wealth = np.full(n, self.initial_wealth, dtype=float)
        self.history = [self.wealth.copy()]

    def simulate(self, dynamic_s=False):
        """
        Run the wealth dynamics simulation for the specified time horizon.

        :param dynamic_s: If True, agents share wealth dynamically based on their relative wealth
        :type dynamic_s: bool
        :return: List of wealth history for all agents at each time step
        :rtype: list
        """
        num_steps = int(self.time / self.timestep)
        for step in range(num_steps):
            # Simulate wealth change for all agents
            data_full = self.process.simulate(t=self.timestep, timestep=self.simulation_timestep, num_instances=self.n)

            # Extract the changes, excluding the time row
            changes = data_full[1:, -1] - data_full[1:, 0]

            # Update wealth
            self.wealth *= (1 + changes)

            # Calculate sharing rate for each agent
            if dynamic_s:
                total_wealth = np.sum(self.wealth)
                s_values = self.base_s * (self.wealth / total_wealth)
            else:
                s_values = np.full(self.n, self.base_s)

            # Calculate shared wealth
            shared_wealth = np.sum(self.wealth * s_values)

            # Update wealth after sharing
            self.wealth = self.wealth * (1 - s_values) + shared_wealth / self.n

            # Record history
            self.history.append(self.wealth.copy())

            # return self.history

    def save_data(self, filename):
        """
        Save the wealth history data to a file.

        :param filename: Name of the file to save the data
        :type filename: str
        :return: None
        :rtype: None
        """
        np.save(filename, np.array(self.history))

    def plot(self):
        """
        Visualize the wealth dynamics of all agents over time.

        :return: None
        :rtype: None
        """
        history_array = np.array(self.history)
        time_points = np.arange(0, self.time + self.timestep, self.timestep)

        plt.figure(figsize=(12, 6))

        # Plot individual agent wealth with faint colors
        for i in range(self.n):
            plt.plot(time_points, history_array[:, i], label=f'Agent {i + 1}', alpha=0.1)

        # Plot average wealth
        plt.plot(time_points, np.mean(history_array, axis=1), 'r--', linewidth=2, label='Average Wealth')

        plt.xlabel('Time')
        plt.ylabel('Wealth')
        plt.title('Wealth Dynamics in Agent Pool')
        plt.legend()
        plt.grid(True)
        plt.show()

    def mean_logarithmic_deviation(self):
        """
        Compute the Mean Logarithmic Deviation (MLD) for each time step.

        :return: array of MLD values for each time step
        :rtype: numpy.ndarray
        """
        history_array = np.array(self.history)
        mean_wealth = np.mean(history_array, axis=1)
        mld = np.mean(np.log(mean_wealth[:, np.newaxis] / history_array), axis=1)
        return mld

    def coefficient_of_variation(self):
        """
        Compute the Coefficient of Variation for each time step.

        :return: Array of CV values for each time step
        :rtype: numpy.ndarray
        """
        history_array = np.array(self.history)
        cv = np.std(history_array, axis=1) / np.mean(history_array, axis=1)
        return cv

    def palma_ratio(self):
        """
        Compute the Palma ratio for each time step.
        The Palma ratio is the ratio of the richest 10% of the population's share of gross national income divided by the poorest 40%'s share.

        :return: Array of Palma ratio values for each time step
        :rtype: numpy.ndarray
        """
        history_array = np.array(self.history)
        palma = []
        for wealth in history_array:
            sorted_wealth = np.sort(wealth)
            n = len(wealth)
            top_10_percent = np.sum(sorted_wealth[int(0.9 * n):])
            bottom_40_percent = np.sum(sorted_wealth[:int(0.4 * n)])
            palma.append(top_10_percent / bottom_40_percent)
        return palma

    def gini_coefficient(self):
        """
        Compute the Gini coefficient for each time step.
        The Gini coefficient is a measure of statistical dispersion intended to represent the income or wealth distribution of a nation's residents.

        :return: Array of Gini coefficient values for each time step
        :rtype: numpy.ndarray
        """
        history_array = np.array(self.history)
        gini = []
        for wealth in history_array:
            sorted_wealth = np.sort(wealth)
            index = np.arange(1, len(wealth) + 1)
            n = len(wealth)
            gini.append((np.sum((2 * index - n - 1) * sorted_wealth)) / (n * np.sum(sorted_wealth)))
        return np.array(gini)

    def plot_inequality_measures(self):
        """
        Plot all implemented inequality measures over time.

        :return: None
        :rtype: None
        """
        history_array = np.array(self.history)
        num_timesteps = len(history_array)
        time_points = np.linspace(0, self.time, num_timesteps)

        mld = self.mean_logarithmic_deviation()
        gini = self.gini_coefficient()
        cv = self.coefficient_of_variation()
        palma = self.palma_ratio()

        plt.figure(figsize=(12, 8))
        plt.plot(time_points, mld, label='Mean Logarithmic Deviation')
        plt.plot(time_points, gini, label='Gini Coefficient')
        plt.plot(time_points, cv, label='Coefficient of Variation')
        plt.plot(time_points, palma, label='Palma Ratio')

        plt.xlabel('Time')
        plt.ylabel('Inequality Measure')
        plt.title('Evolution of Inequality Measures')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_and_plot_wealth_distribution(self, filename_prefix):
        """
        Save and plot the final wealth distribution in normal and log-log scales.

        :param filename_prefix: Prefix for saving the plot files
        :type filename_prefix: str
        :return: None
        :rtype: None
        """
        final_wealth = self.history[-1]

        # Save the final wealth distribution
        np.savetxt(f"{filename_prefix}_wealth_distribution.csv", final_wealth, delimiter=",")

        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Normal scale histogram
        ax1.hist(final_wealth, bins=50, edgecolor='black')
        ax1.set_title('Wealth Distribution (Normal Scale)')
        ax1.set_xlabel('Wealth')
        ax1.set_ylabel('Frequency')

        # Log-log scale histogram
        final_wealth_nonzero = final_wealth[final_wealth > 0]
        wealth_range = np.logspace(np.log10(min(final_wealth_nonzero)), np.log10(max(final_wealth_nonzero)), num=50)
        hist, bins = np.histogram(final_wealth_nonzero, bins=wealth_range)
        center = (bins[:-1] + bins[1:]) / 2

        # Remove zero counts for log-log fit
        nonzero = hist > 0
        log_center = np.log10(center[nonzero])
        log_hist = np.log10(hist[nonzero])

        # Linear regression in log-log space
        fit = stats.linregress(log_center, log_hist)

        # Compute the fitted power-law line
        x_fit = np.logspace(np.log10(min(center[nonzero])), np.log10(max(center[nonzero])), 100)
        y_fit = 10 ** (fit.intercept + fit.slope * np.log10(x_fit))

        # Plotting
        ax2.loglog(center, hist, 'k.', markersize=10)
        # ax2.plot(x_fit, y_fit, 'r-', label=f'Power Law Fit (α ≈ {fit.slope:.2f})')
        ax2.set_title('Wealth Distribution (Log-Log Scale)')
        ax2.set_xlabel('Wealth (log scale)')
        ax2.set_ylabel('Frequency (log scale)')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_wealth_distribution.png", dpi=300)
        plt.show()

        print(f"Wealth distribution data saved to {filename_prefix}_wealth_distribution.csv")
        print(f"Wealth distribution plot saved to {filename_prefix}_wealth_distribution.png")


import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_wealth_3d(process, w, time, simulation_timestep, timestep, s_range, n_range, save_html=False):
    """
    Plot a 3D graph of average wealth as a function of sharing rate s and number of agents n.

    :param process: The stochastic process to use (e.g., GeometricBrownianMotion instance)
    :type process: StochasticProcess
    :param w: Initial wealth for each agent
    :type w: float
    :param time: Total simulation time
    :type time: float
    :param simulation_timestep: Timestep for the simulation
    :type simulation_timestep: float
    :param timestep: Timestep for the sharing and update of AgentPool
    :type timestep: float
    :param s_range: Range of sharing rates to plot (e.g., np.linspace(0, 0.5, 20))
    :type s_range: numpy.ndarray
    :param n_range: Range of number of agents to plot (e.g., range(5, 105, 5))
    :type n_range: range
    :param save_html: If True, save an interactive 3D plot as an HTML file
    :type save_html: bool
    :return: None
    :rtype: None
    """
    s_values, n_values = np.meshgrid(s_range, n_range)
    average_wealth = np.zeros_like(s_values, dtype=float)

    for i, n in enumerate(n_range):
        for j, s in enumerate(s_range):
            pool = AgentPool(process, n, w, s, time, simulation_timestep, timestep)
            pool.simulate(dynamic_s=False)
            average_wealth[i, j] = np.mean(pool.wealth)

    # Create static 3D plot using Matplotlib
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(s_values, n_values, average_wealth, cmap='viridis')

    ax.set_xlabel('Sharing Rate (s)')
    ax.set_ylabel('Number of Agents (n)')
    ax.set_zlabel('Average Wealth')
    ax.set_title('Average Wealth as a Function of Sharing Rate and Number of Agents')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    # Create and save interactive 3D plot using Plotly
    if save_html:
        fig_plotly = go.Figure(data=[go.Surface(z=average_wealth, x=s_values, y=n_values)])
        fig_plotly.update_layout(
            title='Average Wealth as a Function of Sharing Rate and Number of Agents',
            scene=dict(
                xaxis_title='Sharing Rate (s)',
                yaxis_title='Number of Agents (n)',
                zaxis_title='Average Wealth'
            )
        )

        html_filename = 'average_wealth_3d_plot.html'
        fig_plotly.write_html(html_filename)
        print(f"3D interactive graph saved as {html_filename}")


