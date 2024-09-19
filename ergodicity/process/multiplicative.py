"""
multiplicative Submodule

This submodule provides a comprehensive framework for simulating and analyzing multiplicative stochastic processes.
These processes model phenomena where changes in value are proportional to the current state, making them
ideal for capturing growth dynamics, financial modeling, and systems exhibiting exponential behavior. The
submodule includes implementations of both univariate and multivariate processes, with support for Brownian
motion, Lévy processes, fractional dynamics, and heavy-tailed distributions.

Key Features:

1. **Multiplicative Nature**:

    - All processes in this submodule exhibit multiplicative growth, meaning that increments are proportional

      to the current value. This feature is critical for modeling systems where non-negativity and relative

      changes are fundamental, such as in asset prices or population dynamics.

2. **Heavy-Tailed Distributions**:

    - Several processes leverage Lévy stable distributions to capture extreme events and heavy-tailed behavior,
      accommodating scenarios with infinite variance. This is essential for accurately modeling rare but impactful
      events, especially in risk management, finance, and natural systems.

3. **Multidimensional Capabilities**:

    - Multivariate extensions of key processes allow for the modeling of correlated, interacting systems. These
      include processes where different components are driven by shared stochastic factors, capturing the
      interdependence between multiple variables, such as portfolios of financial assets or ecological populations.

4. **Flexible Simulation and Analysis**:

    - Built-in methods support the simulation of paths, growth rates, and increments, as well as the calculation
      of expected values, variances, and higher-order moments. Visualizations through Matplotlib and Plotly
      further aid in analyzing the behavior of these processes over time.

Available Processes:

- **Geometric Brownian Motion (GBM)**:

    A continuous-time stochastic process commonly used in finance to model stock prices. In this submodule, GBM
    is extended with customizable drift and volatility parameters and supports both closed-form solutions and
    simulation of growth rates.

- **Geometric Lévy Process**:

    Combines the heavy-tailed properties of Lévy stable distributions with multiplicative dynamics. Useful for
    modeling processes with large, unpredictable jumps and extreme events.

- **Multivariate Geometric Brownian Motion (MGBM)**:

    An extension of GBM to multiple dimensions, allowing for the simulation of correlated stochastic processes
    with multiplicative growth in each dimension. Ideal for portfolios, interrelated economic indicators, and
    other systems with multiple interacting components.

- **Geometric Fractional Brownian Motion**:

    A fractional extension of Brownian motion incorporating the Hurst parameter to model long-range dependence.
    Suitable for applications requiring memory effects, such as geophysical or economic time series.

- **Geometric Cauchy Process**:

    A specific case of a Lévy process with Cauchy distribution, providing a mechanism for modeling extremely
    heavy-tailed behavior where variance is infinite.

- **Multivariate Geometric Lévy Process**:

    Extends the Lévy process framework to multiple dimensions, allowing for correlated heavy-tailed behaviors
    across multiple interacting components. This class is particularly useful in fields such as finance, where
    multiple asset prices or risk factors may exhibit simultaneous extreme behaviors.

- **Geometric Generalized Hyperbolic Process**:

    A generalization of the Lévy process that includes the generalized hyperbolic distribution, providing a
    flexible framework for modeling heavy-tailed phenomena with varying skewness and kurtosis.

- **Geometric Bessel Process**:

    A Lévy process based on the Bessel distribution, useful for modeling processes with infinite activity and
    non-negative jumps. This process is particularly relevant in insurance and risk management contexts.

- **Geometric Squared Bessel Process**:

    A variant of the Bessel process where the squared values of the process are considered, leading to different
    behavior and statistical properties. 

Helper Functions:

- **implied_levy_correction**:

    Calculates and visualizes the correction term for a range of alpha and beta parameters, helping to understand
    the behavior of Lévy processes under various conditions.

- **estimate_sigma**:

    Provides an estimate of the sigma parameter for Geometric Lévy Processes across a grid of alpha and time values,
    with options for linear and non-linear regression to capture sigma dynamics.

Applications:

This submodule is versatile and applicable across various fields:

- **Finance**:

    For modeling asset prices, portfolios, or other financial quantities where multiplicative dynamics and heavy-tailed
    behavior are important.

- **Risk Management**:

    Particularly useful in scenarios where extreme, rare events (such as market crashes or catastrophic losses)
    need to be modeled.

- **Natural Systems**:

    For capturing exponential growth dynamics, population models, or interacting ecological systems.

- **Telecommunications**:

    In modeling bursty network traffic or data flows, where traffic patterns exhibit heavy-tailed behavior and
    large fluctuations.

By leveraging these processes, researchers and practitioners can model and analyze complex, real-world systems
that exhibit multiplicative growth, heavy tails, and interdependencies, offering rich insights into stochastic
dynamics.

"""
from typing import List, Any, Type, Callable, Tuple, Literal
from .definitions import ItoProcess
from .definitions import NonItoProcess
from .definitions import Process
import inspect
import numpy as np
from ergodicity.process.definitions import simulation_decorator
from ergodicity.process.definitions import check_simulate_with_differential
from ergodicity.tools.helper import plot_simulate_ensemble
from ergodicity.process.default_values import *
from ergodicity.configurations import *
import sympy as sp
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from stochastic.processes.continuous import GeometricBrownianMotion as StochasticGBM
from .basic import BrownianMotion

class GeometricBrownianMotion(BrownianMotion):
    """
    GeometricBrownianMotion represents a fundamental continuous-time stochastic process used to model the
    dynamics of various phenomena, particularly in finance and economics. The process, denoted as (S_t)_{t≥0}.
    """
    def __init__(self, name: str = "Geometric Brownian Motion", process_class: Type[Any] = StochasticGBM, drift: float = drift_term_default, volatility: float = stochastic_term_default):
        """
        Initialize the Geometric Brownian Motion process.

        :param name: Name of the process
        :type name: str
        :param process_class: Class of the process
        :type process_class: Type[Any]
        :param drift: Drift term of the process
        :type drift: float
        :param volatility: Volatility term of the process
        :type volatility: float
        """
        super().__init__(name, process_class, drift, volatility)
        self.types = ["geometric"]
        self._multiplicative = True
        self._drift = drift
        self._volatility = volatility
        self._drift_term = self._drift
        self._has_wrong_params = False
        self._external_simulator = use_external_simulators
        if use_external_simulators is True:
            self._has_wrong_params = True
            print('External simulator for this process may not work properly. We advice to use the internal simulator or make sure you know exactly what you are doing.')
        self._drift_term_sympy = self._drift*sp.Symbol('x')
        self._stochastic_term_sympy = self._volatility*sp.Symbol('x')

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Calculate the custom increment for the process.

        :param X:  Current value of the process
        :type X: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :return: Increment for the process
        :rtype: float
        """
        dX = X * timestep * self._drift + X * (timestep ** 0.5) * self._volatility * np.random.normal(0, 1)
        return dX

    def simulate_growth_rate(self, t: float = t_default, timestep: float = timestep_default,
                             num_instances: int = num_instances_default, n_simulations: int = num_instances_default, save: bool = False,
                             plot: bool = False) -> np.ndarray:
        """
        Simulate the growth rate of the Geometric Brownian Motion process.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param num_instances: Number of instances to simulate
        :type num_instances: int
        :param save: Whether to save the results to a file
        :type save: bool
        :param plot: Whether to plot the results
        :type plot: bool
        :return: Array of simulated growth rates
        :rtype: np.ndarray
        """
        average_rates = np.empty(n_simulations, dtype=object)
        for i in range(n_simulations):
            # Simulate weights
            weight_data = self.simulate_weights(t, timestep, num_instances, save=False, plot=False)

            # Extract times and weights
            times = weight_data[0, :]
            weights = weight_data[1:, :]

            # Calculate growth rate using the formula
            growth_rates = np.zeros(len(times))
            for j in range(len(times)):
                w = weights[:, j]
                term1 = np.sum(w * self._drift)
                term2 = -0.5 * np.sum(w ** 2 * self._volatility ** 2)
                growth_rates[j] = term1 + term2

            # Combine times and growth rates
            growth_rate_data = np.vstack((times, growth_rates))
            average_rates[i] = growth_rate_data

        # Calculate average growth rates
        growth_rate_data = np.mean(average_rates, axis=0)

        if save:
            filename = f"growth_rate_simulation_{self.get_params()}, t:{t}, timestep:{timestep}, num_instances:{num_instances}.csv"
            header = 'time,growth_rate'
            np.savetxt(filename, growth_rate_data.T, delimiter=',', header=header, comments='')
            print(f"Growth rates saved to {filename}")

        if plot:
            self.plot_growth_rate(times, growth_rates, save)

        return growth_rate_data

    def plot_growth_rate(self, times, growth_rates, save):
        """
        Plot the simulated growth rate of the Geometric Brownian Motion process.

        :param times: Array of time values
        :type times: np.ndarray
        :param growth_rates: Array of growth rates
        :type growth_rates: np.ndarray
        :param save: Whether to save the plot
        :type save: bool
        :return: None
        :rtype: None
        """
        plt.figure(figsize=(10, 6))
        plt.plot(times, growth_rates, label='Growth Rate')
        plt.xlabel('Time')
        plt.ylabel('Growth Rate')
        plt.title('Simulated Growth Rate of Geometric Brownian Motion')
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(f"growth_rate_plot_{self.get_params()}.png")
        plt.show()

    def self_averaging_time_theory(self, num_instances: int = num_instances_default):
        """
        Calculate the theoretical estimate of self-averaging time for the Geometric Brownian Motion process.
        Self-averaging time is the time when the process ensemble exits self-averaging regime.
        It means that the ensemble average of the process is no more equal to the expected value.

        :param num_instances: Number of instances to simulate
        :type num_instances: int
        :return: Theoretical estimate of self-averaging time
        :rtype: float
        """
        sat = (2 * np.log(num_instances)) / (self._volatility ** 2)
        return sat

    def relative_variance_pea_theory(self, num_instances: int = num_instances_default, t: float = t_default):
        """
        Calculate the theoretical relative variance of a partial ensemble average (PEA).
        It is used to estimate if PEA is close to its expectation value.
        If the results is <<1, then PEA is close to its expectation value.
        Otherwise, the process has exited the self-averaging regime.

        :param num_instances: Number of instances to simulate
        :type num_instances: int
        :param t: Total time for the simulation
        :type t: float
        :return: Relative variance of the process
        :rtype: float
        """
        rv = (np.exp((self._volatility ** 2) * t) - 1) / num_instances
        return rv

from .basic import LevyStableProcess
from scipy.stats import levy_stable

import ergodicity.configurations as config

def comments_false():
    config.default_comments = False

def comments_true():
    config.default_comments = True

from ergodicity.tools.compute import *

class GeometricLevyProcess(LevyStableProcess):
    """
    GeometricLevyProcess represents a stochastic process that combines the heavy-tailed
    characteristics of Lévy stable distributions with the multiplicative nature of geometric processes.
    This continuous-time process, denoted as (S_t)_{t≥0}, extends the concept of Geometric Brownian Motion
    to accommodate more extreme fluctuations and asymmetry often observed in complex systems.

    where X_t is a Lévy stable process characterized by four key parameters:

    1. α (alpha): Stability parameter (0 < α ≤ 2), controlling tail heaviness. Smaller values lead to
       heavier tails and more extreme events.

    2. β (beta): Skewness parameter (-1 ≤ β ≤ 1), determining the asymmetry of the distribution.

    3. σ (scale): Scale parameter, influencing the spread of the distribution.

    4. μ (loc): Location parameter, affecting the central tendency of the process.

    Key properties of the Geometric Lévy Process include:

    1. Multiplicative nature: Changes are proportional to the current value, preserving non-negativity.

    2. Heavy-tailed behavior: Capable of modeling extreme events more effectively than Gaussian-based processes.

    3. Potential for infinite variance: For α < 2, capturing highly volatile phenomena.

    4. Self-similarity: Exhibiting fractal-like behavior in certain parameter regimes.

    This implementation inherits from LevyStableProcess, adapting it to a geometric framework. It's
    explicitly set as multiplicative (_multiplicative = True) and uses an internal simulator
    (_external_simulator = False) for precise control over the process generation.

    The class is versatile, finding applications in various fields:

    - Financial modeling: Asset prices with extreme movements, particularly in volatile markets.

    - Risk management: Modeling scenarios with potential for large, sudden changes.

    - Physics: Describing growth processes in complex systems with potential for rapid fluctuations.

    - Telecommunications: Modeling bursty traffic patterns in networks.

    Notable features:

    - Flexible parameterization: Allows fine-tuning of tail behavior, skewness, scale, and location.

    - Simulation control: Uses differential-based simulation (_simulate_with_differential = True) for
      accurate trajectory generation.

    - Type categorization: Classified under both "geometric" and "levy" types, reflecting its dual nature.

    Researchers and practitioners should be aware of the increased complexity in parameter estimation
    and interpretation compared to Gaussian-based models. The rich behavior of this process, especially
    for α < 2, requires careful consideration in application and analysis. While powerful in capturing
    extreme behaviors, users should ensure the chosen parameters align with the underlying phenomena
    being modeled and be prepared for potentially counterintuitive results in statistical analyses.
    """
    def __init__(self, name: str = "Geometric Levy Process", process_class: Type[Any] = None, alpha: float = alpha_default, beta: float = beta_default, scale: float = scale_default, loc: float = 0.5):
        """
        Initialize the Geometric Levy Process with the specified parameters.

        :param name: Name of the process
        :type name: str
        :param process_class: Class of the process
        :type process_class: Type[Any]
        :param alpha: Stability parameter (0 < α ≤ 2)
        :type alpha: float
        :param beta: Skewness parameter (-1 ≤ β ≤ 1)
        :type beta: float
        :param scale: Scale parameter
        :type scale: float
        :param loc: Location parameter
        :type loc: float
        """
        super().__init__(name, process_class)
        self.types = ["geometric", "levy"]
        self._multiplicative = True
        self._alpha = alpha
        self._beta = beta
        self._scale = scale
        self._loc = loc
        self._loc_scaled = loc / 10
        self._simulate_with_differential = True
        self._external_simulator = False

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Calculate the custom increment for the process.

        :param X: Current value of the process
        :type X: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :return: Increment for the process
        :rtype: float
        """
        if simulate_with_differential is True:
            L = LevyStableProcess(alpha=self._alpha, beta=self._beta, scale=self._scale, loc=self._loc, comments=False)
            dL = L.increment(timestep_increment=timestep)
            dX = X * dL
        else:
            dX = X * self._loc * timestep + X * (timestep ** (1/self._alpha)) * self._scale * levy_stable.rvs(alpha=self._alpha, beta=self._beta, loc=0,
                                            scale=1)
        return dX

    def sigma_divergence(self, t: float = t_default, timestep: float = timestep_default) -> Any:
        """
        Calculate the divergence between the theoretical and empirical sigma values for the process.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :return: Theoretical and empirical sigma values
        :rtype: Any
        """
        theoretical_sigma = self._scale / ((1 / 2) ** 0.5)
        data = self.simulate(t=t, timestep=timestep, num_instances=1)
        times, mean, variance, skewness, kurtosis, mad = self.moments(data=data, t=t, timestep=timestep,
                                                                      num_instances=1)

        # Convert variance to a 1D array if it's not already
        if isinstance(variance, np.ndarray) and variance.ndim > 1:
            variance = variance.flatten()

        empirical_sigma = np.sqrt(variance)

        # Ensure times and empirical_sigma have the same length
        if len(times) != len(empirical_sigma):
            times = times[:len(empirical_sigma)]

        # Plot theoretical vs empirical sigma
        plt.plot(times, [theoretical_sigma] * len(times), label='Theoretical Sigma')
        plt.plot(times, empirical_sigma, label='Empirical Sigma')
        plt.xlabel('Time')
        plt.ylabel('Sigma')
        plt.title('Theoretical vs Empirical Sigma')
        plt.legend()
        plt.show()

        return theoretical_sigma, empirical_sigma

    def implied_correction(self, t: float = t_default, timestep: float = timestep_default, save: bool = True, plot: bool = False) -> Any:
        """
        Calculate the implied correction term for the Geometric Levy Process (analogoys to -0.5 * sigma^2 * t for GBM).

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param save: Whether to save the results
        :type save: bool
        :param plot: Whether to plot the results
        :type plot: bool
        :return: Implied correction
        :rtype: float
        """

        data = self.simulate(t=t, timestep=timestep, num_instances=1)
        times, mean, variance, skewness, kurtosis, mad = self.moments(data=data, t=t, timestep=timestep,
                                                                      num_instances=1)
        # theoretical_sigma = self._scale / ((1 / 2) ** 0.5)

        average_growth_rate_values = average_growth_rate(data, visualize=False, save=save)
        times = average_growth_rate_values[0, :]
        data_increments = relative_increments(data, visualize=False)
        mu, sigma = mu_sigma(data_increments)

        mu_implied = self._loc
        sigma_implied = self._scale / ((1 / 2) ** 0.5)

        growth_function = mu * times - 0.5 * sigma ** 2 * times

        # implied_growth_function = mu_implied * times - correction * sigma_implied ** 2 * times

        naive_growth_function = mu_implied * times - 0.5 * sigma_implied ** 2 * times

        correction = (mu_implied - mu + 0.5 * sigma ** 2) / (sigma_implied ** 2)

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(times, average_growth_rate_values[1, :], label="Growth Rate", lw=1)
            plt.plot(times, growth_function, label="Growth function", lw=1)
            plt.plot(times, naive_growth_function, label="Naive Growth function", lw=1)
            plt.title("Comparison of growth functions")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            if save:
                plt.savefig('comparison_of_averages_vs_gbm.png')
            plt.show()

        print(f"Implied correction: {correction}")

        return correction

from .basic import MultivariateBrownianMotion
class MultivariateGeometricBrownianMotion(MultivariateBrownianMotion):
    """
    MultivariateGeometricBrownianMotion represents a sophisticated extension of Geometric Brownian Motion
    to multiple dimensions, providing a powerful framework for modeling correlated, exponentially growing
    processes subject to random fluctuations. This continuous-time stochastic process, denoted as
    (S_t)_{t≥0} where S_t is a vector in R^n, is characterized by the system of stochastic differential
    equations:

    dS_i(t) = μ_i S_i(t) dt + Σ_ij S_i(t) dW_j(t)  for i = 1, ..., n

    where:

    - μ (drift) is a vector representing the average rates of return or growth for each component

    - Σ (scale) is a matrix capturing both volatilities and correlations between components

    - W_t is a vector of standard Brownian motions

    Key properties include:

    1. Multiplicative nature in each dimension: Changes are proportional to current values, preserving
       non-negativity of each component.

    2. Log-normality: The logarithm of each component follows a multivariate normal distribution.

    3. Complex correlation structure: Allows modeling of intricate dependencies between components.

    4. Non-stationary: Variances and covariances increase over time.

    This implementation extends the MultivariateBrownianMotion class, adapting it to the geometric nature
    of the process. It's explicitly set as multiplicative (_multiplicative = True) and uses an internal
    simulator (_external_simulator = False) for precise control over the process generation.

    Notable features:

    - Flexible initialization: Accepts drift vector and scale matrix as inputs, allowing for detailed
      specification of growth rates and interdependencies.

    - Numpy integration: Utilizes numpy arrays for efficient computation and manipulation of
      multi-dimensional data.

    - Parameter handling: The _has_wrong_params flag is set to True, indicating potential need for
      parameter adjustment in certain contexts.

    - Initial state: _X is initialized to a vector of ones, reflecting the typical starting point for
      geometric processes.

    Researchers and practitioners should be aware of the increased complexity in parameter estimation
    and interpretation compared to univariate models. The interplay between drift components and the
    scale matrix requires careful consideration, particularly in high-dimensional settings. While
    powerful in capturing complex, correlated growth phenomena, users should ensure the model's
    assumptions align with the characteristics of the system being studied. The log-normal nature of
    the process may not be suitable for all applications, and consideration of alternative multivariate
    processes (e.g., based on Lévy processes) might be necessary for scenarios involving more extreme
    events or heavier tails.
    """
    def __init__(self, name: str = "Multivariate Geometric Brownian Motion", drift: List[float] = mean_list_default, scale: List[List[float]] = variance_matrix_default):
        """
        Initialize the Multivariate Geometric Brownian Motion process.

        :param name: Name of the process
        :type name: str
        :param drift: Drift term of the process
        :type drift: List[float]
        :param scale: Volatility term of the process
        :type scale: List[List[float]]
        :raises ValueError: If drift and scale have different shapes
        :raises ValueError: If the scale matrix is not symmetric and positive definite
        """
        super().__init__(name, drift, scale)
        self.types = ["geometric"]
        self._multiplicative = True
        self._drift = np.array(drift)
        self._scale = np.array(scale)
        self._drift_term = self._drift
        self._stochastic_term = self._scale
        self._has_wrong_params = True
        self._X = np.ones(len(self._drift))
        self._external_simulator = False
        if np.shape(self._drift) != np.shape(self._scale):
            raise ValueError("Drift and scale must have the same shape")
        # check if the scale matrix is symmetric and positive definite
        if not np.allclose(self._scale, self._scale.T) or not np.all(np.linalg.eigvals(self._scale) > 0):
            raise ValueError("Scale matrix must be symmetric and positive definite")

    def custom_increment(self, X: List[float], timestep: float = timestep_default) -> Any:
        """
        Calculate the custom increment for the process.

        :param X: Current values of the process
        :type X: List[float]
        :param timestep: Time step for the simulation
        :type timestep: float
        :return: Increment for the process
        :rtype: List[float]
        """
        # mvbm = MultivariateBrownianMotion(drift=[0]*len(self._drift), scale=self._scale)
        dW = np.random.multivariate_normal(mean=[0]*len(self._drift), cov=self._scale * timestep)
        dX = np.array(X) * timestep * np.array(self._drift) + np.array(X) * dW
        return dX

    def simulate_weights(self, t: float = t_default, timestep: float = timestep_default, save: bool = True,
                                  plot: bool = False) -> np.ndarray:
        """
        Simulate the weights of the Multivariate Geometric Brownian Motion process.

        :param t: Total simulation time
        :type t: float
        :param timestep: Time step for simulation
        :type timestep: float
        :param save: Whether to save the results
        :type save: bool
        :param plot: Whether to plot the results
        :type plot: bool
        :return: Array of simulated weights
        :rtype: np.ndarray
        """
        # Simulate the process
        data = self.simulate(t, timestep, save=False, plot=False)

        # Extract the simulated values (exclude the time column)
        simulated_values = data[1:, :]

        # Calculate weights (shares)
        total = np.sum(simulated_values, axis=0)
        weights = simulated_values / total

        # Prepare data for saving
        times = data[0, :]
        weight_data = np.vstack((times, weights))

        self.save_to_file(data,
                          f"weights_simulation_{self.get_params()}, t:{t}, timestep:{timestep}, num_instances:{self._dims}.csv",
                          save)

        if plot:
            plt.figure(figsize=(10, 6))
            for i in range(self._dims):
                plt.plot(times, weights[i, :], label=f'Weight {i}')
            plt.xlabel('Time')
            plt.ylabel('Weight')
            plt.title('Simulated Weights of Multivariate Brownian Motion')
            plt.legend()
            plt.grid(True)
            if save:
                plt.savefig(f"weights_plot_{self.get_params()}, t:{t}, timestep:{timestep}.png")
            plt.show()

        return weight_data

    def calculate_expected_log_growth_rate(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate the expected log growth rate at each time step.

        :param weights: numpy array of shape (num_instances, num_time_steps)
        :type weights: np.ndarray
        :return: numpy array of expected log growth rates for each time step
        :rtype: np.ndarray
        """
        num_time_steps = weights.shape[1]
        log_growth_rates = np.zeros(num_time_steps)

        for t in range(num_time_steps):
            w = weights[:, t]

            # First term: sum(w_i * mu_i)
            term1 = np.sum(w * self._drift)

            # Second term: -0.5 * (sum(w_i^2 * sigma_i^2) + sum(sum(w_i * w_j * cov(i,j))))
            variance_term = np.sum(w ** 2 * np.diag(self._scale))

            covariance_term = 0
            for i in range(self._dims):
                for j in range(i + 1, self._dims):
                    covariance_term += w[i] * w[j] * self._scale[i, j]

            term2 = -0.5 * (variance_term + 2 * covariance_term)

            log_growth_rates[t] = term1 + term2

        return log_growth_rates

    def simulate_growth_rate(self, t: float = t_default, timestep: float = timestep_default, n_simulations: int = num_instances_default,
                                           save: bool = True, plot: bool = False) -> np.ndarray:
        """
        Simulate the expected log growth rate of the Multivariate Geometric Brownian Motion process.

        :param t: Total simulation time
        :type t: float
        :param timestep: Time step for simulation
        :type timestep: float
        :param n_simulations: Number of simulations to average
        :type n_simulations: int
        :param save: Whether to save the results
        :type save: bool
        :param plot: Whether to plot the results
        :type plot: bool
        :return: Array of simulated expected log growth rates
        :rtype: np.ndarray
        """
        average_data = np.empty(n_simulations, dtype=object)
        for i in range(n_simulations):
          # Simulate weights
            weight_data = self.simulate_weights(t, timestep, save=False, plot=False)

            # Extract weights (exclude time column)
            weights = weight_data[1:, :]

            # Calculate expected log growth rate
            log_growth_rates = self.calculate_expected_log_growth_rate(weights)

            # Combine data
            times = weight_data[0, :]
            combined_data = np.vstack((times, weights, log_growth_rates))
            average_data[i] = combined_data

        # Calculate average data
        combined_data = np.mean(average_data, axis=0)

        if save:
            filename = f"weights_and_growth_rate_{self.get_params()}, t:{t}, timestep:{timestep}.csv"
            header = ','.join(['time'] + [f'weight_{i}' for i in range(self._dims)] + ['log_growth_rate'])
            np.savetxt(filename, combined_data.T, delimiter=',', header=header, comments='')
            print(f"Weights and growth rate saved to {filename}")

        if plot:
            self.plot_weights_and_growth_rate(times, weights, log_growth_rates, save)

        return combined_data

    def plot_weights_and_growth_rate(self, times, weights, log_growth_rates, save):
        """
        Plot the simulated weights and expected log growth rate of the Multivariate Geometric Brownian Motion process.

        :param times: Array of time values
        :type times: np.ndarray
        :param weights: Array of weights
        :type weights: np.ndarray
        :param log_growth_rates: Array of expected log growth rates
        :type log_growth_rates: np.ndarray
        :param save: Whether to save the plot
        :type save: bool
        :return: None
        :rtype: None
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

        # Plot weights
        for i in range(self._dims):
            ax1.plot(times, weights[i, :], label=f'Weight {i}')
        ax1.set_ylabel('Weight')
        ax1.set_title('Simulated Weights of Multivariate Brownian Motion')
        ax1.legend()
        ax1.grid(True)

        # Plot log growth rate
        ax2.plot(times, log_growth_rates, label='Log Growth Rate', color='red')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Log Growth Rate')
        ax2.set_title('Expected Log Growth Rate')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        if save:
            plt.savefig(f"weights_and_growth_rate_plot_{self.get_params()}.png")
        plt.show()

    def simulate_ensemble(self, t: float = t_default, n: int = num_instances_default, timestep: float = timestep_default, save: bool = False) -> Any:
        """
        Simulate a single path for a portfolio consisting of multiple instances.

        :param t: Total simulation time
        :type t: float
        :param timestep: Time step for simulation
        :type timestep: float
        :param num_instances: Number of instances in the portfolio
        :type num_instances: int
        :param save: Whether to save the results
        :type save: bool
        :return: Array of portfolio values and geometric means over time
        :rtype: Any
        """

        num_steps = int(t / timestep)
        dim = len(self._drift)
        num_instances = dim

        # Initialize arrays
        portfolio = np.ones((num_instances, num_steps + 1))
        geometric_means = np.ones(num_steps + 1)
        weights = np.ones((num_instances, num_steps + 1)) / num_instances  # Initialize weights

        # Ensure _scale is a numpy array for matrix operations
        covariance_matrix = np.array(self._scale)

        for step in range(1, num_steps + 1):
            # Calculate weights
            weights[:, step] = portfolio[:, step - 1] / np.sum(portfolio[:, step - 1])

            # Generate increments according to the formula
            drift_term = np.array(self._drift) * weights[:, step] * timestep
            variance_term = 0.5 * np.diag(np.dot(weights[:, step].reshape(1, -1),
                                                 np.dot(covariance_matrix,
                                                        weights[:, step].reshape(-1, 1))).flatten()) * timestep
            increments_list = []
            for i in range(n):
                # Generate random normal values for the stochastic term
                random_term = np.random.multivariate_normal(np.zeros(dim), covariance_matrix * timestep)
                increments = drift_term - variance_term + random_term
                increments_list.append(increments)

            increments = np.mean(increments_list, axis=0)

            # Update portfolio values
            portfolio[:, step] = portfolio[:, step - 1] * np.exp(increments)

            # Calculate the new geometric mean
            geometric_means[step] = np.prod(portfolio[:, step] ** weights[:, step])

            if step/num_steps % 0.01 == 0 and verbose:
                print(f'{step/num_steps*100}% done')

        result = {
            'portfolio': portfolio,
            'geometric_means': geometric_means,
            'weights': weights
        }

        if save:
            np.save('portfolio_simulation.npy', result)

        # Call the plotting function
        plot_simulate_ensemble(result, t, save)

        return result

from ergodicity.process.basic import StandardFractionalBrownianMotion

class GeometricFractionalBrownianMotion(NonItoProcess):
    """
    MultivariateGeometricBrownianMotion represents a sophisticated extension of Geometric Brownian Motion
    to multiple dimensions, providing a powerful framework for modeling correlated, exponentially growing
    processes subject to random fluctuations. This continuous-time stochastic process, denoted as
    (S_t)_{t≥0} where S_t is a vector in R^n, is characterized by the system of stochastic differential
    equations:

    dS_i(t) = μ_i S_i(t) dt + Σ_ij S_i(t) dW_j(t)  for i = 1, ..., n

    where:

    - μ (drift) is a vector representing the average rates of return or growth for each component

    - Σ (scale) is a matrix capturing both volatilities and correlations between components

    - W_t is a vector of standard Brownian motions

    Key properties include:

    1. Multiplicative nature in each dimension: Changes are proportional to current values, preserving
       non-negativity of each component.

    2. Log-normality: The logarithm of each component follows a multivariate normal distribution.

    3. Complex correlation structure: Allows modeling of intricate dependencies between components.

    4. Non-stationary: Variances and covariances increase over time.

    This implementation extends the MultivariateBrownianMotion class, adapting it to the geometric nature
    of the process. It's explicitly set as multiplicative (_multiplicative = True) and uses an internal
    simulator (_external_simulator = False) for precise control over the process generation.

    Notable features:

    - Flexible initialization: Accepts drift vector and scale matrix as inputs, allowing for detailed
      specification of growth rates and interdependencies.

    - Numpy integration: Utilizes numpy arrays for efficient computation and manipulation of
      multi-dimensional data.

    - Parameter handling: The _has_wrong_params flag is set to True, indicating potential need for
      parameter adjustment in certain contexts.

    - Initial state: _X is initialized to a vector of ones, reflecting the typical starting point for
      geometric processes.

    Researchers and practitioners should be aware of the increased complexity in parameter estimation
    and interpretation compared to univariate models. The interplay between drift components and the
    scale matrix requires careful consideration, particularly in high-dimensional settings. While
    powerful in capturing complex, correlated growth phenomena, users should ensure the model's
    assumptions align with the characteristics of the system being studied. The log-normal nature of
    the process may not be suitable for all applications, and consideration of alternative multivariate
    processes (e.g., based on Lévy processes) might be necessary for scenarios involving more extreme
    events or heavier tails.
    """
    def __init__(self, name: str = "Geometric Fractional Brownian Motion", process_class: Type[Any] = None, mean: float = drift_term_default, volatility: float = stochastic_term_default, hurst: float = hurst_default):
        """
        Initialize the Geometric Fractional Brownian Motion process.

        :param name: Name of the process
        :type name: str
        :param process_class: Class of the process
        :type process_class: Type[Any]
        :param mean: Drift term of the process
        :type mean: float
        :param volatility: Volatility term of the process
        :type volatility: float
        :param hurst: Hurst parameter of the process
        :type hurst: float
        :raises ValueError: If the Hurst parameter is outside the valid range (0, 1)
        :raises ValueError: If the volatility is non-positive
        """
        super().__init__(name, process_class)
        self.types = ["geometric", "fractional"]
        self._multiplicative = True
        if hurst <= 0 or hurst >= 1:
            raise ValueError("Hurst parameter must be in the range (0, 1)")
        else:
            self._hurst = hurst
        self._drift_term = mean
        if volatility <= 0:
            raise ValueError("Volatility must be positive")
        else:
            self._stochastic_term = volatility
        self._has_wrong_params = True
        self._external_simulator = False

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Calculate the custom increment for the process.
        :param X:   Current value of the process
        :param timestep:    Time step for the simulation
        :return:    Increment for the process
        """
        FBM = StandardFractionalBrownianMotion(hurst=self._hurst)
        dW = FBM.increment(timestep_increment=timestep)
        dX = X * self._drift_term * timestep + X * self._stochastic_term * (timestep ** self._hurst) * dW
        return dX

class GeometricCauchyProcess(GeometricLevyProcess):
    """
    MultivariateGeometricBrownianMotion represents a sophisticated extension of Geometric Brownian Motion
    to multiple dimensions, providing a powerful framework for modeling correlated, exponentially growing
    processes subject to random fluctuations. This continuous-time stochastic process, denoted as
    (S_t)_{t≥0} where S_t is a vector in R^n, is characterized by the system of stochastic differential
    equations:

    dS_i(t) = μ_i S_i(t) dt + Σ_ij S_i(t) dW_j(t)  for i = 1, ..., n

    where:

    - μ (drift) is a vector representing the average rates of return or growth for each component

    - Σ (scale) is a matrix capturing both volatilities and correlations between components

    - W_t is a vector of standard Brownian motions

    Key properties include:

    1. Multiplicative nature in each dimension: Changes are proportional to current values, preserving
       non-negativity of each component.

    2. Log-normality: The logarithm of each component follows a multivariate normal distribution.

    3. Complex correlation structure: Allows modeling of intricate dependencies between components.

    4. Non-stationary: Variances and covariances increase over time.

    This implementation extends the MultivariateBrownianMotion class, adapting it to the geometric nature
    of the process. It's explicitly set as multiplicative (_multiplicative = True) and uses an internal
    simulator (_external_simulator = False) for precise control over the process generation.

    Notable features:

    - Flexible initialization: Accepts drift vector and scale matrix as inputs, allowing for detailed
      specification of growth rates and interdependencies.

    - Numpy integration: Utilizes numpy arrays for efficient computation and manipulation of
      multi-dimensional data.

    - Parameter handling: The _has_wrong_params flag is set to True, indicating potential need for
      parameter adjustment in certain contexts.

    - Initial state: _X is initialized to a vector of ones, reflecting the typical starting point for
      geometric processes.

    Researchers and practitioners should be aware of the increased complexity in parameter estimation
    and interpretation compared to univariate models. The interplay between drift components and the
    scale matrix requires careful consideration, particularly in high-dimensional settings. While
    powerful in capturing complex, correlated growth phenomena, users should ensure the model's
    assumptions align with the characteristics of the system being studied. The log-normal nature of
    the process may not be suitable for all applications, and consideration of alternative multivariate
    processes (e.g., based on Lévy processes) might be necessary for scenarios involving more extreme
    events or heavier tails.
    """
    def __init__(self, name: str = "Geometric Cauchy Process", scale: float = scale_default, loc: float = 0.5):
        """
        Initialize the Geometric Cauchy Process with the specified parameters.

        :param name: Name of the process
        :type name: str
        :param scale: Scale parameter
        :type scale: float
        :param loc: Location parameter
        :type loc: float
        """
        super().__init__(name, alpha=1, beta=0, scale=scale, loc=loc)
        self.types = ["geometric", "cauchy"]

from ergodicity.process.basic import MultivariateLevy
class MultivariateGeometricLevy(MultivariateLevy):
    """
    MultivariateGeometricLevy represents an advanced stochastic process that combines the heavy-tailed
    characteristics of multivariate Lévy stable distributions with the multiplicative nature of geometric
    processes in multiple dimensions. This sophisticated continuous-time process, denoted as (S_t)_{t≥0}
    where S_t is a vector in R^n, is defined by the exponential of a multivariate Lévy stable process:

    S_i(t) = S_i(0) * exp(X_i(t))  for i = 1, ..., n

    where X(t) = (X_1(t), ..., X_n(t)) is a multivariate Lévy stable process.

    Key parameters:

    1. alpha (α): Stability parameter (0 < α ≤ 2), controlling tail heaviness across all dimensions.

    2. beta (β): Skewness parameter (-1 ≤ β ≤ 1), determining asymmetry.

    3. scale: Global scale parameter influencing the spread of the distribution.

    4. loc: Location vector (μ ∈ R^n), shifting the process in each dimension.

    5. correlation_matrix: Specifies the correlation structure between dimensions.

    6. pseudovariances: Vector of pseudovariances for each dimension, generalizing the concept of variance.

    Key properties:

    1. Heavy-tailed behavior: Capable of modeling extreme events in multiple dimensions simultaneously.

    2. Complex dependency structure: Captures intricate correlations between components.

    3. Multiplicative nature: Preserves non-negativity in each dimension, suitable for modeling quantities
       like prices or sizes.

    4. Potential for infinite variance: For α < 2, allowing for highly volatile phenomena in multiple dimensions.

    This implementation extends the MultivariateLevy class, adapting it to a geometric framework. It's
    explicitly set as multiplicative (_multiplicative = True) and uses an internal simulator
    (_external_simulator = False) for precise control over the process generation.

    Notable features:

    - Flexible parameterization: Allows fine-tuning of tail behavior, skewness, scale, and multidimensional dependencies.

    - Initialization with unit values: _X is initialized to a vector of ones, reflecting the typical starting point
      for geometric processes.

    - Parameter handling: The _has_wrong_params flag is set to True, indicating potential need for parameter
      adjustment in certain contexts.

    Researchers and practitioners should be aware of several important considerations:

    1. Increased complexity in parameter estimation and interpretation compared to Gaussian-based multivariate models.

    2. Computational challenges in simulating and analyzing high-dimensional heavy-tailed processes.

    3. The need for specialized statistical techniques to handle the lack of finite moments when α < 2.

    4. Careful interpretation of results, especially in risk assessment and forecasting, due to the process's
       capacity for extreme behaviors.

    While the MultivariateGeometricLevy process offers a powerful framework for modeling complex, correlated,
    heavy-tailed phenomena in multiple dimensions, its sophisticated nature requires judicious application.
    Users should ensure that the chosen parameters align with the underlying phenomena being modeled and be
    prepared for potentially counterintuitive results in statistical analyses. The process's rich behavior,
    especially for α < 2, necessitates careful consideration in both theoretical development and practical applications.
    """
    def __init__(self, name: str = "Multivariate Geometric Levy Process",
                 alpha: float = 1.5, beta: float = 0, scale: float = 1,
                 loc: np.ndarray = None, correlation_matrix: np.ndarray = None,
                 pseudovariances: np.ndarray = None):
        """
        Initialize the Multivariate Geometric Levy Process with the specified parameters.

        :param name: Name of the process
        :type name: str
        :param alpha: Stability parameter (0 < α ≤ 2)
        :type alpha: float
        :param beta: Skewness parameter (-1 ≤ β ≤ 1)
        :type beta: float
        :param scale: Scale parameter
        :type scale: float
        :param loc: Location parameter
        :type loc: np.ndarray
        :param correlation_matrix: Correlation matrix
        :type correlation_matrix: np.ndarray
        :param pseudovariances: Pseudovariances
        :type pseudovariances: np.ndarray
        """
        super().__init__(name, alpha=alpha, beta=beta, scale=scale, loc=loc,
                         correlation_matrix=correlation_matrix, pseudovariances=pseudovariances)
        self.types = ["geometric"]
        self._multiplicative = True
        self._has_wrong_params = True
        self._X = np.ones(self._dims)
        self._external_simulator = False

    def custom_increment(self, X: np.ndarray, timestep: float = 1.0) -> np.ndarray:
        """
        Calculate the custom increment for the process.

        :param X: Current values of the process
        :type X: np.ndarray
        :param timestep: Time step for the simulation
        :type timestep: float
        :return: Increment for the process
        :rtype: np.ndarray
        """
        # Generate Levy increments
        dL = super().custom_increment(X, timestep)

        # Calculate the drift term
        drift_term = self._loc * timestep

        # Calculate the variance term
        variance_term = 0.5 * np.diag(np.dot(self._A, self._A.T)) * timestep

        # Calculate the increments for the geometric process
        dX = X * np.exp(drift_term - variance_term + dL) - X

        return dX

    def simulate_ensemble(self, t: float = 1.0, n: int = 1000, timestep: float = 0.01, save: bool = False) -> dict:
        """
        Simulate a single path for a portfolio consisting of multiple instances.

        :param t: Total simulation time
        :type t: float
        :param n: Number of simulations for each time step
        :type n: int
        :param timestep: Time step for simulation
        :type timestep: float
        :param save: Whether to save the results
        :type save: bool
        :return: Dictionary containing portfolio values, geometric means, and weights over time
        :rtype: dict
        """
        num_steps = int(t / timestep)
        num_instances = self._dims

        # Initialize arrays
        portfolio = np.ones((num_instances, num_steps + 1))
        geometric_means = np.ones(num_steps + 1)
        weights = np.ones((num_instances, num_steps + 1)) / num_instances

        for step in range(1, num_steps + 1):
            # Calculate weights
            weights[:, step] = portfolio[:, step - 1] / np.sum(portfolio[:, step - 1])

            # Generate increments
            increments_list = []
            for _ in range(n):
                dX = self.custom_increment(portfolio[:, step - 1], timestep)
                increments_list.append(dX)

            increments = np.mean(increments_list, axis=0)

            # Update portfolio values
            portfolio[:, step] = portfolio[:, step - 1] + increments

            # Calculate the new geometric mean
            geometric_means[step] = np.prod(portfolio[:, step] ** weights[:, step])

        result = {
            'portfolio': portfolio,
            'geometric_means': geometric_means,
            'weights': weights
        }

        if save:
            np.save('portfolio_simulation_levy.npy', result)

        plot_simulate_ensemble(result, t, save)

        return result

from .basic import GeneralizedHyperbolicProcess
class GeometricGeneralizedHyperbolicProcess(GeneralizedHyperbolicProcess):
    """
    GeometricGeneralizedHyperbolicProcess represents a multiplicative version of the Generalized Hyperbolic Process.
    This continuous-time stochastic process combines the flexibility of the Generalized Hyperbolic distribution
    with multiplicative dynamics, making it suitable for modeling phenomena where changes are proportional to
    the current state and exhibit complex distributional characteristics.

    The process is defined as:
    dS(t) = S(t) * dX(t)

    where X(t) is a Generalized Hyperbolic Process.

    Key properties include:
    1. Multiplicative nature: Changes are proportional to the current value, preserving non-negativity.
    2. Flexible distribution: Can model a wide range of tail behaviors and asymmetries.
    3. Nests several important distributions: Including normal, Student's t, variance-gamma, and normal-inverse Gaussian.

    This implementation extends the GeneralizedHyperbolicProcess class, adapting it to a multiplicative framework.
    It's explicitly set as multiplicative (_multiplicative = True) and uses an internal simulator for precise
    control over the process generation.

    Researchers and practitioners should be aware of the increased complexity in parameter estimation and
    interpretation compared to simpler processes. The rich behavior of this process requires careful consideration
    in both theoretical development and practical applications.
    """

    def __init__(self, name: str = "Multiplicative Generalized Hyperbolic Process", process_class: Type[Any] = None,
                 plambda: float = 0, alpha: float = 1.7, beta: float = 0, loc: float = 0.0005, delta: float = 0.01,
                 t_scaling: Callable[[float], float] = lambda t: t ** 0.5, **kwargs):
        """
        Initialize the Geometric Generalized Hyperbolic Process.

        :param name: Name of the process
        :type name: str
        :param process_class: Class of the process
        :type process_class: Type[Any]
        :param plambda: The shape parameter (λ)
        :type plambda: float
        :param alpha: The shape parameter (α)
        :type alpha: float
        :param beta: The skewness parameter (β)
        :type beta: float
        :param loc: The location parameter (μ)
        :type loc: float
        :param delta: The scale parameter (δ)
        :type delta: float
        :param t_scaling: The scaling function for time increments
        :type t_scaling: Callable[[float], float]
        :param kwargs: Additional keyword arguments for the process
        """
        super().__init__(name, process_class, plambda, alpha, beta, loc, delta, t_scaling, **kwargs)
        self.types.append("multiplicative")
        self._multiplicative = True
        self._external_simulator = False

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Calculate the custom increment for the process.

        :param X: Current value of the process
        :type X: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :return: Increment for the process
        :rtype: float
        """
        # Generate increment from the base Generalized Hyperbolic Process
        dX_base = super().custom_increment(X, timestep)

        # Convert to multiplicative increment
        dX = X * dX_base

        return dX

from ergodicity.process.basic import StandardBesselProcess
class GeometricBesselProcess(StandardBesselProcess):
    """
    GeometricBesselProcess represents a multiplicative extension of the StandardBesselProcess.
    This process combines the characteristics of a Bessel process with the multiplicative nature of geometric processes.
    The resulting stochastic process, denoted as (S_t)_{t≥0}, is defined as:

    dS_t = S_t * dR_t

    where R_t is the StandardBesselProcess of dimension d.

    Key features of GeometricBesselProcess include:
    1. Multiplicative nature: Changes are proportional to the current value, preserving non-negativity.
    2. Dimension-dependent behavior: The underlying Bessel process characteristics (recurrence, transience) are preserved.
    3. Non-negative: The process is always positive, making it suitable for modeling quantities that cannot be negative.

    This implementation extends the StandardBesselProcess class, adapting it to a geometric framework.
    It inherits the dimension-dependent properties of the Bessel process while providing a multiplicative growth model.

    Applications of GeometricBesselProcess span various fields:
    - Finance: Modeling asset prices or interest rates with specific volatility structures.
    - Physics: Studying particle diffusion processes with multiplicative growth.
    - Biology: Analyzing population dynamics with radial growth patterns.

    Researchers and practitioners should be aware of the increased complexity in interpretation compared to the standard Bessel process.
    The multiplicative nature introduces new dynamics that require careful consideration in both theoretical development and practical applications.
    """

    def __init__(self, name: str = "Geometric Bessel Process",
                 process_class: Type[Any] = None,
                 dim: int = dim_default):
        """
        Constructor method for the GeometricBesselProcess class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param dim: The dimension of the underlying Bessel process
        :type dim: int
        """
        super().__init__(name, process_class, dim)
        self.types.append("geometric")
        self._multiplicative = True
        self._external_simulator = False

    def custom_increment(self, X: float, timestep: float = timestep_default) -> float:
        """
        Calculate the custom increment for the process.

        :param X: Current value of the process
        :type X: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :return: Increment for the process
        :rtype: float
        """
        # Generate increment from the base Bessel Process
        BP = StandardBesselProcess(dim=self._dim)
        dR = BP.increment(timestep)

        # Convert to multiplicative increment
        dX = X * dR

        return dX

from .basic import SquaredBesselProcess
from typing import Type, Any

class GeometricSquaredBesselProcess(SquaredBesselProcess):
    """
    GeometricSquaredBesselProcess represents a multiplicative extension of the SquaredBesselProcess.
    This process combines the characteristics of a squared Bessel process with the multiplicative nature of geometric processes.
    The resulting stochastic process, denoted as (S_t)_{t≥0}, is defined as:

    dS_t = S_t * dR²_t

    where R²_t is the SquaredBesselProcess of dimension d.

    Key features of GeometricSquaredBesselProcess include:
    1. Multiplicative nature: Changes are proportional to the current value, preserving non-negativity.
    2. Dimension-dependent behavior: The underlying squared Bessel process characteristics are preserved.
    3. Non-negative: The process is always positive, making it suitable for modeling quantities that cannot be negative.
    4. Inherits properties of squared Bessel process: Including recurrence/transience behavior based on dimension.

    This implementation extends the SquaredBesselProcess class, adapting it to a geometric framework.
    It inherits the dimension-dependent properties of the squared Bessel process while providing a multiplicative growth model.

    Applications of GeometricSquaredBesselProcess span various fields:
    - Finance: Modeling volatility of asset prices or interest rates with specific structures.
    - Physics: Studying particle diffusion processes with multiplicative squared radial growth.
    - Biology: Analyzing population dynamics with quadratic radial growth patterns.
    - Queueing theory: Modeling busy periods with multiplicative squared characteristics.

    Researchers and practitioners should be aware of the increased complexity in interpretation compared to the standard squared Bessel process.
    The multiplicative nature introduces new dynamics that require careful consideration in both theoretical development and practical applications.
    The dimension parameter d plays a crucial role in determining the behavior of the process, affecting its recurrence properties and long-term behavior.
    """

    def __init__(self, name: str = "Geometric Squared Bessel Process",
                 process_class: Type[Any] = None,
                 dim: int = dim_default):
        """
        Constructor method for the GeometricSquaredBesselProcess class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param dim: The dimension of the underlying squared Bessel process
        :type dim: int
        """
        super().__init__(name, process_class, dim)
        self.types.append("geometric")
        self._multiplicative = True
        self._external_simulator = False

    def custom_increment(self, X: float, timestep: float = timestep_default) -> float:
        """
        Calculate the custom increment for the process.

        :param X: Current value of the process
        :type X: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :return: Increment for the process
        :rtype: float
        """
        # Generate increment from the base Squared Bessel Process
        SBP = SquaredBesselProcess(dim=self._dim)
        dR2 = SBP.increment(timestep)

        # Convert to multiplicative increment
        dX = X * dR2

        return dX

def implied_levy_correction(alpha_range: Tuple[float, float],
                            beta_range: Tuple[float, float],
                            time_range: Tuple[float, float],
                            alpha_step: float,
                            beta_step: float,
                            time_step: float,
                            loc: float,
                            scale: float,
                            timestep: float,
                            save_path: str = None,
                            save_html: bool = True) -> Any:
    """
    Calculate and plot the correction term as a function of alpha and beta for a given loc and scale.

    :param alpha_range: Tuple of (min_alpha, max_alpha)
    :type alpha_range: Tuple[float, float]
    :param beta_range: Tuple of (min_beta, max_beta)
    :type beta_range: Tuple[float, float]
    :param time_range: Tuple of (min_time, max_time) or a single time value
    :type time_range: Tuple[float, float] or float
    :param alpha_step: Step size for alpha
    :type alpha_step: float
    :param beta_step: Step size for beta
    :type beta_step: float
    :param time_step: Step size for time
    :type time_step: float
    :param loc: Fixed loc parameter
    :type loc: float
    :param scale: Fixed scale parameter
    :type scale: float
    :param timestep: Fixed timestep for simulation
    :type timestep: float
    :param num_instances: Number of instances to simulate for each parameter combination
    :type num_instances: int
    :param save_path: Path to save the results (if None, results won't be saved)
    :type save_path: str
    :param save_html: If True, save the graph as an interactive HTML file
    :type save_html: bool
    :return: 3D numpy array of correction terms
    :rtype: NumPy array
    """

    alphas = np.arange(*alpha_range, alpha_step)
    betas = np.arange(*beta_range, beta_step)

    # if times is float just convert it to list
    if isinstance(time_range, float):
        times = [time_range]

    else:
        times = np.arange(*time_range, time_step)

    corrections = np.zeros((len(alphas), len(betas), len(times)))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            for k, time in enumerate(times):
                process = GeometricLevyProcess(alpha=alpha, beta=beta, scale=scale, loc=loc)
                correction = process.implied_correction(t=time, timestep=timestep, save=False, plot=False)
                corrections[i, j, k] = correction

    # Create the 3D plot using matplotlib
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(betas, alphas)
    surf = ax.plot_surface(X, Y, corrections[:, :, -1], cmap='viridis')

    ax.set_xlabel('Beta')
    ax.set_ylabel('Alpha')
    ax.set_zlabel('Correction Term')
    ax.set_title(f'Correction Term vs Alpha and Beta (loc={loc}, scale={scale})')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    if save_path:
        plt.savefig(save_path)
        np.save(save_path.replace('.png', '.npy'), corrections)

    plt.show()

    # Create and save interactive 3D plot using Plotly
    if save_html:
        fig_plotly = go.Figure(data=[go.Surface(z=corrections[:, :, -1], x=betas, y=alphas)])
        fig_plotly.update_layout(title=f'Correction Term vs Alpha and Beta (loc={loc}, scale={scale})',
                                 scene=dict(xaxis_title='Beta',
                                            yaxis_title='Alpha',
                                            zaxis_title='Correction Term'))

        html_filename = f'levy_correction_loc_{loc}_scale_{scale}.html'
        fig_plotly.write_html(html_filename)
        print(f"3D interactive graph saved as {html_filename}")

    return corrections

def estimate_sigma(
        num_steps_alpha: int,
        num_steps_time: int,
        max_time: float,
        min_time: float = 1.0,
        max_alpha: float = 2.0,
        min_alpha: float = 1.0,
        timestep: float = 0.001,
        beta: float = 0,
        loc: float = 0.0001,
        scale: float = 0.0001,
        time_scale: Literal['linear', 'log'] = 'linear',
        alpha_scale: Literal['linear', 'log'] = 'linear',
        save_path: str = None,
        save_html: bool = True
) -> Tuple[np.ndarray, Any, Any]:
    """
    Estimate sigma(alpha, t) for GeometricLevyProcess.

    :param num_steps_alpha: Number of steps for alpha
    :type num_steps_alpha: int
    :param num_steps_time: Number of steps for time
    :type num_steps_time: int
    :param max_time: Maximum time
    :type max_time: float
    :param min_time: Minimum time (default: 1.0)
    :type min_time: float
    :param max_alpha: Maximum alpha (default: 2.0)
    :type max_alpha: float
    :param min_alpha: Minimum alpha (default: 0.01)
    :type min_alpha: float
    :param timestep: Fixed timestep for simulation
    :type timestep: float
    :param beta: Fixed beta parameter
    :type beta: float
    :param loc: Fixed loc parameter
    :type loc: float
    :param scale: Fixed scale parameter
    :type scale: float
    :param time_scale: 'linear' or 'log' for time sampling
    :type time_scale: str
    :param alpha_scale: 'linear' or 'log' for alpha sampling
    :type alpha_scale: str
    :param save_path: Path to save the results (if None, results won't be saved)
    :type save_path: str
    :param save_html: If True, save the graph as an interactive HTML file
    :type save_html: bool
    :return: 3D numpy array of sigma values, linear regression results, non-linear regression results
    :rtype: Tuple[np.ndarray, Any, Any]
    """
    if not 0 < min_alpha < max_alpha <= 2:
        raise ValueError("Alpha range must be within (0, 2] and min_alpha must be less than max_alpha")

    if alpha_scale == 'linear':
        alphas = np.linspace(min_alpha, max_alpha, num_steps_alpha)
    elif alpha_scale == 'log':
        alphas = np.logspace(np.log10(min_alpha), np.log10(max_alpha), num_steps_alpha)
    else:
        raise ValueError("alpha_scale must be either 'linear' or 'log'")

    if time_scale == 'linear':
        times = np.linspace(min_time, max_time, num_steps_time)
    elif time_scale == 'log':
        times = np.logspace(np.log10(min_time), np.log10(max_time), num_steps_time)
    else:
        raise ValueError("time_scale must be either 'linear' or 'log'")

    sigma_values = np.zeros((len(alphas), len(times)))

    for i, alpha in enumerate(alphas):
        process = GeometricLevyProcess(alpha=alpha, beta=beta, scale=scale, loc=loc)
        for j, t in enumerate(times):
            process.simulate(t=t, timestep=timestep, num_instances=1)
            sigma_values[i, j] = process.implied_correction(t, timestep, save=False)

    # Create 3D plot using Plotly
    fig = go.Figure(data=[go.Surface(z=sigma_values, x=times, y=alphas)])
    fig.update_layout(
        title=f'Sigma vs Alpha and Time (beta={beta}, loc={loc}, scale={scale})',
        scene=dict(
            xaxis_title='Time',
            yaxis_title='Alpha',
            zaxis_title='Sigma',
            xaxis_type='log' if time_scale == 'log' else 'linear',
            yaxis_type='log' if alpha_scale == 'log' else 'linear'
        ),
        width=1000,
        height=800
    )
    # Create and save interactive 3D plot using Plotly
    if save_html:
        fig_plotly = go.Figure(data=[go.Surface(z=sigma_values, x=times, y=alphas)])
        fig_plotly.update_layout(
            title=f'Sigma vs Alpha and Time (beta={beta}, loc={loc}, scale={scale})',
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Alpha',
                zaxis_title='Sigma',
                xaxis_type='log' if time_scale == 'log' else 'linear',
                yaxis_type='log' if alpha_scale == 'log' else 'linear'
            )
        )

        html_filename = f'sigma_estimation_beta_{beta}_loc_{loc}_scale_{scale}.html'
        fig_plotly.write_html(html_filename)
        print(f"3D interactive graph saved as {html_filename}")

    fig.show()
    # Perform regression analysis

    # Prepare data for regression
    X = np.column_stack((alphas.repeat(len(times)), np.tile(times, len(alphas))))
    y = sigma_values.flatten()

    # Linear regression
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_pred_linear = linear_model.predict(X)
    r2_linear = r2_score(y, y_pred_linear)

    # Non-linear regression
    def non_linear_model(X, a, b, c, d):
        alpha, t = X[:, 0], X[:, 1]
        return a * np.exp(-b * alpha) * t ** c + d

    try:
        non_linear_params, _ = curve_fit(non_linear_model, X, y, maxfev=10000)
        y_pred_non_linear = non_linear_model(X, *non_linear_params)
        r2_non_linear = r2_score(y, y_pred_non_linear)
        non_linear_success = True
    except RuntimeError as e:
        print(f"Non-linear regression failed to converge: {str(e)}")
        non_linear_params = None
        y_pred_non_linear = None
        r2_non_linear = None
        non_linear_success = False

    # Plot regression results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Linear regression plot
    ax1.scatter(y, y_pred_linear, alpha=0.5)
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Sigma')
    ax1.set_ylabel('Predicted Sigma (Linear)')
    ax1.set_title(f'Linear Regression (R² = {r2_linear:.4f})')

    # Non-linear regression plot
    if non_linear_success:
        ax2.scatter(y, y_pred_non_linear, alpha=0.5)
        ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax2.set_xlabel('Actual Sigma')
        ax2.set_ylabel('Predicted Sigma (Non-linear)')
        ax2.set_title(f'Non-linear Regression (R² = {r2_non_linear:.4f})')
    else:
        ax2.text(0.5, 0.5, 'Non-linear regression failed to converge',
                 ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    plt.show()

    return sigma_values, linear_model, non_linear_params, r2_linear, r2_non_linear

