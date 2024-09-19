"""
compute Submodule

The `compute` submodule provides a comprehensive set of utilities and functions for performing various numerical computations and visualizations related to stochastic processes. This includes tools for calculating growth rates, visualizing time series data, and generating sample paths for different types of stochastic processes. Additionally, it offers advanced numerical methods, such as solving partial differential equations like the Fokker-Planck equation, which is essential for understanding the dynamics of stochastic processes.

Key Features:

1. **Visualization of Time Series Data**:

   - Functions like `visualize_function` and `average` allow for easy visualization of data over time, aiding in the analysis of simulated processes.

   - The submodule also supports visualizing more complex quantities such as growth rates and comparing them with theoretical models like Geometric Brownian Motion (GBM).

2. **Increment Calculation and Validation**:

   - Functions are decorated with `validate_input_shape` to ensure correct input data structure and shape.

   - Tools such as `growth_rates`, `growth_rate_of_average`, and `relative_increments` calculate the growth rates and relative increments for a variety of stochastic processes.

3. **Alpha-Stable Processes**:

   - The `demonstrate_alpha_stable_self_similarity` function illustrates the self-similarity property of alpha-stable processes.

   - It also allows for the comparison of different scaling factors and their effects on the distribution of end values, making it useful for simulating heavy-tailed distributions.

4. **Lévy and Gaussian Processes**:

   - This submodule includes functions to generate and simulate Lévy processes and Gaussian processes, such as `generate_levy_process` and `create_multivariate_gaussian_process`.

   - The matrix `A` is constructed from correlation matrices and variances to model multivariate Gaussian processes, enhancing the ability to work with correlated variables.

5. **Fokker-Planck Equation Solver**:

   - The `solve_fokker_planck_numerically` function numerically solves the Fokker-Planck equation using finite difference methods.

   - The solution is visualized in 3D, making it an invaluable tool for studying drift-diffusion processes and their time evolution.

6. **Statistical Tools**:

   - Functions such as `mu_sigma` and `compare_distributions` provide statistical insights, allowing the user to compute averages, variances, and distributions of stochastic processes over time.

Applications:

This submodule is useful in various fields, including:

- **Financial Modeling**: Simulation of processes like Geometric Brownian Motion and Lévy processes to study asset price dynamics and other market phenomena.

- **Physics and Environmental Science**: Modeling diffusion processes, random walks, and other phenomena where stochastic differential equations apply.

- **Machine Learning and Data Science**: Incorporating stochastic processes in optimization algorithms and reinforcement learning environments.

The `compute` submodule equips users with robust tools for simulating, visualizing, and analyzing stochastic processes. Whether working on theoretical models or practical applications, this submodule offers both simplicity and flexibility in dealing with complex stochastic dynamics.
"""

from .helper import separate
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from typing import Tuple, Any
from scipy.stats import levy_stable
import functools
import sympy as sp
from scipy.integrate import quad
from scipy import integrate
from scipy.interpolate import interp1d

def visualize_function(time: np.ndarray, data: np.ndarray, name: str, save: bool = False) -> None:
    """
    Visualize the given precomputed data evolution over time. The data should be a 2D array with the first row.
    This function is intended to work with the outputs of the other functions in this module.

    :param time: The time points for the data.
    :type time: np.ndarray
    :param data: The data to visualize.
    :type data: np.ndarray
    :param name: The name of the data to display.
    :type name: str
    :param save: Whether to save the plot as a PNG file.
    :type save: bool
    :return: None
    :rtype: None
    """
    times = time
    num_instances = data.shape[0]
    print('instance', num_instances)
    plt.figure(figsize=(10, 6))
    for i in range(num_instances):
        plt.plot(times, data[i], lw=0.5)
    plt.title(f'Simulation of {name}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    if save:
        plt.savefig(f'{name}.png')
    plt.show()

    return None

def validate_input_shape(func):
    """
    Decorator to validate the input shape of the data array for the functions in this module.
    The correct shape is a 2D array with the first row representing the time points and the remaining rows representing the instances.

    :param func: The function to decorate.
    :type func: Any
    :return: The decorated function.
    :rtype: Any
    """
    @functools.wraps(func)
    def wrapper(data: np.ndarray, *args, **kwargs):
        """
        Wrapper function to validate the input shape of the data array.

        :param data: The input data array.
        :type data: np.ndarray
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: The result of the decorated function.
        :rtype: Any
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        if data.ndim != 2:
            raise ValueError("Input data must be a 2D array")
        return func(data, *args, **kwargs)
    return wrapper

@validate_input_shape
def average(data: np.ndarray, visualize: bool = True, name = "average", save: bool = False) -> np.ndarray:
    """
    Calculate the average over all instances for each time step.

    :param data: The input data array of shape (1 + number of instances, number of time steps).
    :type data: np.ndarray
    :param visualize: Whether to visualize the average values.
    :type visualize: bool
    :param name: The name of the data to display.
    :type name: str
    :param save: Whether to save the plot as a PNG file.
    :type save: bool
    :return: The average values for each time step.
    :rtype: np.ndarray
    """
    # Calculate the average over all instances for each time step
    average = np.mean(data[1:], axis=0)
    average = np.array([average])
    times = data[0, :]
    if visualize:
        visualize_function(times, average, name, save=save)
    result = np.vstack([times, average])
    return result

@validate_input_shape
def growth_rates(data: np.ndarray, visualize: bool = True, name = "growth rates", save: bool = False) -> np.ndarray:
    """
    Calculate cumulative log growth rates for each time step based on the input data.

    :param data: The input data array of shape (1 + number of instances, number of time steps).
    :type data: np.ndarray
    :param visualize: Whether to visualize the growth rates.
    :type visualize: bool
    :param name: The name of the data to display.
    :type name: str
    :param save: Whether to save the plot as a PNG file.
    :type save: bool
    :return: The cumulative log growth rates for each time step.
    :rtype: np.ndarray
    """
    # Separate time row and instances data
    times = data[0, 1:]
    instances = data[1:]

    # Calculate cumulative log growth rates for each time step
    # Maybe bring it back instead of the next line
    growth_rates_data = np.log(instances[:, 1:] / instances[:, 1][:, np.newaxis])

    # Visualize the log growth rates
    if visualize:
        visualize_function(times, growth_rates_data, name, save=save)

    # Combine the time row and the average log growth rates into a new array
    result = np.vstack([times, growth_rates_data])

    return result

@validate_input_shape
def growth_rate_of_average(data: np.ndarray, visualize: bool = True, save: bool = False) -> np.ndarray:
    """
    Calculate the cumulative log growth rates for the average of all instances.

    :param data: The input data array of shape (1 + number of instances, number of time steps).
    :type data: np.ndarray
    :param visualize: Whether to visualize the growth rates.
    :type visualize: bool
    :param save: Whether to save the plot as a PNG file.
    :type save: bool
    :return: The cumulative log growth rates for the average of all instances.
    :rtype: np.ndarray
    """
    average_values = average(data, visualize=False)
    growth_rate_values = growth_rates(average_values, visualize=visualize, name="growth rate of average", save=save)

    return growth_rate_values

@validate_input_shape
def growth_rate_of_average_per_time(data: np.ndarray) -> np.ndarray:
    """
    Calculate the cumulative log growth rates for the average of all instances per time.

    :param data: The input data array of shape (1 + number of instances, number of time steps).
    :type data: np.ndarray
    :return: The cumulative log growth rates for the average of all instances per time.
    :rtype: np.ndarray
    """
    growth_rate_values = growth_rate_of_average(data, visualize=False)

    increments = np.diff(growth_rate_values[1, :])

    average_increment = np.mean(increments)

    # divide by the time step
    average_increment = average_increment / (data[0, 1] - data[0, 0])

    return average_increment

@validate_input_shape
def average_growth_rate(data: np.ndarray, visualize: bool = True, save: bool = False) -> np.ndarray:
    """
    Calculate the average of the cumulative log growth rates for all instances.

    :param data: The input data array of shape (1 + number of instances, number of time steps).
    :type data: np.ndarray
    :param visualize: Whether to visualize the average growth rates.
    :type visualize: bool
    :param save: Whether to save the plot as a PNG file.
    :type save: bool
    :return: The average of the cumulative log growth rates for all instances.
    :rtype: np.ndarray
    """
    growth_rate_values = growth_rates(data, visualize=False)
    average_growth_rate_values = average(growth_rate_values, visualize=visualize, name="average growth rate", save=save)

    return average_growth_rate_values

@validate_input_shape
def mu_sigma(data: np.ndarray):
    """
    Calculate the average and standard deviation of the dataset.

    :param data: The input data array of shape (1 + number of instances, number of time steps).
    :type data: np.ndarray
    :return: The average and standard deviation of the dataset.
    :rtype: tuple
    """
    instances = data[1:]

    # Calculate the average and standard deviation of the dataset
    average = np.mean(instances)
    std_dev = np.std(instances)

    timestep = data[0, 1] - data[0, 0]
    average = average / timestep
    std_dev = std_dev / np.sqrt(timestep)

    return average, std_dev

@validate_input_shape
def average_growth_rate_vs_gbm(data: np.ndarray, save: bool = False) -> np.ndarray:
    """
    Calculate the average of the cumulative log growth rates for all instances and compare with the graph
    mu*t - 0.5*sigma^2*t

    :param data: The input data array of shape (1 + number of instances, number of time steps).
    :type data: np.ndarray
    :param save: Whether to save the plot as a PNG file.
    :type save: bool
    :return: The average of the cumulative log growth rates for all instances.
    :rtype: np.ndarray
    """

    average_growth_rate_values = average_growth_rate(data, visualize=False, save=save)
    times = average_growth_rate_values[0, :]
    data_increments = relative_increments(data, visualize=False)
    mu, sigma = mu_sigma(data_increments)
    print(mu, sigma)
    gbm = mu*times - 0.5*sigma**2*times
    plt.figure(figsize=(10, 6))
    plt.plot(times, average_growth_rate_values[1, :], label="Average Growth Rate", lw=1)
    plt.plot(times, gbm, label="GBM", lw=1)
    plt.title("Comparison of Average Growth Rate and GBM")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig('comparison_of_averages_vs_gbm.png')
    plt.show()

    return average_growth_rate_values

@validate_input_shape
def compare_averages(data: np.ndarray, save: bool = False) -> np.ndarray:
    """
    Compare and plot average growth rate and growth rate of average.

    :param data: The input data array of shape (1 + number of instances, number of time steps).
    :type data: np.ndarray
    :param save: Whether to save the plot as a PNG file.
    :type save: bool
    :return: None
    :rtype: None
    """
    average_growth_rate_values = average_growth_rate(data, visualize=False, save=save)
    growth_rate_of_average_values = growth_rate_of_average(data, visualize=False, save=save)

    # Visualize the comparison of average growth rate and growth rate of average
    times = average_growth_rate_values[0, :]
    plt.figure(figsize=(10, 6))
    plt.plot(times, average_growth_rate_values[1, :], label="Average Growth Rate", lw=1)
    plt.plot(times, growth_rate_of_average_values[1, :], label="Growth Rate of Average", lw=1)
    plt.title("Comparison of Average Growth Rate and Growth Rate of Average")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig('comparison_of_averages.png')
    plt.show()

    return None

def demonstrate_alpha_stable_self_similarity(alpha, c, n_points, n_simulations):
    """
    Demonstrate the self-similarity of an α-stable process.

    :param alpha: Stability parameter (0 < alpha <= 2).
    :type alpha: float
    :param c: Scaling factor for time.
    :type c: float
    :param n_points: Number of points in each process realization.
    :type n_points: int
    :param n_simulations: Number of process realizations to generate.
    :type n_simulations: int
    :return: Figure and axes objects for the plot.
    :rtype: tuple
    """
    # Generate time points
    t = np.linspace(0, 1, n_points)
    ct = c * t

    # Generate α-stable increments
    increments = levy_stable.rvs(alpha, 0, size=(n_simulations, n_points-1))

    # Generate processes
    X_t = np.cumsum(increments, axis=1)
    X_ct = np.cumsum(c**(1/alpha) * increments, axis=1)

    # Prepend zeros to align with time points
    X_t = np.hstack((np.zeros((n_simulations, 1)), X_t))
    X_ct = np.hstack((np.zeros((n_simulations, 1)), X_ct))

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot sample paths
    for i in range(min(5, n_simulations)):
        ax1.plot(t, X_t[i], label=f'X(t) - Sim {i+1}', alpha=0.7)
        ax1.plot(ct, X_ct[i], label=f'X(ct) - Sim {i+1}', linestyle='--', alpha=0.7)
    ax1.set_title(f'Sample Paths of α-Stable Process (α={alpha}, c={c})')
    ax1.set_xlabel('t')
    ax1.set_ylabel('X(t)')
    ax1.legend()

    # Plot distributions
    ax2.hist(X_t[:, -1], bins=30, density=True, alpha=0.5, label='X(1)')
    ax2.hist(X_ct[:, -1]/c**(1/alpha), bins=30, density=True, alpha=0.5, label='X(c)/c^(1/α)')
    ax2.set_title('Distribution Comparison at t=1 and t=c')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    ax2.legend()

    plt.tight_layout()

    fig.show()

    return fig, (ax1, ax2)

@validate_input_shape
def relative_increments(data: np.ndarray, visualize: bool = True) -> np.ndarray:
    """
    Calculate the relative increments for each time step based on the input data.

    :param data: The input data array of shape (1 + number of instances, number of time steps).
    :type data: np.ndarray
    :param visualize: Whether to visualize the relative increments.
    :type visualize: bool
    :return: The relative increments for each time step.
    :rtype: np.ndarray
    """
    # Separate time row and instances data
    times = data[0, 1:]
    instances = data[1:]

    # Calculate the relative increments for each time step
    relative_increments = (instances[:, 1:] - instances[:, :-1])/ instances[:, :-1]

    # Visualize the relative increments
    if visualize:
        visualize_function(times, relative_increments, "relative increments")

    # Combine the time row and the relative increments into a new array
    result = np.vstack([times, relative_increments])

    return result

# Demonstration
if __name__ == "__main__":
    alpha = 1.5  # Stability parameter
    c = 2.0      # Scaling factor
    n_points = 1000
    n_simulations = 1000

    fig, axes = demonstrate_alpha_stable_self_similarity(alpha, c, n_points, n_simulations)

    # Additional statistical comparison
    X_1 = levy_stable.rvs(alpha, 0, size=n_simulations)
    X_c = c**(1/alpha) * levy_stable.rvs(alpha, 0, size=n_simulations)

    print(f"Mean of X(1): {np.mean(X_1):.4f}")
    print(f"Mean of X(c)/c^(1/α): {np.mean(X_c/c**(1/alpha)):.4f}")
    print(f"Std of X(1): {np.std(X_1):.4f}")
    print(f"Std of X(c)/c^(1/α): {np.std(X_c/c**(1/alpha)):.4f}")

    plt.show()

def generate_levy_process(T=10, N=10, alpha=1.5, beta=0, loc=0, scale=1):
    """
    A Function to generate a Lévy process with the given parameters.

    :param T: Time horizon.
    :type T: float
    :param N: Number of time steps.
    :type N: int
    :param alpha: Stability parameter (0 < alpha <= 2).
    :return: The generated Lévy process dataset.
    :rtype: np.ndarray
    """
    dt = T / N
    dL = levy_stable.rvs(alpha=alpha, beta=beta, size=N, scale=scale, loc=loc) * dt ** (1 / alpha)
    return np.cumsum(dL)


def compare_distributions(alpha, T=1, N=1000, scales=[1, 10, 100], num_realizations=1000):
    """
    Compare the scaled distributions of end values for Lévy processes with different scales to check self-similarity.

    :param alpha: Stability parameter (0 < alpha <= 2).
    :type alpha: float
    :param T: Time horizon.
    :type T: float
    :param N: Number of time steps.
    :type N: int
    :param scales: List of scales to compare.
    :type scales: list
    :param num_realizations:
    :type num_realizations: int
    :return: None
    :rtype: None
    """
    plt.figure(figsize=(12, 6))

    for scale in scales:
        end_values = []
        for _ in range(num_realizations):
            # Generate process for each realization
            process = generate_levy_process(T * scale, N, alpha)

            # Store the end value of each process
            end_values.append(process[-1])

        # Scale the end values
        scaled_values = np.array(end_values) / scale ** (1 / alpha)

        plt.hist(scaled_values, bins=50, density=True, alpha=0.5, label=f'Scale: {scale}x')

    plt.title(f'Lévy Process (α={alpha}) - Scaled Distributions of End Values')
    plt.xlabel('Scaled Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def create_A_from_correlation_and_variances(correlation_matrix, variances):
    """
    Create matrix A from a given correlation matrix and variance vector.

    :param correlation_matrix: Desired correlation matrix R.
    :type correlation_matrix: np.ndarray
    :param variances: Desired variances for each process.
    :type variances: np.ndarray
    :return: The matrix used to create the multivariate Gaussian process.
    :rtype: np.ndarray
    """
    # Check if correlation matrix is positive definite (all eigenvalues are positive)
    if not np.all(np.linalg.eigvals(correlation_matrix) > 0):
        raise ValueError("Correlation matrix must be positive definite.")

    # Step 1: Perform Cholesky decomposition on the correlation matrix
    L = np.linalg.cholesky(correlation_matrix)

    # Step 2: Create the diagonal matrix of standard deviations
    D = np.diag(np.sqrt(variances))

    # Step 3: Scale L by the standard deviations to obtain A
    A = np.dot(D, L)

    return A

def simulate_independent_gaussian_processes(n, size, mean=0, std=1):
    """
    Simulate n independent Gaussian processes.

    :param n: Number of processes.
    :type n: int
    :param size: Number of time steps.
    :type size: int
    :param mean: Mean of the Gaussian distribution.
    :type mean: float
    :param std: Standard deviation of the Gaussian distribution.
    :type std: float
    :return: Matrix of independent Gaussian processes.
    :rtype: np.ndarray
    """
    return np.random.normal(mean, std, size=(n, size))

def create_multivariate_gaussian_process(A, independent_processes):
    """
    Create a multivariate Gaussian process by linearly combining independent Gaussian processes.

    :param A: Matrix to combine the independent processes.
    :type A: np.ndarray
    :param independent_processes: Matrix of independent Gaussian processes.
    :type independent_processes: np.ndarray
    :return: The multivariate Gaussian process.
    :rtype: np.ndarray
    """
    return np.dot(A, independent_processes)

def random_variable_from_pdf(pdf, x, num_samples, t=1):
    """
    Generate random variable samples from a given probability density function (pdf).

    :param pdf: Probability density function.
    :type pdf: sympy.core.add.Add
    :param x: Symbolic variable for the random variable.
    :type x: sympy.core.symbol.Symbol
    :param num_samples: Number of random variable samples to generate.
    :type num_samples: int
    :param t: Time parameter for the PDF (default is 1).
    :type t: float
    :return: Random variable samples.
    :rtype: np.ndarray
    """
    # Convert sympy expression to numpy function
    pdf_func = sp.lambdify((x, sp.symbols('t')), pdf, "numpy")

    # Create a grid for x
    x_vals = np.linspace(-10, 10, 1000)  # Adjust range as needed

    # Compute the CDF numerically
    cdf_vals = np.array([integrate.quad(lambda xi: pdf_func(xi, t), -np.inf, x)[0] for x in x_vals])

    # Normalize CDF
    cdf_vals = (cdf_vals - cdf_vals.min()) / (cdf_vals.max() - cdf_vals.min())

    # Inverse CDF (interpolation)
    inverse_cdf = interp1d(cdf_vals, x_vals, bounds_error=False, fill_value="extrapolate")

    # Generate uniform random numbers and apply inverse CDF
    uniform_random = np.random.uniform(0, 1, num_samples)
    random_samples = inverse_cdf(uniform_random)

    return random_samples


if __name__=="__main__":
    # Example correlation matrix and variances
    correlation_matrix = np.array([
        [1.0, 0.8, 0.5],
        [0.8, 1.0, 0.3],
        [0.5, 0.3, 1.0]
    ])

    variances = np.array([1.0, 2.0, 0.5])

    # Create matrix A
    A = create_A_from_correlation_and_variances(correlation_matrix, variances)

    print("Matrix A:\n", A)


def solve_fokker_planck_numerically(mu_func, sigma_func, P0_func, x_range, t_range, Nx, Nt, boundary_conditions):
    """
    Solve the Fokker-Planck equation numerically using finite differences and visualize the result in 3D.

    :param mu_func: Function for drift term, mu(x, t).
    :type mu_func: function
    :param sigma_func: Function for diffusion term, sigma(x, t).
    :type sigma_func: function
    :param P0_func: Function for the initial condition P(x, 0).
    :type P0_func: function
    :param x_range: Tuple (x_min, x_max) defining the spatial domain.
    :type x_range: Tuple[float, float]
    :param t_range: Tuple (t_min, t_max) defining the time domain.
    :type t_range: Tuple[float, float]
    :param Nx: Number of spatial grid points.
    :type Nx: int
    :param Nt: Number of time steps.
    :type Nt: int
    :param boundary_conditions: Tuple specifying Dirichlet boundary conditions as (P(x_min, t), P(x_max, t)).
    :type boundary_conditions: Tuple[float, float]
    :return: Arrays for the solution P(x, t) at each time step and the spatial grid points.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    # Discretize space and time
    x_min, x_max = x_range
    t_min, t_max = t_range
    dx = (x_max - x_min) / (Nx - 1)  # Spatial step
    dt = (t_max - t_min) / (Nt - 1)  # Time step (fixed for entire range of t)

    x = np.linspace(x_min, x_max, Nx)
    t = np.linspace(t_min, t_max, Nt)

    # Initialize solution array
    P = np.zeros((Nt, Nx))

    # Set initial condition P(x, 0)
    P[0, :] = P0_func(x)

    # Boundary conditions
    P[:, 0] = boundary_conditions[0]  # P(x_min, t)
    P[:, -1] = boundary_conditions[1]  # P(x_max, t)

    # Finite difference loop
    for n in range(0, Nt - 1):
        for i in range(1, Nx - 1):  # Skip the boundary points
            mu = mu_func(x[i], t[n])
            sigma = sigma_func(x[i], t[n])

            # Central difference for second derivative (diffusion term)
            d2P_dx2 = (P[n, i + 1] - 2 * P[n, i] + P[n, i - 1]) / dx ** 2

            # Central difference for first derivative (drift term)
            dP_dx = (P[n, i + 1] - P[n, i - 1]) / (2 * dx)

            # Update P using finite differences
            P[n + 1, i] = P[n, i] + dt * (-mu * dP_dx + 0.5 * sigma ** 2 * d2P_dx2)

    # Plot the result in 3D
    X, T = np.meshgrid(x, t)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, P, cmap='viridis')

    ax.set_xlabel('Space (x)')
    ax.set_ylabel('Time (t)')
    ax.set_zlabel('P(x, t)')
    ax.set_title('Numerical Solution of Fokker-Planck Equation')

    plt.show()

    return x, t, P


if __name__ == "__main__":
    # Example usage with arbitrary drift and diffusion terms

    # Define drift term: mu(x,t) = x - t
    mu_func = lambda x, t: x - t

    # Define diffusion term: sigma(x,t) = x**2 + t
    sigma_func = lambda x, t: x ** 2 + t

    # Define initial condition: P(x, 0) = exp(-x**2)
    P0_func = lambda x: np.exp(-x ** 2)

    # Define boundary conditions: P(x_min, t) = 0, P(x_max, t) = 0
    boundary_conditions = (0, 0)  # Dirichlet boundary conditions at both ends

    # Spatial and time ranges
    x_range = (-5, 5)  # x from -5 to 5
    t_range = (0, 1)  # t from 0 to 1

    # Number of spatial grid points and time steps
    Nx = 100
    Nt = 1000

    # Solve the Fokker-Planck equation numerically and visualize
    x, t, P = solve_fokker_planck_numerically(mu_func, sigma_func, P0_func, x_range, t_range, Nx, Nt, boundary_conditions)
