"""
fit Submodule

The `fit` submodule provides a set of tools for fitting various stochastic processes and probability distributions to data. It leverages techniques such as maximum likelihood estimation (MLE) and optimization to find the best parameters that describe the observed data. This submodule is particularly useful in research and simulations that involve probabilistic models and stochastic processes.

Key Features:

1. **Lévy Stable Distribution Fitting**:

   - The function `levy_stable_fit` performs maximum likelihood estimation for fitting Lévy stable distributions to data, a family of distributions used in finance, physics, and other fields that involve heavy tails and skewness.

2. **Distribution Fitting and Model Comparison**:

   - `fit_distributions`: Fits various probability distributions (e.g., normal, lognormal, exponential, Lévy stable) to a given dataset and provides goodness-of-fit measures like the Kolmogorov-Smirnov test, AIC (Akaike Information Criterion), and BIC (Bayesian Information Criterion).

   - `print_results`: Summarizes and compares the fitted distributions based on their AIC and BIC values to help identify the best model.

3. **Visualization of Fitted Distributions**:

   - `plot_fitted_distributions`: Visualizes the fitted distributions by overlaying them on the histogram of the data, providing a clear comparison between the data and the fitted models.

4. **Stochastic Process Parameter Fitting**:

   - `fit_stochastic_process`: Fits the parameters of a given stochastic process to observed data using optimization techniques. This is particularly useful when simulating stochastic processes like Ornstein-Uhlenbeck or Brownian motion and comparing them to real-world or simulated data.

5. **Fitting Success Testing**:

   - `test_fitting_success`: Generates synthetic data using known parameters and tests the success of fitting the process to the data, evaluating the accuracy of the fitted parameters across multiple tests.

6. **Distribution Comparison**:

   - `compare_distributions`: Generates data from a specified probability distribution and compares the fitted parameters with the original generating parameters. This function is helpful for understanding the reliability of different fitting techniques.

Typical Use Cases:

- **Research and Data Analysis**:

  Provides tools for fitting complex stochastic models to experimental or observed data, enabling researchers to validate theoretical models.

- **Simulation Studies**:

  Enables parameter fitting for stochastic simulations, especially when simulating processes like Lévy flights, Brownian motion, or other random walks.

- **Model Selection**:

  Facilitates the selection of the best probabilistic model using AIC, BIC, and goodness-of-fit tests.

Example Usage:

data = np.random.normal(0, 1, 1000)  # Example data

# Fit various distributions

fitted_dists = fit_distributions(data)

# Print the results

print_results(fitted_dists)

# Plot the fitted distributions

plot_fitted_distributions(data, fitted_dists)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from typing import List, Any, Type, Callable, Dict, Tuplemak

def levy_stable_fit(data):
    """
    Fit Lévy stable distribution to data using maximum likelihood estimation.

    :param data: The data to fit the Lévy stable distribution to
    :type data: array
    :return: The fitted parameters (alpha, beta, loc, scale) for the Lévy stable distribution
    :rtype: array
    """

    def negative_log_likelihood(params, data):
        alpha, beta, loc, scale = params
        return -np.sum(stats.levy_stable.logpdf(data, alpha, beta, loc, scale))

    # Initial guess
    initial_guess = [1.5, 0, np.mean(data), np.std(data)]

    # Bounds for parameters
    bounds = [(0.01, 2), (-1, 1), (None, None), (1e-6, None)]

    # Minimize negative log-likelihood
    result = minimize(negative_log_likelihood, initial_guess, args=(data,), bounds=bounds)

    return result.x

def fit_distributions(data):
    """
    Fit various probability distributions to the given data, including Lévy stable.

    :param data: The data to fit the distributions to
    :type data: array
    :return: A dictionary containing the fitted distributions, parameters, and goodness-of-fit measures
    :rtype: dict
    """
    distributions = {
        'normal': stats.norm,
        'lognormal': stats.lognorm,
        'exponential': stats.expon,
        'gamma': stats.gamma,
        't': stats.t,
        'cauchy': stats.cauchy,
        'levy_stable': stats.levy_stable
    }

    fitted_dists = {}

    for name, distribution in distributions.items():
        if name == 'levy_stable':
            params = levy_stable_fit(data)
            fitted_dist = distribution(*params)
        else:
            params = distribution.fit(data)
            fitted_dist = distribution(*params)

        # Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.kstest(data, fitted_dist.cdf)

        # Log-likelihood
        log_likelihood = np.sum(fitted_dist.logpdf(data))

        # Number of parameters
        if name in ['normal', 'exponential', 'cauchy']:
            n_params = 2
        elif name in ['lognormal', 'gamma', 't']:
            n_params = 3
        elif name == 'levy_stable':
            n_params = 4

        # AIC and BIC
        n = len(data)
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n) - 2 * log_likelihood

        fitted_dists[name] = {
            'distribution': fitted_dist,
            'params': params,
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'aic': aic,
            'bic': bic
        }

    return fitted_dists

def plot_fitted_distributions(data, fitted_dists):
    """
    Plot the histogram of the data with fitted distributions.

    :param data: The data to plot the histogram of
    :type data: array
    :param fitted_dists: A dictionary containing the fitted distributions and their parameters
    :type fitted_dists: dict
    :return: None
    :rtype: None
    """
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=50, density=True, alpha=0.7, color='skyblue')

    x = np.linspace(min(data), max(data), 1000)

    for name, dist_info in fitted_dists.items():
        plt.plot(x, dist_info['distribution'].pdf(x), label=name)

    plt.title('Data Histogram with Fitted Distributions')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def print_results(fitted_dists):
    """
    Print the results of distribution fitting, including goodness-of-fit and information criteria.

    :param fitted_dists: A dictionary containing the fitted distributions and their information
    :type fitted_dists: dict
    :return: None
    :rtype: None
    """
    print("\nDistribution Fitting Results:")
    print("-----------------------------")
    for name, info in fitted_dists.items():
        print(f"\n{name.capitalize()} Distribution:")
        if name == 'levy_stable':
            print(f"Parameters: alpha={info['params'][0]:.4f}, beta={info['params'][1]:.4f}, "
                  f"loc={info['params'][2]:.4f}, scale={info['params'][3]:.4f}")
        else:
            print(f"Parameters: {info['params']}")
        print(f"Kolmogorov-Smirnov Statistic: {info['ks_statistic']:.4f}")
        print(f"P-value: {info['p_value']:.4f}")
        print(f"AIC: {info['aic']:.2f}")
        print(f"BIC: {info['bic']:.2f}")

    # Find best model according to AIC and BIC
    best_aic = min(fitted_dists.items(), key=lambda x: x[1]['aic'])
    best_bic = min(fitted_dists.items(), key=lambda x: x[1]['bic'])

    print("\nModel Selection:")
    print(f"Best model according to AIC: {best_aic[0]} (AIC: {best_aic[1]['aic']:.2f})")
    print(f"Best model according to BIC: {best_bic[0]} (BIC: {best_bic[1]['bic']:.2f})")

def fit_stochastic_process(process_func, external_data, initial_params, param_names, bounds=None, num_fits=10):
    """
    Fit parameters for a given stochastic process function to external data.

    :param process_func: The stochastic process function decorated with @custom_simulate
    :type process_func: callable
    :param external_data: External observed data points
    :type external_data: array
    :param initial_params: Initial guess for the parameters
    :type initial_params: dict
    :param param_names: Names of the parameters to fit
    :type param_names: list
    :param bounds: Bounds for the parameters [(low1, high1), (low2, high2), ...]
    :type bounds: list of tuples, optional
    :param num_fits: Number of fitting attempts with different initial conditions
    :type num_fits: int
    :return: Optimized parameters
    :rtype: dict
    """

    def objective(params_to_fit):
        # Create kwargs for the process function
        kwargs = {name: value for name, value in zip(param_names, params_to_fit)}
        kwargs.update(initial_params)  # Add any fixed parameters
        kwargs['num_instances'] = 1  # We only need one instance for fitting
        kwargs['t'] = 10
        kwargs['timestep'] = 0.01
        kwargs['initial_value'] = external_data[0]  # Set initial value to match external data

        # Run the simulation
        simulated_data = process_func(**kwargs)

        # Calculate the error (using only the simulated values, not the time points)
        error = np.mean((simulated_data[1] - external_data[1]) ** 2)
        return error

    best_fit = None
    best_error = np.inf

    for _ in range(num_fits):
        # Randomize initial values within bounds
        if bounds:
            initial_values = [np.random.uniform(low, high) for (low, high) in bounds]
        else:
            initial_values = [initial_params[name] for name in param_names]

        # Optimize
        result = minimize(objective, initial_values, bounds=bounds, method='L-BFGS-B')

        if result.fun < best_error:
            best_error = result.fun
            best_fit = result.x

    fitted_params = {name: value for name, value in zip(param_names, best_fit)}

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    t = np.linspace(0, params['t'], len(external_data))
    ax.plot(t, external_data, label='External Data', alpha=0.7)

    simulated_data = process_func(**fitted_params, num_instances=1, initial_value=external_data[0])[1]
    ax.plot(t, simulated_data, label='Fitted Process', alpha=0.7)

    ax.set_title(f'Stochastic Process: External Data vs Fitted\nFitted Parameters: {fitted_params}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

    return fitted_params, fig

def test_fitting_success(process_func: Callable,
                         true_params: Dict[str, float],
                         param_names: List[str],
                         bounds: List[Tuple[float, float]] = None,
                         num_tests: int = 10,
                         t: float = 10,
                         timestep: float = 0.01,
                         initial_value: float = 1.0) -> Dict[str, List[float]]:
    """
    Test the success of fitting by generating data with known parameters and then fitting it.

    :param process_func: The stochastic process function decorated with @custom_simulate
    :type process_func: callable
    :param true_params: The true parameters of the process to generate data
    :type true_params: dict
    :param param_names: Names of the parameters to fit
    :type param_names: list
    :param bounds: Bounds for the parameters [(low1, high1), (low2, high2), ...]
    :type bounds: list of tuples, optional
    :param num_tests: Number of tests to run
    :type num_tests: int
    :param t: Total time for each simulation
    :type t: float
    :param timestep: Time step for simulations
    :type timestep: float
    :param initial_value: Initial value for the process
    :type initial_value: float
    :return: Dictionary with lists of true and fitted values for each parameter
    :rtype: dict
    """
    results = {param: {'true': [], 'fitted': []} for param in param_names}

    for _ in range(num_tests):
        # Generate data using true parameters
        generated_data = \
        process_func(t=t, timestep=timestep, num_instances=1, initial_value=initial_value, **true_params)[1]

        # Fit the process to the generated data
        initial_guess = {param: np.random.uniform(bounds[i][0], bounds[i][1]) for i, param in enumerate(param_names)}
        fitted_params, _ = fit_stochastic_process(process_func, generated_data, initial_guess, param_names, bounds)

        # Store results
        for param in param_names:
            results[param]['true'].append(true_params[param])
            results[param]['fitted'].append(fitted_params[param])

    # Plot results
    fig, axs = plt.subplots(len(param_names), 1, figsize=(10, 5 * len(param_names)), squeeze=False)
    for i, param in enumerate(param_names):
        axs[i, 0].scatter(results[param]['true'], results[param]['fitted'], alpha=0.5)
        axs[i, 0].plot([min(results[param]['true']), max(results[param]['true'])],
                       [min(results[param]['true']), max(results[param]['true'])], 'r--')
        axs[i, 0].set_xlabel(f'True {param}')
        axs[i, 0].set_ylabel(f'Fitted {param}')
        axs[i, 0].set_title(f'{param}: True vs Fitted')

    plt.tight_layout()
    plt.show()

    return results

def compare_distributions(dist_name, params, size=1000):
    """
    Generates a dataset from a specified probability distribution, fits the dataset back to the distribution,
    and compares the fitted distribution with the generating distribution.

    :param dist_name: Name of the probability distribution (e.g., 'norm', 'expon', 'gamma')
    :type dist_name: str
    :param params: Parameters for the generating distribution (e.g., (mean, std) for 'norm')
    :type params: tuple
    :param size: Number of samples to generate
    :type size: int
    :return: None
    :rtype: None
    """
    # Generate data from the specified distribution
    distribution = getattr(stats, dist_name)
    generated_data = distribution.rvs(*params, size=size)

    # Fit the data to the specified distribution
    fitted_params = distribution.fit(generated_data)

    # Compare the parameters
    print(f"Generating parameters: {params}")
    print(f"Fitted parameters: {fitted_params}")

    # Plot the generated data and the fitted distribution
    plt.figure(figsize=(10, 6))

    # Plot histogram of the generated data
    plt.hist(generated_data, bins=30, density=True, alpha=0.6, color='g', label='Generated Data')

    # Plot the generating distribution
    x = np.linspace(min(generated_data), max(generated_data), 100)
    plt.plot(x, distribution.pdf(x, *params), 'r-', label='Generating Distribution')

    # Plot the fitted distribution
    plt.plot(x, distribution.pdf(x, *fitted_params), 'b--', label='Fitted Distribution')

    plt.title(f'Comparison of Generating and Fitted Distributions ({dist_name})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Generate some external data (replace this with your actual external data)
    np.random.seed(42)
    t = np.linspace(0, params['t'], int(params['t'] / params['timestep']))
    external_data = np.cumsum(np.random.normal(0, 0.1, len(t))) + np.sin(t)

    # Fit the process
    initial_guess = {'theta': 0.5, 'sigma': 0.5}
    param_names = ['theta', 'sigma']
    bounds = [(0, 2), (0, 2)]  # bounds for theta and sigma

    fitted_params = fit_stochastic_process(OrnsteinUhlenbeckSimulation, external_data, initial_guess, param_names,
                                           bounds)

    print("Fitted parameters:", fitted_params)


