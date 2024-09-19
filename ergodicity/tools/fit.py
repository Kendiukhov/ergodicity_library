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
from typing import List, Any, Type, Callable, Dict, Tuple
import warnings
from scipy import stats
from ergodicity.tools.helper import separate
from scipy.optimize import minimize, differential_evolution
from scipy.stats import levy_stable

def levy_stable_fit(data):
    """
    Estimate the parameters of a stable distribution using the Empirical Characteristic Function method.

    :param data: The data to fit the stable distribution to
    :type data: array-like
    :return: The estimated parameters of the stable distribution
    :rtype: dict
    """

    # Empirical Characteristic Function (ECF)
    def ecf(t):
        return np.mean(np.exp(1j * t * data))

    # Theoretical Characteristic Function (TCF) of stable distribution
    def tcf(t, alpha, beta, gamma, delta):
        return np.exp(1j * delta * t - gamma ** alpha * np.abs(t) ** alpha * (
                    1 - 1j * beta * np.sign(t) * np.tan(np.pi * alpha / 2)))

    # Objective function to minimize
    def objective(params):
        alpha, beta, gamma, delta = params
        t_values = np.linspace(-10, 10, 100)
        ecf_values = np.array([ecf(t) for t in t_values])
        tcf_values = np.array([tcf(t, alpha, beta, gamma, delta) for t in t_values])
        error = np.sum(np.abs(ecf_values - tcf_values) ** 2)
        return error

    # Initial guesses
    alpha0 = 1.5
    beta0 = 0.0
    gamma0 = np.std(data)
    delta0 = np.mean(data)
    print(f'Initial guesses: alpha={alpha0}, beta={beta0}, gamma={gamma0}, delta={delta0}')

    initial_guess = [alpha0, beta0, gamma0, delta0]

    # Bounds
    bounds = [(0.5, 2), (-1, 1), (1e-3, None), (None, None)]

    # Minimization
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

    if result.success:
        alpha_est, beta_est, gamma_est, delta_est = result.x
        return {
            'alpha': alpha_est,
            'beta': beta_est,
            'scale': gamma_est,
            'loc': delta_est
        }
    else:
        print("Optimization failed:", result.message)
        return None

# Example Usage
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import levy_stable

    # Set random seed for reproducibility
    np.random.seed(0)

    # Define true parameters for the Lévy stable distribution
    true_alpha = 1.8  # Stability parameter
    true_beta = -0.2  # Skewness parameter
    true_loc = 0.1  # Location parameter
    true_scale = 0.6  # Scale parameter

    # Generate sample data
    sample_size = 10000
    data_raw = levy_stable.rvs(
        alpha=true_alpha,
        beta=true_beta,
        loc=true_loc,
        scale=true_scale,
        size=sample_size
    )

    # Extract increments
    data = np.diff(data_raw)

    # Visualize the increments
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=100, density=True, alpha=0.6, color='g', label='Increments Histogram')
    plt.title('Histogram of Increments')
    plt.xlabel('Increment Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Plot empirical cumulative distribution
    plt.figure(figsize=(10, 6))
    sorted_data = np.sort(data)
    ecdf = np.arange(1, sample_size) / sample_size
    plt.plot(sorted_data, ecdf, label='Empirical CDF')

    # Overlay the true CDF
    true_cdf = levy_stable.cdf(sorted_data, true_alpha, true_beta, loc=true_loc, scale=true_scale)
    plt.plot(sorted_data, true_cdf, 'r--', label='True CDF')
    plt.title('Empirical vs. True CDF')
    plt.xlabel('Increment Value')
    plt.ylabel('CDF')
    plt.legend()
    plt.show()

    # Fit the Lévy stable distribution to the increments
    levy_params = levy_stable_fit(
        data,
        alpha_bounds=(0.5, 2),
        beta_bounds=(-1, 1),
        scale_bounds=(1e-3, None),
        fix_loc=True,
        verbose=True  # Enable detailed logging
    )

    # Display the fitted parameters
    print("\nFitted Lévy stable parameters:")
    print(f"Alpha (stability): {levy_params['alpha']}")
    print(f"Beta (skewness): {levy_params['beta']}")
    print(f"Loc (location): {levy_params['loc']}")
    print(f"Scale (scale): {levy_params['scale']}")
    print(f"Optimization Success: {levy_params['success']}")
    print(f"Message: {levy_params['message']}")

    # Extract fitted parameters
    fitted_alpha = levy_params['alpha']
    fitted_beta = levy_params['beta']
    fitted_loc = levy_params['loc']
    fitted_scale = levy_params['scale']

    # Define a range for plotting the fitted PDF
    x = np.linspace(min(data), max(data), 1000)
    fitted_pdf = levy_stable.pdf(x, fitted_alpha, fitted_beta, loc=fitted_loc, scale=fitted_scale)

    # Plot the histogram and fitted PDF
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=100, density=True, alpha=0.6, color='g', label='Increments Histogram')
    plt.plot(x, fitted_pdf, 'r-', lw=2, label='Fitted Lévy Stable PDF')
    plt.title('Histogram of Increments with Fitted Lévy Stable PDF')
    plt.xlabel('Increment Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Overlay the true PDF for comparison
    true_pdf = levy_stable.pdf(x, true_alpha, true_beta, loc=true_loc, scale=true_scale)
    plt.figure(figsize=(10, 6))
    plt.plot(x, fitted_pdf, 'r-', lw=2, label='Fitted Lévy Stable PDF')
    plt.plot(x, true_pdf, 'b--', lw=2, label='True Lévy Stable PDF')
    plt.title('Fitted vs. True Lévy Stable PDF')
    plt.xlabel('Increment Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Q-Q plot against the fitted distribution
    from scipy.stats import probplot

    fitted_dist = stats.levy_stable(alpha=fitted_alpha, beta=fitted_beta, loc=fitted_loc, scale=fitted_scale)
    probplot(data, dist=fitted_dist, plot=plt)
    plt.title('Q-Q Plot Against Fitted Lévy Stable Distribution')
    plt.show()


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
        try:
            if name == 'levy_stable':
                param_dict = levy_stable_fit(data)
                if param_dict is None:
                    print(f"Failed to fit Levy stable distribution")
                    continue
                params = (param_dict['alpha'], param_dict['beta'], param_dict['loc'], param_dict['scale'])
            else:
                params = distribution.fit(data)

            print(f"Fitted parameters for {name}: {params}")  # Debugging line

            # Check if all parameters are numerical
            if not all(isinstance(p, (int, float)) for p in params):
                print(f"Warning: Non-numerical parameters for {name}: {params}")
                continue

            fitted_dist = distribution(*params)

            # Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.kstest(data, fitted_dist.cdf)

            # If ks_statistic or p_value is an array, take the first element
            ks_statistic = ks_statistic[0] if isinstance(ks_statistic, np.ndarray) else ks_statistic
            p_value = p_value[0] if isinstance(p_value, np.ndarray) else p_value

            print(f"KS test results for {name}: statistic={ks_statistic}, p-value={p_value}")  # Debug line

            # Log-likelihood
            log_likelihood = np.sum(fitted_dist.logpdf(data))

            # Number of parameters
            n_params = len(params)

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
        except Exception as e:
            print(f"Error fitting {name} distribution: {str(e)}")

    return fitted_dists


def plot_fitted_distributions(data, fitted_dists):
    """
    Plot the histogram of the data with fitted distributions.

    :param data: The data to plot the histogram of
    :type data: array or list of arrays
    :param fitted_dists: A dictionary containing the fitted distributions and their parameters
    :type fitted_dists: dict
    :return: None
    :rtype: None
    """
    plt.figure(figsize=(12, 6))

    # Check if data is a list of datasets or a single dataset
    if isinstance(data[0], (list, np.ndarray)):
        # Multiple datasets
        for dataset in data:
            plt.hist(dataset, bins=50, density=True, alpha=0.1, color='skyblue')
        # Compute global min and max across all datasets
        global_min = min(np.min(dataset) for dataset in data)
        global_max = max(np.max(dataset) for dataset in data)
    else:
        # Single dataset
        plt.hist(data, bins=50, density=True, alpha=0.7, color='skyblue')
        global_min = np.min(data)
        global_max = np.max(data)

    x = np.linspace(global_min, global_max, 1000)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(fitted_dists)))

    for (name, dist_info), color in zip(fitted_dists.items(), colors):
        try:
            y = dist_info['distribution'].pdf(x)
            plt.plot(x, y, label=name, color=color)
            print(f"Successfully plotted {name} distribution")
        except Exception as e:
            print(f"Error plotting {name} distribution: {str(e)}")

    plt.title('Data Histogram with Fitted Distributions')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Print debug information
    print("\nDebug Information:")
    for name, dist_info in fitted_dists.items():
        print(f"{name}: {dist_info['params']}")

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

def fit_stochastic_process(
    process_func,
    external_data,
    initial_params,
    param_names,
    bounds=None,
    num_fits=10,
    t=10,
    timestep=0.01
):
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
    :param t: Total simulation time
    :type t: float
    :param timestep: Time step for the simulation
    :type timestep: float
    :return: Optimized parameters and the plot figure
    :rtype: dict, matplotlib.figure.Figure
    """

    def objective(params_to_fit, external_data=external_data):
        # Create kwargs for the process function
        kwargs = {name: value for name, value in zip(param_names, params_to_fit)}
        kwargs.update(initial_params)  # Add any fixed parameters
        kwargs['num_instances'] = 1  # We only need one instance for fitting
        kwargs['t'] = t
        kwargs['timestep'] = timestep

        # Run the simulation
        simulated_output = process_func(**kwargs)
        times_simulated, simulated_data = separate(simulated_output)

        # Ensure that simulated_data and external_data are numpy arrays
        simulated_data = np.array(simulated_data).flatten()
        external_data_flat = np.array(external_data).flatten()

        # Check that they have the same length
        if len(simulated_data) != len(external_data_flat):
            # Optionally, you can interpolate or truncate to match lengths
            min_length = min(len(simulated_data), len(external_data_flat))
            simulated_data = simulated_data[:min_length]
            external_data_flat = external_data_flat[:min_length]
            print("Warning: Truncated data to match lengths.")

        # Calculate the error
        error = np.mean((simulated_data - external_data_flat) ** 2)
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
        result = minimize(
            objective,
            initial_values,
            bounds=bounds,
            method='L-BFGS-B'
        )

        if result.fun < best_error:
            best_error = result.fun
            best_fit = result.x

    # Build kwargs for the final simulation with fitted parameters
    fitted_params = {name: value for name, value in zip(param_names, best_fit)}
    kwargs = {}
    kwargs.update(fitted_params)
    kwargs['num_instances'] = 1
    kwargs['t'] = t
    kwargs['timestep'] = timestep

    # Run the process_func with fitted parameters to get simulated data
    simulated_output = process_func(**kwargs)
    times_simulated, simulated_data = separate(simulated_output)

    # Flatten or squeeze simulated_data
    simulated_data = np.squeeze(simulated_data)

    # Correct the shapes of times_external and external_data
    external_data_flat = np.array(external_data).flatten()  # Ensure external_data is 1D
    times_external = np.linspace(0, t, len(external_data_flat))  # Create a 1D array for times_external

    # Debug: Print shapes
    # print("Shape of times_simulated:", times_simulated.shape)
    # print("Shape of simulated_data:", simulated_data.shape)
    # print("Shape of times_external:", times_external.shape)
    # print("Shape of external_data_flat:", external_data_flat.shape)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot external data once
    ax.plot(times_external, external_data_flat, label='External Data', alpha=0.7)

    # Plot fitted process
    ax.plot(times_simulated, simulated_data, label='Fitted Process', alpha=0.7)

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
        process_func(t=t, timestep=timestep, num_instances=1, **true_params)[1]

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


