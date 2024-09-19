"""
evaluate Submodule

The `evaluate` submodule provides tools for assessing the statistical properties and characteristics of stochastic processes. This includes methods for testing ergodicity, self-similarity, fat-tailedness, and more. It offers a variety of functions for time series analysis, focusing on stochastic processes commonly used in finance, physics, and other fields requiring statistical modeling.

Key Features:

1. **Ergodicity Testing**:

   - `test_ergodicity_increments`: Tests ergodicity by comparing time and ensemble averages of a process's increments. It also checks for stationarity and normality of the increments.

2. **Self-Similarity Analysis**:

   - `hurst_exponent`: Estimates the Hurst exponent using the Rescaled Range (R/S) method, a key indicator of self-similarity and long-term memory.

   - `aggregate_variance`: Uses the aggregate variance method to estimate the scaling parameter.

   - `test_self_similarity`: Combines multiple methods to test for self-similarity, including Hurst exponent estimation and scaling parameter analysis.

   - `test_self_similarity_wavelet`: Uses wavelet analysis to estimate the Hurst exponent, offering an alternative method for testing self-similarity.

3. **Fat-Tailedness Testing**:

   - `test_fat_tailedness`: Performs multiple tests to detect fat-tailedness in data, including kurtosis tests, Jarque-Bera tests, and tail index estimation using the Hill estimator. It also generates Q-Q plots for visual analysis.

   - `kurtosis_test`, `jarque_bera_test`, `tail_index_hill`: Individual functions to test for fat tails in distributions.

4. **Wavelet-Based Analysis**:

   - `wavelet_estimation`: Estimates the Hurst exponent using wavelet-based techniques. This is particularly useful for processes with fractal or self-similar properties.

   - `plot_wavelet_analysis`: Provides visual analysis of wavelet decomposition, allowing users to assess the variance of wavelet coefficients across scales.

5. **Visualization**:

   - The submodule includes multiple plotting functions such as `plot_ergodicity_results` and `qqplot_analysis`, which help visualize important statistical properties of time series data.

Applications:

This submodule is highly applicable in fields such as:

- **Financial Modeling**: Understanding stock prices, asset returns, and market dynamics through ergodicity and self-similarity testing.

- **Physics**: Analyzing systems governed by stochastic dynamics, such as diffusion processes.

- **Environmental Science**: Studying long-range dependence in environmental data (e.g., temperature, rainfall).

- **Machine Learning**: Utilizing statistical tools to evaluate the properties of learning algorithms that incorporate random processes.

The `evaluate` submodule serves as a powerful toolkit for researchers and practitioners who need to test key statistical properties of their models or real-world data. By providing a range of methods, it simplifies complex evaluations and enhances the understanding of stochastic processes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from scipy.stats import linregress
import pywt
import scipy.stats as stats
from ergodicity.tools.helper import *
from ergodicity.tools.compute import relative_increments as ri

def test_ergodicity_increments(data, relative_increments=False):
    """
    Test the ergodicity of a stochastic process based on its increments.

    :param data: Array of shape (num_time_steps, num_instances + 1) where the first row is time reprsenting the time steps and subsequent rows are instances of the process
    :type data: numpy.ndarray
    :param dt:vTime step size (1/number of steps in a time unit)
    :type dt: float
    :return: Dictionary containing test results and metrics
    :rtype: dict
    """
    times, process_data = separate(data)
    if not relative_increments:
        increments = np.diff(process_data, axis=1)

    else:
        increments = ri(data)
        times, increments = separate(increments)

    num_instances, num_time_steps = process_data.shape
    dt = len(times) / (num_time_steps - 1)

    results = {}

    # 1. Test stationarity of increments (ADF test)
    adf_results = [adfuller(inc) for inc in increments]
    results['stationarity_p_values'] = [result[1] for result in adf_results]
    results['increments_stationary'] = np.mean(results['stationarity_p_values']) < 0.05

    # 2. Calculate time averages
    if relative_increments:
        process_data = np.log(process_data)
    time_averages = (process_data[:, -1] - process_data[:, 0]) / (num_time_steps - 1)
    time_averages_per_unit = time_averages / dt

    # 3. Calculate ensemble averages of increments
    ensemble_averages = np.mean(increments, axis=0)
    ensemble_average_per_step = np.mean(ensemble_averages)
    ensemble_average_per_unit = ensemble_average_per_step / dt

    results['time_average_per_unit'] = np.mean(time_averages_per_unit)
    results['ensemble_average_per_unit'] = ensemble_average_per_unit

    # 4. Compare time and ensemble averages
    results['average_relative_difference'] = np.abs(
        results['time_average_per_unit'] - results['ensemble_average_per_unit']) / results['ensemble_average_per_unit']

    # 5. Test normality of increments
    _, normality_p_value = stats.normaltest(increments.flatten())
    results['increments_normality_p_value'] = normality_p_value

    return results


def plot_ergodicity_results(data, results):
    """
    Plot visualizations to help assess ergodicity of increments.

    :param data: Array of shape (num_time_steps, num_instances + 1) where the first row is time and subsequent rows are instances of the process
    :type data: numpy.ndarray
    :param results: Dictionary containing test results from test_ergodicity_increments function
    :type results: dict
    :return: None
    :rtype: None
    """
    times = data[0]
    process_data = data[1:]
    increments = np.diff(process_data, axis=1)

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    # Plot 1: Sample paths
    for i in range(min(5, process_data.shape[0])):
        axs[0, 0].plot(times, process_data[i], alpha=0.5)
    axs[0, 0].set_title("Sample Paths")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Value")

    # Plot 2: Increment distribution
    axs[0, 1].hist(increments.flatten(), bins=50, density=True, alpha=0.7)
    axs[0, 1].set_title(f"Increment Distribution\nNormality p-value: {results['increments_normality_p_value']:.3e}")
    axs[0, 1].set_xlabel("Increment Value")
    axs[0, 1].set_ylabel("Density")

    # Plot 3: QQ plot of increments
    stats.probplot(increments.flatten(), dist="norm", plot=axs[1, 0])
    axs[1, 0].set_title("Q-Q Plot of Increments")

    # Plot 4: Autocorrelation of increments
    lag_max = min(100, increments.shape[1] - 1)
    acf = np.mean([np.correlate(inc, inc, mode='full')[lag_max:] for inc in increments], axis=0)
    acf /= acf[0]
    axs[1, 1].plot(acf[:lag_max])
    axs[1, 1].set_title("Mean Autocorrelation of Increments")
    axs[1, 1].set_xlabel("Lag")
    axs[1, 1].set_ylabel("Autocorrelation")

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate some example data (replace this with your actual process data)
    num_instances = 100
    num_time_steps = 1001  # Odd number to have 1000 increments
    dt = 0.01

    # Example: Brownian motion (which has stationary, ergodic increments)
    times = np.linspace(0, (num_time_steps - 1) * dt, num_time_steps)
    increments = np.random.normal(0, np.sqrt(dt), (num_instances, num_time_steps - 1))
    process_data = np.cumsum(increments, axis=1)
    process_data = np.insert(process_data, 0, 0, axis=1)  # Start at 0
    data = np.vstack((times, process_data))

    # Test ergodicity
    results = test_ergodicity_increments(data, dt)

    # Print results
    print("Ergodicity Test Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    # Plot results
    plot_ergodicity_results(data, results)


def detrended_fluctuation_analysis(time_series):
    """
    Estimate the Hurst exponent using Detrended Fluctuation Analysis (DFA).

    :param time_series: The time series to analyze
    :type time_series: array_like
    :return: Estimated Hurst exponent
    :rtype: float
    """
    N = len(time_series)
    # Cumulative sum of deviations from the mean
    Y = np.cumsum(time_series - np.mean(time_series))

    # Define scales (window sizes)
    scales = np.floor(np.logspace(np.log10(10), np.log10(N // 4), num=20)).astype(int)
    scales = np.unique(scales)

    flucts = []
    for scale in scales:
        if scale < 2:
            continue
        # Number of non-overlapping windows
        n_windows = N // scale
        if n_windows < 2:
            continue
        F_nu = []
        for i in range(n_windows):
            segment = Y[i * scale:(i + 1) * scale]
            # Fit a polynomial of order 1 (linear detrending)
            coeffs = np.polyfit(range(scale), segment, 1)
            fit = np.polyval(coeffs, range(scale))
            # Calculate the fluctuation (standard deviation) after detrending
            F_nu.append(np.sqrt(np.mean((segment - fit) ** 2)))
        # Average fluctuation over all segments of this scale
        F = np.mean(F_nu)
        flucts.append((scale, F))

    if len(flucts) < 2:
        return np.nan

    scales, Fs = zip(*flucts)
    log_scales = np.log(scales)
    log_Fs = np.log(Fs)
    slope, intercept, r_value, p_value, std_err = linregress(log_scales, log_Fs)
    hurst = slope-1
    return hurst


def test_self_similarity(time_series):
    """
    Test the self-similarity of a given time series.

    :param time_series: The time series to analyze
    :type time_series: array_like
    :return: Dictionary containing the estimated Hurst exponents
    :rtype: dict
    """
    hurst_dfa = detrended_fluctuation_analysis(time_series)

    results = {
        "Hurst Exponent (DFA)": hurst_dfa,
    }

    return results

# Example usage
if __name__ == "__main__":
    # Generate a sample time series (fractional Brownian motion with H=0.7)
    from numpy.random import RandomState

    def fbm(n, H):
        rng = RandomState(0)
        t = np.arange(n)
        dt = 1
        X = np.zeros(n)
        X[1:] = np.cumsum(rng.standard_normal(n - 1) * dt ** H)
        return X

    n_points = 10000
    H_true = 0.7
    time_series = fbm(n_points, H_true)

    # Test self-similarity
    results = test_self_similarity(time_series)

    print("Self-similarity test results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    print(f"\nTrue Hurst exponent: {H_true:.4f}")
    print(f"True alpha: {1 / H_true:.4f}")

    # Plot the time series
    plt.figure(figsize=(12, 6))
    plt.plot(time_series)
    plt.title(f"Sample Time Series (Fractional Brownian Motion, H={H_true})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

def wavelet_estimation(time_series, wavelet='db4', max_level=None):
    """
    Estimate the Hurst exponent using wavelet-based method.

    :param time_series: The time series to analyze
    :type time_series: array_like
    :param wavelet: The wavelet to use for the analysis (default is 'db4')
    :type wavelet: str, optional
    :param max_level: The maximum decomposition level (default is None, which means it will be automatically determined)
    :type max_level: int, optional
    :return: Estimated Hurst exponent
    :rtype: float
    """
    n = len(time_series)

    # Determine the maximum decomposition level if not provided
    if max_level is None:
        max_level = int(np.log2(n)) - 1

    # Perform the wavelet transform
    coeffs = pywt.wavedec(time_series, wavelet, level=max_level)

    # Compute the variance of the wavelet coefficients at each level
    variances = [np.var(coeff) for coeff in coeffs[1:]]  # Exclude the approximation coefficients

    # Compute the scales
    scales = [2 ** i for i in range(1, len(variances) + 1)]

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(np.log2(scales), np.log2(variances))

    # The Hurst exponent is related to the slope
    H = -(slope + 1) / 2

    return H


def test_self_similarity_wavelet(time_series, wavelet='db4', max_level=None):
    """
    Test the self-similarity of a given time series using wavelet-based method.

    :param time_series: The time series to analyze
    :type time_series: array_like
    :param wavelet: The wavelet to use for the analysis (default is 'db4')
    :type wavelet: str, optional
    :param max_level: The maximum decomposition level (default is None, which means it will be automatically determined)
    :type max_level: int, optional
    :return: Dictionary containing the estimated Hurst exponent and scaling parameter
    :rtype: dict
    """
    H = wavelet_estimation(time_series, wavelet, max_level)

    results = {
        "Hurst Exponent (Wavelet)": H,
        "Estimated Alpha (Wavelet)": 1 / H if H > 0 else np.nan
    }

    return results


def plot_wavelet_analysis(time_series, wavelet='db4', max_level=None):
    """
    Perform wavelet analysis and plot the results.

    :param time_series: The time series to analyze
    :type time_series: array_like
    :param wavelet: The wavelet to use for the analysis (default is 'db4')
    :type wavelet: str, optional
    :param max_level: The maximum decomposition level (default is None, which means it will be automatically determined)
    :type max_level: int, optional
    :return: None
    :rtype: None
    """
    n = len(time_series)

    if max_level is None:
        max_level = int(np.log2(n)) - 1

    coeffs = pywt.wavedec(time_series, wavelet, level=max_level)
    variances = [np.var(coeff) for coeff in coeffs[1:]]
    scales = [2 ** i for i in range(1, len(variances) + 1)]

    slope, intercept, r_value, p_value, std_err = linregress(np.log2(scales), np.log2(variances))
    H = -(slope + 1) / 2

    plt.figure(figsize=(12, 6))
    plt.loglog(scales, variances, 'bo-')
    plt.loglog(scales, [2 ** (intercept + slope * np.log2(scale)) for scale in scales], 'r--')
    plt.title(f"Wavelet Variance Plot (Estimated H = {H:.4f})")
    plt.xlabel("Scale")
    plt.ylabel("Variance of Wavelet Coefficients")
    plt.legend(['Data', 'Fitted Line'])
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate a sample time series (fractional Brownian motion with H=0.7)
    from numpy.random import RandomState


    def fbm(n, H):
        rng = RandomState(0)
        t = np.arange(n)
        dt = 1
        X = np.zeros(n)
        X[1:] = np.cumsum(rng.standard_normal(n - 1) * dt ** H)
        return X

    n_points = 2 ** 14  # Use a power of 2 for efficiency in wavelet transform
    H_true = 0.7
    time_series = fbm(n_points, H_true)

    # Test self-similarity using wavelet method
    results = test_self_similarity_wavelet(time_series)

    print("Wavelet-based self-similarity test results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    print(f"\nTrue Hurst exponent: {H_true:.4f}")
    print(f"True alpha: {1 / H_true:.4f}")

    # Plot wavelet analysis
    plot_wavelet_analysis(time_series)

    # Plot the time series
    plt.figure(figsize=(12, 6))
    plt.plot(time_series)
    plt.title(f"Sample Time Series (Fractional Brownian Motion, H={H_true})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

def kurtosis_test(data):
    """
    Test for fat-tailedness using excess kurtosis.

    :param data: The data to analyze
    :type data: array_like
    :return: Dictionary containing the kurtosis and its interpretation
    :rtype: dict
    """
    kurt = stats.kurtosis(data)
    result = {
        "Excess Kurtosis": kurt,
        "Interpretation": "Fat-tailed" if kurt > 0 else "Not fat-tailed"
    }
    return result

def jarque_bera_test(data):
    """
    Perform Jarque-Bera test for normality.

    :param data: The data to analyze
    :type data: array_like
    :return: Dictionary containing the test statistic, p-value, and interpretation
    :rtype: dict
    """
    statistic, p_value = stats.jarque_bera(data)
    result = {
        "JB Statistic": statistic,
        "p-value": p_value,
        "Interpretation": "Likely fat-tailed" if p_value < 0.05 else "Not enough evidence for fat-tailedness"
    }
    return result

def tail_index_hill(data, tail_fraction=0.1):
    """
    Estimate the tail index using Hill estimator.

    :param data: The data to analyze
    :type data: array_like
    :param tail_fraction: Fraction of the data to consider as the tail
    :type tail_fraction: float
    :return: Dictionary containing the tail index and its interpretation
    :rtype: dict
    """
    sorted_data = np.sort(np.abs(data))
    n = len(sorted_data)
    k = int(n * tail_fraction)
    tail_index = 1 / np.mean(np.log(sorted_data[-k:]) - np.log(sorted_data[-k]))
    result = {
        "Tail Index": tail_index,
        "Interpretation": "Fat-tailed" if tail_index < 3 else "Not fat-tailed"
    }
    return result

def qqplot_analysis(data):
    """
    Perform Q-Q plot analysis against normal distribution.

    :param data: The data to analyze
    :type data: array_like
    :return: None (displays the Q-Q plot)
    :rtype: None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot against Normal Distribution")
    plt.show()


def test_fat_tailedness(data):
    """
    Perform a comprehensive test for fat-tailedness.

    :param data: The data to analyze
    :type data: array_like
    :return: Dictionary containing results from various tests
    :rtype: dict
    """
    results = {}
    results["Kurtosis Test"] = kurtosis_test(data)
    results["Jarque-Bera Test"] = jarque_bera_test(data)
    results["Tail Index (Hill Estimator)"] = tail_index_hill(data)

    print("Fat-tailedness Test Results:")
    for test_name, test_results in results.items():
        print(f"\n{test_name}:")
        for key, value in test_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print("\nGenerating Q-Q Plot...")
    qqplot_analysis(data)

    return results

# Example usage
if __name__ == "__main__":
    # Generate sample data from a fat-tailed distribution (Student's t with 3 degrees of freedom)
    np.random.seed(0)
    fat_tailed_data = stats.t.rvs(df=3, size=10000)

    # Test for fat-tailedness
    results = test_fat_tailedness(fat_tailed_data)

    # Generate and test normal distribution for comparison
    normal_data = np.random.normal(size=10000)
    print("\n\nFor comparison, results for normal distribution:")
    _ = test_fat_tailedness(normal_data)
