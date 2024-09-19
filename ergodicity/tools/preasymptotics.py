"""
preasymptotic Submodule

The `Preasymptotics` submodule provides tools to estimate, model, and test the speed of convergence to a limiting distribution, such as the normal distribution or other asymptotic limits. The core functionality is divided into three main components:

1. **Preasymptotic Behavior Quantification**: This class analyzes the rate at which a time series converges to its asymptotic value and investigates transient behaviors before reaching the steady state.

2. **Visualization Tools**: This class visualizes various aspects of the time series, including distribution evolution, multi-scale analysis, and scaling behavior.

3. **Statistical Testing and Validation**: This class performs statistical tests, such as bootstrapping, surrogate data testing, and cross-validation, to validate hypotheses about the time series.

Key Features:

1. **Preasymptotic Behavior Quantification**:

   - **Convergence Rate Estimation**: Estimates how quickly a time series converges to its asymptotic value using an exponential decay model.

   - **Transient Fluctuation Analysis**: Examines fluctuations in the time series over time, estimating the decay rate of these fluctuations.

   - **Time-to-Stationarity Estimation**: Determines how long it takes for the time series to become approximately stationary using the Augmented Dickey-Fuller (ADF) test.

2. **Visualization Tools**:

   - **Time Series Plotting**: Plots the time series over different time windows to visualize transient behavior.

   - **Distribution Evolution**: Shows how the distribution of the time series changes over different time intervals.

   - **Scaling Analysis**: Creates log-log plots for different statistical properties of the time series (e.g., variance, mean, max).

   - **Heatmap and 3D Surface Plotting**: Provides heatmaps for time-dependent statistics and 3D surface plots for multi-scale analysis.

3. **Statistical Testing and Validation**:

   - **Bootstrap Confidence Interval Estimation**: Provides confidence intervals for statistical metrics through bootstrapping.

   - **Surrogate Data Testing**: Validates time series characteristics against surrogate data using methods such as phase randomization.

   - **Cross-Validation for Time Series**: Implements time series cross-validation with user-defined models to validate forecasting performance.

4. **Reporting and Exporting**:

   - **PDF/HTML Report Generation**: Creates detailed reports summarizing the preasymptotic behavior and statistical testing results in PDF or HTML format.

   - **Result Export**: Exports results in CSV and JSON formats.

   - **Visualization Export**: Saves visualizations as image files (PNG or SVG).

Example Usage:

if __name__ == "__main__":

    # Generate a sample non-stationary time series

    np.random.seed(0)

    t = np.linspace(0, 10, 1000)

    non_stationary_series = 10 * np.exp(-0.5 * t) * np.sin(t) + np.cumsum(np.random.normal(0, 0.1, 1000))

    # Initialize the PreasymptoticBehaviorQuantification object

    pbq = PreasymptoticBehaviorQuantification(non_stationary_series, t)

    # Perform analyses

    convergence_results = pbq.convergence_rate_estimation()

    fluctuation_results = pbq.transient_fluctuation_analysis()

    stationarity_results = pbq.time_to_stationarity_estimation()

    # Visualize the results

    pbq.plot_results(convergence_results, fluctuation_results, stationarity_results)

    # Generate and export reports

    reporter = ReportingAndExport(non_stationary_series, t)

    reporter.generate_pdf_report()

    reporter.export_results_csv()

    reporter.export_visualizations(format='png')
"""
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import curve_fit
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import acf
import json
import csv
import matplotlib.pyplot as plt
from fpdf import FPDF
from jinja2 import Environment, FileSystemLoader
import base64
from io import BytesIO

class PreasymptoticBehaviorQuantification:
    """
    Class to quantify preasymptotic behavior in time series data.
    It provides methods to estimate convergence rates, transient fluctuations, and time to stationarity.

    Attributes:

        time_series (array): Time series data

        time_values (array): Time values corresponding to the data points
    """
    def __init__(self, time_series, time_values=None):
        """
        Initialize the PreasymptoticBehaviorQuantification object.

        :param time_series: Time series data
        :type time_series: array
        :param time_values: Time values corresponding to the data points
        :type time_values: array, optional
        """
        self.time_series = np.array(time_series)
        self.time_values = time_values if time_values is not None else np.arange(len(time_series))

    def convergence_rate_estimation(self, window_size=100, asymptotic_value=None):
        """
        Estimate the convergence rate to asymptotic value.

        :param window_size: Size of the moving window
        :type window_size: int
        :param asymptotic_value: Known asymptotic value. If None, uses the mean of the last window as an estimate.
        :type asymptotic_value: float, optional
        :return: Dictionary containing convergence rate and related information
        :rtype: dict
        """
        if asymptotic_value is None:
            asymptotic_value = np.mean(self.time_series[-window_size:])

        moving_average = np.convolve(self.time_series, np.ones(window_size) / window_size, mode='valid')
        time_points = self.time_values[window_size - 1:]

        distance_to_asymptote = np.abs(moving_average - asymptotic_value)

        def convergence_model(t, rate, amplitude):
            return amplitude * np.exp(-rate * t)

        popt, _ = curve_fit(convergence_model, time_points, distance_to_asymptote, p0=[0.1, 1])
        rate, amplitude = popt

        return {
            "convergence_rate": rate,
            "initial_amplitude": amplitude,
            "asymptotic_value": asymptotic_value,
            "time_points": time_points,
            "distance_to_asymptote": distance_to_asymptote
        }

    def transient_fluctuation_analysis(self, window_size=100):
        """
        Analyze transient fluctuations in the time series.

        :param window_size: Size of the moving window
        :type window_size: int
        :return: Dictionary containing transient fluctuation analysis results
        :rtype: dict
        """
        std_dev = np.array(
            [np.std(self.time_series[i:i + window_size]) for i in range(0, len(self.time_series) - window_size + 1)])
        time_points = self.time_values[window_size - 1:]

        # Fit a curve to characterize the decay of fluctuations
        def fluctuation_model(t, a, b, c):
            return a * np.exp(-b * t) + c

        popt, _ = curve_fit(fluctuation_model, time_points, std_dev, p0=[1, 0.1, 0])
        a, b, c = popt

        return {
            "time_points": time_points,
            "standard_deviation": std_dev,
            "decay_rate": b,
            "initial_amplitude": a,
            "long_term_fluctuation": c
        }

    def time_to_stationarity_estimation(self, window_size=100, p_threshold=0.05):
        """
        Estimate the time to reach approximate stationarity.

        :param window_size: Size of the moving window for the ADF test
        :type window_size: int
        :param p_threshold: p-value threshold for the ADF test
        :type p_threshold: float
        :return: Dictionary containing time-to-stationarity estimation results
        :rtype: dict
        """
        adf_results = []
        for i in range(0, len(self.time_series) - window_size + 1, window_size // 10):
            result = adfuller(self.time_series[i:i + window_size])
            middle_index = i + window_size // 2
            adf_results.append((self.time_values[middle_index], result[1]))  # Store time and p-value

        adf_results = np.array(adf_results)

        # Find the first time when the p-value is below the threshold
        stationary_indices = np.where(adf_results[:, 1] < p_threshold)[0]
        time_to_stationarity = adf_results[stationary_indices[0], 0] if len(stationary_indices) > 0 else None

        return {
            "time_to_stationarity": time_to_stationarity,
            "adf_results": adf_results,
            "p_threshold": p_threshold
        }

    def plot_results(self, convergence_results, fluctuation_results, stationarity_results):
        """
        Plot the results of preasymptotic behavior quantification.

        :param convergence_results: Results from convergence_rate_estimation
        :type convergence_results: dict
        :param fluctuation_results: Results from transient_fluctuation_analysis
        :type fluctuation_results: dict
        :param stationarity_results: Results from time_to_stationarity_estimation
        :type stationarity_results: dict
        :return: None
        :rtype: None
        """
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))

        # Convergence rate plot
        axs[0].plot(convergence_results['time_points'], convergence_results['distance_to_asymptote'], label='Data')
        axs[0].plot(convergence_results['time_points'],
                    convergence_results['initial_amplitude'] * np.exp(
                        -convergence_results['convergence_rate'] * convergence_results['time_points']),
                    'r--', label='Fitted Curve')
        axs[0].set_title('Convergence to Asymptotic Value')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Distance to Asymptote')
        axs[0].legend()

        # Transient fluctuation plot
        axs[1].plot(fluctuation_results['time_points'], fluctuation_results['standard_deviation'], label='Data')
        axs[1].plot(fluctuation_results['time_points'],
                    fluctuation_results['initial_amplitude'] * np.exp(
                        -fluctuation_results['decay_rate'] * fluctuation_results['time_points']) + fluctuation_results[
                        'long_term_fluctuation'],
                    'r--', label='Fitted Curve')
        axs[1].set_title('Transient Fluctuation Analysis')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Standard Deviation')
        axs[1].legend()

        # Time to stationarity plot
        axs[2].plot(stationarity_results['adf_results'][:, 0], stationarity_results['adf_results'][:, 1],
                    label='ADF p-value')
        axs[2].axhline(y=stationarity_results['p_threshold'], color='r', linestyle='--', label='Threshold')
        if stationarity_results['time_to_stationarity'] is not None:
            axs[2].axvline(x=stationarity_results['time_to_stationarity'], color='g', linestyle='--',
                           label='Time to Stationarity')
        axs[2].set_title('Time to Stationarity Estimation')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('ADF Test p-value')
        axs[2].legend()

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate a sample non-stationary time series
    np.random.seed(0)
    t = np.linspace(0, 10, 1000)
    non_stationary_series = 10 * np.exp(-0.5 * t) * np.sin(t) + np.cumsum(np.random.normal(0, 0.1, 1000))

    # Initialize the PreasymptoticBehaviorQuantification object
    pbq = PreasymptoticBehaviorQuantification(non_stationary_series, t)

    # Perform analyses
    convergence_results = pbq.convergence_rate_estimation()
    fluctuation_results = pbq.transient_fluctuation_analysis()
    stationarity_results = pbq.time_to_stationarity_estimation()

    # Print some results
    print(f"Convergence Rate: {convergence_results['convergence_rate']:.4f}")
    print(f"Transient Fluctuation Decay Rate: {fluctuation_results['decay_rate']:.4f}")
    print(f"Time to Stationarity: {stationarity_results['time_to_stationarity']:.4f}" if stationarity_results[
                                                                                             'time_to_stationarity'] is not None else "Stationarity not reached")

    # Plot results
    pbq.plot_results(convergence_results, fluctuation_results, stationarity_results)

class PreasymptoticVisualizationTools:
    """
    Class to visualize preasymptotic behavior in time series data. It provides tools to plot time series windows,

    Attributes:

        time_series: array

        time_values: array
    """
    def __init__(self, time_series, time_values=None):
        """
        Initialize the PreasymptoticVisualizationTools object.

        :param time_series: Time series data
        :type time_series: array
        :param time_values: Time values corresponding to the data points
        :type time_values: array, optional
        """
        self.time_series = np.array(time_series)
        self.time_values = time_values if time_values is not None else np.arange(len(time_series))

    def plot_time_series_windows(self, window_sizes=[100, 500, 1000]):
        """
        Plot time series with variable time windows.

        :param window_sizes: Sizes of time windows to plot
        :type window_sizes: list of int
        :return: None
        :rtype: None
        """
        fig, axs = plt.subplots(len(window_sizes), 1, figsize=(12, 4 * len(window_sizes)), sharex=True)
        if len(window_sizes) == 1:
            axs = [axs]

        for ax, window in zip(axs, window_sizes):
            ax.plot(self.time_values, self.time_series)
            ax.set_title(f'Time Window: {window}')
            ax.set_ylabel('Value')

            # Highlight windows
            for i in range(0, len(self.time_series), window):
                ax.axvspan(self.time_values[i], self.time_values[min(i + window, len(self.time_series) - 1)],
                           alpha=0.2, color='red')

        axs[-1].set_xlabel('Time')
        plt.tight_layout()
        plt.show()

    def plot_distribution_evolution(self, num_windows=5):
        """
        Plot the evolution of the distribution over time.

        :param num_windows: Number of time windows to use for distribution evolution
        :type num_windows: int
        :return: None
        :rtype: None
        """
        window_size = len(self.time_series) // num_windows
        fig, axs = plt.subplots(1, num_windows, figsize=(4 * num_windows, 4), sharey=True)

        for i, ax in enumerate(axs):
            start = i * window_size
            end = (i + 1) * window_size
            data = self.time_series[start:end]
            sns.histplot(data, kde=True, ax=ax)
            ax.set_title(f'Time: {self.time_values[start]:.2f} - {self.time_values[end - 1]:.2f}')
            ax.set_xlabel('Value')

        axs[0].set_ylabel('Density')
        plt.tight_layout()
        plt.show()

    def plot_scaling_analysis(self, statistic='variance', num_points=20):
        """
        Create log-log plots of various statistics.

        :param statistic: Statistic to analyze ('variance', 'abs_mean', or 'max')
        :type statistic: str
        :param num_points: Number of points to use in the log-log plot
        :type num_points: int
        :return: None
        :rtype: None
        """
        window_sizes = np.logspace(1, np.log10(len(self.time_series) // 2), num_points, dtype=int)
        statistics = []

        for size in window_sizes:
            if statistic == 'variance':
                stat = np.var(self.time_series[:size])
            elif statistic == 'abs_mean':
                stat = np.mean(np.abs(self.time_series[:size]))
            elif statistic == 'max':
                stat = np.max(np.abs(self.time_series[:size]))
            else:
                raise ValueError("Invalid statistic. Choose 'variance', 'abs_mean', or 'max'.")
            statistics.append(stat)

        plt.figure(figsize=(10, 6))
        plt.loglog(window_sizes, statistics, 'o-')
        plt.title(f'Scaling Analysis: {statistic.capitalize()}')
        plt.xlabel('Window Size')
        plt.ylabel(statistic.capitalize())
        plt.grid(True)
        plt.show()

    def plot_time_dependent_heatmap(self, window_size=100, step_size=10):
        """
        Create a heatmap for time-dependent analyses.
        It allows visualization of time-dependent statistics like mean, standard deviation, and skewness.

        :param window_size: Size of the sliding window
        :type window_size: int
        :param step_size: Step size for sliding the window
        :type step_size: int
        :return: None
        :rtype: None
        """
        num_windows = (len(self.time_series) - window_size) // step_size + 1
        statistics = np.zeros((3, num_windows))

        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            window_data = self.time_series[start:end]
            statistics[0, i] = np.mean(window_data)
            statistics[1, i] = np.std(window_data)
            statistics[2, i] = stats.skew(window_data)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(statistics, ax=ax, cmap='viridis')
        ax.set_title('Time-Dependent Statistics Heatmap')
        ax.set_xlabel('Time Window')
        ax.set_ylabel('Statistic')
        ax.set_yticklabels(['Mean', 'Std Dev', 'Skewness'])
        plt.show()

    def plot_3d_multiscale_analysis(self, max_scale=10):
        """
        Create a 3D surface plot for multi-scale analyses.

        :param max_scale: Maximum scale for the analysis
        :type max_scale: int
        :return: None
        :rtype: None
        """
        scales = range(1, max_scale + 1)
        times = self.time_values[::max_scale]  # Downsample for clarity

        fluctuations = np.zeros((len(scales), len(times)))

        for i, scale in enumerate(scales):
            for j, t in enumerate(range(0, len(self.time_series) - scale * max_scale, max_scale)):
                segment = self.time_series[t:t + scale * max_scale]
                fluctuations[i, j] = np.std(segment)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(times, scales)
        surf = ax.plot_surface(X, Y, fluctuations, cmap='viridis')

        ax.set_title('Multi-Scale Analysis')
        ax.set_xlabel('Time')
        ax.set_ylabel('Scale')
        ax.set_zlabel('Fluctuation')
        fig.colorbar(surf)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate a sample non-stationary time series
    np.random.seed(0)
    t = np.linspace(0, 10, 1000)
    non_stationary_series = 10 * np.exp(-0.5 * t) * np.sin(t) + np.cumsum(np.random.normal(0, 0.1, 1000))

    # Initialize the visualization tools
    vis_tools = PreasymptoticVisualizationTools(non_stationary_series, t)

    # 11.1 Time series plots with variable time windows
    vis_tools.plot_time_series_windows([100, 250, 500])

    # 11.2 Distribution evolution plots
    vis_tools.plot_distribution_evolution()

    # 11.3 Scaling plots
    vis_tools.plot_scaling_analysis(statistic='variance')

    # 11.4 Heatmaps for time-dependent analyses
    vis_tools.plot_time_dependent_heatmap()

    # 11.5 3D surface plots for multi-scale analyses
    vis_tools.plot_3d_multiscale_analysis()

class StatisticalTestingValidation:
    """
    Class to perform statistical testing and validation on time series data with the emphasis on preasymptotics.

    Attributes:

        time_series: array

        time_values: array
    """
    def __init__(self, time_series, time_values=None):
        """
        Initialize the StatisticalTestingValidation object.

        :param time_series: Time series data
        :type time_series: array
        :param time_values: Time values corresponding to the data points
        :type time_values: array, optional
        """
        self.time_series = np.array(time_series)
        self.time_values = time_values if time_values is not None else np.arange(len(time_series))

    def bootstrap_confidence_interval(self, statistic_func, n_bootstraps=1000, confidence_level=0.95):
        """
        Estimate confidence intervals using bootstrap method.

        :param statistic_func: Function to compute the statistic of interest
        :type statistic_func: callable
        :param n_bootstraps: Number of bootstrap samples
        :type n_bootstraps: int
        :param confidence_level: Confidence level for the interval
        :type confidence_level: float
        :return: Lower and upper bounds of the confidence interval
        :rtype: tuple
        """
        bootstrap_statistics = []
        for _ in range(n_bootstraps):
            resampled_series = np.random.choice(self.time_series, size=len(self.time_series), replace=True)
            bootstrap_statistics.append(statistic_func(resampled_series))

        lower_percentile = (1 - confidence_level) / 2
        upper_percentile = 1 - lower_percentile
        return np.percentile(bootstrap_statistics, [lower_percentile * 100, upper_percentile * 100])

    def surrogate_data_test(self, test_statistic_func, n_surrogates=1000, method='phase_randomization'):
        """
        Perform surrogate data testing.

        :param test_statistic_func: Function to compute the test statistic
        :type test_statistic_func: callable
        :param n_surrogates: Number of surrogate series to generate
        :type n_surrogates: int
        :param method: Method for generating surrogates ('phase_randomization' or 'bootstrap')
        :type method: str
        :return: Tuple of p-value, original statistic, and surrogate statistics
        :rtype: tuple
        """
        original_statistic = test_statistic_func(self.time_series)
        surrogate_statistics = []

        for _ in range(n_surrogates):
            if method == 'phase_randomization':
                surrogate = self._phase_randomization()
            elif method == 'bootstrap':
                surrogate = np.random.choice(self.time_series, size=len(self.time_series), replace=True)
            else:
                raise ValueError("Invalid method. Choose 'phase_randomization' or 'bootstrap'.")

            surrogate_statistics.append(test_statistic_func(surrogate))

        p_value = (sum(s >= original_statistic for s in surrogate_statistics) + 1) / (n_surrogates + 1)
        return p_value, original_statistic, surrogate_statistics

    def _phase_randomization(self):
        """
        Generate a surrogate time series using phase randomization.

        :returns np.array: Surrogate time series
        :rtype: np.array
        """
        fft = np.fft.fft(self.time_series)
        amplitudes = np.abs(fft)
        phases = np.angle(fft)
        random_phases = np.random.uniform(0, 2*np.pi, len(phases))
        new_fft = amplitudes * np.exp(1j * random_phases)
        return np.real(np.fft.ifft(new_fft))

    def cross_validation(self, model_func, n_splits=5):
        """
        Perform time series cross-validation.

        :param model_func: Function that takes training data and returns predictions for test data
        :type model_func: callable
        :param n_splits: Number of splits for cross-validation
        :type n_splits: int
        :return: List of error scores for each split
        :rtype: list
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for train_index, test_index in tscv.split(self.time_series):
            train_data = self.time_series[train_index]
            test_data = self.time_series[test_index]
            predictions = model_func(train_data, test_data)
            mse = np.mean((test_data - predictions)**2)
            scores.append(mse)

        return scores

    def plot_results(self, bootstrap_results, surrogate_results, cv_results):
        """
        Plot the results of statistical testing and validation.

        :param bootstrap_results: Results from bootstrap_confidence_interval
        :type bootstrap_results: tuple
        :param surrogate_results: Results from surrogate_data_test
        :type surrogate_results: tuple
        :param cv_results: Results from cross_validation
        :type cv_results: list
        :return: None
        :rtype: None
        """
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))

        # Bootstrap results
        axs[0].hist(bootstrap_results[2], bins=30, edgecolor='black')
        axs[0].axvline(bootstrap_results[0], color='r', linestyle='--', label='Lower CI')
        axs[0].axvline(bootstrap_results[1], color='r', linestyle='--', label='Upper CI')
        axs[0].set_title('Bootstrap Confidence Interval')
        axs[0].set_xlabel('Statistic Value')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()

        # Surrogate data results
        axs[1].hist(surrogate_results[2], bins=30, edgecolor='black')
        axs[1].axvline(surrogate_results[1], color='r', linestyle='--', label='Original Statistic')
        axs[1].set_title(f'Surrogate Data Test (p-value: {surrogate_results[0]:.4f})')
        axs[1].set_xlabel('Statistic Value')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()

        # Cross-validation results
        axs[2].plot(range(1, len(cv_results) + 1), cv_results, 'o-')
        axs[2].set_title('Cross-Validation Results')
        axs[2].set_xlabel('Split')
        axs[2].set_ylabel('Mean Squared Error')

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate a sample time series
    np.random.seed(0)
    t = np.linspace(0, 10, 1000)
    time_series = 10 * np.exp(-0.5 * t) * np.sin(t) + np.cumsum(np.random.normal(0, 0.1, 1000))

    # Initialize the StatisticalTestingValidation object
    stv = StatisticalTestingValidation(time_series, t)

    # 12.1 Bootstrap confidence interval
    def mean_statistic(data):
        return np.mean(data)

    ci_lower, ci_upper = stv.bootstrap_confidence_interval(mean_statistic)
    print(f"Bootstrap 95% CI for mean: ({ci_lower:.2f}, {ci_upper:.2f})")

    # 12.2 Surrogate data testing
    def test_statistic(data):
        return np.max(np.abs(acf(data)))

    p_value, orig_stat, surr_stats = stv.surrogate_data_test(test_statistic)
    print(f"Surrogate data test p-value: {p_value:.4f}")

    # 12.3 Cross-validation
    def simple_model(train, test):
        # Simple model: predict the mean of the training data
        return np.full(len(test), np.mean(train))

    cv_scores = stv.cross_validation(simple_model)
    print(f"Cross-validation MSE scores: {cv_scores}")

    # Plot results
    stv.plot_results((ci_lower, ci_upper, np.random.normal(np.mean(time_series), np.std(time_series), 1000)),
                     (p_value, orig_stat, surr_stats),
                     cv_scores)

class ReportingAndExport:
    """
    Class to generate reports and export results of preasymptotic analysis.
    The reports can be generated in PDF or HTML format.

    Attributes:

        time_series: array

        time_values: array
    """
    def __init__(self, time_series, time_values=None):
        """
        Initialize the ReportingAndExport object.

        :param time_series: Time series data
        :type time_series: array
        :param time_values: Time values corresponding to the data points
        :type time_values: array, optional
        """
        self.time_series = time_series
        self.time_values = time_values if time_values is not None else range(len(time_series))
        self.pbq = PreasymptoticBehaviorQuantification(time_series, time_values)
        self.pvt = PreasymptoticVisualizationTools(time_series, time_values)
        self.stv = StatisticalTestingValidation(time_series, time_values)

    def generate_pdf_report(self, filename="preasymptotic_analysis_report.pdf"):
        """
        Generate a PDF report of the preasymptotic analysis.

        :param filename: name of the PDF file, optional
        :type filename: str
        :return: None
        :rtype: None
        """
        pdf = FPDF()
        pdf.add_page()

        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Preasymptotic Analysis Report", 0, 1, "C")
        pdf.ln(10)

        # Preasymptotic Behavior Quantification
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "1. Preasymptotic Behavior Quantification", 0, 1)

        convergence_results = self.pbq.convergence_rate_estimation()
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Convergence Rate: {convergence_results['convergence_rate']:.4f}", 0, 1)

        fluctuation_results = self.pbq.transient_fluctuation_analysis()
        pdf.cell(0, 10, f"Fluctuation Decay Rate: {fluctuation_results['decay_rate']:.4f}", 0, 1)

        stationarity_results = self.pbq.time_to_stationarity_estimation()
        pdf.cell(0, 10, f"Time to Stationarity: {stationarity_results['time_to_stationarity']:.4f}", 0, 1)

        # Visualizations
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "2. Visualizations", 0, 1)

        # Time series plot
        plt.figure(figsize=(10, 6))
        self.pvt.plot_time_series_windows()
        plt.savefig("temp_time_series.png")
        pdf.image("temp_time_series.png", x=10, y=pdf.get_y(), w=190)
        pdf.ln(100)

        # Distribution evolution plot
        plt.figure(figsize=(10, 6))
        self.pvt.plot_distribution_evolution()
        plt.savefig("temp_distribution.png")
        pdf.image("temp_distribution.png", x=10, y=pdf.get_y(), w=190)
        pdf.ln(100)

        # Statistical Testing and Validation
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "3. Statistical Testing and Validation", 0, 1)

        # Bootstrap results
        ci_lower, ci_upper = self.stv.bootstrap_confidence_interval(lambda x: np.mean(x))
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Bootstrap 95% CI for mean: ({ci_lower:.2f}, {ci_upper:.2f})", 0, 1)

        # Surrogate data test results
        p_value, _, _ = self.stv.surrogate_data_test(lambda x: np.max(np.abs(np.fft.fft(x))))
        pdf.cell(0, 10, f"Surrogate data test p-value: {p_value:.4f}", 0, 1)

        pdf.output(filename)
        print(f"PDF report generated: {filename}")

    def generate_html_report(self, filename="preasymptotic_analysis_report.html"):
        """
        Generate an HTML report of the preasymptotic analysis.

        :param filename: Name of the HTML file, optional
        :type filename: str
        :return: None
        :rtype: None
        """
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template('report_template.html')

        # Perform analyses
        convergence_results = self.pbq.convergence_rate_estimation()
        fluctuation_results = self.pbq.transient_fluctuation_analysis()
        stationarity_results = self.pbq.time_to_stationarity_estimation()

        # Generate plots and convert to base64
        plt.figure(figsize=(10, 6))
        self.pvt.plot_time_series_windows()
        time_series_plot = self._fig_to_base64()

        plt.figure(figsize=(10, 6))
        self.pvt.plot_distribution_evolution()
        distribution_plot = self._fig_to_base64()

        # Statistical testing results
        ci_lower, ci_upper = self.stv.bootstrap_confidence_interval(lambda x: np.mean(x))
        p_value, _, _ = self.stv.surrogate_data_test(lambda x: np.max(np.abs(np.fft.fft(x))))

        # Render template
        html_out = template.render(
            convergence_rate=convergence_results['convergence_rate'],
            fluctuation_decay_rate=fluctuation_results['decay_rate'],
            time_to_stationarity=stationarity_results['time_to_stationarity'],
            time_series_plot=time_series_plot,
            distribution_plot=distribution_plot,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value
        )

        # Write HTML file
        with open(filename, 'w') as f:
            f.write(html_out)
        print(f"HTML report generated: {filename}")

    def export_results_csv(self, filename="preasymptotic_analysis_results.csv"):
        """
        Export analysis results to a CSV file.

        :param filename: Name of the CSV file, optional
        :type filename: str
        :return: None
        :rtype: None
        """
        results = self._gather_results()
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            for key, value in results.items():
                writer.writerow([key, value])
        print(f"Results exported to CSV: {filename}")

    def export_results_json(self, filename="preasymptotic_analysis_results.json"):
        """
        Export analysis results to a JSON file.

        :param filename: Name of the JSON file, optional
        :type filename: str
        :return: None
        :rtype: None
        """
        results = self._gather_results()
        with open(filename, 'w') as jsonfile:
            json.dump(results, jsonfile, indent=4)
        print(f"Results exported to JSON: {filename}")

    def export_visualizations(self, format='png'):
        """
        Export visualizations as image files.

        :param format: Format for the exported files ('png' or 'svg')
        :type format: str
        :return: None
        :rtype: None
        """
        if format not in ['png', 'svg']:
            raise ValueError("Format must be 'png' or 'svg'")

        # Time series plot
        plt.figure(figsize=(10, 6))
        self.pvt.plot_time_series_windows()
        plt.savefig(f"time_series_plot.{format}")
        plt.close()

        # Distribution evolution plot
        plt.figure(figsize=(10, 6))
        self.pvt.plot_distribution_evolution()
        plt.savefig(f"distribution_evolution_plot.{format}")
        plt.close()

        # Scaling analysis plot
        plt.figure(figsize=(10, 6))
        self.pvt.plot_scaling_analysis()
        plt.savefig(f"scaling_analysis_plot.{format}")
        plt.close()

        print(f"Visualizations exported as {format.upper()} files")

    def _gather_results(self):
        """
        Gather all analysis results.

        :return: Dictionary of analysis results
        :rtype: dict
        """
        convergence_results = self.pbq.convergence_rate_estimation()
        fluctuation_results = self.pbq.transient_fluctuation_analysis()
        stationarity_results = self.pbq.time_to_stationarity_estimation()
        ci_lower, ci_upper = self.stv.bootstrap_confidence_interval(lambda x: np.mean(x))
        p_value, _, _ = self.stv.surrogate_data_test(lambda x: np.max(np.abs(np.fft.fft(x))))

        return {
            'convergence_rate': convergence_results['convergence_rate'],
            'fluctuation_decay_rate': fluctuation_results['decay_rate'],
            'time_to_stationarity': stationarity_results['time_to_stationarity'],
            'bootstrap_ci_lower': ci_lower,
            'bootstrap_ci_upper': ci_upper,
            'surrogate_test_p_value': p_value
        }

    def _fig_to_base64(self):
        """
        Convert the current matplotlib figure to a base64 string.

        :return: Base64-encoded image
        :rtype: str
        """
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(0)
    t = np.linspace(0, 10, 1000)
    time_series = 10 * np.exp(-0.5 * t) * np.sin(t) + np.cumsum(np.random.normal(0, 0.1, 1000))

    # Initialize ReportingAndExport
    reporter = ReportingAndExport(time_series, t)

    # Generate reports
    reporter.generate_pdf_report()
    reporter.generate_html_report()

    # Export results
    reporter.export_results_csv()
    reporter.export_results_json()

    # Export visualizations
    reporter.export_visualizations(format='png')
    reporter.export_visualizations(format='svg')
