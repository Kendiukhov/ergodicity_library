"""
evaluation Submodule Overview

The **`evaluation`** submodule provides tools for analyzing, comparing, and optimizing utility functions in decision-making environments. It allows users to evaluate how agents which learnt their behaviour using different optimization algorithms interact with stochastic processes and fit utility functions to observed behaviors using various optimization techniques.

Key Features:

1. **Utility Function Definition and Evaluation**:

   - The submodule allows users to define custom utility functions and optimize them based on stochastic process trajectories.

   - The **`UtilityFunction`** class is used to define utility functions with initial parameters, and it supports fitting parameters to minimize negative log-likelihood.

2. **Utility Function Inference**:

   - The **`UtilityFunctionInference`** class facilitates the fitting of utility functions to agent decisions.

   - Users can use both maximum likelihood estimation (MLE) and Bayesian inference (using Metropolis-Hastings sampling) to fit utility functions.

   - This class also includes methods for generating datasets, fitting models, and visualizing results such as utility functions and parameter distributions.

3. **Regression Analysis**:

   - A neural network-based regression model can be trained to predict agent preferences based on process parameters.

   - The **`regression_fit()`** method fits a neural network to the dataset, while **`plot_regression_results()`** provides visualizations of training results.

4. **Utility Function Tester**:

   - The **`UtilityFunctionTester`** class allows users to test and compare multiple utility functions by simulating processes and calculating optimal utility values for each function.

   - Includes methods for generating process parameters, simulating process trajectories, and optimizing utility functions for given trajectories.

5. **Inverse Reinforcement Learning (IRL)**:

   - The **`MaxEntIRL`** (Maximum Entropy Inverse Reinforcement Learning) class infers reward weights from agent behavior using observed trajectories.

   - The IRL approach fits a reward model that explains observed agent decisions by maximizing the likelihood of the agent's actions.

Example Usage:

### Fitting Utility Functions to Agent's Choices:

from ergodicity.evaluation import UtilityFunctionInference, UtilityFunction

from ergodicity.process.basic import BrownianMotion, GeometricBrownianMotion

# Initialize UtilityFunctionInference

ufi = UtilityFunctionInference('path/to/your/model.h5', param_ranges={
    'BrownianMotion': {'mu': (0, 0.5), 'sigma': (0.1, 0.5)},
    'GeometricBrownianMotion': {'mu': (0, 0.5), 'sigma': (0.1, 0.5)}
})

# Add utility functions to be fitted

ufi.add_utility_function(UtilityFunction('Power', lambda x, beta: x ** beta, [1.0]))

ufi.add_utility_function(UtilityFunction('Exponential', lambda x, alpha: 1 - np.exp(-alpha * x), [1.0]))

# Generate dataset of stochastic processes

dataset = ufi.generate_dataset(100)

# Get agent's choices based on the dataset

choices = ufi.get_agent_choices(dataset)

# Fit utility functions based on the agent's choices

ufi.fit_utility_functions(dataset, choices)

# Plot the fitted utility functions

ufi.plot_utility_functions()
"""
from tensorflow.python.framework.ops import strip_name_scope
from ergodicity.agents.sml import *
from ergodicity.tools.helper import ProcessEncoder
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Callable, Tuple
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from ergodicity.process.basic import *
from ergodicity.process.multiplicative import *
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from ergodicity.tools.compute import *

class UtilityFunction:
    """
    UtilityFunction Class

    This class represents a utility function used in decision-making models and optimization problems.

    Attributes:

        name (str): The name of the utility function.

        func (Callable): The actual utility function implementation.

        initial_params (List[float]): Initial parameters for the utility function.

        fitted_params (List[float] or None): Fitted parameters after optimization, if any.

        nll (float or None): Negative log-likelihood of the fitted function, if calculated.

    The UtilityFunction class encapsulates both the definition and the behavior of a utility function.
    It supports initial parametrization and subsequent fitting, making it suitable for use in
    optimization processes where the parameters of the utility function are adjusted based on data.

    Usage:

        def power_utility(x, beta):

            return x ** beta

        util_func = UtilityFunction("Power Utility", power_utility, [1.0])

        # Before fitting

        value = util_func(2.0)  # Uses initial parameter

        # After fitting (assuming fitting has been done elsewhere)

        util_func.fitted_params = [0.8]

        value = util_func(2.0)  # Uses fitted parameter

    This class is particularly useful in contexts where multiple utility functions need to be
    defined, compared, and optimized, such as in economic models or decision theory applications.
    """
    def __init__(self, name: str, func: Callable, initial_params: List[float]):
        """
        Initialize a utility function with a name, function, and initial parameters.

        :param name: Name of the utility function
        :type name: str
        :param func: The utility function as a callable
        :type func: Callable
        :param initial_params: Initial parameters for the utility function
        :type initial_params: List[float]
        """
        self.name = name
        self.func = func
        self.initial_params = initial_params
        self.fitted_params = None
        self.nll = None

    def __call__(self, x: float) -> float:
        """
        Evaluate the utility function at a given input.

        :param x: Input value
        :type x: float
        :return: Utility value
        :rtype: float
        """
        if self.fitted_params is None:
            return self.func(x, *self.initial_params)
        return self.func(x, *self.fitted_params)


class UtilityFunctionInference:
    """
    UtilityFunctionInference Class

    This class is designed for inferring and analyzing utility functions based on agent behavior
    in stochastic processes. It combines machine learning, Bayesian inference, and economic modeling
    to understand decision-making patterns.

    Attributes:

        model (tf.keras.Model): A neural network model loaded from a file, used for decision prediction.

        agent (NeuralNetworkAgent): An agent that uses the loaded model for decision-making.

        process_encoder (ProcessEncoder): Encoder for converting stochastic processes into a format suitable for the model.

        utility_functions (List[UtilityFunction]): Collection of utility functions to be analyzed.

        param_ranges (Dict[str, Dict[str, Tuple[float, float]]]): Ranges of parameters for different stochastic processes.

        mcmc_samples (Dict): Stores samples from Markov Chain Monte Carlo simulations for Bayesian inference.

        regression_model (tf.keras.Model or None): A regression model for preference prediction, if trained.

        regression_history (tf.keras.callbacks.History or None): Training history of the regression model, if available.

    This class provides a comprehensive toolkit for analyzing decision-making behavior in the context
    of stochastic processes. It supports various methods of utility function inference, including
    maximum likelihood estimation, Bayesian inference, and inverse reinforcement learning. The class
    also includes functionality for visualizing results and analyzing feature importance in the
    decision-making process.

    Usage:

        ufi = UtilityFunctionInference('path/to/model.h5', param_ranges)

        ufi.add_utility_function(UtilityFunction('Power', utility_power, [1.0]))

        dataset = ufi.generate_dataset(1000)

        choices = ufi.get_agent_choices(dataset)

        ufi.fit_utility_functions(dataset, choices)

        ufi.plot_utility_functions()

    This class is particularly useful for researchers and practitioners in fields such as
    economics, decision theory, and reinforcement learning, where understanding the underlying
    utility functions driving agent behavior is crucial.
    """
    def __init__(self, model_path: str, param_ranges: Dict[str, Dict[str, Tuple[float, float]]], model=None):
        """
        Initialize the UtilityFunctionInference class with a trained model and parameter ranges.

        :param model_path: Path to the trained model file
        :type model_path: str
        :param param_ranges: Dictionary of process types and their parameter ranges
        :type param_ranges: Dict[str, Dict[str, Tuple[float, float]]]
        :param model: Optional Keras model instance to use instead of loading from file. If None, the model is loaded from the file.
        :type model: tf.keras.Model or None
        """
        if model is not None:
            self.model = model
        else:
            self.model = self.load_model(model_path)
        self.agent = NeuralNetworkAgent(self.model)
        self.process_encoder = ProcessEncoder()
        self.utility_functions = []
        self.param_ranges = param_ranges
        self.mcmc_samples = {}
        self.regression_model = None
        self.regression_history = None
        self.dataset = None
        self.choices = None

    def load_model(self, model_path: str):
        """
        Load a Keras model from a file, handling potential compatibility issues.

        :param model_path: Path to the model file
        :type model_path: str
        :return: Loaded Keras model
        :rtype: tf.keras.Model
        """
        try:
            # Try to load the model normally
            return tf.keras.models.load_model(model_path)
        except (TypeError, ValueError) as e:
            print(f"Error loading model: {e}")
            print("Attempting to load model with custom objects...")

            # Define custom objects if necessary
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError,
                # Add any other custom objects here if needed
            }

            try:
                # Try to load the model with custom objects
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                print("Model loaded successfully with custom objects.")
                return model
            except Exception as e:
                print(f"Failed to load model even with custom objects: {e}")
                print("Attempting to load model architecture and weights separately...")

                # As a last resort, try to load just the architecture and weights
                try:
                    # Load model architecture from json
                    with open(model_path.replace('.h5', '_architecture.json'), 'r') as json_file:
                        model_json = json_file.read()
                    model = tf.keras.models.model_from_json(model_json)

                    # Load weights
                    model.load_weights(model_path)

                    print("Model architecture and weights loaded successfully.")
                    return model
                except Exception as e:
                    print(f"Failed to load model architecture and weights: {e}")
                    raise ValueError("Unable to load the model. Please check the model file and its compatibility.")

    def add_utility_function(self, utility_function):
        """
        Add a utility function to the collection for analysis.

        :param utility_function: UtilityFunction instance to add
        :type utility_function: UtilityFunction
        :return: None
        :rtype: None
        """
        self.utility_functions.append(utility_function)

    def generate_dataset(self, n_processes: int, n_options: int = 2, simulate_method=True) -> List[List[np.ndarray]]:
        """
        Generate a dataset of stochastic processes for analysis.

        :param n_processes: Number of datasets to generate (i.e., number of decision instances)
        :type n_processes: int
        :param n_options: Number of process options per decision instance
        :type n_options: int
        :return:  List of datasets, each containing process options
        :rtype: List[List[np.ndarray]]
        """
        dataset = []
        process_types = list(self.param_ranges.keys())

        for _ in range(n_processes):
            process_options = []
            for _ in range(n_options):
                process_type = np.random.choice(process_types)
                process_type_code = process_types.index(process_type) + 1  # 1-indexed

                params = {param: np.random.uniform(low, high)
                          for param, (low, high) in self.param_ranges[process_type].items()}

                # Create a process instance
                process_class = globals()[process_type]
                process = process_class(**params)

                # Simulate the process
                if simulate_method:
                    trajectory = process.simulate(
                        t=1, timestep=0.1, num_instances=1000, plot=False)
                else:
                    trajectory = process.simulate_until(
                        timestep=0.1, num_instances=1000, condition=lambda X: X >= 2, X0=1, plot=False)

                # Prepare data: trajectory values
                # Assuming trajectory is of shape (num_time_steps, num_instances)
                # We take the final values for each instance
                final_values = trajectory[-1, :]

                process_data = np.column_stack((
                    np.full(final_values.shape, process_type_code),
                    np.tile(list(params.values()), (final_values.shape[0], 1)),
                    final_values.reshape(-1, 1)
                ))

                process_options.append(process_data)

            dataset.append(process_options)

        return dataset

    def get_agent_choices(self, data: List[np.ndarray]) -> int:
        """
        Get the agent's choice based on a dataset of process options.

        :param data: A list of process options, each containing process data (np.ndarray)
        :type data: List[np.ndarray]
        :return: The index of the agent's choice
        :rtype: int
        """
        all_encoded_processes = []
        for process_data in data:
            try:
                # Extract process_type_code and parameters from the first row
                first_row = process_data[0]
                process_type_code = first_row[0]
                process_type = self.get_process_type(process_type_code)
                num_params = len(self.param_ranges[process_type])
                params_values = first_row[1:1 + num_params]
                params = {k: v for k, v in zip(self.param_ranges[process_type].keys(), params_values)}
                process_class = globals()[process_type]
                process = process_class(**params)
                encoded_process = self.process_encoder.encode_process_with_time(process, 1.0)
                all_encoded_processes.append(encoded_process)
            except ValueError as e:
                print(f"Error processing data point: {e}")
                # Skip this process option
                continue

        if not all_encoded_processes:
            raise ValueError("No valid processes to choose from")

        # Present all encoded processes to the agent at once
        all_encoded_processes = np.array(all_encoded_processes)
        choice = self.agent.select_process(all_encoded_processes)

        print(f"Agent's choice: {choice}")
        return choice  # Return the index of the chosen process

    def get_process_type(self, type_code) -> str:
        """
        Get the process type name based on the type code.

        :param type_code: Process type code
        :type type_code: int
        :return: Process type name
        :rtype: str
        """
        process_types = list(self.param_ranges.keys())
        type_mapping = {i + 1: process_type for i, process_type in enumerate(process_types)}

        # Convert type_code to float and round to nearest integer
        type_code_int = round(float(type_code))

        if type_code_int in type_mapping:
            return type_mapping[type_code_int]
        else:
            raise ValueError(f"Invalid type_code: {type_code}. Valid codes are {list(type_mapping.keys())}")

    def generate_choices(self, n_options: int = 2, n_choices: int = 100):
        """
        Generate agent choices among stochastic processes for testing utility functions with the corresponding dataset of processes to choose from.

        :param n_options: Number of process options per decision instance
        :param n_choices: Number of choices to generate (number of decision instances)
        :return dataset: A list of datasets, each containing process options
        :rtype dataset: List[List[np.ndarray]]
        :return: A list of choices where each choice is an index of the selected process
        :rtype: List[int]
        """
        choices = []
        dataset = []
        for i in range(n_choices):
            data = self.generate_dataset(n_processes=1, n_options=n_options)
            data = data[0]  # Since generate_dataset returns a list of datasets
            choice = self.get_agent_choices(data)
            choices.append(choice)
            dataset.append(data)

        return dataset, choices

    def negative_log_likelihood(self, params: List[float], utility_func: Callable, dataset: List[List[np.ndarray]],
                                choices: List[int]) -> float:
        """
        Calculate the negative log-likelihood of the utility function given the dataset and choices,
        assuming the agent maximizes expected utility.

        :param params: Utility function parameters
        :type params: List[float]
        :param utility_func: Utility function to evaluate
        :type utility_func: Callable
        :param dataset: List of datasets, each containing lists of process trajectories
        :type dataset: List[List[np.ndarray]]
        :param choices: List of agent choices
        :type choices: List[int]
        :return: Negative log-likelihood value
        :rtype: float
        """
        total_nll = 0
        for data, choice in zip(dataset, choices):
            expected_utilities = []
            for process_data in data:
                final_values = process_data[:, -1]  # Final values of trajectories
                utilities = utility_func(final_values, *params)
                expected_utility = np.mean(utilities)
                expected_utilities.append(expected_utility)
            expected_utilities = np.array(expected_utilities)
            # Softmax over expected utilities
            exp_utilities = np.exp(expected_utilities - np.max(expected_utilities))
            probs = exp_utilities / np.sum(exp_utilities)
            total_nll -= np.log(probs[choice])
        return total_nll

    def fit_utility_functions(self, dataset: List[List[np.ndarray]], choices: List[int]):
        """
        Fit the utility functions to the observed choices using maximum likelihood estimation (MLE),
        assuming the agent maximizes expected utility.

        :param dataset: List of datasets, each containing lists of process trajectories
        :type dataset: List[List[np.ndarray]]
        :param choices: List of agent choices
        :type choices: List[int]
        :return: None
        :rtype: None
        """
        for utility_function in self.utility_functions:
            res = minimize(
                self.negative_log_likelihood,
                x0=utility_function.initial_params,
                args=(utility_function.func, dataset, choices),
                method='Nelder-Mead',
                options={'maxiter': 1000}
            )
            utility_function.fitted_params = res.x
            utility_function.nll = res.fun

    def print_results(self):
        """
        Print the fitted utility functions and their parameters, including negative log-likelihood.

        :return: None
        :rtype: None
        """
        for utility_function in self.utility_functions:
            print(f"{utility_function.name} utility function:")
            print(f"  Parameters: {', '.join([f'{p:.4f}' for p in utility_function.fitted_params])}")
            print(f"  Negative log-likelihood: {utility_function.nll:.4f}")
            print()

    def plot_utility_functions(self, x_range: Tuple[float, float] = (0, 0.5)):
        """
        Plot the fitted utility functions for visualization.

        :param x_range: Range of x values to plot
        :type x_range: Tuple[float, float]
        :return: None
        :rtype: None
        """
        x = np.linspace(x_range[0], x_range[1], 100)
        plt.figure(figsize=(12, 8))
        for utility_function in self.utility_functions:
            y = [utility_function(xi) for xi in x]
            plt.plot(x, y, label=utility_function.name)
        plt.xlabel('Process value')
        plt.ylabel('Utility')
        plt.title('Fitted Utility Functions')
        plt.legend()
        plt.grid(True)
        plt.show()

    def bayesian_fit_utility_functions(self, dataset: List[List[np.ndarray]], choices: List[int],
                                       n_samples: int = 10000, burn_in: int = 1000):
        """
        Perform Bayesian inference on utility functions using Metropolis-Hastings sampling.
        It generates samples from the posterior distribution of the utility function parameters.
        Provides a distribution of parameter values instead of a single point estimate.

        :param dataset: List of datasets, each containing lists of process trajectories
        :type dataset: List[List[np.ndarray]]
        :param choices: List of agent choices
        :type choices: List[int]
        :param n_samples: Number of MCMC samples to generate
        :type n_samples: int
        :param burn_in: Number of burn-in samples to discard
        :type burn_in: int
        :return: None
        :rtype: None
        """
        # Store dataset and choices as instance variables
        self.dataset = dataset
        self.choices = choices

        for utility_function in self.utility_functions:
            samples = self.metropolis_hastings(utility_function, dataset, choices, n_samples, burn_in)
            self.mcmc_samples[utility_function.name] = samples

    def metropolis_hastings(self, utility_function, dataset: List[List[np.ndarray]], choices: List[int],
                            n_samples: int, burn_in: int) -> np.ndarray:
        """
        Perform Metropolis-Hastings sampling for Bayesian inference on utility functions.
        Generates samples from the posterior distribution of the utility function parameters.

        :param utility_function: Utility function to fit
        :type utility_function: UtilityFunction
        :param dataset: List of datasets, each containing lists of process trajectories
        :type dataset: List[List[np.ndarray]]
        :param choices: List of agent choices
        :type choices: List[int]
        :param n_samples: Number of samples to generate
        :type n_samples: int
        :param burn_in: Number of burn-in samples to discard
        :type burn_in: int
        :return: Array of MCMC samples
        :rtype: np.ndarray
        """
        current_params = np.array(utility_function.initial_params)
        n_params = len(current_params)
        samples = np.zeros((n_samples, n_params))

        for i in range(n_samples + burn_in):
            # Propose new parameters
            proposal_params = current_params + norm.rvs(scale=0.1, size=n_params)

            # Calculate log likelihoods
            current_nll = self.negative_log_likelihood(current_params, utility_function.func, dataset, choices)
            proposal_nll = self.negative_log_likelihood(proposal_params, utility_function.func, dataset, choices)

            # Calculate acceptance probability
            acceptance_prob = np.exp(-(proposal_nll - current_nll))

            # Accept or reject the new parameters
            if np.random.rand() < acceptance_prob:
                current_params = proposal_params

            # Store the sample after burn-in period
            if i >= burn_in:
                samples[i - burn_in] = current_params

        return samples

    def print_bayesian_results(self):
        """
        Print the mean and standard deviation of the fitted parameters from Bayesian inference.

        :return:    None
        """
        for name, samples in self.mcmc_samples.items():
            mean_params = np.mean(samples, axis=0)
            std_params = np.std(samples, axis=0)
            print(f"{name} utility function:")
            for i, (mean, std) in enumerate(zip(mean_params, std_params)):
                print(f"  Parameter {i + 1}: Mean = {mean:.4f}, Std = {std:.4f}")
            print()

    def plot_bayesian_results(self, x_range: Tuple[float, float] = None):
        """
        Plot the fitted utility functions based on Bayesian inference.

        :param x_range: Range of x values to plot. If None, it will be determined based on the data.
        :type x_range: Tuple[float, float] or None
        :return: None
        :rtype: None
        """
        if x_range is None:
            # Determine x_range based on the final values in the dataset
            all_final_values = []
            for data in self.dataset:
                for process_data in data:
                    final_values = process_data[:, -1]
                    all_final_values.extend(final_values)
            x_min, x_max = np.min(all_final_values), np.max(all_final_values)
            x_range = (x_min, x_max)

        x = np.linspace(x_range[0], x_range[1], 100)
        plt.figure(figsize=(12, 8))

        for name, samples in self.mcmc_samples.items():
            utility_function = next(uf for uf in self.utility_functions if uf.name == name)
            mean_params = np.mean(samples, axis=0)

            # Plot mean function
            y_mean = [utility_function.func(xi, *mean_params) for xi in x]
            plt.plot(x, y_mean, label=f"{name} (Mean)")

            # Plot credible intervals
            y_samples = np.array([[utility_function.func(xi, *params) for xi in x] for params in samples])
            y_5 = np.percentile(y_samples, 5, axis=0)
            y_95 = np.percentile(y_samples, 95, axis=0)
            plt.fill_between(x, y_5, y_95, alpha=0.3)

        plt.xlabel('Process Value')
        plt.ylabel('Utility')
        plt.title('Fitted Utility Functions (Bayesian Inference)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_parameter_distributions(self):
        """
        Plot the distributions of fitted parameters from Bayesian inference.

        :return: None
        :rtype: None
        """
        n_functions = len(self.mcmc_samples)
        fig, axes = plt.subplots(n_functions, 1, figsize=(10, 5 * n_functions), squeeze=False)

        for i, (name, samples) in enumerate(self.mcmc_samples.items()):
            ax = axes[i, 0]
            for j in range(samples.shape[1]):
                ax.hist(samples[:, j], bins=50, alpha=0.5, label=f'Param {j + 1}')
            ax.set_title(f'{name} Parameter Distributions')
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Frequency')
            ax.legend()

        plt.tight_layout()
        plt.show()

    def regression_fit(self, n_processes: int = 10000, n_options: int = 2, test_size: float = 0.2, epochs: int = 100,
                       batch_size: int = 32):
        """
        Train a regression model to predict agent preferences based on process parameters.
        The model is trained using a dataset of stochastic processes and agent choices,
        assuming the agent maximizes expected utility.

        :param n_processes: Number of decision instances to generate for training
        :type n_processes: int
        :param n_options: Number of process options per decision instance
        :type n_options: int
        :param test_size: Fraction of data to use for testing
        :type test_size: float
        :param epochs: Number of training epochs
        :type epochs: int
        :param batch_size: Batch size for training
        :type batch_size: int
        :raises ValueError: If input parameters are invalid
        :return: None
        :rtype: None
        """
        # Input Validation
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        if epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        # Generate dataset and get agent choices
        dataset, choices = self.generate_choices(n_options=n_options, n_choices=n_processes)

        # Assign dataset and choices to instance variables for potential use in plotting
        self.dataset = dataset
        self.choices = choices

        # Prepare input data (process type code and parameters) and output data (choice)
        X = []
        y = []
        for data, choice in zip(dataset, choices):
            for idx, process_data in enumerate(data):
                # Extract process_type_code and parameters from the first row
                first_row = process_data[0]
                process_type_code = first_row[0]
                process_type = self.get_process_type(process_type_code)
                num_params = len(self.param_ranges[process_type])
                params_values = first_row[1:1 + num_params]
                process_params = list(params_values)

                # Create feature vector: [process_type_code, param1, param2, ...]
                feature_vector = np.concatenate(([process_type_code], params_values))
                X.append(feature_vector)

                # Create label: 1 if this process option is the chosen one, else 0
                label = 1 if idx == choice else 0
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Print shapes for debugging
        print(f"X shape: {X.shape}")  # Should be (n_processes * n_options, num_features)
        print(f"y shape: {y.shape}")  # Should be (n_processes * n_options,)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Print final shapes for debugging
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

        # Create and compile the regression model
        self.regression_model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        self.regression_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the regression model
        self.regression_history = self.regression_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )

    def plot_regression_results(self):
        """
        Plot the results of the regression model training.
        It includes accuracy and loss curves for both training and validation sets.

        :return: None
        :rtype: None
        """
        if self.regression_history is None:
            print("No regression results to plot. Run regression_fit() first.")
            return

        plt.figure(figsize=(12, 4))

        # Plot training & validation accuracy values
        plt.subplot(121)
        plt.plot(self.regression_history.history['accuracy'])
        plt.plot(self.regression_history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(122)
        plt.plot(self.regression_history.history['loss'])
        plt.plot(self.regression_history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.tight_layout()
        plt.show()

    def predict_preference(self, process_type: str, params: Dict[str, float]) -> float:
        """
        Predict the agent's preference for a given process type and parameters.
        It is done by encoding the process and passing it through the regression model.

        :param process_type: Type of the stochastic process
        :type process_type: str
        :param params: Parameters of the stochastic process
        :type params: Dict[str, float]
        :return: Predicted preference value
        :rtype: float
        """
        if self.regression_model is None:
            raise ValueError("Regression model not trained. Run regression_fit() first.")

        # Encode the process type and parameters
        encoded_process = self.process_encoder.encode_process(globals()[process_type](**params))

        # Make prediction
        preference = self.regression_model.predict(np.array([encoded_process]))[0][0]
        return float(preference)

    def plot_preference_heatmap(self, process_type: str, param1: str, param2: str, n_points: int = 20):
        """
        Plot a heatmap of agent preferences for different parameter values of a process type.
        The heatmap shows how the agent's preference changes with different parameter combinations.
        It allows visualizing the utility landscape for the agent.

        :param process_type: Type of the stochastic process
        :type process_type: str
        :param param1: First parameter to vary
        :type param1: str
        :param param2: Second parameter to vary
        :type param2: str
        :param n_points: Number of points to sample for each parameter
        :type n_points: int
        :return: None
        :rtype: None
        """
        if self.regression_model is None:
            raise ValueError("Regression model not trained. Run regression_fit() first.")

        # Create grid of parameter values
        param1_range = np.linspace(*self.param_ranges[process_type][param1], n_points)
        param2_range = np.linspace(*self.param_ranges[process_type][param2], n_points)
        param1_grid, param2_grid = np.meshgrid(param1_range, param2_range)

        # Compute preferences for each point in the grid
        preferences = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                params = {param1: param1_grid[i, j], param2: param2_grid[i, j]}
                preferences[i, j] = self.predict_preference(process_type, params)

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(preferences, origin='lower',
                   extent=[param1_range[0], param1_range[-1], param2_range[0], param2_range[-1]])
        plt.colorbar(label='Preference')
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f'Preference Heatmap for {process_type}')
        plt.show()

    def analyze_feature_importance(self):
        """
        Analyze the importance of different features in the regression model.
        The method calculates the feature importance based on the weights of the first layer of the model.
        It helps understand which features have the most influence on the agent's decision-making process.

        :return: Dictionary of feature names and their importance scores
        :rtype: Dict[str, float]
        """
        if self.regression_model is None:
            raise ValueError("Regression model not trained. Run regression_fit() first.")

        # Get the weights of the first layer
        first_layer_weights = self.regression_model.layers[0].get_weights()[0]

        # Calculate the absolute mean of weights for each feature
        feature_importance = np.mean(np.abs(first_layer_weights), axis=1)

        # Normalize the importance scores
        feature_importance = feature_importance / np.sum(feature_importance)

        # Get feature names
        feature_names = ['Process Type'] + list(self.param_ranges[list(self.param_ranges.keys())[0]].keys())

        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importance = feature_importance[sorted_idx]

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features, sorted_importance)
        plt.xlabel('Normalized Importance')
        plt.title('Feature Importance Analysis')
        plt.tight_layout()
        plt.show()

        # Print feature importance
        print("Feature Importance:")
        for feature, importance in zip(sorted_features, sorted_importance):
            print(f"{feature}: {importance:.4f}")

        return dict(zip(sorted_features, sorted_importance))

    def analyze_feature_interactions(self):
        """
        Analyze the interaction strength between features in the regression model.
        It is done by calculating the dot product of the weights of the first layer.
        The resulting matrix shows how features interact with each other in the decision-making process.

        :return: Interaction strength matrix
        :rtype: np.ndarray
        """
        if self.regression_model is None:
            raise ValueError("Regression model not trained. Run regression_fit() first.")

        # Get the weights of the first layer
        first_layer_weights = self.regression_model.layers[0].get_weights()[0]

        # Calculate the interaction strength between features
        interaction_strength = np.dot(first_layer_weights.T, first_layer_weights)

        # Get feature names
        feature_names = ['Process Type'] + list(self.param_ranges[list(self.param_ranges.keys())[0]].keys())

        # Plot interaction heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(interaction_strength, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Interaction Strength')
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
        plt.yticks(range(len(feature_names)), feature_names)
        plt.title('Feature Interaction Analysis')
        plt.tight_layout()
        plt.show()

        return interaction_strength

    def plot_partial_dependence(self, feature_index, num_points=100):
        """
        Plot the partial dependence of the predicted preference on a selected feature.
        The partial dependence shows how the predicted preference changes with variations in a single feature.
        It helps understand the relationship between the feature and the agent's decision-making process.
        It is calculated by fixing all other features at a reference point and varying the selected feature.

        :param feature_index: Index of the feature to analyze
        :type feature_index: int
        :param num_points: Number of points to sample for the selected feature
        :type num_points: int
        :return: None
        :rtype: None
        """
        if self.regression_model is None:
            raise ValueError("Regression model not trained. Run regression_fit() first.")

        feature_names = ['Process Type'] + list(self.param_ranges[list(self.param_ranges.keys())[0]].keys())
        feature_name = feature_names[feature_index]

        # Generate a range of values for the selected feature
        if feature_index == 0:  # Process Type
            x_range = np.arange(len(self.param_ranges))
        else:
            param_name = list(self.param_ranges[list(self.param_ranges.keys())[0]].keys())[feature_index - 1]
            param_range = self.param_ranges[list(self.param_ranges.keys())[0]][param_name]
            x_range = np.linspace(param_range[0], param_range[1], num_points)

        # Create a copy of the first data point and vary only the selected feature
        base_input = self.regression_model.layers[0].get_weights()[0][:, 0].reshape(1, -1)
        predictions = []

        for value in x_range:
            input_data = base_input.copy()
            input_data[0, feature_index] = value
            prediction = self.regression_model.predict(input_data)[0, 0]
            predictions.append(prediction)

        # Plot partial dependence
        plt.figure(figsize=(10, 6))
        plt.plot(x_range, predictions)
        plt.xlabel(feature_name)
        plt.ylabel('Predicted Preference')
        plt.title(f'Partial Dependence Plot for {feature_name}')
        plt.tight_layout()
        plt.show()

    def perform_irl(self, n_processes=1000, n_options=2):
        """
        Perform Inverse Reinforcement Learning (IRL) to infer the reward function from agent choices.
        It uses the MaxEntIRL algorithm to learn the reward weights based on agent behavior.
        It is designed to understand the underlying reward structure that drives the agent's decisions.

        :param n_processes: Number of decision instances to generate for IRL
        :type n_processes: int
        :param n_options: Number of process options per decision instance
        :type n_options: int
        :return: Inferred reward weights
        :rtype: np.ndarray
        """

        # Generate dataset and agent choices
        dataset, choices = self.generate_choices(n_options=n_options, n_choices=n_processes)

        # Assign dataset and choices to instance variables for potential use in plotting
        self.dataset = dataset
        self.choices = choices

        # Prepare trajectories for IRL
        trajectories = []
        for data, choice in zip(dataset, choices):
            trajectory = []
            for idx, process_data in enumerate(data):
                # Extract process features from the first trajectory (all trajectories have the same features)
                first_row = process_data[0]
                process_type_code = first_row[0]
                process_type = self.get_process_type(process_type_code)
                num_params = len(self.param_ranges[process_type])
                params_values = first_row[1:1 + num_params]
                state = np.concatenate(([process_type_code], params_values))

                # Define action: 1 if chosen, 0 otherwise
                action = 1 if idx == choice else 0

                # Append the (state, action) pair to the trajectory
                trajectory.append((state, action))
            trajectories.append(trajectory)

        # Initialize and fit IRL model
        # Number of features corresponds to the length of the state vector
        n_features = len(trajectories[0][0][0])
        # Number of actions: binary (chosen or not chosen)
        n_actions = 2

        # Initialize MaxEntIRL model
        irl_model = MaxEntIRL(n_features, n_actions)

        # Fit the IRL model using the prepared trajectories
        reward_weights = irl_model.fit(trajectories)

        # Print and plot results
        print("Inferred Reward Weights:")
        # Feature names: ['Process Type', 'Param1', 'Param2', ...]
        process_types = list(self.param_ranges.keys())
        feature_names = ['Process Type'] + list(self.param_ranges[process_types[0]].keys())
        for name, weight in zip(feature_names, reward_weights):
            print(f"{name}: {weight:.4f}")

        # Plot the inferred reward weights
        plt.figure(figsize=(10, 6))
        plt.bar(feature_names, reward_weights)
        plt.title("Inferred Reward Weights")
        plt.xlabel("Features")
        plt.ylabel("Weight")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        return reward_weights

class MaxEntIRL:
    """
    MaxEntIRL (Maximum Entropy Inverse Reinforcement Learning) Class

    This class implements the Maximum Entropy Inverse Reinforcement Learning algorithm, which aims to
    recover the underlying reward function from observed optimal behavior in a Markov Decision Process (MDP).

    Attributes:

        n_features (int): Number of features in the state space.

        n_actions (int): Number of possible actions in the MDP.

        learning_rate (float): Learning rate for the optimization process.

        n_iterations (int): Number of iterations for the optimization process.

        reward_weights (np.ndarray): Weights representing the reward function.

    The MaxEntIRL class implements the core algorithm of Maximum Entropy Inverse Reinforcement Learning.
    It aims to find a reward function that makes the observed behavior appear near-optimal. The algorithm
    works by iteratively updating the reward weights to maximize the likelihood of the observed trajectories
    under the maximum entropy distribution.

    Key aspects of the implementation:

    1. It uses feature expectations to characterize the observed behavior.

    2. It computes state visitation frequencies to understand the importance of different states.

    3. The optimization process uses the L-BFGS-B algorithm to find the optimal reward weights.

    4. The resulting policy is computed using a softmax over Q-values.

    This implementation is particularly useful in scenarios where we want to understand the underlying
    motivations or rewards that drive observed behavior, such as in robotics, autonomous systems,
    or behavioral economics.

    Usage:

        irl_model = MaxEntIRL(n_features=5, n_actions=3)

        trajectories = [...]  # List of observed state-action trajectories

        learned_rewards = irl_model.fit(trajectories)

        # Predict reward for a new state

        new_state = np.array([...])

        predicted_reward = irl_model.predict_reward(new_state)

    Note: This implementation assumes discrete state and action spaces and may require modifications
    for continuous domains or large-scale problems.
    """
    def __init__(self, n_features, n_actions, learning_rate=0.01, n_iterations=100):
        """
        Initialize the MaxEntIRL model with the given parameters.

        :param n_features: Number of features in the state space
        :type n_features: int
        :param n_actions: Number of actions in the action space
        :type n_actions: int
        :param learning_rate: Learning rate for optimization
        :type learning_rate: float
        :param n_iterations: Number of iterations for optimization
        :type n_iterations: int
        """
        self.n_features = n_features
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reward_weights = np.random.rand(n_features)

    def feature_expectations(self, trajectories):
        """
        Compute empirical feature expectations.
        It is the average feature value observed in the given trajectories.

        :param trajectories: List of observed state-action trajectories
        :type trajectories: List[List[Tuple[np.ndarray, int]]]
        :return: Empirical feature expectations
        :rtype: np.ndarray
        """
        feature_exp = np.zeros(self.n_features)
        for trajectory in trajectories:
            for state, action in trajectory:
                feature_exp += state
        return feature_exp / len(trajectories)

    def compute_state_visitation_freq(self, trajectories, policy):
        """
        Compute state visitation frequencies.
        It calculates the frequency of visiting each state under a given policy.

        :param trajectories: List of observed state-action trajectories
        :type trajectories: List[List[Tuple[np.ndarray, int]]]
        :param policy: Policy function mapping states to action probabilities
        :type policy: Callable
        :return: State visitation frequencies
        :rtype: np.ndarray
        """
        n_states = len(trajectories[0])
        t_matrix = np.zeros((n_states, n_states))
        for trajectory in trajectories:
            for i in range(n_states - 1):
                t_matrix[i, i + 1] += 1
        t_matrix /= len(trajectories)

        state_freq = np.zeros(n_states)
        state_freq[0] = 1.0
        for _ in range(n_states):
            state_freq = state_freq.dot(t_matrix)
        return state_freq

    def compute_expected_svf(self, trajectories, policy):
        """
        Compute expected state visitation frequency.
        It estimates the expected frequency of visiting each state under the given policy.

        :param trajectories: List of observed state-action trajectories
        :type trajectories: List[List[Tuple[np.ndarray, int]]]
        :param policy: Policy function mapping states to action probabilities
        :type policy: Callable
        :return: Expected state visitation frequency
        :rtype: np.ndarray
        """
        svf = self.compute_state_visitation_freq(trajectories, policy)
        exp_svf = np.zeros(self.n_features)
        for trajectory in trajectories:
            for i, (state, _) in enumerate(trajectory):
                exp_svf += state * svf[i]
        return exp_svf / len(trajectories)

    def compute_gradient(self, feat_exp, exp_svf):
        """
        Compute the gradient for the optimization.
        It represents the difference between empirical feature expectations and expected state visitation frequency.

        :param feat_exp: Empirical feature expectations
        :type feat_exp: np.ndarray
        :param exp_svf: Expected state visitation frequency
        :type exp_svf: np.ndarray
        :return: Gradient vector
        :rtype: np.ndarray
        """
        return feat_exp - exp_svf

    def optimize_reward(self, trajectories):
        """
        Optimize the reward function.
        It uses the L-BFGS-B algorithm to find the optimal reward weights.
        The result is a reward function that explains the observed behavior.

        :param trajectories: List of observed state-action trajectories
        :type trajectories: List[List[Tuple[np.ndarray, int]]]
        :return: None
        :rtype: None
        """
        feat_exp = self.feature_expectations(trajectories)

        def obj_func(weights):
            self.reward_weights = weights
            policy = self.compute_policy(trajectories)
            exp_svf = self.compute_expected_svf(trajectories, policy)
            return -np.dot(weights, feat_exp - exp_svf) + np.sum(np.log(policy))

        result = minimize(obj_func, self.reward_weights, method='L-BFGS-B')
        self.reward_weights = result.x

    def compute_policy(self, trajectories):
        """
        Compute the policy based on current reward weights.
        It uses a softmax over Q-values to derive the action probabilities.
        The result is a policy that explains the observed behavior.

        :param trajectories: List of observed state-action trajectories
        :type trajectories: List[List[Tuple[np.ndarray, int]]]
        :return: Computed policy
        :rtype: np.ndarray
        """
        policy = np.zeros((len(trajectories[0]), self.n_actions))
        for i, (state, _) in enumerate(trajectories[0]):
            q_values = np.dot(state, self.reward_weights)
            policy[i] = np.exp(q_values) / np.sum(np.exp(q_values))
        return policy

    def fit(self, trajectories):
        """
        Fit the IRL model to the given trajectories.
        The method allows to learn the reward function that explains the observed behavior.

        :param trajectories: List of observed state-action trajectories
        :type trajectories: List[List[Tuple[np.ndarray, int]]]
        :return: Inferred reward
        :rtype: np.ndarray
        """
        for _ in range(self.n_iterations):
            self.optimize_reward(trajectories)
        return self.reward_weights

    def predict_reward(self, state):
        """
        Predict the reward for a given state.
        It uses the learned reward weights to estimate the reward value.

        :param state: Input state for reward prediction
        :type state: np.ndarray
        :return: Predicted reward value
        :rtype: float
        """
        return np.dot(state, self.reward_weights)

# Utility function definitions
def utility_power(x: float, beta: float) -> float:
    """
    Power utility function.
    The expression is: U(x) = x^beta

    :param x: Input value
    :type x: float
    :param beta: Utility parameter
    :type beta: float
    :return: Utility value
    :rtype: float
    """
    return x ** beta

def utility_exp(x: float, alpha: float) -> float:
    """
    Exponential utility function.
    The expression is: U(x) = 1 - exp(-alpha * x)

    :param x: Input value
    :type x: float
    :param alpha: Utility parameter
    :type alpha: float
    :return: Utility value
    :rtype: float
    """
    return 1 - np.exp(-alpha * x)

def utility_log(x: float, gamma: float) -> float:
    """
    Logarithmic utility function.
    The expression is: U(x) = log(1 + gamma * x)

    :param x: Input value
    :type x: float
    :param gamma: Utility parameter
    :type gamma: float
    :return: Utility value
    :rtype: float
    """
    return np.log(1 + gamma * x)

def utility_quadratic(x: float, a: float, b: float) -> float:
    """
    Quadratic utility function.
    The expression is: U(x) = a * x - b * x^2

    :param x: Input value
    :type x: float
    :param a: Utility parameter corresponding to linear term
    :type a: float
    :param b: Utility parameter corresponding to quadratic term
    :type b: float
    :return: Utility value
    :rtype: float
    """
    return a * x - b * x ** 2

def utility_arctan(x: float, k: float) -> float:
    """
    Arctan utility function.
    The expression is: U(x) = arctan(k * x)

    :param x: Input value
    :type x: float
    :param k: Utility parameter
    :type k: float
    :return: Utility value
    :rtype: float
    """
    return np.arctan(k * x)

def utility_sigmoid(x: float, k: float, x0: float) -> float:
    """
    Sigmoid utility function.
    The expression is: U(x) = 1 / (1 + exp(-k * (x - x0)))

    :param x: Input value
    :type x: float
    :param k: Utility parameter controlling slope
    :type k: float
    :param x0: Utility parameter controlling inflection point
    :type x0: float
    :return: Utility value
    :rtype: float
    """
    return 1 / (1 + np.exp(-k * (x - x0)))

def utility_linear_threshold(x: float, a: float, b: float) -> float:
    """
    Linear threshold utility function.
    The expression is: U(x) = max(0, a * x - b)

    :param x: Input value
    :type x: float
    :param a: Utility parameter corresponding to linear term
    :type a: float
    :param b: Utility parameter   corresponding to threshold
    :type b: float
    :return: Utility value
    :rtype: float
    """
    return np.maximum(0, a * x - b)

def utility_cobb_douglas(x: float, alpha: float, beta: float) -> float:
    """
    Cobb-Douglas utility function.
    The expression is: U(x) = x^alpha * (1 - x)^beta

    :param x: Input value
    :type x: float
    :param alpha: Utility parameter corresponding to first term
    :type alpha: float
    :param beta: Utility parameter corresponding to second term
    :type beta: float
    :return: Utility value
    :rtype: float
    """
    return x ** alpha * (1 - x) ** beta

def utility_prospect_theory(x: float, alpha: float, lambda_: float) -> float:
    """
    Prospect Theory utility function.

    :param x: Input value
    :type x: float
    :param alpha: Utility parameter corresponding to loss aversion
    :type alpha: float
    :param lambda_: Utility parameter corresponding to risk-seeking behavior
    :type lambda_: float
    :return: Utility value
    :rtype: float
    """
    return np.where(x >= 0, x ** alpha, -lambda_ * (-x) ** alpha)


if __name__=='__main__':

    # Initialize the UtilityFunctionInference with the path to your trained model
    ufi = UtilityFunctionInference('path/to/your/model.h5')

    # Add utility functions to be considered
    ufi.add_utility_function(UtilityFunction('Power', utility_power, [1.0]))
    ufi.add_utility_function(UtilityFunction('Exponential', utility_exp, [1.0]))
    ufi.add_utility_function(UtilityFunction('Logarithmic', utility_log, [1.0]))
    ufi.add_utility_function(UtilityFunction('Quadratic', utility_quadratic, [1.0, 0.5]))
    ufi.add_utility_function(UtilityFunction('Arctan', utility_arctan, [1.0]))
    ufi.add_utility_function(UtilityFunction('Sigmoid', utility_sigmoid, [10.0, 0.25]))
    ufi.add_utility_function(UtilityFunction('Linear Threshold', utility_linear_threshold, [1.0, 0.1]))
    ufi.add_utility_function(UtilityFunction('Cobb-Douglas', utility_cobb_douglas, [0.5, 0.5]))
    ufi.add_utility_function(UtilityFunction('Prospect Theory', utility_prospect_theory, [0.88, 2.25]))

    # Define processes and parameter ranges
    processes = [
        {'type': 'BrownianMotion'},
        {'type': 'GeometricBrownianMotion'},
        # Add more process types here as needed
    ]
    param_ranges = {
        'BrownianMotion': {'mu': (0, 0.5), 'sigma': (0.1, 0.5)},
        'GeometricBrownianMotion': {'mu': (0, 0.5), 'sigma': (0.1, 0.5)},
        # Add parameter ranges for other process types here
    }

    # Generate dataset
    dataset = ufi.generate_dataset(processes, param_ranges, num_instances=10, n_simulations=100)

    # Get agent choices
    choices = ufi.get_agent_choices(dataset)

    # Fit utility functions
    ufi.fit_utility_functions(dataset, choices)

    # Print results
    ufi.print_results()

    # Plot utility functions
    ufi.plot_utility_functions()

    # You can also access individual fitted utility functions
    best_utility_function = min(ufi.utility_functions, key=lambda uf: uf.nll)
    print(f"The best-fitting utility function is: {best_utility_function.name}")
    print(f"With parameters: {best_utility_function.fitted_params}")

    # Use the best-fitting utility function
    x = 0.3  # Example input
    utility = best_utility_function(x)
    print(f"The utility of {x} according to the best-fitting function is: {utility}")

class UtilityFunctionTester:
    """
    UtilityFunctionTester Class

    This class is designed to test and analyze various utility functions against stochastic processes.
    It provides tools for generating process parameters, simulating processes, optimizing utility
    functions, and analyzing the results through statistical and visual methods.

    Attributes:

        process_class: The class of the stochastic process to be tested.

        param_ranges (Dict[str, Tuple[float, float]]): Ranges for each parameter of the process.

        utility_functions (Dict[str, Callable]): Dictionary of utility functions to be tested.

        results (List): Stores the results of the tests.

    This class provides a comprehensive framework for evaluating and comparing different utility
    functions in the context of stochastic processes. It is particularly useful for researchers
    and practitioners in fields such as economics, finance, and decision theory, where
    understanding the performance of utility functions under various stochastic conditions is crucial.

    Key features:

    1. Automatic generation of process parameters for comprehensive testing.

    2. Parallel processing capability for efficient large-scale testing.

    3. Advanced statistical analysis including correlation, PCA, and clustering.

    4. Visualization tools for intuitive interpretation of results.

    Usage:

        process_class = BrownianMotion

        param_ranges = {'mu': (0, 0.5), 'sigma': (0.1, 0.5)}

        utility_functions = {
            'power': lambda x, beta: x ** beta,
            'exponential': lambda x, alpha: 1 - np.exp(-alpha * x),
            'logarithmic': lambda x, gamma: np.log(1 + gamma * x)
        }

        tester = UtilityFunctionTester(process_class, param_ranges, utility_functions)

        tester.run_tests(n_processes=1000, n_steps=1000)

        tester.analyze_results()

        tester.plot_optimal_utility_vs_process_params()

    This class enables researchers to gain insights into how different utility functions perform
    under various stochastic process conditions, helping in the selection and refinement of
    utility models for specific applications.
    """
    def __init__(self, process_class, param_ranges: Dict[str, Tuple[float, float]],
                 utility_functions: Dict[str, Callable]):
        """
        Initialize the UtilityFunctionTester with the given parameters.

        :param process_class: Class representing the stochastic process to simulate
        :type: process_class: type
        :param param_ranges: Dictionary of parameter ranges for the process
        :type param_ranges: Dict[str, Tuple[float, float]]
        :param utility_functions: Dictionary of utility functions to test
        :type utility_functions: Dict[str, Callable]
        """
        self.process_class = process_class
        self.param_ranges = param_ranges
        self.utility_functions = utility_functions
        self.results = []

    def generate_process_parameters(self, n_samples: int) -> List[Dict[str, float]]:
        """
        Generate random process parameters within specified ranges.

        :param n_samples: Number of parameter sets to generate
        :type n_samples: int
        :return: List of process parameters
        :rtype: List[Dict[str, float
        """
        params_list = []
        for _ in range(n_samples):
            params = {param: np.random.uniform(low, high)
                      for param, (low, high) in self.param_ranges.items()}
            params_list.append(params)
        return params_list

    def simulate_process(self, params: Dict[str, float], n_steps: int) -> np.ndarray:
        """
        Simulate the stochastic process with given parameters.

        :param params: Parameters of the process
        :type params: Dict[str, float]
        :param n_steps: Number of steps to simulate
        :type n_steps: int
        :return: Trajectory of the process
        :rtype: np.ndarray
        """
        process = self.process_class(**params)
        return process.simulate(n_steps)

    def calculate_utility(self, utility_func: Callable, trajectory: np.ndarray, utility_params: List[float]) -> float:
        """
        Calculate the utility of a given stochastic process instance using a utility function.

        :param utility_func: Utility function to evaluate
        :type utility_func: Callable
        :param trajectory: Trajectory of the stochastic process
        :type trajectory: np.ndarray
        :param utility_params: Parameters of the utility function
        :type utility_params: List[float]
        :return: Utility value
        """
        return np.mean([utility_func(x, *utility_params) for x in trajectory])

    def optimize_utility_function(self, utility_func: Callable, trajectory: np.ndarray) -> Tuple[List[float], float]:
        """
        Optimize the parameters of a utility function for a given trajectory.
        It is done by maximizing the utility value using numerical optimization.

        :param utility_func: Utility function to optimize
        :type utility_func: Callable
        :param trajectory: Trajectory of the stochastic process
        :type trajectory: np.ndarray
        :return: Tuple of optimal parameters and maximum utility value
        :rtype: Tuple[List[float], float]
        """
        def objective(params):
            return -self.calculate_utility(utility_func, trajectory, params)

        initial_guess = [1.0] * len(signature(utility_func).parameters[1:])
        result = minimize(objective, initial_guess, method='Nelder-Mead')
        return result.x, -result.fun

    def test_utility_functions(self, process_params: Dict[str, float], n_steps: int) -> Dict:
        """
        Test all utility functions against a simulated process trajectory.
        It evaluates each utility function's performance and optimizes its parameters.

        :param process_params:
        :type process_params: Dict[str, float]
        :param n_steps:
        :type n_steps: int
        :return:    Dictionary of results for each utility function
        :rtype: Dict
        """
        trajectory = self.simulate_process(process_params, n_steps)
        results = {'process_params': process_params}

        for name, func in self.utility_functions.items():
            optimal_params, optimal_utility = self.optimize_utility_function(func, trajectory)
            results[f'{name}_params'] = optimal_params
            results[f'{name}_utility'] = optimal_utility

        return results

    def run_tests(self, n_processes: int, n_steps: int, n_jobs: int = 1):
        """
        Run tests for multiple processes in parallel.

        :param n_processes: Number of processes to test
        :type n_processes: int
        :param n_steps: Number of steps to simulate for each process
        :type n_steps: int
        :param n_jobs: Number of parallel jobs to run (-1 for all cores)
        :type n_jobs: int
        :return: None
        :rtype: None
        """
        process_params_list = self.generate_process_parameters(n_processes)

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            self.results = list(executor.map(lambda params: self.test_utility_functions(params, n_steps),
                                             process_params_list))

    def analyze_results(self):
        """
        Perform statistical analysis on the test results for all utility functions.
        It includes correlation analysis, PCA, and clustering to understand the relationships.
        The results are aimed to demonstrate the performance and characteristics of each utility function.

        :return: None
        :rtype: None
        """
        df = pd.DataFrame(self.results)

        # Correlation analysis
        correlation_matrix = df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation between Process Parameters and Utility Function Parameters')
        plt.tight_layout()
        plt.show()

        # PCA analysis
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df.drop('process_params', axis=1))

        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['process_params'].apply(lambda x: x['mu']))
        plt.colorbar(label='Process mu')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA of Utility Function Parameters')
        plt.show()

        # Clustering analysis
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(df.drop('process_params', axis=1))

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters)
        plt.legend(*scatter.legend_elements(), title='Clusters')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Clustering of Utility Function Parameters')
        plt.show()

    def plot_optimal_utility_vs_process_params(self):
        """
        Plot the relationship between process parameters and optimal utility values.
        It visualizes how the utility functions perform under different process conditions.

        :return: None
        :rtype: None
        """
        df = pd.DataFrame(self.results)

        for param in self.param_ranges.keys():
            plt.figure(figsize=(12, 6))
            for name in self.utility_functions.keys():
                plt.scatter(df['process_params'].apply(lambda x: x[param]),
                            df[f'{name}_utility'],
                            alpha=0.5,
                            label=name)
            plt.xlabel(f'Process {param}')
            plt.ylabel('Optimal Utility')
            plt.title(f'Optimal Utility vs Process {param}')
            plt.legend()
            plt.show()

if __name__=='__main__':

# Usage example:
    from ergodicity.process.basic import BrownianMotion


    def utility_power(x, beta):
       return x ** beta


    def utility_exp(x, alpha):
        return 1 - np.exp(-alpha * x)


    def utility_log(x, gamma):
        return np.log(1 + gamma * x)


    process_class = BrownianMotion
    param_ranges = {'mu': (0, 0.5), 'sigma': (0.1, 0.5)}
    utility_functions = {
        'power': utility_power,
        'exponential': utility_exp,
        'logarithmic': utility_log
    }

    tester = UtilityFunctionTester(process_class, param_ranges, utility_functions)
    tester.run_tests(n_processes=1000, n_steps=1000)
    tester.analyze_results()
    tester.plot_optimal_utility_vs_process_params()


# Utility functions as functions of x without parameters
def utility_power(x):
    return x  # Equivalent to x ** 0.5

def utility_exp(x):
    return x**2  # Equivalent to alpha = 1.0

def utility_log(x):
    return np.log(x + 1e-8)  # Small constant to avoid log(0)

# Updated utility_functions list
utility_functions = [
    {'name': 'Power', 'function': utility_power},
    {'name': 'Exponential', 'function': utility_exp},
    {'name': 'Logarithmic', 'function': utility_log},
]

# Define the agent class
class AgentEvaluation:
    """
    AgentEvaluation Class
    """
    def __init__(self, model):
        """
        Initialize the AgentEvaluation with the given model.

        :param model: Trained model for selecting processes
        :type model: Model
        """
        self.model = model

    def select_process(self, encoded_processes, select_max=True):
        """
        Select a process based on the model's predictions.

        :param encoded_processes: Encoded processes to choose from
        :param select_max: Whether to select the process with the maximum score
        :return: Index of the selected process
        """
        # Compute scores for each process
        scores = self.model.predict(encoded_processes)
        if select_max:
            choice = np.argmax(scores)
        else:
            choice = np.argmin(scores)
        return choice

# Main function
def evaluate_utility_functions(utility_functions, agent, processes, param_ranges, n_process_batches=100, n_options=2, num_instances=1000, select_max=True):
    """
    Evaluate utility functions based on agent choices.
    The utility functions are estimated according to the maximum likelihood of the agent's choices.

    :param utility_functions: List of utility functions to evaluate
    :rtype: List[Dict[str, float]]
    :param agent: An instance of the AgentEvaluation class
    :rtype agent: AgentEvaluation
    :param processes: A list of process types
    :rtype processes: List[Dict[str, str]]
    :param param_ranges: A dictionary of parameter ranges for each process type
    :rtype param_ranges: Dict[str, Dict[str, Tuple[float, float]]]
    :param n_process_batches: Number of process batches to evaluate
    :rtype n_process_batches: int
    :param n_options: Number of process options per batch
    :rtype n_options: int
    :param num_instances: Number of instances used in the expected utility calculation
    :rtype num_instances: int
    :param select_max: Whether to select the process with the maximum expected utility or with the minimum
    :rtype select_max: bool
    :return: Dictionary of likelihood scores for each utility function
    :rtype: Dict[str, float]
    """
    # Define the process encoder
    process_encoder = ProcessEncoder()

    # Initialize counters for utility functions
    utility_counts = {uf['name']: 0 for uf in utility_functions}

    # Total number of batches
    total_batches = n_process_batches

    for batch_index in range(n_process_batches):
        # For each batch, generate n_options processes
        process_options = []
        for option_index in range(n_options):
            # Randomly select a process type
            process_info = np.random.choice(processes)
            process_type = process_info['type']
            process_class = globals()[process_type]
            # Randomly select parameters within specified ranges
            params = {}
            for param_name, (low, high) in param_ranges[process_type].items():
                params[param_name] = np.random.uniform(low, high)
            # Create process instance
            process = process_class(**params)
            # Simulate the process to get final values x
            x = process.simulate(t=1.0, num_instances=num_instances)
            times, x = separate(x)
            # Get final values
            x = x[:, -1]
            # x = np.maximum(x, 0.001)  # Ensure x is positive
            # Store the process data
            process_data = {
                'process': process,
                'x': x
            }
            process_options.append(process_data)

        # Encode the processes using ProcessEncoder
        encoded_processes = []
        for process_data in process_options:
            process = process_data['process']
            encoded_process = process_encoder.encode_process_with_time(process, time=1.0)
            encoded_processes.append(encoded_process)
        encoded_processes = np.array(encoded_processes)

        # The agent makes its choice
        agent_choice = agent.select_process(encoded_processes, select_max=select_max)

        # For each utility function, compute expected utilities and determine the process with the highest expected utility
        for uf in utility_functions:
            expected_utilities = []
            for p_data in process_options:
                x = p_data['x']
                u_x = uf['function'](x)
                expected_u = np.mean(u_x)
                expected_utilities.append(expected_u)
            # Determine the process with the maximum expected utility
            if select_max:
                utility_choice = np.argmax(expected_utilities)
            else:
                utility_choice = np.argmin(expected_utilities)
            # Compare with agent's choice
            if utility_choice == agent_choice:
                utility_counts[uf['name']] += 1

    # After all batches, calculate likelihood scores
    likelihood_scores = {}
    for uf in utility_functions:
        count = utility_counts[uf['name']]
        likelihood = count / total_batches
        likelihood_scores[uf['name']] = likelihood

    # Return the likelihood scores
    return likelihood_scores

# Example usage
if __name__ == "__main__":
    # Instantiate the agent
    agent = AgentEvaluation()

    # Evaluate utility functions
    likelihood_scores = evaluate_utility_functions(
        utility_functions=utility_functions,
        agent=agent,
        n_process_batches=100,
        n_options=2,
        num_instances=1000,
        select_max=True
    )

    # Print the likelihood scores
    print("Likelihood scores for each utility function:")
    for name, score in likelihood_scores.items():
        print(f"{name}: {score:.4f}")
