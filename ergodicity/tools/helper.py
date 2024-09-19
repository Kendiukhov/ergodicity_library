"""
helper Submodule

The `helper` submodule provides various utility functions designed to assist with data manipulation, visualization, and process encoding within stochastic process simulations. These helper functions streamline the workflow for saving results, plotting simulation data, encoding processes, and handling system-level data transformations. It is commonly used in conjunction with other modules such as `process`, `compute`, and `fit`.

Key Features:

1. **Data Manipulation**:

   - `separate`: A simple utility that separates time and process data from a given dataset. Often used to split simulation data into its respective components.

2. **Saving and Visualization**:

   - `save_to_file`: Saves simulation data to a specified directory. It can create necessary directories if they do not exist.

   - `visualize_process_from_data`: Visualizes the output of a stochastic process simulation.

   - `plot_simulate_ensemble`: Plots the results of an ensemble of simulated processes, showing portfolio evolution, asset weights, and geometric means.

   - `plot`: Plots the simulation results of custom processes with multiple instances.

   - `plot_system`: A more advanced plotting function designed for systems of stochastic differential equations (SDEs), where multiple equations are simulated simultaneously.

3. **Process Encoding**:

   - `ProcessEncoder`: A class designed to encode different stochastic processes and their parameters into a numeric representation. This is useful for handling multiple processes programmatically and standardizing their representation in simulations.

     - **Methods**:

       - `encode`: Encodes a process type as a unique integer.

       - `decode`: Decodes an integer back into the corresponding process type.

       - `encode_process`: Encodes a process instance into a list of floats, allowing for the serialization of its parameters.

       - `pad_encoded_process`: Pads the encoded process representation to ensure uniform length.

       - `encode_process_with_time`: Encodes a process along with a time value to preserve temporal information for the simulation.

4. **Parallelism**:

   - The module imports `ProcessPoolExecutor` to enable parallel execution, although no explicit parallel functionality is currently implemented in the given code. This suggests potential future extensions for parallel processing of simulations or fitting routines.

Example Usage:

data = np.random.randn(100, 50)  # Example data

# Save the data

save_to_file(data, output_dir="results", file_name="simulation_data", save=True)

# Plot the data

plot(data, num_instances=10, save=True, plot=True)

# Encode a process instance

encoder = ProcessEncoder()

encoded_process = encoder.encode_process(MyProcessClass())

print("Encoded process:", encoded_process)
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Union
from ergodicity.configurations import *
from ergodicity.process.default_values import *
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime

def separate(data):
    """
    Separate time and data from the given dataset.

    :param data: The dataset to separate
    :type data: numpy.ndarray
    :return: A tuple containing the time values and the data values
    :rtype: tuple
    """
    times = data[0]
    data = data[1:]
    return times, data

def save_to_file(data, output_dir: str, file_name: str, save: bool = False):
    """
    Save the given data to a file in the specified output directory.

    :param data: The data to save
    :type data: numpy.ndarray
    :param output_dir: The directory to save the file in
    :type output_dir: str
    :param file_name: The name of the file to save
    :type file_name: str
    :param save: Whether to save the data
    :type save: bool
    :return: None
    :rtype: None
    """
    if save is True:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save((os.path.join(output_dir, file_name)), data)
        print(f"Data saved to {output_dir}/{file_name}")

def plot_simulate_ensemble(simulation_result: Dict[str, Any], t: float, save: bool = False) -> None:
    """
    Plot the results of the simulate_ensemble method.

    :param simulation_result: Dictionary containing 'portfolio', 'geometric_means', and 'weights'
    :type simulation_result: Dict[str, Any]
    :param t: Total simulation time
    :type t: float
    :param save: Whether to save the plots as image files
    :type save: bool
    :return: None
    :rtype: None
    """
    portfolio = simulation_result['portfolio']
    geometric_means = simulation_result['geometric_means']
    weights = simulation_result['weights']

    num_steps = portfolio.shape[1] - 1
    num_instances = portfolio.shape[0]

    # Plot portfolio value and geometric mean
    plt.figure(figsize=(12, 8))
    portfolio_value = np.mean(portfolio, axis=0)
    plt.plot(np.linspace(0, t, num_steps + 1), portfolio_value, label='Portfolio Value')
    plt.plot(np.linspace(0, t, num_steps + 1), geometric_means, label='Geometric Mean')
    plt.title('Evolution of Portfolio Value and Geometric Mean')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig('portfolio_evolution.png')
    plt.show()

    # Plot weight evolution
    plt.figure(figsize=(12, 8))
    for i in range(num_instances):
        plt.plot(np.linspace(0, t, num_steps + 1), weights[i, :], label=f'Asset {i + 1}')
    plt.title('Evolution of Asset Weights')
    plt.xlabel('Time')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig('weight_evolution.png')
    plt.show()

    # Plot individual asset paths
    plt.figure(figsize=(12, 8))
    for i in range(num_instances):
        plt.plot(np.linspace(0, t, num_steps + 1), portfolio[i, :], label=f'Asset {i + 1}')
    plt.title('Individual Asset Paths')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig('asset_paths.png')
    plt.show()

def plot(data_full, num_instances: int, save: bool = False, plot: bool = False):
    """
    Plot the simulation results of custom processes with multiple instances.

    :param data_full: The simulation data to plot
    :type data_full: numpy.ndarray
    :param num_instances: The number of instances to plot
    :type num_instances: int
    :param save: Whether to save the plots as image files
    :type save: bool
    :param plot: Whether to display the plots
    :type plot: bool
    :return: None
    :rtype: None
    """
    if plot:
        times, data = separate(data_full)
        t = times[-1]
        timestep = times[1] - times[0]

        # Visualization
        plt.figure(figsize=(10, 6))
        for i in range(num_instances):
            plt.plot(times, data[i, :], lw=0.5)
        plt.title(f'Simulation of the custom process')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True)
        if save:
            plt.savefig(os.path.join(output_dir_default, f'custom_process_simulation:{t}_timestep:{timestep}_num_instances:{num_instances}_process_simulation.png'))
        plt.show()

def plot_system(data_full, num_instances: int, num_equations: int, save: bool = False, plot: bool = False):
    """
    Plot the simulation results of a system of stochastic differential equations (SDEs).

    :param data_full: The simulation data to plot
    :type data_full: numpy.ndarray
    :param num_instances: The number of instances to plot
    :type num_instances: int
    :param num_equations: The number of equations in the system
    :type num_equations: int
    :param save: Whether to save the plots as image files
    :type save: bool
    :param plot: Whether to display the plots
    :type plot: bool
    :return: None
    :rtype: None
    """
    if plot:
        times = data_full[0]
        data = data_full[1:]
        t = times[-1]
        timestep = times[1] - times[0]

        plt.figure(figsize=(12, 8))
        for eq in range(num_equations):
            plt.subplot(num_equations, 1, eq + 1)
            for i in range(num_instances):
                plt.plot(times, data[i * num_equations + eq], lw=0.5)
            plt.title(f'Equation {eq + 1}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.grid(True)

        plt.tight_layout()
        if save:
            plt.savefig(
                os.path.join(output_dir_default, f'system_sde_simulation_t{t}_dt{timestep}_n{num_instances}.png'))
        plt.show()

class ProcessEncoder:
    """
    A class to encode different stochastic processes and their parameters into a numeric representation.
    It is neeeded for handling multiple processes programmatically and standardizing their representation in simulations.
    It is used in the submodules 'multiprocessing' and 'automate' of the 'compute' module and in the 'agents' module.

    Attributes:

        process_types (dict): A dictionary mapping process types to their encoded integer values.

        reverse_mapping (dict): A dictionary mapping encoded integer values to their corresponding process types.

        next_id (int): The next available integer value for encoding a new process type.
    """
    def __init__(self):
        """
        Initialize the ProcessEncoder with default process types and an empty reverse mapping.

        The default process types are:
        - BrownianMotion: 1
        - GeometricBrownianMotion: 2
        """
        self.process_types = {'BrownianMotion': 1, 'GeometricBrownianMotion': 2}
        self.reverse_mapping = {}
        self.next_id = 3

    def encode(self, process_type: str) -> int:
        """
        Encode a process type as a unique integer.

        :param process_type: The type of the process to encode
        :type process_type: str
        :return: The encoded integer value
        :rtype: int
        """
        if process_type not in self.process_types:
            self.process_types[process_type] = self.next_id
            self.reverse_mapping[self.next_id] = process_type
            self.next_id += 1
        return self.process_types[process_type]

    def decode(self, process_id: int) -> str:
        """
        Decode an integer back into the corresponding process type.

        :param process_id: The integer value to decode
        :type process_id: int
        :return: The decoded process type
        :rtype: str
        """
        return self.reverse_mapping.get(process_id, "Unknown")

    def get_encoding(self) -> Dict[str, int]:
        """
        Get the encoding of process types.

        :return: A dictionary mapping process types to their encoded integer values
        :rtype: Dict[str, int]
        """
        return self.process_types

    def get_decoding(self) -> Dict[int, str]:
        """
        Get the decoding of process types.

        :return: A dictionary mapping encoded integer values to their corresponding process types
        :rtype: Dict[int, str]
        """
        return self.reverse_mapping

    def encode_process(self, process: object) -> List[float]:
        """
        Encode a process instance into a list of floats.

        :param process: A process instance
        :type process: object
        :return: A list of floats representing the encoded process
        :rtype: List[float]
        """
        process_type = type(process).__name__
        encoded = [float(self.encode(process_type))]

        # Use the get_params method to retrieve process parameters
        params = process.get_params()
        # print(f"Process parameters from the ProcessEncoder: {params}")
        for param_value in params.values():
            try:
                encoded.append(float(param_value))
            except (ValueError, TypeError):
                print(f"Warning: Skipping non-numeric parameter with value {param_value}")

        # print(f"Encoded process from the ProcessEncoder: {encoded}")

        return encoded

    def pad_encoded_process(self, encoded_process: List[float], max_params: int = 10) -> List[float]:
        """
        Pad the encoded process representation to ensure uniform length.

        :param encoded_process: The encoded process to pad
        :type encoded_process: List[float]
        :param max_params: The maximum number of parameters to include
        :type max_params: int
        :return: The padded encoded process
        :rtype: List[float]
        """
        padded = encoded_process[:1]  # Keep the process type
        padded.extend(encoded_process[1:max_params + 1])  # Take up to max_params
        padded.extend([0.0] * (max_params - len(encoded_process[1:])))  # Pad with zeros if needed
        return padded

    def encode_process_with_time(self, process: Union[Dict, object], time: float) -> List[float]:
        """
        Encode a process with its time value, maintaining the original total length.

        :param process: The process to encode (either a dictionary or an object)
        :type process: Union[Dict, object]
        :param time: The time value to include in the encoding
        :type time: float
        :return: A list of floats representing the encoded process with time
        :rtype: List[float]
        """
        encoded_process = self.pad_encoded_process(self.encode_process(process))
        return [encoded_process[0]] + [time] + encoded_process[1:-1]

def covariance_to_correlation(covariance_matrix):
    """
    Convert a covariance matrix to a correlation matrix.

    :param covariance_matrix: The covariance matrix to convert
    :type covariance_matrix: numpy.ndarray
    :return: The correlation matrix
    :rtype: numpy.ndarray
    """
    # Extract the diagonal elements (variances)
    variances = np.diag(covariance_matrix)

    # Calculate standard deviations
    std_devs = np.sqrt(variances)

    # Create a diagonal matrix of inverse standard deviations
    D_inv = np.diag(1 / std_devs)

    # Calculate correlation matrix
    correlation_matrix = D_inv @ covariance_matrix @ D_inv

    return correlation_matrix


