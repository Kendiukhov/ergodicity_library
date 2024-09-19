"""
sml Submodule Overview

The **`sml`** (Stochastic machine Learning) submodule integrates stochastic processes, utility functions, and machine learning techniques to enable the study and optimization of decision-making processes in non-ergodic systems. It allows users to infer optimal behaviors in uncertain environments by simulating processes, fitting utility functions, and applying machine learning models.

Key Features:

1. **Utility Functions**:

   - Allows users to define utility functions, which model the preferences or satisfaction of an agent.

   - Utility functions can be optimized to find the parameters that best explain observed decisions.

2. **Utility Function Inference**:

   - **UtilityFunctionInference** class fits utility functions to agents’ decisions by minimizing negative log-likelihood or using Bayesian inference.

   - Includes Metropolis-Hastings MCMC sampling for fitting utility functions in a Bayesian framework.

   - Allows the user to analyze, visualize, and compare different utility functions using the provided dataset.

3. **Agent-based Process Selection**:

   - Agents are created with neural network models that can be trained to select optimal stochastic processes based on encoded inputs.

   - Simulates multiple stochastic processes (e.g., Brownian motion, Geometric Brownian motion) to study agent preferences and process performance.

4. **Utility Function Testing**:

   - The **UtilityFunctionTester** class provides tools to test different utility functions against stochastic processes.

   - Generates random process parameters and simulates trajectories, calculating and comparing the utilities of different functions for the same process.

5. **Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL)**:

   - Enables inference of reward weights from agent behavior using **MaxEntIRL**.

   - Learns reward weights for different features by optimizing the expected state visitation frequencies to match observed behavior.

6. **Machine Learning and Regression**:

   - The submodule includes a regression-based approach to predict agent preferences and analyze feature importance using neural networks.

   - Includes tools to visualize model training results, such as loss and accuracy plots, and feature importance analysis.

Example Usage:

# Fitting Utility Functions to an Agent's Choices:

from ergodicity.sml import UtilityFunctionInference, UtilityFunction

from ergodicity.process.basic import BrownianMotion, GeometricBrownianMotion

# Initialize the inference object

ufi = UtilityFunctionInference('path/to/your/model.h5', param_ranges={
    'BrownianMotion': {'mu': (0, 0.5), 'sigma': (0.1, 0.5)},
    'GeometricBrownianMotion': {'mu': (0, 0.5), 'sigma': (0.1, 0.5)}
})

# Add utility functions

ufi.add_utility_function(UtilityFunction('Power', lambda x, beta: x ** beta, [1.0]))

ufi.add_utility_function(UtilityFunction('Exponential', lambda x, alpha: 1 - np.exp(-alpha * x), [1.0]))

# Generate dataset

dataset = ufi.generate_dataset(100)

# Get agent's choices based on the dataset

choices = ufi.get_agent_choices(dataset)

# Fit the utility functions

ufi.fit_utility_functions(dataset, choices)

# Plot results

ufi.plot_utility_functions()
"""
import tensorflow as tf
from sympy.physics.control import step_response_plot
from sympy.physics.units import length
from tensorflow import keras
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Callable
from ergodicity.process.basic import *
from ergodicity.process.multiplicative import *
import os
import csv
from datetime import datetime
from ergodicity.configurations import *
from ergodicity.process.default_values import *
from ergodicity.tools.helper import ProcessEncoder
import sympy as sp
import matplotlib.pyplot as plt

def create_model(hidden_layers, output_shape):
    """
    Create a neural network model with the specified hidden layers and output shape.

    :param hidden_layers: List of integers specifying the number of units in each hidden layer.
    :type hidden_layers: List[int]
    :param output_shape: Integer specifying the output shape of the model.
    :type output_shape: int
    :return: A compiled Keras model.
    :rtype: tf.keras.Model
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(11,)),  # 11 input features
        keras.layers.Dense(hidden_layers[0], activation='relu')
    ])

    # Hidden layers
    for units in hidden_layers[1:]:
        model.add(keras.layers.Dense(units, activation='relu'))

    # Output layer
    model.add(keras.layers.Dense(output_shape, activation='linear'))

    model.compile(optimizer='adam', loss='mse')

    return model

def x_reach_2(X):
    """
    Check if the value of X is greater than or equal to 2.

    :param X: Value of X.
    :type X: float
    :return: Boolean indicating if X is greater than or equal to 2.
    :rtype: bool
    """
    return X >= 2

def simulate_process(process_type: str, params: Dict[str, float]) -> np.ndarray:
    """
    Simulate a stochastic process with the specified parameters.

    :param process_type: Type of stochastic process to simulate.
    :type process_type: str
    :param params: Dictionary of parameters for the stochastic process.
    :type params: Dict[str, float]
    :return: Array of simulated process values.
    :rtype: np.ndarray
    """
    if process_type == "BrownianMotion":
        process = BrownianMotion(**params)
    elif process_type == "GeometricBrownianMotion":
        process = GeometricBrownianMotion(**params)
    else:
        raise ValueError(f"Unknown process type: {process_type}")

    data = process.simulate_until(timestep=0.1, num_instances=1, condition=x_reach_2, X0=1, plot=False)
    return data

def pad_array(arr, target_shape):
    """
    Pad an array to the target shape with NaN values.

    :param arr: Array to pad.
    :type arr: np.ndarray
    :param target_shape: Target shape to pad the array to.
    :type target_shape: Tuple[int]
    :return: Padded array.
    :rtype: np.ndarray
    """
    pad_width = [(0, max(0, target_shape[i] - arr.shape[i])) for i in range(len(target_shape))]
    # print(f'Padding array with shape {arr.shape} to target shape {target_shape}')
    return np.pad(arr, pad_width, mode='constant', constant_values=np.nan)

def worker(args):
    """
    Worker function for parallel processing.
    It simulates a stochastic process with random parameters and returns the results.

    :param args: Tuple of arguments for the worker function.
    :type args: Tuple
    :return: List of results from the worker function.
    :rtype: List
    """

    process_type, param_ranges, num_instances, n_simulations = args
    results = []
    for sim in range(n_simulations):
        params = {k: np.random.uniform(v[0], v[1]) for k, v in param_ranges.items()}
        instances = []
        max_shape = None
        for inst in range(num_instances):
            instance = simulate_process(process_type, params)
            instances.append(instance)
            if max_shape is None:
                max_shape = instance.shape
            else:
                max_shape = tuple(max(s1, s2) for s1, s2 in zip(max_shape, instance.shape))

        # Pad all instances to the same shape
        padded_instances = [pad_array(inst, max_shape) for inst in instances]
        result = np.stack(padded_instances, axis=0)
        results.append((result, process_type, params, sim))
    return results

def generate_dataset(processes: List[Dict[str, Any]], param_ranges: Dict[str, Dict[str, tuple]], num_instances: int,
                     n_simulations: int, output_dir: str = 'output_general', save: bool = False, simulate_method=False) -> List[np.ndarray]:
    """
    Generate a dataset of simulated processes with random parameters.

    :param processes: List of dictionaries specifying the processes to simulate.
    :type processes: List[Dict[str, Any]]
    :param param_ranges: Dictionary of parameter ranges for each process type.
    :type param_ranges: Dict[str, Dict[str, tuple]]
    :param num_instances: Number of instances to simulate for each process.
    :type num_instances: int
    :param n_simulations: Number of simulations to run for each process. It sets how many process objects will be created within the specified parameter ranges.
    :type n_simulations: int
    :param output_dir: Output directory to save the results.
    :type output_dir: str
    :param save: Boolean indicating whether to save the results to files.
    :type save: bool
    :param simulate_method: Boolean indicating whether to use the simulation method or the simulate_until method.
    :type simulate_method: bool
    :return: List of arrays containing the simulated process data.
    :rtype: List[np.ndarray]
    """
    if simulate_method:
        dataset = []
        for process in processes:
            process_type = process['type']
            params = {k: np.mean(v) for k, v in param_ranges[process_type].items()}  # Use mean of parameter ranges
            process_instance = globals()[process_type](**params)
            data = process_instance.simulate(t=1, timestep=timestep_default, num_instances=1)
            dataset.append(data)
        return dataset

    else:
        os.makedirs(output_dir, exist_ok=True)

        with ProcessPoolExecutor() as executor:
            args = [(process['type'], param_ranges[process['type']], num_instances, n_simulations) for process in processes]
            all_results = list(executor.map(worker, args))

        # Flatten the list of lists
        flat_results = [item for sublist in all_results for item in sublist]

        # Save results to files and prepare return data
        return_data = []
        for result, process_type, params, sim in flat_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{process_type}_{sim}_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)

            if save:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['process_type'] + list(params.keys()) + ['instance', 'time'] + [f'value_{i}' for i in
                                                                                                     range(result.shape[2])])
                    for inst in range(num_instances):
                        for t in range(result.shape[1]):
                            row = [process_type] + list(params.values()) + [inst, t * 0.01] + list(result[inst, t])
                            writer.writerow(row)

            # print(f"Saved simulation to {filepath}")

            # Prepare data for return, including parameter values
            times = np.arange(result.shape[1]) * 0.01
            param_values = np.array(list(params.values()))
            return_data.append(
                np.column_stack((times[:, np.newaxis], np.tile(param_values, (result.shape[1], 1)), result[0])))

        return return_data

class NeuralNetworkAgent:
    """
    NeuralNetworkAgent Class

    This class represents an agent that uses a neural network to make decisions in a stochastic
    process environment. It encapsulates the neural network model along with methods for process
    selection and wealth management.

    Attributes:
        model (tf.keras.Model): The neural network model used for decision making.
        wealth (float): The current wealth of the agent, initialized to 1.0.
        performance_history (list): A list tracking the agent's wealth over time.


    This class is designed for use in reinforcement learning scenarios where an agent learns to
    select optimal processes in a stochastic environment. The neural network model is used to
    predict the best process based on encoded representations, and the agent's performance is
    tracked through its wealth accumulation.

    Key Features:

    1. Integration with TensorFlow Keras models for decision making.

    2. Wealth tracking to measure agent performance over time.

    3. Process selection based on minimizing the predicted value from the neural network.

    4. Ability to reset wealth for multiple episodes or experiments.

    Usage:

        model = tf.keras.Sequential([...])  # Define your neural network model

        agent = NeuralNetworkAgent(model)

        # In each step of your simulation:

        selected_process = agent.select_process(encoded_processes)

        process_outcome = simulate_process(selected_process)

        agent.update_wealth(process_outcome)

        # To start a new episode:

        agent.reset_wealth()
    """
    def __init__(self, model: tf.keras.Model):
        """
        Initialize the NeuralNetworkAgent with a Keras model.

        :param model: Keras model for the agent.
        :type model: tf.keras.Model
        """
        self.model = model
        self.wealth = 1.0
        self.performance_history = []

    def select_process(self, encoded_processes: np.ndarray) -> int:
        """
        Select the optimal process based on the encoded processes.

        :param encoded_processes: Array of encoded processes.
        :type encoded_processes: np.ndarray
        :return: Index of the selected process.
        :rtype: int
        """
        if encoded_processes.shape[1] != 11:
            raise ValueError(f"Expected 11 elements per process, but got {encoded_processes.shape[1]}")

        predictions = self.model.predict(encoded_processes)
        return np.argmin(predictions)

    def update_wealth(self, process_value: float):
        """
        Update the agent's wealth based on the outcome of a selected process.

        :param process_value: The return value of the selected process.
        :type process_value: float
        :return: None
        :rtype: None
        """
        self.wealth *= process_value
        self.performance_history.append(self.wealth)

    def reset_wealth(self):
        """
        Reset the agent's wealth to the initial value (1.0) and clear the performance history.

        :return: None
        :rtype: None
        """
        self.wealth = 1.0
        self.performance_history = []

def save_model(model, filepath):
    """
    Save the full model (architecture + weights) to a file.

    :param model: Keras model to save.
    :type model: tf.keras.Model
    :param filepath: Path to save the model file.
    :type filepath: str
    :return: None
    :rtype: None
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save the entire model (architecture + weights)
    model.save(filepath)
    print(f"Full model saved to {filepath}")

def ranked_array(arr):
    """
    Rank the elements of an array in descending order.

    :param arr: Input array.
    :type arr: np.ndarray
    :return: Array with elements and their corresponding ranks.
    :rtype: np.ndarray
    """
    sorted_indices = np.argsort(-arr)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(arr))
    result = np.column_stack((arr, ranks))
    return result

def train_agent_time(agent: NeuralNetworkAgent,
                processes: List[Dict[str, Any]],
                param_ranges: Dict[str, Dict[str, tuple]],
                n_episodes: int,
                n_steps: int,
                num_instances: int = 10,
                n_simulations: int = 10,
                output_dir: str = 'output_general'):
    """
    Train the agent to select optimal processes in a stochastic environment.

    :param agent: NeuralNetworkAgent object to train.
    :type agent: NeuralNetworkAgent
    :param processes: List of process dictionaries to simulate.
    :type processes: List[Dict[str, Any]]
    :param param_ranges: Dictionary of parameter ranges for each process type.
    :type param_ranges: Dict[str, Dict[str, tuple]]
    :param n_episodes: Number of episodes to train the agent.
    :type n_episodes: int
    :param n_steps: Number of steps per episode.
    :type n_steps: int
    :param num_instances: Number of instances to simulate for each process.
    :type num_instances: int
    :param n_simulations: Number of simulations to run for each process.
    :type n_simulations: int
    :param output_dir: Output directory to save the results.
    :type output_dir: str
    :return: Tuple of agent, episode wealth history, episode performance history, min times, and actual min times.
    :rtype: Tuple[NeuralNetworkAgent, List[List[float]], List[float], List[float], List[float]]
    """

    process_encoder = ProcessEncoder()
    episode_wealth_history = []
    episode_performance_history = []
    min_times = []
    actual_min_times = []

    for episode in range(n_episodes):
        agent.reset_wealth()
        step_wealth_history = []

        for step in range(n_steps):
            # Generate a batch of processes
            dataset = generate_dataset(processes, param_ranges, num_instances, n_simulations, output_dir)

            # Encode processes
            encoded_processes = []
            for i, data in enumerate(dataset):
                process_type = processes[i % len(processes)]['type']

                # Extract parameters from the data
                params = {k: data[0, i + 1] for i, k in enumerate(param_ranges[process_type].keys())}

                # Create process instance
                process_instance = globals()[process_type](**params)

                # Get the final time
                final_time = 1.0

                # Use the ProcessEncoder's encode_process_with_time method
                encoded_process = process_encoder.encode_process_with_time(process_instance, final_time)

                encoded_processes.append(encoded_process)

            encoded_processes = np.array(encoded_processes)

            # Select process
            selected_process = agent.select_process(encoded_processes)

            actual_times = []
            for i in range(len(dataset)):
                time_values = dataset[i][0]
                non_nan_time_values = time_values[~np.isnan(time_values)]
                actual_times.append(non_nan_time_values[-1])

            # Ensure encoded_processes and actual_times have correct shapes
            encoded_processes = np.array(encoded_processes)
            actual_times = np.array(actual_times)
            print(f'length of actual times', len(actual_times))

            # Train the model
            agent.model.fit(encoded_processes, actual_times, epochs=1, verbose=0)

            # Compare predictions to actual results
            predictions = agent.model.predict(encoded_processes).flatten()
            min_time_index = np.argmin(predictions)
            min_time = actual_times[min_time_index]

            # Get the corresponding process type and parameters
            process_type = processes[min_time_index % len(processes)]['type']
            params = {k: dataset[min_time_index][0, i + 1] for i, k in enumerate(param_ranges[process_type].keys())}

            # Create and simulate the process instance
            process_instance = globals()[process_type](**params)
            simulated_data = process_instance.simulate(t=1, timestep=timestep_default, num_instances=1)

            # Update agent's wealth
            agent.update_wealth(simulated_data[1, -1])

            step_wealth_history.append(agent.wealth)

            actual_min_time = np.min(actual_times)

            ranked_predictions = ranked_array(predictions)
            ranked_actual_times = ranked_array(actual_times)

            print('ranked_predictions:', ranked_predictions)
            print('ranked_actual_times:', ranked_actual_times)

            print('min_time:', min_time)
            min_times.append(min_time)
            actual_min_times.append(actual_min_time)
            print('min_times:', min_times)
            print('actual_min_times:', actual_min_times)
            mse = np.mean((predictions - actual_times) ** 2)
            print(f"Episode {episode + 1}, Step {step + 1}: MSE = {mse:.4f}")

        episode_wealth_history.append(step_wealth_history)
        episode_performance_history.append(agent.wealth)
        print(f"Episode {episode + 1}/{n_episodes}: Final wealth = {agent.wealth:.2f}")

        # Save the full model after each episode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f'model_episode_{episode+1}_{timestamp}.h5'
        save_model(agent.model, os.path.join(output_dir, model_filename))

    return agent, episode_wealth_history, episode_performance_history, min_times, actual_min_times

def save_model_weights(model, filepath):
    """
    Save the weights of a Keras model to a file.

    :param model: Keras model to save.
    :type model: tf.keras.Model
    :param filepath: Path to save the model weights.
    :type filepath: str
    :return: None
    :rtype: None
    """
    model.save_weights(filepath)
    print(f"Model weights saved to {filepath}")

def visualize_results(episode_wealth_history, episode_performance_history, min_times, actual_min_times, output_dir):
    """
    Visualize the results of the agent training.

    :param episode_wealth_history: List of episode wealth histories.
    :type episode_wealth_history: List[np.ndarray]
    :param episode_performance_history: List of episode performance histories.
    :type episode_performance_history: List[float]
    :param min_times: List of predicted min times.
    :type min_times: List[float]
    :param actual_min_times: List of actual min times.
    :type actual_min_times: List[float]
    :param output_dir: Output directory to save the visualizations.
    :type output_dir: str
    :return: None
    :rtype: None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot wealth dynamics within episodes
    plt.figure(figsize=(12, 6))
    for i, wealth_history in enumerate(episode_wealth_history):
        plt.plot(wealth_history, label=f'Episode {i + 1}')
    plt.title('Wealth Dynamics within Episodes')
    plt.xlabel('Step')
    plt.ylabel('Wealth')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'wealth_dynamics.png'))
    plt.close()

    # Plot final wealth per episode
    plt.figure(figsize=(12, 6))
    plt.plot(episode_performance_history)
    plt.title('Final Wealth per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Final Wealth')
    plt.savefig(os.path.join(output_dir, 'final_wealth_per_episode.png'))
    plt.close()

    # Plot min times
    plt.figure(figsize=(12, 6))
    plt.plot(min_times, label='Predicted Min Times')
    plt.plot(actual_min_times, label='Actual Min Times')
    plt.title('Min Times')
    plt.xlabel('Episode')
    plt.ylabel('Min Time')
    plt.savefig(os.path.join(output_dir, 'min_times.png'))
    plt.close()

    print(f"Visualization results saved in {output_dir}")

def train_agent_value(agent: NeuralNetworkAgent,
                      processes: List[Dict[str, Any]],
                      param_ranges: Dict[str, Dict[str, tuple]],
                      n_episodes: int,
                      n_steps: int,
                      output_dir: str = 'output_general',
                      ergodicity_transform: bool = False,
                      early_stopping_patience: int = 10,
                      early_stopping_min_delta: float = 1e-4):
    """
    Train the agent to predict and select stochastic processes based on their values.

    :param agent: NeuralNetworkAgent object to train.
    :param processes: List of process dictionaries to simulate.
    :param param_ranges: Dictionary of parameter ranges for each process type.
    :param n_episodes: Number of episodes to train the agent.
    :param n_steps: Number of steps per episode.
    :param output_dir: Output directory to save the results.
    :param ergodicity_transform: Whether to use ergodicity transform of process values.
    :param early_stopping_patience: Number of episodes with no improvement after which training will be stopped.
    :param early_stopping_min_delta: Minimum change in MSE to qualify as an improvement.
    :return: Tuple of agent, episode performance history, and best MSE.
    """
    process_encoder = ProcessEncoder()
    episode_performance_history = []
    best_mse = float('inf')
    patience_counter = 0

    for episode in range(n_episodes):
        episode_mse = 0

        for step in range(n_steps):
            # Generate a batch of processes
            encoded_processes = []
            actual_values = []

            for process in processes:
                process_type = process['type']
                params = {k: np.random.uniform(v[0], v[1]) for k, v in param_ranges[process_type].items()}
                process_instance = globals()[process_type](**params)

                encoded_process = process_encoder.encode_process_with_time(process_instance, t_default)
                encoded_processes.append(encoded_process)

                # Simulate the process
                print(f"Simulating {process_type} with parameters: {params}")
                simulated_data = process_instance.simulate(t=1, timestep=timestep_default, num_instances=1)
                print(f"Simulation completed. Shape of simulated data: {simulated_data.shape}")

                if simulated_data.shape[1] > 2:  # Ensure we have at least two time points
                    process_value = simulated_data[1, -1]  # Last value of the process (excluding time)
                else:
                    print(f"Warning: Insufficient data points in simulation. Using initial value.")
                    process_value = simulated_data[1, 0]  # Use the initial value

                print(f"Process value: {process_value}")

                if ergodicity_transform:
                    # Apply ergodicity transform
                    istrue, transform_expr, a_u, b_u = process_instance.ergodicity_transform()
                    transform_func = sp.lambdify('x', transform_expr, 'numpy')
                    process_value = transform_func(process_value)
                    print(f"Transformed process value: {process_value}")

                actual_values.append(process_value)

            encoded_processes = np.array(encoded_processes)
            actual_values = np.array(actual_values)

            print(f"Shape of encoded_processes: {encoded_processes.shape}")
            print(f"Shape of actual_values: {actual_values.shape}")

            # Predict process values
            predicted_values = agent.model.predict(encoded_processes).flatten()

            print(f"Shape of predicted_values: {predicted_values.shape}")

            # Train the model
            agent.model.fit(encoded_processes, actual_values, epochs=1, verbose=0)

            # Calculate MSE
            mse = np.mean((predicted_values - actual_values) ** 2)
            episode_mse += mse

            # Select process with highest predicted value
            selected_process_index = np.argmax(predicted_values)
            selected_process_value = actual_values[selected_process_index]

            # Update agent's wealth
            agent.update_wealth(selected_process_value)

        # Calculate average MSE for the episode
        avg_episode_mse = episode_mse / n_steps
        episode_performance_history.append(avg_episode_mse)

        print(f"Episode {episode + 1}/{n_episodes}: MSE = {avg_episode_mse:.4f}, Wealth = {agent.wealth:.2f}")

        # Early stopping check
        if avg_episode_mse < best_mse - early_stopping_min_delta:
            best_mse = avg_episode_mse
            patience_counter = 0
            # Save the best model
            save_model(agent.model,
                       os.path.join(output_dir, f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'))
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {episode + 1} episodes.")
            break

    return agent, episode_performance_history, best_mse

