"""
evolutionary_nn Submodule Overview

The **`evolutionary_nn`** submodule combines neural networks with evolutionary strategies to solve problems involving stochastic processes, ergodicity, and time-optimal behaviour. This submodule allows users to create neural network-based agents that evolve over time, optimize their behavior, and select processes to maximize their wealth or other objectives. The agents can be trained using evolutionary algorithms, with or without reinforcement learning techniques.

Key Features:

1. **Neural Network Creation**:

   - Easily create customizable feedforward neural networks with various configurations.

   - Features include batch normalization, dropout, and several activation functions (ReLU, Leaky ReLU, Tanh, etc.).

   - Supports multiple weight initialization methods (Xavier, He, etc.) and optimizers (Adam, SGD, RMSprop).

2. **Agent-Based Evolution**:

   - The `NeuralNetworkAgent` class represents agents with a neural network-based decision-making process.

   - Agents accumulate wealth based on their decisions, and their performance (fitness) is measured by their accumulated wealth.

   - Agents can mutate, reproduce, and be cloned, allowing for evolutionary strategies.

3. **Evolutionary Neural Network Training**:

   - The `EvolutionaryNeuralNetworkTrainer` class enables training a population of agents using evolutionary algorithms.

   - Agents are evaluated based on their wealth, and top-performing agents are selected to produce offspring for the next generation.

   - The training process supports wealth sharing and mutation of agents to explore new strategies.

4. **Reinforcement Learning with Evolution**:

   - The `ReinforcementEvolutionaryTrainer` class combines evolutionary strategies with reinforcement learning.

   - Agents are trained to select processes that maximize their wealth using reinforcement learning principles.

   - Crossover and mutation are applied to create new agents based on elite performers.

5. **Process Encoding**:

   - The `ProcessEncoder` class encodes stochastic processes (e.g., Brownian motion, Geometric Brownian motion) into a numeric format for input into neural networks.

   - This allows agents to process different types of stochastic processes and make decisions based on encoded data.

6. **Visualization**:

   - Visualize the performance of agents, including wealth evolution over time, using Matplotlib and animations.

   - Generate visualizations of neural network evolution and save the parameters of the best-performing agents to files.

Example Usage:

### Creating and Training Neural Network Agents:

from ergodicity.process.multiplicative import GeometricBrownianMotion, BrownianMotion

from ergodicity.evolutionary_nn import NeuralNetwork, NeuralNetworkAgent, EvolutionaryNeuralNetworkTrainer

# Define stochastic processes

process_types = [GeometricBrownianMotion, BrownianMotion]

processes = generate_processes(100, process_types, param_ranges)

# Initialize the ProcessEncoder

encoder = ProcessEncoder()

# Create a neural network for an agent

net = NeuralNetwork(input_size=11, hidden_sizes=[20, 10], output_size=1)

# Create an agent with the neural network

agent = NeuralNetworkAgent(net)

# Train a population of agents using evolutionary strategies

trainer = EvolutionaryNeuralNetworkTrainer(
    population_size=10,
    input_size=11,
    hidden_sizes=[20, 10],
    output_size=1,
    processes=processes,
    process_encoder=encoder,
    process_times=[1.0, 5.0, 10.0],
)

population, history = trainer.train(n_steps=100, save_interval=10)

# Get the best agent

best_agent = max(population, key=lambda agent: agent.accumulated_wealth)

print(f"Best agent accumulated wealth: {best_agent.accumulated_wealth}")
"""
from typing import List, Callable, Type, Union, Dict, Any, Tuple
import torch.nn as nn
import numpy as np
import random
import os
import csv
from typing import List, Union, Type
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from ergodicity.tools.helper import ProcessEncoder
import torch
import torch.optim as optim

# class ProcessEncoder:
#     def __init__(self):
#         self.process_types = {'BrownianMotion': 1, 'GeometricBrownianMotion': 2}
#         self.reverse_mapping = {}
#         self.next_id = 3
#
#     def encode(self, process_type: str) -> int:
#         if process_type not in self.process_types:
#             self.process_types[process_type] = self.next_id
#             self.reverse_mapping[self.next_id] = process_type
#             self.next_id += 1
#         return self.process_types[process_type]
#
#     def decode(self, process_id: int) -> str:
#         return self.reverse_mapping.get(process_id, "Unknown")
#
#     def get_encoding(self) -> Dict[str, int]:
#         return self.process_types
#
#     def get_decoding(self) -> Dict[int, str]:
#         return self.reverse_mapping
#
#     def encode_process(self, process: object) -> List[float]:
#         """
#         Encode a process instance into a list of floats.
#
#         :param process: A process instance
#         :return: A list of floats representing the encoded process
#         """
#         process_type = type(process).__name__
#         encoded = [float(self.encode(process_type))]
#
#         # Use the get_params method to retrieve process parameters
#         params = process.get_params()
#         print(f"Process parameters from the ProcessEncoder: {params}")
#         for param_value in params.values():
#             try:
#                 encoded.append(float(param_value))
#             except (ValueError, TypeError):
#                 print(f"Warning: Skipping non-numeric parameter with value {param_value}")
#
#         return encoded
#
#     def pad_encoded_process(self, encoded_process: List[float], max_params: int = 10) -> List[float]:
#         padded = encoded_process[:1]  # Keep the process type
#         padded.extend(encoded_process[1:max_params + 1])  # Take up to max_params
#         padded.extend([0.0] * (max_params - len(encoded_process[1:])))  # Pad with zeros if needed
#         return padded
#
#
#     def encode_process_with_time(self, process: Union[Dict, object], time: float) -> List[float]:
#         """
#         Encode a process with its time value, maintaining the original total length.
#
#         :param process: The process to encode (either a dictionary or an object)
#         :param time: The time value to include in the encoding
#         :return: A list of floats representing the encoded process with time
#         """
#         encoded_process = self.pad_encoded_process(self.encode_process(process))
#         return [encoded_process[0]] + [time] + encoded_process[1:-1]


class DynamicBatchNorm1d(nn.Module):
    """
    DynamicBatchNorm1d Class

    This class implements a dynamic version of 1D Batch Normalization, designed to handle both
    batch and single-sample inputs. It extends PyTorch's nn.Module.

    Attributes:

        bn (nn.BatchNorm1d): Standard PyTorch 1D Batch Normalization layer.

        running_mean (nn.Parameter): Running mean of the features, not updated during training.

        running_var (nn.Parameter): Running variance of the features, not updated during training.

    The DynamicBatchNorm1d class addresses a common issue in batch normalization where
    single-sample inputs (batch size of 1) can cause problems due to the lack of batch statistics.
    This implementation provides a solution by using running statistics for single samples,
    ensuring stable behavior regardless of batch size.

    Key Features:

    1. Seamless handling of both batch and single-sample inputs.

    2. Uses standard BatchNorm1d for batches to leverage its optimizations.

    3. Fallback to running statistics for single samples to avoid statistical instability.

    Usage:

        layer = DynamicBatchNorm1d(num_features=64)

        output = layer(input_tensor)

    This class is particularly useful in scenarios where the model might receive inputs of
    varying batch sizes, including single samples, such as in online learning or
    when processing sequential data of varying lengths.

    Note: The running mean and variance are not updated during training in this implementation.
    For applications requiring adaptive statistics, additional logic for updating these values
    may be necessary.
    """
    def __init__(self, num_features):
        """
        Initialization of a custom Batch Normalization layer that allows for batch size of 1.

        :param num_features: Number of features in the input tensor
        :type num_features: int
        :return: None
        :rtype: None
        """
        super(DynamicBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(num_features), requires_grad=False)

    def forward(self, x):
        """
        Forward pass of the dynamic Batch Normalization layer.

        :param x: Input tensor of shape (batch_size, num_features)
        :type x: torch.Tensor
        :return: Normalized tensor
        :rtype: torch.Tensor
        """
        if x.size(0) > 1:
            return self.bn(x)
        else:
            return (x - self.running_mean) / torch.sqrt(self.running_var + self.bn.eps)

class NeuralNetwork(nn.Module):
    """
    NeuralNetwork Class

    This class implements a flexible and customizable neural network using PyTorch. It provides a
    wide range of options for network architecture, activation functions, regularization, and
    optimization.

    Attributes:

        input_size (int): Size of the input layer.

        hidden_sizes (List[int]): Sizes of the hidden layers.

        output_size (int): Size of the output layer.

        dropout_rate (float): Dropout rate for regularization.

        batch_norm (bool): Whether to use batch normalization.

        weight_init (str): Weight initialization method.

        learning_rate (float): Learning rate for the optimizer.

        optimizer_name (str): Name of the optimizer to use.

        model (nn.Sequential): The PyTorch sequential model containing all layers.

        optimizer (torch.optim.Optimizer): The optimizer for training the network.

    This NeuralNetwork class offers a high degree of flexibility and customization:

    1. Supports arbitrary numbers and sizes of hidden layers.

    2. Offers multiple activation functions (ReLU, LeakyReLU, Tanh, Sigmoid, ELU).

    3. Includes options for dropout and batch normalization for regularization.

    4. Provides various weight initialization methods (Xavier, He initialization).

    5. Supports different optimizers (Adam, SGD, RMSprop).

    6. Includes methods for genetic algorithm-style operations (mutation, cloning).

    7. Implements save and load functionality for model persistence.

    Usage:

        model = NeuralNetwork(
            input_size=10,
            hidden_sizes=[64, 32],
            output_size=1,
            activation='relu',
            dropout_rate=0.1,
            batch_norm=True,
            weight_init='he_uniform',
            optimizer='adam'
        )

        output = model(input_tensor)

        model.save('model.pth')

        loaded_model = NeuralNetwork.load('model.pth')

    This class is particularly useful for experiments involving neural architecture search,
    evolutionary algorithms, or any scenario requiring dynamic creation and modification of
    neural networks.
    """
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 activation='relu',
                 output_activation=None,
                 dropout_rate=0.0,
                 batch_norm=False,
                 weight_init='xavier_uniform',
                 learning_rate=0.001,
                 optimizer='adam'):

        """
        Initialize a feedforward neural network with customizable hyperparameters.

        :param input_size: Number of input features
        :type input_size: int
        :param hidden_sizes: List of hidden layer sizes
        :type hidden_sizes: List[int]
        :param output_size: Number of output units
        :type output_size: int
        :param activation: Activation function for hidden layers
        :type activation: str
        :param output_activation: Activation function for the output layer
        :type output_activation: str
        :param dropout_rate: Dropout rate (0.0 to 1.0)
        :type dropout_rate: float
        :param batch_norm: Whether to use batch normalization
        :type batch_norm: bool
        :param weight_init: Weight initialization method
        :type weight_init: str
        :param learning_rate: Learning rate for the optimizer
        :type learning_rate: float
        :param optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
        :type optimizer: str
        """

        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.weight_init = weight_init
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer

        # Define activation functions
        self.activation_functions = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            None: nn.Identity()
        }
        self.activation = self.activation_functions[activation]
        self.output_activation = self.activation_functions[output_activation]

        # Create layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if self.batch_norm:
                layers.append(DynamicBatchNorm1d(hidden_size))
            layers.append(self.activation)
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(self.output_activation)

        self.model = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

        # Set up optimizer
        self.optimizer = self._get_optimizer()

    def forward(self, x):
        """
        Forward pass through the neural network.

        :param x: Input tensor
        :type x: torch.Tensor
        :return: Output tensor
        :rtype: torch.Tensor
        """
        return self.model(x)

    def _init_weights(self, module):
        """
        Initialize the weights of the network based on the specified method.

        :param module: Neural network module
        :type module: nn.Module
        :return: None
        :rtype: None
        """
        if isinstance(module, nn.Linear):
            if self.weight_init == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif self.weight_init == 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            elif self.weight_init == 'he_uniform':
                nn.init.kaiming_uniform_(module.weight)
            elif self.weight_init == 'he_normal':
                nn.init.kaiming_normal_(module.weight)
            else:
                raise ValueError(f"Unsupported weight initialization: {self.weight_init}")
            nn.init.zeros_(module.bias)

    def _get_optimizer(self):
        """
        Set up the optimizer based on the specified name.

        :return: Optimizer instance
        :rtype: torch.optim.Optimizer
        """
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        """
        Apply random mutations to the network parameters.
        The mutations are applied by adding Gaussian noise to the parameters.

        :param mutation_rate: Probability of mutating each parameter
        :type mutation_rate: float
        :param mutation_scale: scale of the mutation (the standard deviation of the Gaussian noise)
        :type mutation_scale: float
        :return: None
        :rtype: None
        """
        with torch.no_grad():
            for param in self.parameters():
                mutation_mask = torch.rand_like(param) < mutation_rate
                mutation = torch.randn_like(param) * mutation_scale
                param += mutation * mutation_mask

    def get_num_parameters(self):
        """
        Return the total number of trainable parameters in the network.

        :return: Number of parameters
        :rtype: int
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def clone(self):
        """
        Create a deep copy of the network.

        :return: Cloned network
        :rtype: NeuralNetwork
        """
        clone = NeuralNetwork(
            self.input_size,
            self.hidden_sizes,
            self.output_size,
            activation=next(name for name, func in self.activation_functions.items() if func == self.activation),
            output_activation=next(
                name for name, func in self.activation_functions.items() if func == self.output_activation),
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            weight_init=self.weight_init,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer_name
        )
        clone.load_state_dict(self.state_dict())
        return clone

    def save(self, path):
        """
        Save the model state and hyperparameters to a file.

        :param path: Path to the output file
        :type path: str
        :return: None
        :rtype: None
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'input_size': self.input_size,
                'hidden_sizes': self.hidden_sizes,
                'output_size': self.output_size,
                'activation': next(name for name, func in self.activation_functions.items() if func == self.activation),
                'output_activation': next(
                    name for name, func in self.activation_functions.items() if func == self.output_activation),
                'dropout_rate': self.dropout_rate,
                'batch_norm': self.batch_norm,
                'weight_init': self.weight_init,
                'learning_rate': self.learning_rate,
                'optimizer': self.optimizer_name
            }
        }, path)

    @classmethod
    def load(cls, path):
        """
        Load a model from a file.

        :param path: Path to the input file
        :type path: str
        :return: NeuralNetwork instance
        :rtype: NeuralNetwork
        """
        checkpoint = torch.load(path)
        hyperparameters = checkpoint['hyperparameters']
        model = cls(**hyperparameters)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model

class NeuralNetworkAgent:
    """
    NeuralNetworkAgent Class

    This class represents an agent that uses a neural network to make decisions in a stochastic
    process environment. It encapsulates the neural network along with methods for process
    selection, wealth management, and evolutionary operations.

    Attributes:

        network (NeuralNetwork): The neural network used for decision making.

        wealth (float): The current wealth of the agent.

        accumulated_wealth (float): The total accumulated wealth over time.

        fitness (float): The fitness score of the agent, based on accumulated wealth.

    The NeuralNetworkAgent class is designed to work in environments where decisions are made
    based on encoded representations of stochastic processes. It's particularly suited for
    evolutionary algorithms and reinforcement learning scenarios in financial or economic
    simulations.

    Key features:

    1. Decision making using a neural network on encoded process representations.

    2. Wealth tracking and accumulation based on process returns.

    3. Fitness calculation for use in evolutionary algorithms.

    4. Support for genetic operations like mutation and cloning.

    5. Persistence through save and load functionality.

    Usage:

        network = NeuralNetwork(input_size=10, hidden_sizes=[64, 32], output_size=1)

        agent = NeuralNetworkAgent(network)

        selected_process = agent.select_process(encoded_processes)

        agent.update_wealth(process_return)

        agent.calculate_fitness()

        mutated_agent = agent.clone()

        mutated_agent.mutate()

        agent.save('agent.pth')

        loaded_agent = NeuralNetworkAgent.load('agent.pth')

    This class is ideal for simulations where agents need to learn and adapt to complex,
    stochastic environments, particularly in financial modeling or economic simulations
    involving decision-making under uncertainty.
    """
    def __init__(self, neural_network: NeuralNetwork, initial_wealth: float = 1.0):
        """
        Initialize a neural network-based agent with an initial wealth value.

        :param neural_network: Neural network model for decision-making
        :type neural_network: NeuralNetwork
        :param initial_wealth: Initial wealth value
        :type initial_wealth: float
        """
        self.network = neural_network
        self.wealth = initial_wealth
        self.accumulated_wealth = initial_wealth
        self.fitness = 0.0

    def select_process(self, encoded_processes: List[List[float]]) -> int:
        """
        Select a process based on the neural network's output.

        :param encoded_processes: List of encoded processes
        :type encoded_processes: List[List[float]]
        :return: Index of the selected process
        :rtype: int
        """
        with torch.no_grad():
            outputs = []
            for process in encoded_processes:
                input_tensor = torch.tensor(process, dtype=torch.float32).unsqueeze(0)
                output = self.network(input_tensor)
                outputs.append(output.item())

            # Select the process with the highest output
            selected_index = np.argmax(outputs)
            return selected_index

    def update_wealth(self, process_return: float):
        """
        Update the agent's wealth based on the process return.

        :param process_return: The return from the selected process
        :type process_return: float
        """
        self.wealth *= process_return
        self.accumulated_wealth *= process_return

    def reset_wealth(self):
        """Reset the agent's wealth to the initial value."""
        self.wealth = 1.0

    def calculate_fitness(self):
        """Calculate the fitness of the agent based on accumulated wealth."""
        self.fitness = np.log(self.accumulated_wealth)

    def mutate(self, mutation_rate: float = 0.1, mutation_scale: float = 0.1):
        """Mutate the agent's neural network."""
        self.network.mutate(mutation_rate, mutation_scale)

    def clone(self):
        """Create a clone of the agent with the same network structure but newly initialized weights."""
        cloned_network = self.network.clone()
        return NeuralNetworkAgent(cloned_network, self.wealth)

    def __str__(self):
        """Return a string representation of the agent."""
        return f"NeuralNetworkAgent(wealth={self.wealth:.2f}, accumulated_wealth={self.accumulated_wealth:.2f}, fitness={self.fitness:.2f})"

    def save(self, path: str):
        """
        Save the agent's state to a file.

        :param path: Path to the output file
        :type path: str
        :return: None
        :rtype: None
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'network_class': type(self.network),
            'network_params': {
                'input_size': self.network.input_size,
                'hidden_sizes': self.network.hidden_sizes,
                'output_size': self.network.output_size,
                'activation': next(name for name, func in self.network.activation_functions.items() if
                                   func == self.network.activation),
                'output_activation': next(name for name, func in self.network.activation_functions.items() if
                                          func == self.network.output_activation),
                'dropout_rate': self.network.dropout_rate,
                'batch_norm': self.network.batch_norm,
                'weight_init': self.network.weight_init,
                'learning_rate': self.network.learning_rate,
                'optimizer': self.network.optimizer_name
            },
            'wealth': self.wealth,
            'accumulated_wealth': self.accumulated_wealth,
            'fitness': self.fitness
        }, path)

    @classmethod
    def load(cls, path: str):
        """
        Load an agent from a file.

        :param path: Path to the input file
        :type path: str
        :return: NeuralNetworkAgent instance
        :rtype: NeuralNetworkAgent
        """
        data = torch.load(path)
        network_class = data['network_class']
        network_params = data['network_params']

        # Recreate the network
        network = network_class(**network_params)

        # Load the state dict
        network.load_state_dict(data['network_state_dict'])

        # Create the agent
        agent = cls(network)
        agent.wealth = data['wealth']
        agent.accumulated_wealth = data['accumulated_wealth']
        agent.fitness = data['fitness']
        return agent

class EvolutionaryNeuralNetworkTrainer:
    """
    EvolutionaryNeuralNetworkTrainer Class

    This class implements an evolutionary algorithm for training neural networks to make decisions
    in stochastic process environments. It manages a population of neural network agents, evolves
    them over time, and provides comprehensive logging and visualization capabilities.

    Attributes:

        population_size (int): The number of agents in the population.

        input_size (int): The size of the input layer for the neural networks.

        hidden_sizes (List[int]): The sizes of the hidden layers.

        output_size (int): The size of the output layer.

        processes (List[Union[dict, object]]): The stochastic processes used for training.

        process_encoder (ProcessEncoder): Encoder for the stochastic processes.

        process_times (List[float]): Time horizons for process simulations.

        mutation_rate (float): Rate of mutation for genetic operations.

        mutation_scale (float): Scale of mutations.

        with_exchange (bool): Whether to use population-wide information exchange.

        top_k (int): Number of top agents to consider in exchange.

        exchange_interval (int): Interval for population-wide information exchange.

        initial_wealth (float): Initial wealth of agents.

        keep_top_n (int): Number of top agents to keep after each removal interval.

        removal_interval (int): Interval for removing underperforming agents.

        process_selection_share (float): Proportion of processes to select in each step.

        output_dir (str): Directory for saving outputs.

    This class combines evolutionary algorithms with neural networks to tackle decision-making
    in stochastic environments. It's particularly suited for financial modeling, economic
    simulations, and other domains with complex, uncertain dynamics.

    Key Features:

    1. Flexible neural network architecture for agents.

    2. Support for various stochastic processes as the environment.

    3. Evolutionary mechanisms including mutation and reproduction.

    4. Option for population-wide information exchange.

    5. Comprehensive logging and visualization of training progress.

    6. Persistence of best models and training statistics.

    Usage:

        trainer = EvolutionaryNeuralNetworkTrainer(
            population_size=100,
            input_size=10,
            hidden_sizes=[64, 32],
            output_size=1,
            processes=stochastic_processes,
            process_encoder=encoder,
            process_times=[1.0, 2.0, 5.0],
            with_exchange=True
        )

        final_population, history = trainer.train(n_steps=1000, save_interval=50)

    This class is ideal for researchers and practitioners in fields such as quantitative finance,
    economics, and artificial intelligence who are interested in evolving adaptive agents for
    decision-making in complex, stochastic environments.
    """
    def __init__(self,
                 population_size: int,
                 hidden_sizes: List[int],
                 processes: List[Union[dict, object]],
                 process_encoder: ProcessEncoder,
                 process_times: List[float],
                 input_size: int = 11,
                 output_size: int = 1,
                 mutation_rate: float = 0.1,
                 mutation_scale: float = 0.1,
                 with_exchange: bool = False,
                 top_k: int = 10,
                 exchange_interval: int = 10,
                 initial_wealth: float = 1.0,
                 keep_top_n: int = 50,
                 removal_interval: int = 10,
                 process_selection_share: float = 0.5,
                 output_dir: str = 'output_nn'):

        """
        Initialize the EvolutionaryNeuralNetworkTrainer.

        :param population_size: Number of agents in the population
        :type population_size: int
        :param input_size: Size of the input layer
        :type input_size: int
        :param hidden_sizes: List of hidden layer sizes
        :type hidden_sizes: List[int]
        :param output_size: Size of the output layer
        :type output_size: int
        :param processes: List of stochastic processes
        :type processes: List[Union[dict, object]]
        :param process_encoder: ProcessEncoder instance
        :type process_encoder: ProcessEncoder
        :param process_times: List of time values for process encoding
        :type process_times: List[float]
        :param mutation_rate: Probability of mutating each parameter
        :type mutation_rate: float
        :param mutation_scale: Scale of the mutation (standard deviation of the Gaussian noise)
        :type mutation_scale: float
        :param with_exchange: Whether to perform exchange of top agents
        :type with_exchange: bool
        :param top_k: Number of top agents to select for reproduction
        :type top_k: int
        :param exchange_interval: Interval for exchanging top agents
        :type exchange_interval: int
        :param initial_wealth: Initial wealth value for agents
        :type initial_wealth: float
        :param keep_top_n: Number of top agents to keep in the population
        :type keep_top_n: int
        :param removal_interval: Interval for removing low-performing agents
        :type removal_interval: int
        :param process_selection_share: Share of processes to select for each agent
        :type process_selection_share: float
        :param output_dir: Directory for saving output files
        :type output_dir: str
        """
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.processes = processes
        self.process_encoder = process_encoder
        self.process_times = process_times
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.with_exchange = with_exchange
        self.top_k = top_k
        self.exchange_interval = exchange_interval
        self.initial_wealth = initial_wealth
        self.keep_top_n = keep_top_n
        self.removal_interval = removal_interval
        self.process_selection_share = process_selection_share

        self.population = self.initialize_population()
        self.history = []
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.log_file = os.path.join(self.output_dir, 'training_log.txt')
        self.performance_file = os.path.join(self.output_dir, 'performance_metrics.csv')
        self.best_weights_file = os.path.join(self.output_dir, 'best_weights.pth')

        self.performance_history = []

    def log(self, message: str):
        """
        Log a message to the log file.

        :param message: Message to log
        :type message: str
        :return: None
        :rtype: None
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp} - {message}\n")

    def save_performance_metrics(self):
        """
        Save performance metrics to a CSV file.

        :return: None
        :rtype: None
        """
        with open(self.performance_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.performance_history[0].keys())
            writer.writeheader()
            writer.writerows(self.performance_history)

    def save_best_weights(self, best_agent: NeuralNetworkAgent):
        """
        Save the weights of the best performing agent.

        :param best_agent: Best performing agent
        :type best_agent: NeuralNetworkAgent
        :return: None
        :rtype: None
        """
        torch.save(best_agent.network.state_dict(), self.best_weights_file)

    def visualize_performance(self):
        """
        Create separate visualizations for average and max wealth during training.

        :return: None
        :rtype: None
        """
        steps = [metric['step'] for metric in self.performance_history]
        avg_wealth = [metric['avg_wealth'] for metric in self.performance_history]
        max_wealth = [metric['max_wealth'] for metric in self.performance_history]

        # Function to create and save a single graph
        def create_wealth_graph(wealth_data, ylabel, title, filename):
            plt.figure(figsize=(12, 6))
            plt.plot(steps, wealth_data)
            plt.xlabel('Step')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.show()
            plt.close()

        # Create and save average wealth graph
        create_wealth_graph(
            avg_wealth,
            'Average Wealth',
            'Average Wealth Evolution During Training',
            'average_wealth_evolution.png'
        )

        # Create and save max wealth graph
        create_wealth_graph(
            max_wealth,
            'Maximum Wealth',
            'Maximum Wealth Evolution During Training',
            'max_wealth_evolution.png'
        )

        self.log("Performance visualization graphs have been created and saved.")

    def visualize_neural_network_evolution(self, output_video_path='neural_network_evolution.mp4', output_csv_path='best_agent_params.csv'):
        """
        Create a video visualization of the neural network evolution and save best agent parameters to CSV.

        :param output_video_path: Path to save the output video
        :type output_video_path: str
        :param output_csv_path: Path to save the CSV file with best agent parameters
        :type output_csv_path: str
        :return: None
        :rtype: None
        """
        # Extract data from history
        steps = [entry['step'] for entry in self.history]
        best_params_history = [entry['best_params'] for entry in self.history]

        # Prepare data for visualization
        param_names = list(best_params_history[0].keys())
        param_values = {name: [params[name].numpy().flatten() for params in best_params_history] for name in
                        param_names}

        # Create figure and axes for animation
        fig, axes = plt.subplots(len(param_names), 1, figsize=(12, 4 * len(param_names)))
        if len(param_names) == 1:
            axes = [axes]

        # Initialize plots
        plots = []
        for ax, name in zip(axes, param_names):
            plot, = ax.plot([], [], 'b-')
            ax.set_xlim(0, len(param_values[name][0]))
            ax.set_ylim(min(np.min(values) for values in param_values[name]),
                        max(np.max(values) for values in param_values[name]))
            ax.set_title(f'Evolution of {name}')
            ax.set_xlabel('Parameter Index')
            ax.set_ylabel('Parameter Value')
            plots.append(plot)

        # Animation update function
        def update(frame):
            for plot, name in zip(plots, param_names):
                plot.set_data(range(len(param_values[name][frame])), param_values[name][frame])
            return plots

        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(steps), interval=200, blit=True)

        # Save animation as video
        writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(output_video_path, writer=writer)

        # Save best agent parameters to CSV
        best_agent_params = best_params_history[-1]
        param_dict = {name: best_agent_params[name].numpy().flatten() for name in param_names}
        max_length = max(len(arr) for arr in param_dict.values())
        param_dict = {name: np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=np.nan)
                      for name, arr in param_dict.items()}
        df = pd.DataFrame(param_dict)
        df.to_csv(output_csv_path, index=False)

        plt.close(fig)

        self.log(f"Neural network evolution video saved to {output_video_path}")
        self.log(f"Best agent parameters saved to {output_csv_path}")

    def initialize_population(self) -> List[NeuralNetworkAgent]:
        """
        Initialize the population of neural network agents.

        :return: List of NeuralNetworkAgent instances
        :rtype: List[NeuralNetworkAgent]
        """
        return [NeuralNetworkAgent(NeuralNetwork(self.input_size, self.hidden_sizes, self.output_size),
                                   self.initial_wealth)
                for _ in range(self.population_size)]

    # def encode_process_with_time(self, process, time):
    #     encoded_process = self.process_encoder.pad_encoded_process(self.process_encoder.encode_process(process))
    #     return [encoded_process[0]] + [time] + encoded_process[1:]

    def select_top_agents(self, k: int) -> List[NeuralNetworkAgent]:
        """
        Select the top k agents based on accumulated wealth.

        :param k: Number of top agents to select
        :type k: int
        :return: List of top agents
        :rtype: List[NeuralNetworkAgent]
        """
        sorted_agents = sorted(self.population, key=lambda agent: agent.accumulated_wealth, reverse=True)
        return sorted_agents[:min(k, len(sorted_agents))]

    def reproduce_with_exchange(self):
        """
        Reproduce agents with population-wide information exchange.

        :return: None
        :rtype: None
        """
        top_agents = self.select_top_agents(self.top_k)

        if not top_agents:
            print("Warning: No top agents to reproduce. Reinitializing population.")
            self.population = self.initialize_population()
            return

        # Calculate average parameters
        avg_params = {}
        for name, param in top_agents[0].network.named_parameters():
            avg_params[name] = sum(agent.network.state_dict()[name] for agent in top_agents) / len(top_agents)

        # Create new population with noise added to averaged parameters
        new_population = []
        for _ in range(self.population_size):
            new_network = NeuralNetwork(self.input_size, self.hidden_sizes, self.output_size)
            with torch.no_grad():
                for name, param in new_network.named_parameters():
                    param.copy_(avg_params[name] + torch.randn_like(param) * self.mutation_scale)
            new_population.append(NeuralNetworkAgent(new_network, self.initial_wealth))

        self.population = new_population

    def reproduce_without_exchange(self):
        new_agents = []
        for agent in self.population:
            if agent.wealth >= 2:
                rounded_wealth = int(np.floor(agent.wealth))
                for _ in range(rounded_wealth - 1):
                    new_agent = agent.clone()
                    new_agent.mutate(self.mutation_rate, self.mutation_scale)
                    new_agent.wealth = 1
                    new_agents.append(new_agent)
                agent.wealth = 1
        self.population.extend(new_agents)

    def train(self, n_steps: int, save_interval: int):
        """
        Run the main training loop for the specified number of steps to train the evolutionary neural network.

        :param n_steps: Number of training steps
        :type n_steps: int
        :param save_interval: Interval for saving metrics and weights
        :type save_interval: int
        :return: Tuple of final population and training history
        :rtype: Tuple[List[NeuralNetworkAgent], List[Dict]]
        """
        os.makedirs(self.output_dir, exist_ok=True)
        intermediate_results_path = os.path.join(self.output_dir, 'intermediate_results.csv')
        with open(intermediate_results_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ['Step', 'Agent_Rank'] + [f'Param_{i}' for i in range(self.input_size)] + ['Total_Wealth'])

        self.log(f"Starting training with {n_steps} steps")

        for step in range(n_steps):
            print(f"Step {step}/{n_steps}")
            self.log(f"Step {step}/{n_steps}")

            if not self.population:
                message = "Warning: Population is empty. Reinitializing."
                print(message)
                self.log(message)
                self.population = self.initialize_population()

            num_processes_to_select = max(1, int(len(self.processes) * self.process_selection_share))
            selected_indices = random.sample(range(len(self.processes)), num_processes_to_select)
            selected_processes = [self.processes[i] for i in selected_indices]

            for agent in self.population:
                agent_utilities = []
                for process in selected_processes:
                    process_time = random.choice(self.process_times)
                    encoded_process_with_time = self.process_encoder.encode_process_with_time(process, process_time)
                    input_tensor = torch.tensor(encoded_process_with_time, dtype=torch.float32).unsqueeze(0)
                    utility = agent.network(input_tensor).item()
                    agent_utilities.append((utility, process, process_time))

                best_utility, best_process, best_time = max(agent_utilities, key=lambda x: x[0])

                if isinstance(best_process, dict):
                    process_params = {k: v for k, v in best_process.items() if k != 'symbolic'}
                    process_instance = best_process['symbolic'](**process_params)
                else:
                    process_instance = best_process

                data = process_instance.simulate(t=best_time, timestep=0.1, num_instances=1)
                process_value = data[-1, -1]
                agent.update_wealth(process_value)

            self.population = [agent for agent in self.population if agent.wealth > 0]

            if not self.population:
                message = "Warning: All agents have been removed. Reinitializing population."
                print(message)
                self.log(message)
                self.population = self.initialize_population()
                continue

            if step % self.removal_interval == 0 and step > 0:
                message = f'There were {len(self.population)} agents.'
                print(message)
                self.log(message)
                self.population.sort(key=lambda a: a.accumulated_wealth, reverse=True)
                self.population = self.population[:min(self.keep_top_n, len(self.population))]
                message = f"Kept top {len(self.population)} agents."
                print(message)
                self.log(message)

            if self.with_exchange and step % self.exchange_interval == 0:
                self.reproduce_with_exchange()
            else:
                self.reproduce_without_exchange()

            # Calculate and log performance metrics
            avg_wealth = np.mean([agent.wealth for agent in self.population])
            max_wealth = np.max([agent.accumulated_wealth for agent in self.population])
            best_agent = max(self.population, key=lambda a: a.accumulated_wealth)

            self.performance_history.append({
                'step': step,
                'avg_wealth': avg_wealth,
                'max_wealth': max_wealth,
                'population_size': len(self.population)
            })

            self.log(f"Step {step} metrics - Avg Wealth: {avg_wealth:.2f}, Max Wealth: {max_wealth:.2f}")

            if step % save_interval == 0:
                self.history.append({
                    'step': step,
                    'n_agents': len(self.population),
                    'avg_wealth': avg_wealth,
                    'best_params': best_agent.network.state_dict()
                })
                self.save_performance_metrics()
                self.save_best_weights(best_agent)
                self.visualize_performance()
                self.log(f"Saved metrics, weights, and visualization at step {step}")

            if step % 10 == 0:
                self.population.sort(key=lambda a: a.accumulated_wealth, reverse=True)
                with open(intermediate_results_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    for rank, agent in enumerate(self.population[:self.keep_top_n]):
                        csv_writer.writerow([step, rank] + list(agent.network.state_dict().values())[0].tolist() + [
                            agent.accumulated_wealth])

            if best_agent.accumulated_wealth > 1000000:
                for agent in self.population:
                    agent.accumulated_wealth /= 1000000

        output_video_path = os.path.join(self.output_dir, 'neural_network_evolution.mp4')
        output_csv_path = os.path.join(self.output_dir, 'best_agent_parameters.csv')
        self.visualize_neural_network_evolution(output_video_path, output_csv_path)

        self.log("Training completed")
        return self.population, self.history

    def load_best_weights(self, agent: NeuralNetworkAgent):
        """
        Load the best weights into an agent's network.

        :param agent: Agent instance
        :type agent: NeuralNetworkAgent
        :return: None
        :rtype
        """
        agent.network.load_state_dict(torch.load(self.best_weights_file))

# example usage with if main

if __name__ == '__main__':

    # Define process types
    process_types = [GeometricBrownianMotion, BrownianMotion]

    # Define parameter ranges for each process type
    param_ranges = {
        'GeometricBrownianMotion': {
            'drift': (-0.2, 0.2),
            'volatility': (0.01, 0.5)
        },
        'BrownianMotion': {
            'drift': (-0.4, 0.5),
            'scale': (0.01, 0.6)
        }
    }

    # Generate processes
    processes = generate_processes(100, process_types, param_ranges)

    # print(processes)

    # Create a ProcessEncoder instance
    encoder = ProcessEncoder()

    # Encode and pad all processes
    encoded_processes = [encoder.pad_encoded_process(encoder.encode_process(p)) for p in processes]

    # print(encoded_processes)

    # Create a network with custom hyperparameters
    net = NeuralNetwork(
        input_size=11,
        hidden_sizes=[20, 10],
        output_size=1,
        activation='leaky_relu',
        output_activation='sigmoid',
        dropout_rate=0.1,
        batch_norm=True,
        weight_init='he_uniform',
        learning_rate=0.001,
        optimizer='adam'
    )

    # Create some dummy input
    input_data = torch.randn(1, 11)

    # Forward pass
    output = net(input_data)

    print(f"Network output: {output.item()}")
    print(f"Number of parameters: {net.get_num_parameters()}")

    # Mutate the network
    net.mutate(mutation_rate=0.1, mutation_scale=0.1)

    # Clone the network
    net_clone = net.clone()

    # Save the network
    net.save('my_network.pth')

    # Load the network
    loaded_net = NeuralNetwork.load('my_network.pth')

    # Create an agent
    agent = NeuralNetworkAgent(net)

    # Example list of encoded processes
    encoded_processes = [
        [1.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.05, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]

    # Select a process
    selected_index = agent.select_process(encoded_processes)
    print(f"Selected process index: {selected_index}")

    # Update wealth (assuming we've simulated the selected process and got a return)
    agent.update_wealth(1.05)  # 5% return
    print(f"Updated wealth: {agent.wealth}")

    # Calculate fitness
    agent.calculate_fitness()
    print(f"Agent fitness: {agent.fitness}")

    # Mutate the agent
    agent.mutate()

    # Clone the agent
    cloned_agent = agent.clone()

    # Save and load the agent
    agent.save("agent_state.pth")
    loaded_agent = NeuralNetworkAgent.load("agent_state.pth")

    process_encoder = ProcessEncoder()

    process_times = [1.0, 2.0, 5.0, 10.0]

    trainer = EvolutionaryNeuralNetworkTrainer(
        population_size=10,
        input_size=11,  # Assuming 11 input features for the encoded process
        hidden_sizes=[20, 10],
        output_size=1,
        processes=processes,
        process_encoder=process_encoder,
        with_exchange=False,  # Set to False for the algorithm without exchange
        top_k=10,
        exchange_interval=10,
        keep_top_n=5,
        removal_interval=3,
        process_selection_share=0.5,
        process_times=process_times,
        output_dir='output_nn/2',
    )

    population, history = trainer.train(n_steps=100, save_interval=50)
    best_agent = max(population, key=lambda agent: agent.accumulated_wealth)
    print(f"Best agent accumulated wealth: {best_agent.accumulated_wealth}")

class ReinforcementEvolutionaryTrainer:
    """
    ReinforcementEvolutionaryTrainer Class
    This class implements a hybrid approach combining reinforcement learning and evolutionary algorithms
    for training neural network agents in stochastic process environments. It manages a population of
    agents, evolves them over time, and applies reinforcement learning techniques to improve their
    decision-making capabilities.

    Attributes:

        population_size (int): The number of agents in the population.

        input_size (int): The size of the input layer for the neural networks.

        hidden_sizes (List[int]): The sizes of the hidden layers.

        output_size (int): The size of the output layer.

        processes (List[Union[dict, object]]): The stochastic processes used for training.

        process_encoder (ProcessEncoder): Encoder for the stochastic processes.

        process_times (List[float]): Time horizons for process simulations.

        learning_rate (float): Learning rate for the reinforcement learning updates.

        mutation_rate (float): Rate of mutation for genetic operations.

        mutation_scale (float): Scale of mutations.

        rl_interval (int): Interval for applying reinforcement learning updates.

        elite_percentage (float): Percentage of top-performing agents to consider as elite.


    This class combines evolutionary algorithms with reinforcement learning to create a powerful
    hybrid approach for training agents in complex, stochastic environments. It's particularly
    well-suited for financial modeling, economic simulations, and other domains with intricate,
    uncertain dynamics where both long-term evolution and short-term learning are beneficial.

    Key Features:

    Flexible neural network architecture for agents.

    Support for various stochastic processes as the environment.

    Reinforcement learning updates to improve agent performance.

    Evolutionary mechanisms including mutation, crossover, and elite selection.

    Periodic population renewal based on fitness.

    Adaptive learning through a combination of exploration and exploitation.

    Usage:

    trainer = ReinforcementEvolutionaryTrainer(
    population_size=100,
    input_size=10,
    hidden_sizes=[64, 32],
    output_size=1,
    processes=stochastic_processes,
    process_encoder=encoder,
    process_times=[1.0, 2.0, 5.0],
    learning_rate=0.001,
    mutation_rate=0.1,
    rl_interval=10
    )

    final_population = trainer.train(n_steps=1000)

    This class is ideal for researchers and practitioners in fields such as quantitative finance,
    economics, and artificial intelligence who are interested in developing sophisticated,
    adaptive agents capable of making informed decisions in complex, stochastic environments.
    The combination of evolutionary algorithms and reinforcement learning provides a robust
    framework for discovering and refining effective strategies in these challenging domains.
    """
    def __init__(
        self,
        population_size: int,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        processes: List[Union[dict, object]],
        process_encoder: ProcessEncoder,
        process_times: List[float],
        learning_rate: float = 0.001,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.1,
        rl_interval: int = 10,
        elite_percentage: float = 0.2,
        output_dir: str = 'output_nn'
    ):
        """
        Initialize the ReinforcementEvolutionaryTrainer.

        :param population_size: Number of agents in the population
        :type population_size: int
        :param input_size: Size of the input layer
        :type input_size: int
        :param hidden_sizes: List of hidden layer sizes
        :type hidden_sizes: List[int]
        :param output_size: Size of the output layer
        :type output_size: int
        :param processes: List of stochastic processes
        :type processes: List[Union[dict, object]]
        :param process_encoder: ProcessEncoder instance
        :type process_encoder: ProcessEncoder
        :param process_times: List of time values for process encoding
        :type process_times: List[float]
        :param learning_rate: Learning rate for the neural networks
        :type learning_rate: float
        :param mutation_rate: Probability of mutating each parameter
        :type mutation_rate: float
        :param mutation_scale: Scale of the mutation (standard deviation of the Gaussian noise)
        :type mutation_scale: float
        :param rl_interval: Interval for reinforcement learning updates
        :type rl_interval: int
        :param elite_percentage: Percentage of elite agents to keep in each generation
        :type elite_percentage: float
        :param output_dir: Directory for saving output files
        :type output_dir: str
        """
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.processes = processes
        self.process_encoder = process_encoder
        self.process_times = process_times
        self.learning_rate = learning_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.rl_interval = rl_interval
        self.elite_percentage = elite_percentage

        self.population = self.initialize_population()
        self.optimizers = [optim.Adam(agent.network.parameters(), lr=self.learning_rate) for agent in self.population]

        self.history = []
        self.performance_history = []
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.log_file = os.path.join(self.output_dir, 'training_log.txt')
        self.performance_file = os.path.join(self.output_dir, 'performance_metrics.csv')
        self.best_weights_file = os.path.join(self.output_dir, 'best_weights.pth')

    def log(self, message: str):
        """
        Log a message to the log file.

        :param message: Message to log
        :type message: str
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp} - {message}\n")

    def save_performance_metrics(self):
        """
        Save performance metrics to a CSV file.
        """
        if not self.performance_history:
            return
        with open(self.performance_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.performance_history[0].keys())
            writer.writeheader()
            writer.writerows(self.performance_history)

    def save_best_weights(self, best_agent: NeuralNetworkAgent):
        """
        Save the weights of the best-performing agent.

        :param best_agent: Best-performing agent
        :type best_agent: NeuralNetworkAgent
        """
        torch.save(best_agent.network.state_dict(), self.best_weights_file)

    def visualize_performance(self):
        """
        Create visualizations for average and maximum fitness during training.
        """
        steps = [metric['step'] for metric in self.performance_history]
        avg_fitness = [metric['avg_fitness'] for metric in self.performance_history]
        max_fitness = [metric['max_fitness'] for metric in self.performance_history]

        # Function to create and save a single graph
        def create_fitness_graph(fitness_data, ylabel, title, filename):
            plt.figure(figsize=(12, 6))
            plt.plot(steps, fitness_data)
            plt.xlabel('Step')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.show()
            plt.close()

        # Create and save average fitness graph
        create_fitness_graph(
            avg_fitness,
            'Average Fitness',
            'Average Fitness Evolution During Training',
            'average_fitness_evolution.png'
        )

        # Create and save maximum fitness graph
        create_fitness_graph(
            max_fitness,
            'Maximum Fitness',
            'Maximum Fitness Evolution During Training',
            'max_fitness_evolution.png'
        )

        self.log("Performance visualization graphs have been created and saved.")

    def visualize_neural_network_evolution(self, output_video_path='neural_network_evolution.mp4',
                                           output_csv_path='best_agent_params.csv'):
        """
        Create a video visualization of the neural network evolution and save best agent parameters to CSV.

        :param output_video_path: Path to save the output video
        :type output_video_path: str
        :param output_csv_path: Path to save the CSV file with best agent parameters
        :type output_csv_path: str
        """
        # Extract data from history
        steps = [entry['step'] for entry in self.history]
        best_params_history = [entry['best_params'] for entry in self.history]

        if not best_params_history:
            self.log("No history data available for visualization.")
            return

        # Prepare data for visualization
        param_names = list(best_params_history[0].keys())
        param_values = {name: [params[name].cpu().numpy().flatten() for params in best_params_history] for name in
                        param_names}

        # Create figure and axes for animation
        num_params = len(param_names)
        fig, axes = plt.subplots(num_params, 1, figsize=(12, 4 * num_params))
        if num_params == 1:
            axes = [axes]

        # Initialize plots
        plots = []
        for ax, name in zip(axes, param_names):
            plot, = ax.plot([], [], 'b-')
            ax.set_xlim(0, len(param_values[name][0]))
            all_values = np.concatenate(param_values[name])
            ax.set_ylim(np.min(all_values), np.max(all_values))
            ax.set_title(f'Evolution of {name}')
            ax.set_xlabel('Parameter Index')
            ax.set_ylabel('Parameter Value')
            plots.append(plot)

        # Animation update function
        def update(frame):
            for plot, name in zip(plots, param_names):
                plot.set_data(range(len(param_values[name][frame])), param_values[name][frame])
            return plots

        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(steps), interval=200, blit=True)

        # Save animation as video
        writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='ReinforcementEvolutionaryTrainer'), bitrate=1800)
        anim.save(os.path.join(self.output_dir, output_video_path), writer=writer)

        # Save best agent parameters to CSV
        best_agent_params = best_params_history[-1]
        param_dict = {name: best_agent_params[name].cpu().numpy().flatten() for name in param_names}
        max_length = max(len(arr) for arr in param_dict.values())
        param_dict = {name: np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=np.nan)
                      for name, arr in param_dict.items()}
        df = pd.DataFrame(param_dict)
        df.to_csv(os.path.join(self.output_dir, output_csv_path), index=False)

        plt.close(fig)

        self.log(f"Neural network evolution video saved to {output_video_path}")
        self.log(f"Best agent parameters saved to {output_csv_path}")

    def initialize_population(self) -> List[NeuralNetworkAgent]:
        """
        Initialize the population of neural network agents.

        :return: List of NeuralNetworkAgent instances
        :rtype: List[NeuralNetworkAgent]
        """
        return [NeuralNetworkAgent(NeuralNetwork(self.input_size, self.hidden_sizes, self.output_size))
                for _ in range(self.population_size)]

    def select_process(self, agent: NeuralNetworkAgent) -> Tuple[object, float]:
        """
        Select a process and time horizon for an agent to interact with.

        :param agent: NeuralNetworkAgent instance
        :type agent: NeuralNetworkAgent
        :return: Tuple of process and time
        :rtype: Tuple[object, float]
        """

        encoded_processes = [self.process_encoder.encode_process_with_time(p, random.choice(self.process_times))
                             for p in random.sample(self.processes, k=min(10, len(self.processes)))]
        process_index = agent.select_process(encoded_processes)
        selected_process = self.processes[process_index]
        selected_time = random.choice(self.process_times)
        return selected_process, selected_time

    def simulate_process(self, process: object, time: float) -> float:
        """
        Simulate the selected process for the given time horizon.

        :param process: Stochastic process instance
        :type process: object
        :param time: Time horizon for simulation
        :type time: float
        :return: Final value of the process
        :rtype: float
        """
        if isinstance(process, dict):
            process_params = {k: v for k, v in process.items() if k != 'symbolic'}
            process_instance = process['symbolic'](**process_params)
        else:
            process_instance = process
        data = process_instance.simulate(t=time, timestep=0.1, num_instances=1)
        return data[-1, -1]

    def calculate_reward(self, initial_wealth: float, final_wealth: float) -> float:
        """
        Calculate the reward based on the change in wealth.

        :param initial_wealth: Initial wealth value
        :type initial_wealth: float
        :param final_wealth: Final wealth value
        :type final_wealth: float
        :return: Reward value
        :rtype: float
        """
        return np.log(final_wealth / initial_wealth)

    def reinforce(self, agent: NeuralNetworkAgent, optimizer: optim.Optimizer, reward: float):
        """
        Apply reinforcement learning update to an agent.
        This is done by performing a gradient ascent step on the agent's neural network.

        :param agent: NeuralNetworkAgent instance
        :type agent: NeuralNetworkAgent
        :param optimizer: Optimizer for the neural network
        :type optimizer: optim.Optimizer
        :param reward: Reward value
        :type reward: float
        :return: None
        """
        optimizer.zero_grad()
        loss = -agent.network(agent.last_input) * reward  # Negative because we want to maximize reward
        loss.backward()
        optimizer.step()

    def mutate(self, agent: NeuralNetworkAgent):
        """
        Apply mutation to an agent's neural network.
        The mutation is applied to the parameters of the network based on the mutation rate and scale.

        :param agent: NeuralNetworkAgent instance
        :type agent: NeuralNetworkAgent
        :return: None
        :rtype: None
        """
        agent.mutate(self.mutation_rate, self.mutation_scale)

    def select_elite(self, population: List[NeuralNetworkAgent]) -> List[NeuralNetworkAgent]:
        """
        Select the elite agents from the population.

        :param population: List of NeuralNetworkAgent instances
        :type population: List[NeuralNetworkAgent]
        :return: List of elite agents
        :rtype: List[NeuralNetworkAgent]
        """
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        elite_count = int(self.population_size * self.elite_percentage)
        return sorted_population[:elite_count]

    def crossover(self, parent1: NeuralNetworkAgent, parent2: NeuralNetworkAgent) -> NeuralNetworkAgent:
        """
        Perform crossover between two parent agents to create a child agent.
        It means that the child agent inherits some parameters from each parent.

        :param parent1: First parent agent
        :type parent1: NeuralNetworkAgent
        :param parent2: Second parent agent
        :type parent2: NeuralNetworkAgent
        :return: Child agent
        :rtype: NeuralNetworkAgent
        """
        child = parent1.clone()
        for child_param, parent2_param in zip(child.network.parameters(), parent2.network.parameters()):
            mask = torch.rand_like(child_param) < 0.5
            child_param.data[mask] = parent2_param.data[mask]
        return child

    def train(self, n_steps: int):
        """
        Run the main training loop for the specified number of steps.

        :param n_steps: Number of training steps
        :type n_steps: int
        :return: List of final agents in the population
        :rtype: List[NeuralNetworkAgent]
        """
        for step in range(n_steps):
            for agent, optimizer in zip(self.population, self.optimizers):
                process, time = self.select_process(agent)
                encoded_process = self.process_encoder.encode_process_with_time(process, time)
                agent.last_input = torch.tensor(encoded_process, dtype=torch.float32).unsqueeze(0)

                initial_wealth = agent.wealth
                process_return = self.simulate_process(process, time)
                agent.update_wealth(process_return)
                reward = self.calculate_reward(initial_wealth, agent.wealth)
                agent.fitness = reward

                if step % self.rl_interval == 0:
                    self.reinforce(agent, optimizer, reward)

            if step % self.rl_interval == 0:
                elite = self.select_elite(self.population)
                new_population = elite.copy()

                while len(new_population) < self.population_size:
                    parent1, parent2 = random.sample(elite, 2)
                    child = self.crossover(parent1, parent2)
                    self.mutate(child)
                    new_population.append(child)

                self.population = new_population
                self.optimizers = [optim.Adam(agent.network.parameters(), lr=self.learning_rate)
                                   for agent in self.population]

            if step % 100 == 0:
                best_agent = max(self.population, key=lambda x: x.fitness)
                print(f"Step {step}, Best Fitness: {best_agent.fitness:.4f}, Best Wealth: {best_agent.wealth:.4f}")

        return self.population

# Example usage
if __name__ == "__main__":
    # Assuming you have defined processes, process_encoder, and process_times as before
    trainer = ReinforcementEvolutionaryTrainer(
        population_size=50,
        input_size=11,
        hidden_sizes=[20, 10],
        output_size=1,
        processes=processes,
        process_encoder=process_encoder,
        process_times=process_times,
        learning_rate=0.001,
        mutation_rate=0.1,
        mutation_scale=0.1,
        rl_interval=10,
        elite_percentage=0.2
    )

    final_population = trainer.train(n_steps=1000)
    best_agent = max(final_population, key=lambda x: x.fitness)
    print(f"Training completed. Best agent fitness: {best_agent.fitness:.4f}, Best agent wealth: {best_agent.wealth:.4f}")



