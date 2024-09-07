"""
agents Submodule Overview

The **`agents`** module focuses on creating and managing economic agents that optimize their utility functions over time in a stochastic environment. It provides various algorithms and tools for simulating agent behavior, evolving utility functions, and running multi-agent evolutionary processes. These agents interact with multiple stochastic processes, making decisions based on expected utility and wealth maximization.

Key Features:

1. **Utility Functions**:

   - Each agent uses a general utility function parameterized by alpha, beta, gamma, delta, and epsilon.

   - Utility functions are used to calculate the expected utility from various stochastic processes.

2. **Expected Utility**:

   - The module allows agents to calculate expected utility either numerically (via process simulations) or symbolically (via integral approximations).

3. **Evolutionary Algorithms**:

   - Agents evolve over time using evolutionary algorithms that simulate wealth accumulation and utility optimization.

   - Agents with higher utility accumulate more wealth and influence the population's evolution.

   - Includes features like mutation, agent reproduction, and selection of top-performing agents.

4. **Processes**:

   - Agents interact with multiple stochastic processes, such as Geometric Brownian Motion or Geometric Lévy processes, to maximize their wealth.

   - Agents select and invest in the best process based on expected utility.

5. **Multi-Agent Simulations**:

   - The module provides tools to simulate the behavior of a population of agents over multiple time steps.

   - Agents are removed or duplicated based on their performance, and new agents are created through mutation.

6. **Visualization**:

   - Includes tools for visualizing agent evolution, utility function evolution, and other trends over time.

   - Provides methods to generate animations and CSV logs for post-simulation analysis.

7. **Support for Numerical and Symbolic Computation**:

   - Both symbolic and numerical approaches are supported for computing expected utility, offering flexibility in different simulation scenarios.

8. **Evolution with Multiple Processes**:

   - Agents can select from multiple stochastic processes, optimizing their utility functions in a diverse environment.

   - The module supports the creation of multiple stochastic processes and allows agents to interact with them over time.

Example Usage:

### Basic Usage of Evolutionary Algorithm:

from ergodicity.process.multiplicative import GeometricBrownianMotion

from ergodicity.agents import Agent_utility

# Define parameters for agents

param_means = np.array([1.0, 1.0, 0.5, 1.0])

param_stds = np.array([0.1, 0.1, 0.05, 0.1])

mutation_rate = 0.01

# Define a set of stochastic processes

processes = [
    GeometricBrownianMotion(drift=0.02, volatility=0.15),
    GeometricBrownianMotion(drift=0.03, volatility=0.18)
]

# Run the evolutionary algorithm

final_agents, history = Agent_utility.evolutionary_algorithm(
    n_agents=100,
    n_steps=1000,
    save_interval=50,
    processes=processes,
    param_means=param_means,
    param_stds=param_stds,
    mutation_rate=mutation_rate,
    stochastic_process_class=GeometricBrownianMotion,
    keep_top_n=50,
    removal_interval=10,
    process_selection_share=0.5
)

# Visualize agent evolution

Agent_utility.visualize_agent_evolution(history)
"""

# from ergodicity.process.multiplicative import GeometricBrownianMotion, GeometricLevyProcess
import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import List, Callable, Type, Union, Dict, Any, Tuple
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import os
from scipy import integrate as scipy_integrate
import warnings
import inspect
import time

from ergodicity.custom_warnings import InDevelopmentWarning


def general_utility_function(x, alpha, beta, gamma, delta, epsilon):
    """
    Calculates the utility for a given x and set of parameters.
    Works with both SymPy symbols and numpy arrays.

    :param x: Input value or array
    :type x: Union[float, np.ndarray, sp.Expr, sp.Symbol]
    :param alpha: Utility function parameter
    :type alpha: float
    :param beta: Utility function parameter
    :type beta: float
    :param gamma: Utility function parameter
    :type gamma: float
    :param delta: Utility function parameter
    :type delta: float
    :param epsilon: Utility function parameter
    :type epsilon: float
    :return: The calculated utility expression or array
    :rtype: Union[float, np.ndarray]
    """
    if isinstance(x, (sp.Expr, sp.Symbol)):
        return alpha * (1 - sp.exp(-beta * x ** gamma)) / (1 - sp.exp(-delta)) + epsilon
    else:
        return alpha * (1 - np.exp(-beta * np.power(np.abs(x), gamma))) / (1 - np.exp(-delta)) + epsilon

@dataclass
class Agent_utility:
    """
    Represents an agent with utility-based decision making in stochastic processes.

    This class implements an agent that can evaluate and interact with various stochastic processes
    based on a parameterized utility function. It supports both symbolic and numerical methods for
    calculating expected utilities, and includes evolutionary algorithms for optimizing agent parameters.

    Attributes:

        params (np.ndarray): Parameters of the agent's utility function [alpha, beta, gamma, delta, epsilon].

        wealth (float): Current wealth of the agent.

        total_accumulated_wealth (float): Total accumulated wealth of the agent over time.

    The class provides comprehensive tools for modeling agent behavior in complex stochastic environments,
    including utility calculation, parameter optimization, and visualization of results. It is particularly
    useful for studying optimal strategies in financial markets, decision making under uncertainty, and
    evolutionary dynamics in economic systems.
    """
    params: np.ndarray  # [alpha, beta, gamma, delta]
    wealth: float = 1.0
    total_accumulated_wealth: float = 1.0  # New field to track total wealth

    @staticmethod
    def expected_utility(process_dict: Dict[str, Any], params: np.ndarray, t: float = None) -> Union[
        Callable[[float], float], float]:
        """
        Calculate the expected utility of a process using robust adaptive numerical integration.

        :param process_dict: Dictionary containing the symbolic function and process parameters
        :type process_dict: Dict[str, Any]
        :param params: Parameters of the utility function [alpha, beta, gamma, delta, epsilon]
        :type params: np.ndarray
        :param t: Time horizon for the process. If None, returns a function of t.
        :type t: float
        :return: Either a function that computes expected utility for any t, or the numerical value for a specific t
        :rtype: Union[Callable[[float], float], float]
        """
        start_time = time.time()

        t_sym = sp.Symbol('t', positive=True)
        W = sp.Symbol('W')
        alpha, beta, gamma, delta, epsilon = params

        symbolic_func = process_dict['symbolic']
        process_params = {k: v for k, v in process_dict.items() if k != 'symbolic'}

        X = symbolic_func(t_sym, W, **process_params)
        utility = Agent_utility.general_utility_function(X, alpha, beta, gamma, delta, epsilon)
        integrand = utility * sp.exp(-W ** 2 / (2 * t_sym)) / sp.sqrt(2 * sp.pi * t_sym)
        np_integrand = sp.lambdify((W, t_sym), integrand, modules=['numpy'])

        def expected_utility_for_t(t_val: float) -> float:
            def integrand_for_quad(w):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = np_integrand(w, t_val)
                return np.where(np.isfinite(result), result, 0.0)

            def adaptive_integrate(func, a, b, tol=1.49e-8):
                def quad_wrapper(a, b):
                    return scipy_integrate.quad(func, a, b, epsabs=tol, epsrel=tol, limit=200)[0]

                result = quad_wrapper(a, b)
                error = np.inf
                iterations = 0
                max_iterations = 10

                while error > tol and iterations < max_iterations:
                    mid = (a + b) / 2
                    left = quad_wrapper(a, mid)
                    right = quad_wrapper(mid, b)
                    new_result = left + right
                    error = abs(new_result - result)
                    result = new_result
                    iterations += 1

                return result

            try:
                result = adaptive_integrate(integrand_for_quad, -10, 10)
                if not np.isfinite(result):
                    raise ValueError("Integration result is not finite")
                return result
            except Exception as e:
                print(f"Integration failed for t={t_val}: {e}")
                try:
                    x = np.linspace(-10, 10, 10000)
                    y = integrand_for_quad(x)
                    result = np.trapz(y, x)
                    if not np.isfinite(result):
                        raise ValueError("Trapz integration result is not finite")
                    return result
                except Exception as e2:
                    print(f"Fallback integration also failed: {e2}")
                    return 0.0  # Return a default value if all integration methods fail

        end_time = time.time()
        execution_time = end_time - start_time
        # print(f"Execution time of the symbolic expected utility method: {execution_time:.2f} seconds")

        if t is None:
            return expected_utility_for_t
        else:
            return expected_utility_for_t(t)

    @staticmethod
    def numerical_expected_utility(process: Union[Dict[str, Any], object], params: np.ndarray,
                                   stochastic_process_class: Type, t: float = None, num_instances: int = 1000) -> Union[
        Callable[[float], float], float]:
        """
        Generate multiple instances of the process and take their average utility to approximate the expected utility.

        :param process: Either a dictionary containing process parameters or an instance of a stochastic process
        :type process: Union[Dict[str, Any], object]
        :param params: Parameters of the utility function [alpha, beta, gamma, delta, epsilon]
        :type params: np.ndarray
        :param stochastic_process_class: The class of the stochastic process
        :type stochastic_process_class: Type
        :param t: Time horizon for the process. If None, returns a function of t.
        :type t: float
        :return: Either a function that computes expected utility for any t, or the numerical value for a specific t
        :rtype: Union[Callable[[float], float], float]
        """
        start_time = time.time()

        def numerical_expected_utility_for_t(t_val: float) -> float:
            if isinstance(process, dict):
                # For dictionary-based processes
                process_params = {k: v for k, v in process.items() if k != 'symbolic'}
                process_instance = stochastic_process_class(**process_params)
            else:
                # For direct instances of stochastic processes
                process_instance = process

            # Simulate the process
            data = process_instance.simulate(t=t_val, timestep=0.1, num_instances=num_instances)
            data = data[1:, -1]
            utility = Agent_utility.general_utility_function_np(data, *params)
            return np.mean(utility)

        end_time = time.time()
        execution_time = end_time - start_time
        # print(f"Execution time of the numerical expected utility method: {execution_time:.2f} seconds")

        if t is None:
            return numerical_expected_utility_for_t
        else:
            return numerical_expected_utility_for_t(t)

    @staticmethod
    def compare_numerical_and_symbolic_expected_utility(process_dict: Dict[str, Any], params: np.ndarray,
                                                        stochastic_process_class: Type, t: float = None):
        """
        Compare the numerical and symbolic expected utility calculations for a given process and parameters.

        :param process_dict: Dictionary containing the symbolic function and process parameters
        :type process_dict: Dict[str, Any]
        :param params: Parameters of the utility function [alpha, beta, gamma, delta, epsilon]
        :type params: np.ndarray
        :param stochastic_process_class: The class of the stochastic process
        :type stochastic_process_class: Type
        :param t: Time horizon for the process. If None, compares functions instead of specific values.
        :type t: float
        :return: None (prints the comparison results)
        :rtype: None
        """
        symbolic_result = Agent_utility.expected_utility(process_dict, params, t)
        numerical_result = Agent_utility.numerical_expected_utility(process_dict, params, stochastic_process_class, t)

        if t is None:
            print("Comparing utility functions:")
            test_t_values = [0.5, 1.0, 2.0]  # Example t values for comparison
            for test_t in test_t_values:
                sym_value = symbolic_result(test_t)
                num_value = numerical_result(test_t)
                print(f"  At t = {test_t}:")
                print(f"    Symbolic utility: {sym_value}")
                print(f"    Numerical utility: {num_value}")
                print(f"    Difference: {abs(sym_value - num_value)}")
        else:
            print(f"Comparing utilities at t = {t}:")
            print(f"  Symbolic utility: {symbolic_result}")
            print(f"  Numerical utility: {numerical_result}")
            print(f"  Difference: {abs(symbolic_result - numerical_result)}")

    # example for GeometricBrownianMotion:
    # process = GeometricBrownianMotion()
    # params = np.array([1, 1, 1, 1, 0])
    # compare_numerical_and_symbolic_expected_utility(process, params, 1.0, GeometricBrownianMotion)

    @staticmethod
    def general_utility_function(x, alpha, beta, gamma, delta, epsilon):
        """
        Symbolic representation of the general utility function.

        :param x: The input value
        :type x: sp.Symbol
        :param alpha, beta, gamma, delta, epsilon: Utility function parameters
        :type alpha, beta, gamma, delta, epsilon: float
        :return: The utility function expression
        :rtype: sp.Expr
        """
        return alpha * (1 - sp.exp(-beta * x ** gamma)) / (1 - sp.exp(-delta)) + epsilon

    @staticmethod
    def general_utility_function_np(x, alpha, beta, gamma, delta, epsilon):
        """
        Numpy version of the general utility function with overflow protection

        :param x: The input value(s)
        :type x: Union[float, np.ndarray]
        :param alpha, beta, gamma, delta, epsilon: Utility function parameters
        :type alpha, beta, gamma, delta, epsilon: float
        :return: The utility function value(s)
        :rtype: Union[float, np.ndarray]
        """
        # Clip extremely large values to prevent overflow
        max_exp = 709  # np.log(np.finfo(np.float64).max)
        clipped_x = np.clip(np.abs(x), 0, max_exp)
        clipped_beta = np.clip(beta, -max_exp, max_exp)
        clipped_delta = np.clip(delta, -max_exp, max_exp)

        term1 = alpha * (1 - np.exp(-clipped_beta * clipped_x ** gamma))
        term2 = 1 - np.exp(-clipped_delta)

        # Avoid division by zero
        result = np.where(term2 != 0, term1 / term2, 0)
        return result + epsilon

    @staticmethod
    def visualize_utility_function_evolution(history, output_video_path, output_csv_path):
        """
        Create an animation of the evolution of the best agent's utility function over time.

        :param history: List of dictionaries containing step, best_params, and other data
        :type history: List[Dict[str, Any]]
        :param output_video_path: Path to save the output video
        :type output_video_path: str
        :param output_csv_path: Path to save the output CSV file
        :type output_csv_path: str
        :return: None
        :rtype: None
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.linspace(0, 10, 1000)  # Extended range to include negative values

        # Open the CSV file outside of the animation function
        csvfile = open(output_csv_path, 'w', newline='')
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Step', 'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'])

        def animate(i):
            ax.clear()
            params = history[i]['best_params']
            y = Agent_utility.general_utility_function_np(x, *params)
            ax.plot(x, y)
            ax.set_title(f'Best Agent Utility Function - Step {history[i]["step"]}')
            ax.set_xlabel('x')
            ax.set_ylabel('Utility')

            # Dynamically set y-axis limits
            y_min, y_max = np.min(y), np.max(y)
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)

            # Add legend with parameter values
            param_names = ['α', 'β', 'γ', 'δ', 'ε']
            legend_text = ', '.join([f'{name}: {value:.2f}' for name, value in zip(param_names, params)])
            ax.legend([legend_text], loc='upper right', fontsize='small')

            # Write to CSV file
            csv_writer.writerow([history[i]['step']] + list(params))

        ani = animation.FuncAnimation(fig, animate, frames=len(history), interval=200, repeat=False)

        # Use a different writer if ffmpeg is not available
        try:
            writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(output_video_path, writer=writer)
        except Exception as e:
            print(f"Error saving with FFMpegWriter: {e}")
            print("Trying to save with PillowWriter...")
            writer = animation.PillowWriter(fps=5)
            ani.save(output_video_path.replace('.mp4', '.gif'), writer=writer)

        plt.close(fig)

        # Close the CSV file after the animation is complete
        csvfile.close()

    @staticmethod
    def initialize_agents(n: int, param_means: np.ndarray, param_stds: np.ndarray) -> List['Agent_utility']:
        """
        Initialize a population of agents with random parameters drawn from normal distributions.

        :param n: Number of agents to create
        :type n: int
        :param param_means: Means of the parameter distributions
        :type param_means: np.ndarray
        :param param_stds: Standard deviations of the parameter distributions
        :type param_stds: np.ndarray
        :return: List of initialized agents
        """
        return [Agent_utility(params=np.random.normal(param_means, param_stds)) for _ in range(n)]

    @staticmethod
    def mutate_params(params: np.ndarray, mutation_rate: float) -> np.ndarray:
        """
        Mutate agent parameters for evolutionary algorithms.

        :param params: Current parameters of the agent
        :type params: np.ndarray
        :param mutation_rate: Rate of mutation for the parameters
        :type mutation_rate: float
        :return: Mutated parameters
        :rtype: np.ndarray
        """
        mutated_params = params * (1 + np.random.normal(0, mutation_rate, size=params.shape))
        return mutated_params

    @staticmethod
    def evolutionary_algorithm(n_agents: int, n_steps: int, save_interval: int,
                               processes: List[Union[dict, object]],
                               param_means: np.ndarray, param_stds: np.ndarray, mutation_rate: float,
                               stochastic_process_class: Type = None, keep_top_n: int = 50,
                               removal_interval: int = 10, process_selection_share: float = 0.5,
                               output_dir: str = 'output', process_time=1.0, numeric_utilities: bool = True):
        """
        Run an evolutionary algorithm to optimize agent parameters based on expected utility.

        :param n_agents: Number of agents in the population
        :type n_agents: int
        :param n_steps: Number of steps to run the algorithm
        :type n_steps: int
        :param save_interval: Interval for saving intermediate results
        :type save_interval: int
        :param processes: List of stochastic processes for agents to interact with
        :type processes: List[Union[dict, object]]
        :param param_means: Means of the parameter distributions
        :type param_means: np.ndarray
        :param param_stds: Standard deviations of the parameter distributions
        :type param_stds: np.ndarray
        :param mutation_rate: Rate of mutation for agent parameters
        :type mutation_rate: float
        :param stochastic_process_class: Class of the stochastic process (if using dict-based processes)
        :type stochastic_process_class: Type
        :param keep_top_n:  Number of top agents to keep after each removal interval
        :type keep_top_n: int
        :param removal_interval: Interval for removing agents based on performance (in time units)
        :type removal_interval: int
        :param process_selection_share: Share of processes to select for each agent (0 to 1)
        :type process_selection_share: float
        :param output_dir: Directory to save output files
        :type output_dir: str
        :param process_time: Time horizon for the stochastic processes
        :type process_time: float
        :param numeric_utilities: Use numerical utility calculation if True, else use symbolic
        :type numeric_utilities: bool
        :raises InDevelopmentWarning: If numeric_utilities is False (feature in development)
        :return: List of final agents and history of the evolutionary process
        :rtype: Tuple[List[Agent_utility], List[Dict[str, Any]]]
        """

        if not numeric_utilities:
            raise InDevelopmentWarning("The feature of the calculation of the expected utility with the integrals of the probability density functions is still in development."
                                       "Calculating utilities with the integration of PDFs may deliver incorrect results."
                                       "Moreover, for many cases calculation of the expected utility via simulation is faster."
                                       "It you still want to use integrals of PDFs, use them with the parameters that are relatively close to 0, because for the large parameters the results diverge from the true values."
                                       "We are working on making this feature more accurate and fast.")

        agents = Agent_utility.initialize_agents(n_agents, param_means, param_stds)
        history = []

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Prepare CSV file for intermediate results
        intermediate_results_path = os.path.join(output_dir, 'intermediate_results.csv')
        with open(intermediate_results_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ['Step', 'Agent_Rank'] + [f'Param_{i}' for i in range(len(param_means))] + ['Total_Wealth'])

        for step in range(n_steps):
            print(f"Step {step}/{n_steps}")

            # Randomly select a subset of processes
            num_processes_to_select = max(1, int(len(processes) * process_selection_share))
            selected_processes = random.sample(processes, num_processes_to_select)

            # Calculate expected utilities for each agent and selected process
            utilities = []
            for agent in agents:
                agent_utilities = []
                for process in selected_processes:
                    try:
                        if numeric_utilities:
                            utility = Agent_utility.numerical_expected_utility(process, agent.params,
                                                                               stochastic_process_class, t=process_time)
                        else:
                            process_dict = process_to_dict(process) if not isinstance(process, dict) else process
                            utility = Agent_utility.expected_utility(process_dict, agent.params, t=process_time)
                        agent_utilities.append(utility)
                    except Exception as e:
                        print(f"Utility calculation failed for agent {agent} and process {process}: {e}")
                        agent_utilities.append(0.0)  # Assign a default utility if calculation fails
                utilities.append(agent_utilities)

            # Invest wealth in the best process
            for agent, agent_utilities in zip(agents, utilities):
                best_process_index = np.argmax(agent_utilities)
                best_process = selected_processes[best_process_index]

                # Simulate the stochastic process
                if isinstance(best_process, dict):
                    if stochastic_process_class is None:
                        raise ValueError("stochastic_process_class must be provided when using dict-based processes")
                    process_params = {k: v for k, v in best_process.items() if k != 'symbolic'}
                    process_instance = stochastic_process_class(**process_params)
                else:
                    process_instance = best_process

                data = process_instance.simulate(t=process_time, timestep=0.1, num_instances=1)

                # Update wealth based on the last value of the process
                process_value = data[-1, -1]  # Last value of the process
                agent.wealth *= process_value
                agent.total_accumulated_wealth *= process_value  # Update total accumulated wealth

            # Remove agents with zero or negative wealth
            agents = [agent for agent in agents if agent.wealth > 0]

            # Keep only top n agents every removal_interval steps
            if step % removal_interval == 0 and step > 0:
                print(f'There were {len(agents)} agents."')
                agents.sort(key=lambda a: a.total_accumulated_wealth, reverse=True)
                agents = agents[:keep_top_n]
                print(f"Kept top {keep_top_n} agents.")

            # Handle agent splitting
            new_agents = []
            for agent in agents:
                if agent.wealth >= 2:
                    # Create new agent with mutated parameters and half the wealth
                    rounded_wealth = int(np.floor(agent.wealth))
                    for i in range(rounded_wealth - 1):
                        new_agent = Agent_utility(
                            params=Agent_utility.mutate_params(agent.params, mutation_rate),
                            wealth=1,
                            total_accumulated_wealth=agent.total_accumulated_wealth  # Inherit total wealth
                        )
                        new_agents.append(new_agent)
                    agent.wealth = 1

            agents.extend(new_agents)

            # Save intermediate data
            if step % save_interval == 0:
                history.append({
                    'step': step,
                    'n_agents': len(agents),
                    'avg_wealth': np.mean([agent.wealth for agent in agents]),
                    'best_params': max(agents, key=lambda a: a.total_accumulated_wealth).params
                })

            # Save intermediate results every 10 steps
            if step % 10 == 0:
                agents.sort(key=lambda a: a.total_accumulated_wealth, reverse=True)
                with open(intermediate_results_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    for rank, agent in enumerate(agents[:keep_top_n]):
                        csv_writer.writerow([step, rank] + list(agent.params) + [agent.total_accumulated_wealth])

            best_agent = max(agents, key=lambda a: a.total_accumulated_wealth)
            if best_agent.total_accumulated_wealth > 1000000:
                # divide all agents' total wealth by 1000000
                for agent in agents:
                    agent.total_accumulated_wealth /= 1000000

        # Generate visualization after the algorithm completes
        output_video_path = os.path.join(output_dir, 'utility_function_evolution.mp4')
        output_csv_path = os.path.join(output_dir, 'best_agent_parameters.csv')
        Agent_utility.visualize_utility_function_evolution(history, output_video_path, output_csv_path)

        return agents, history

    @staticmethod
    def evolutionary_algorithm_with_exchange(n_agents: int, n_steps: int, save_interval: int,
                                             processes: List[Union[dict, object]],
                                             param_means: np.ndarray, param_stds: np.ndarray, noise_std: float,
                                             stochastic_process_class: Type = None, top_k: int = 10,
                                             process_selection_share: float = 0.5,
                                             output_dir: str = 'output', s: int = 10, process_time=1.0,
                                             numeric_utilities: bool = True):
        """
        Run an evolutionary algorithm to optimize agent parameters based on expected utility with the exchange of parameters (evolution-like mixing of genetic information).

        :param n_agents: Number of agents in the population
        :type n_agents: int
        :param n_steps: Number of steps to run the algorithm
        :type n_steps: int
        :param save_interval: Interval for saving intermediate results
        :type save_interval: int
        :param processes: List of stochastic processes for agents to interact with
        :type processes: List[Union[dict, object]]
        :param param_means: Means of the parameter distributions
        :type param_means: np.ndarray
        :param param_stds: Standard deviations of the parameter distributions
        :type param_stds: np.ndarray
        :param noise_std: Standard deviation of the noise added to averaged parameters
        :type noise_std: float
        :param stochastic_process_class: The class of the stochastic process (if using dict-based processes)
        :type stochastic_process_class: Type
        :param top_k: Number of top agents to keep after each selection interval
        :type top_k: int
        :param process_selection_share: Share of processes to select for each agent (0 to 1)
        :type process_selection_share: float
        :param output_dir: Directory to save output files
        :type output_dir: str
        :param s: Interval for selection, averaging, and multiplication
        :type s: int
        :param process_time: Time horizon for the stochastic processes
        :type process_time: float
        :param numeric_utilities: Use numerical utility calculation if True, else use symbolic
        :type numeric_utilities: bool
        :raises InDevelopmentWarning: If numeric_utilities is False (feature in development)
        :return: List of final agents and history of the evolutionary process
        :rtype: Tuple[List[Agent_utility], List[Dict[str, Any]]]
        """
        agents = Agent_utility.initialize_agents(n_agents, param_means, param_stds)
        history = []

        if not numeric_utilities:
            raise InDevelopmentWarning("The feature of the calculation of the expected utility with the integrals of the probability density functions is still in development."
                                       "Calculating utilities with the integration of PDFs may deliver incorrect results."
                                       "Moreover, for many cases calculation of the expected utility via simulation is faster."
                                       "It you still want to use integrals of PDFs, use them with the parameters that are relatively close to 0, because for the large parameters the results diverge from the true values."
                                       "We are working on making this feature more accurate and fast.")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Prepare CSV file for intermediate results
        intermediate_results_path = os.path.join(output_dir, 'intermediate_results.csv')
        with open(intermediate_results_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ['Step', 'Agent_Rank'] + [f'Param_{i}' for i in range(len(param_means))] + ['Total_Wealth'])

        for step in range(n_steps):
            print(f"Step {step}/{n_steps}")

            # Randomly select a subset of processes
            num_processes_to_select = max(1, int(len(processes) * process_selection_share))
            selected_processes = random.sample(processes, num_processes_to_select)

            # Calculate expected utilities for each agent and selected process
            utilities = []
            for agent in agents:
                agent_utilities = []
                for process in selected_processes:
                    try:
                        if numeric_utilities:
                            utility = Agent_utility.numerical_expected_utility(process, agent.params,
                                                                               stochastic_process_class, t=process_time)
                        else:
                            process_dict = process_to_dict(process) if not isinstance(process, dict) else process
                            utility = Agent_utility.expected_utility(process_dict, agent.params, t=process_time)
                        agent_utilities.append(utility)
                    except Exception as e:
                        print(f"Utility calculation failed for agent {agent} and process {process}: {e}")
                        agent_utilities.append(0.0)  # Assign a default utility if calculation fails
                utilities.append(agent_utilities)

            # Invest wealth in the best process
            for agent, agent_utilities in zip(agents, utilities):
                best_process_index = np.argmax(agent_utilities)
                best_process = selected_processes[best_process_index]

                # Simulate the stochastic process
                if isinstance(best_process, dict):
                    if stochastic_process_class is None:
                        raise ValueError("stochastic_process_class must be provided when using dict-based processes")
                    process_params = {k: v for k, v in best_process.items() if k not in ['symbolic']}
                    process_instance = stochastic_process_class(**process_params)
                else:
                    process_instance = best_process

                data = process_instance.simulate(t=process_time, timestep=0.1, num_instances=1)

                # Update wealth based on the last value of the process
                process_value = data[-1, -1]  # Last value of the process
                agent.wealth *= process_value
                agent.total_accumulated_wealth *= process_value  # Update total accumulated wealth

            # Remove agents with zero or negative wealth
            agents = [agent for agent in agents if agent.wealth > 0]

            # Perform selection, averaging, and multiplication every s steps
            if step % s == 0 and step > 0:
                # Total wealth of all agents
                total_wealth = sum([agent.total_accumulated_wealth for agent in agents])
                print(f'Total wealth: {total_wealth}')
                # Sort agents by total accumulated wealth
                agents.sort(key=lambda a: a.total_accumulated_wealth, reverse=True)

                # Select top k agents
                top_agents = agents[:top_k]

                # Calculate average parameters of top k agents
                avg_params = np.mean([agent.params for agent in top_agents], axis=0)

                # Create new generation with noise added to averaged parameters
                new_agents = []
                for _ in range(n_agents):
                    new_params = avg_params + np.random.normal(0, noise_std, size=avg_params.shape)
                    new_agent = Agent_utility(params=new_params, wealth=1.0, total_accumulated_wealth=1.0)
                    new_agents.append(new_agent)

                agents = new_agents

            # Save intermediate data
            if step % save_interval == 0:
                history.append({
                    'step': step,
                    'n_agents': len(agents),
                    'avg_wealth': np.mean([agent.wealth for agent in agents]),
                    'best_params': max(agents, key=lambda a: a.total_accumulated_wealth).params
                })

            # Save intermediate results every 10 steps
            if step % 10 == 0:
                agents.sort(key=lambda a: a.total_accumulated_wealth, reverse=True)
                with open(intermediate_results_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    for rank, agent in enumerate(agents[:top_k]):
                        csv_writer.writerow([step, rank] + list(agent.params) + [agent.total_accumulated_wealth])

            best_agent = max(agents, key=lambda a: a.total_accumulated_wealth)
            if best_agent.total_accumulated_wealth > 1000000:
                # divide all agents' total wealth by 1000000
                for agent in agents:
                    agent.total_accumulated_wealth /= 1000000

        # Generate visualization after the algorithm completes
        output_video_path = os.path.join(output_dir, 'utility_function_evolution.mp4')
        output_csv_path = os.path.join(output_dir, 'best_agent_parameters.csv')
        Agent_utility.visualize_utility_function_evolution(history, output_video_path, output_csv_path)

        return agents, history

    @staticmethod
    def evolutionary_algorithm_with_multiple_processes(
            n_agents: int, n_steps: int, save_interval: int,
            processes: List[Union[dict, object]],
            param_means: np.ndarray, param_stds: np.ndarray,
            mutation_rate: float,
            keep_top_n: int = 50, removal_interval: int = 10,
            process_selection_share: float = 0.5, output_dir: str = 'output',
            process_time: float = 1.0, numeric_utilities: bool = True):

        """
        Run an evolutionary algorithm to optimize agent parameters based on expected utility with multiple process types.

        :param n_agents: Number of agents in the population
        :type n_agents: int
        :param n_steps: Number of steps to run the algorithm
        :type n_steps: int
        :param save_interval: Interval for saving intermediate results
        :type save_interval: int
        :param processes: List of stochastic processes for agents to interact with
        :type processes: List[Union[dict, object]]
        :param param_means: Means of the parameter distributions
        :type param_means: np.ndarray
        :param param_stds: Standard deviations of the parameter distributions
        :type param_stds: np.ndarray
        :param mutation_rate: Rate of mutation for agent parameters
        :type mutation_rate: float
        :param keep_top_n: Number of top agents to keep after each removal interval
        :type keep_top_n: int
        :param removal_interval: Interval for removing agents based on performance (in time units)
        :type removal_interval: int
        :param process_selection_share: Share of processes to select for each agent (0 to 1)
        :type process_selection_share: float
        :param output_dir: Directory to save output files
        :type output_dir: str
        :param process_time: Time horizon for the stochastic processes
        :type process_time: float
        :param numeric_utilities: Use numerical utility calculation if True, else use symbolic
        :type numeric_utilities: bool
        :raises InDevelopmentWarning: If numeric_utilities is False (feature in development)
        :return: List of final agents and history of the evolutionary process
        :rtype: Tuple[List[Agent_utility], List[Dict[str, Any]]]
        """

        if not numeric_utilities:
            raise InDevelopmentWarning("The feature of the calculation of the expected utility with the integrals of the probability density functions is still in development."
                                       "Calculating utilities with the integration of PDFs may deliver incorrect results."
                                       "Moreover, for many cases calculation of the expected utility via simulation is faster."
                                       "It you still want to use integrals of PDFs, use them with the parameters that are relatively close to 0, because for the large parameters the results diverge from the true values."
                                       "We are working on making this feature more accurate and fast.")

        def initialize_agents(n: int, param_means: np.ndarray, param_stds: np.ndarray) -> List['Agent_utility']:
            return [Agent_utility(params=np.random.normal(param_means, param_stds)) for _ in range(n)]

        def mutate_params(params: np.ndarray, mutation_rate: float) -> np.ndarray:
            return params * (1 + np.random.normal(0, mutation_rate, size=params.shape))

        agents = initialize_agents(n_agents, param_means, param_stds)
        history = []

        os.makedirs(output_dir, exist_ok=True)
        intermediate_results_path = os.path.join(output_dir, 'intermediate_results.csv')

        with open(intermediate_results_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            header = ['Step', 'Agent_Rank']
            for i in range(param_means.shape[0]):
                header.extend([f'Param_{i}'])
            header.append('Total_Wealth')
            csv_writer.writerow(header)

        for step in range(n_steps):
            print(f"Step {step}/{n_steps}")

            num_processes_to_select = max(1, int(len(processes) * process_selection_share))
            selected_processes = random.sample(processes, num_processes_to_select)

            utilities = []
            for agent in agents:
                agent_utilities = []
                for process in selected_processes:
                    try:
                        if numeric_utilities:
                            utility = Agent_utility.numerical_expected_utility(
                                process, agent.params, type(process) if not isinstance(process, dict) else None,
                                t=process_time, num_instances = 10000
                            )
                        else:
                            process_dict = process_to_dict(process) if not isinstance(process, dict) else process
                            utility = Agent_utility.expected_utility(
                                process_dict, agent.params, t=process_time
                            )
                        agent_utilities.append(utility)
                    except Exception as e:
                        print(f"Utility calculation failed for agent {agent} and process {process}: {e}")
                        agent_utilities.append(0.0)
                utilities.append(agent_utilities)

            for agent, agent_utilities in zip(agents, utilities):
                best_process_index = np.argmax(agent_utilities)
                best_process = selected_processes[best_process_index]

                if isinstance(best_process, dict):
                    process_params = {k: v for k, v in best_process.items() if k != 'symbolic'}
                    process_instance = best_process['symbolic'](**process_params)
                else:
                    process_instance = best_process

                data = process_instance.simulate(t=process_time, timestep=0.1, num_instances=1)
                process_value = data[-1, -1]
                agent.wealth *= process_value
                agent.total_accumulated_wealth *= process_value

            agents = [agent for agent in agents if agent.wealth > 0]

            if step % removal_interval == 0 and step > 0:
                print(f'There were {len(agents)} agents.')
                agents.sort(key=lambda a: a.total_accumulated_wealth, reverse=True)
                agents = agents[:keep_top_n]
                print(f"Kept top {keep_top_n} agents.")

            new_agents = []
            for agent in agents:
                if agent.wealth >= 2:
                    rounded_wealth = int(np.floor(agent.wealth))
                    for _ in range(rounded_wealth - 1):
                        new_agent = Agent_utility(
                            params=mutate_params(agent.params, mutation_rate)
                        )
                        new_agent.total_accumulated_wealth = agent.total_accumulated_wealth
                        new_agents.append(new_agent)
                    agent.wealth = 1

            agents.extend(new_agents)

            if step % save_interval == 0:
                history.append({
                    'step': step,
                    'n_agents': len(agents),
                    'avg_wealth': np.mean([agent.wealth for agent in agents]),
                    'best_params': max(agents, key=lambda a: a.total_accumulated_wealth).params
                })

            if step % 10 == 0:
                agents.sort(key=lambda a: a.total_accumulated_wealth, reverse=True)
                with open(intermediate_results_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    for rank, agent in enumerate(agents[:keep_top_n]):
                        row = [step, rank]
                        row.extend(list(agent.params))
                        row.append(agent.total_accumulated_wealth)
                        csv_writer.writerow(row)

            best_agent = max(agents, key=lambda a: a.total_accumulated_wealth)
            if best_agent.total_accumulated_wealth > 1000000:
                for agent in agents:
                    agent.total_accumulated_wealth /= 1000000

        output_video_path = os.path.join(output_dir, 'utility_function_evolution.mp4')
        output_csv_path = os.path.join(output_dir, 'best_agent_parameters.csv')
        Agent_utility.visualize_utility_function_evolution(history, output_video_path, output_csv_path)

        return agents, history

def visualize_agent_evolution(history: List[Dict], top_n: int = 5):
    """
    Visualize the evolution of top agents and their utility functions.

    :param history: List of dictionaries containing historical data
    :type history: List[Dict]
    :param top_n: Number of top agents to visualize
    :type top_n: int
    :return: None (displays plots)
    :rtype: None
    """
    steps = [entry['step'] for entry in history]

    # Plot average wealth over time
    plt.figure(figsize=(12, 6))
    plt.plot(steps, [entry['avg_wealth'] for entry in history])
    plt.title('Average Wealth Over Time')
    plt.xlabel('Step')
    plt.ylabel('Average Wealth')
    plt.show()

    # Plot number of agents over time
    plt.figure(figsize=(12, 6))
    plt.plot(steps, [entry['n_agents'] for entry in history])
    plt.title('Number of Agents Over Time')
    plt.xlabel('Step')
    plt.ylabel('Number of Agents')
    plt.show()

    # Visualize utility functions of top agents
    plt.figure(figsize=(12, 6))
    params = history[-1]['best_params']
    x = np.linspace(0, 10, 100)
    x_sym = sp.Symbol('x')

    if isinstance(params, np.ndarray) and params.ndim == 1:
        # If params is a 1D array, it's for a single agent
        symbolic_utility = general_utility_function(x_sym, *params)
        numeric_utility = sp.lambdify(x_sym, symbolic_utility, 'numpy')
        y = numeric_utility(x)
        plt.plot(x, y, label='Best Agent')
    else:
        # If params is a 2D array, it's for multiple agents
        for i in range(min(top_n, len(params))):
            symbolic_utility = general_utility_function(x_sym, *params[i])
            numeric_utility = sp.lambdify(x_sym, symbolic_utility, 'numpy')
            y = numeric_utility(x)
            plt.plot(x, y, label=f'Agent {i + 1}')

    plt.title(f'Utility Functions of Top Agents')
    plt.xlabel('x')
    plt.ylabel('Utility')
    plt.legend()
    plt.show()

    # Analyze parameter trends
    param_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    if isinstance(history[0]['best_params'], np.ndarray) and history[0]['best_params'].ndim == 1:
        # If best_params is a 1D array, it's for a single agent
        param_trends = {name: [float(entry['best_params'][i]) for entry in history] for i, name in
                        enumerate(param_names)}
    else:
        # If best_params is a 2D array, take the parameters of the best agent
        param_trends = {name: [float(entry['best_params'][0][i]) for entry in history] for i, name in
                        enumerate(param_names)}

    # Create a 3x2 grid of subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 20))
    for i, (name, values) in enumerate(param_trends.items()):
        ax = axs[i // 2, i % 2]
        ax.plot(steps, values)
        ax.set_title(f'{name} Over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel(name)

    # Remove the empty subplot
    fig.delaxes(axs[2, 1])

    plt.tight_layout()
    plt.show()

def recursive_flatten(data: Any) -> List[float]:
    """
    Recursively flatten any nested structure into a 1D list of floats.

    :param data: Nested structure to flatten
    :type data: Any
    :return: 1D list of floats
    :rtype: List[float]
    """
    if isinstance(data, (int, float)):
        return [float(data)]
    elif isinstance(data, np.ndarray):
        return data.flatten().tolist()
    elif isinstance(data, list):
        return [item for sublist in data for item in Agent_utility.recursive_flatten(sublist)]
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")

def analyze_utility_function_trends(history: List[Dict], num_agents: int = 10):
    """
    Analyze and visualize trends in the utility functions' evolutionary dynamics.

    :param history: List of dictionaries containing historical data
    :type history: List[Dict]
    :param num_agents: Maximum number of top agents to analyze
    :type num_agents: int
    :return: None (displays plots)
    :rtype: None
    """
    if not history:
        print("Error: History is empty.")
        return

    param_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    param_trends = {name: [] for name in param_names}

    for entry in history:
        if 'best_params' not in entry:
            print(f"Warning: 'best_params' not found in history entry: {entry}")
            continue
        best_params = entry['best_params']
        if isinstance(best_params, np.ndarray) and best_params.ndim > 1:
            best_params = best_params[:num_agents]
        elif isinstance(best_params, (list, np.ndarray)):
            best_params = [best_params]
        else:
            print(f"Warning: Unexpected type for best_params: {type(best_params)}")
            continue

        for params in best_params:
            for i, name in enumerate(param_names):
                if i < len(params):
                    param_trends[name].append(params[i])
                else:
                    print(f"Warning: Parameter {name} not found in entry: {params}")

    print("Param trends structure:")
    for name, trend in param_trends.items():
        print(f"{name}: {len(trend)} entries, types: {set(type(item) for item in trend)}")

    # Visualize parameter distributions over time
    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
    for i, (name, values) in enumerate(param_trends.items()):
        ax = axs[i // 2, i % 2]
        sns.violinplot(data=values, ax=ax)
        ax.set_title(f'{name} Distribution Over Time')
        ax.set_ylabel('Value')
        ax.set_xlabel('Step')
        ax.tick_params(axis='x', rotation=45)

    fig.delaxes(axs[2, 1])
    plt.tight_layout()
    plt.show()

    # Analyze parameter correlations
    corr_matrix = np.corrcoef([param_trends[name] for name in param_names])
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, xticklabels=param_names, yticklabels=param_names)
    plt.title('Parameter Correlations')
    plt.show()

    # Analyze parameter convergence
    # plt.figure(figsize=(12, 6))
    # for name, values in param_trends.items():
    #     stds = np.std(values, axis=0)
    #     plt.plot(stds, label=name)
    # plt.title('Parameter Convergence')
    # plt.xlabel('Step')
    # plt.ylabel('Standard Deviation')
    # plt.legend()
    # plt.show()

    # Analyze utility function diversity
    x = np.linspace(0, 10, 200)
    # diversity = []
    # for entry in history:
    #     params = entry['best_params']
    #     if isinstance(params, np.ndarray) and params.ndim > 1:
    #         params = params[:num_agents]
    #     else:
    #         params = [params]
    #     utilities = [Agent_utility.general_utility_function_np(x, *p) for p in params]
    #     diversity.append(np.mean(np.std(utilities, axis=0)) if len(utilities) > 1 else 0)
    #
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(len(history)), diversity)
    # plt.title('Utility Function Diversity Over Time')
    # plt.xlabel('Step')
    # plt.ylabel('Average Standard Deviation of Utilities')
    # plt.show()

    # Visualize final utility functions
    plt.figure(figsize=(12, 6))
    final_params = history[-1]['best_params']
    if isinstance(final_params, np.ndarray) and final_params.ndim > 1:
        final_params = final_params[:num_agents]
    else:
        final_params = [final_params]
    for i, params in enumerate(final_params):
        y = Agent_utility.general_utility_function_np(x, *params)
        plt.plot(x, y, label=f'Agent {i + 1}')
    plt.title('Final Utility Functions')
    plt.xlabel('x')
    plt.ylabel('Utility')
    plt.legend()
    plt.grid(True)
    plt.show()


def process_to_dict(process: Any) -> Dict[str, Any]:
    """
    Converts a process object with a closed_formula method to a dictionary format
    compatible with Agent_utility.compare_numerical_and_symbolic_expected_utility.

    :param process: An object with a closed_formula method
    :type process: Any
    :return: A dictionary with 'symbolic' key for the formula and additional keys for process parameters
    :rtype: Dict[str, Any]
    """
    if not hasattr(process, 'closed_formula'):
        raise AttributeError("The provided process object does not have a closed_formula method")

    formula = process.closed_formula()

    # print(f'formula: {formula}')

    params = process.get_params()

    # Create symbolic parameters
    param_symbols = {param: sp.Symbol(param) for param in params}
    # create a tuple with parameter symbols
    param_symbols_tuple = tuple(param_symbols.values())
    # print(f'param_symbols: {param_symbols}')
    t, W = sp.symbols('t W')

    # Call the formula with symbolic parameters
    symbolic_result = formula(t, W)
    # print('symbolic_result:', symbolic_result)

    # Replace the process attributes with symbolic parameters
    # for param, symbol in param_symbols.items():
    #     symbolic_result = symbolic_result.subs(getattr(process, f"_{param}"), symbol)

    # print(f'parameter symbol values: {list(param_symbols.values())}')

    # Create the lambda function using SymPy's lambdify
    lambda_func = sp.lambdify([t, W] + list(param_symbols.values()), symbolic_result, modules=['sympy'])

    # Create the dictionary
    result = {
        'symbolic': lambda_func,
    }

    # Add process parameters to the dictionary
    for param in params:
        result[param] = getattr(process, f"_{param}")


    return result

def generate_processes(num_processes, process_types, param_ranges):
    """
    Generate a list of stochastic processes.

    :param num_processes: Number of processes to generate
    :type num_processes: int
    :param process_types: List of process classes or functions to use
    :type process_types: list
    :param param_ranges: Dictionary of parameter ranges for each process type
    :type param_ranges: dict
    :return: List of generated processes
    :rtype: list
    """
    processes = []

    for i in range(num_processes):
        # Randomly select a process type
        process_type = np.random.choice(process_types)

        # Generate parameters for the selected process type
        params = {}
        for param, range_values in param_ranges[process_type.__name__].items():
            params[param] = np.random.uniform(*range_values)

        # Create the process instance
        if isinstance(process_type, type):
            # If it's a class, instantiate it
            process = process_type(**params)
        else:
            # If it's a function, create a dictionary with the function and its params
            process = {
                'symbolic': process_type,
                **params
            }

        processes.append(process)

    print(f"Generated {len(processes)} processes")  # Debug print
    return processes

if __name__ == '__main__':
    # Example usage with multiple processes
    from ergodicity.process.multiplicative import GeometricBrownianMotion, GeometricLevyProcess

    n_agents = 100
    n_steps = 1000
    save_interval = 50
    removal_percentage = 0.1  # Remove 10% of worst-performing agents
    removal_interval = 10  # Remove agents every 10 steps

    # Generate multiple processes with varying parameters
    processes = [
        GeometricBrownianMotion(drift=0.02, volatility=0.15),
        GeometricBrownianMotion(drift=0.03, volatility=0.18),
        GeometricBrownianMotion(drift=0.01, volatility=0.12),
        GeometricBrownianMotion(drift=0.025, volatility=0.20),
        GeometricBrownianMotion(drift=0.015, volatility=0.10)
    ]

    param_means = np.array([1.0, 1.0, 0.5, 1.0])
    param_stds = np.array([0.1, 0.1, 0.05, 0.1])
    mutation_rate = 0.01

    final_agents, history = Agent_utility.evolutionary_algorithm(
        n_agents, n_steps, save_interval, processes,
        param_means, param_stds, mutation_rate,
        GeometricBrownianMotion,
        keep_top_n=50,  # Specify the number of top agents to keep
        removal_interval=10,
        process_selection_share=0.5
    )

    # Print some results
    print(f"Number of agents at the end: {len(final_agents)}")
    print(f"Average wealth at the end: {np.mean([agent.wealth for agent in final_agents]):.2f}")
    best_agent = max(final_agents, key=lambda a: a.total_accumulated_wealth)
    print(f"Best agent's wealth: {best_agent.wealth:.2f}")
    print(f"Best agent's total accumulated wealth: {best_agent.total_accumulated_wealth:.2f}")
    print(f"Best agent's parameters: {best_agent.params}")

    # Example usage:
    # Assuming you have run the evolutionary algorithm and obtained the history
    visualize_agent_evolution(history)
    analyze_utility_function_trends(history)
