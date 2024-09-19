"""
The `definitions.py` module lays the foundation for the Ergodicity Library, introducing core concepts and defining abstract classes for stochastic processes.

**Key Components:**

* **`Process` Class:**

    * The central abstract class for all stochastic processes in the library.

    * Provides a standardized interface for process initialization, property access, and common methods like simulation, moments calculation, and visualization.

* **`ItoProcess` and `NonItoProcess` Classes:**

    * Abstract subclasses of `Process` that categorize processes into Ito (subject to Ito calculus) and non-Ito types.

* **`CustomProcess` Class:**

    * Empowers users to create their own custom stochastic processes within the library's framework.

* **Decorators and Helper Functions:**

    * `simulation_decorator`: Enhances simulation methods with verbose output for easier tracking.

    * `check_simulate_with_differential`: Checks if a process's simulation uses differential methods.

    * `create_correlation_matrix` and `correlation_to_covariance`: Facilitate working with correlated processes.

**Core Functionalities:**

* **Process Initialization and Properties:**

    * Standardizes the creation and attribute access of processes, including names, multiplicative nature, independence of increments, and more.

* **Simulation:**

    * `simulate()`: Primary method for generating sample paths of a process.

    * `simulate_2d()` and `simulate_3d()`: Extend simulation to 2D and 3D processes.

    * `simulate_live_*`: Create dynamic visualizations of simulations.

* **Moments and Distributions:**

    * `moments()`: Calculates the cumulative moments (mean, variance, etc.) of a process.

    * `k_moments()`: Extends moment calculation to higher orders.

    * `simulate_distribution()`: Simulates the probability distribution of a process.

* **Analysis and Visualization:**

    * `ensemble_average()` and `time_average()`: Calculate ensemble and time averages.

    * `self_averaging_time()`: Estimates the time when a process transitions from ensemble-averaged to time-averaged behavior.

    * `plot*`, `visualize_moments`: Provide various plotting and visualization capabilities.

* **Ito-Specific Functionality:**

    * `solve()`: Attempts to find analytical solutions for Ito processes using Ito calculus.

    * `differential()`: Represents the stochastic differential equation of an Ito process.

**Intended Audience:**

* **Researchers and Practitioners:** Leverage the library's core definitions to work with and analyze a wide variety of stochastic processes.

* **Students and Learners:** Gain a deeper understanding of stochastic processes through the clear structure and implementation of fundamental concepts.

* **Developers:** Extend the library's capabilities by building upon these base classes and methods to create new process types and functionalities.

**Dependencies:**

    - abc.ABC: Provides support for defining abstract base classes (ABC) for ensuring proper subclass implementation.

    - typing.List, Any, Type, Callable: Used for type annotations to enforce type safety and clarity in function signatures.

    - numpy (np): Fundamental package for numerical computation, array manipulation, and random number generation.

    - matplotlib.pyplot (plt): Library for creating static, animated, and interactive visualizations.

    - matplotlib.animation: For creating dynamic animations of the simulated processes.

    - inspect: Used to inspect live objects, retrieve information about classes, functions, methods, and more.

    - sympy (sp): Symbolic mathematics library for defining and solving algebraic expressions and differential equations.

    - pandas.core.tools.times.to_time: Time handling and manipulation tool.

    - .default_values: Internal module that defines default constants used in simulations.

    - ergodicity.configurations: Module that defines custom configurations for the stochastic processes and models.

    - warnings: Provides a framework for issuing runtime warnings, used here for custom warnings.

    - threading: Module to run code concurrently via threads for efficient simulations.

    - os: Interface for interacting with the operating system, primarily for file handling and environment queries.

    - csv: For reading from and writing to CSV files.

    - subprocess: Allows for spawning new processes, connecting to their input/output/error pipes, and obtaining their return codes.

    - plotly.graph_objects (go): Advanced plotting library for interactive visualizations.

    - mpl_toolkits.mplot3d: Used for creating 3D plots with matplotlib.

    - ..custom_warnings: Defines custom warnings specific to this module, like `InDevelopmentWarning` and `KnowWhatYouDoWarning`.

    - ergodicity.tools.compute.growth_rate_of_average_per_time: Utility function for computing growth rates over time.

    - ergodicity.tools.compute.average: Utility function for calculating average values over a data set.
"""

from abc import ABC
from typing import List, Any, Type, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import inspect
import sympy as sp
from pandas.core.tools.times import to_time
from .default_values import *
from ergodicity.configurations import *
import warnings
import threading
import os
import csv
import subprocess
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from ..custom_warnings import InDevelopmentWarning, KnowWhatYouDoWarning
from ergodicity.tools.compute import growth_rate_of_average_per_time, average_growth_rate
from ergodicity.tools.compute import average
from ..tools.helper import separate


class Process(ABC):
    """
    The Process class is the base class for all processes in the Ergodicity Library.
    This is an abstract class.
    Here, the majority of the methods available for all classes related to stochastic processes are defined.

    Attributes:

        name (str): The name of the process.

        multiplicative (bool): Whether the process is multiplicative.

        independent (bool): Whether the increments of the process are independent.

        ito (bool): Whether the process is an Ito process.

        process_class (Type[Any]): The class in the stochastic library that corresponds to the process.

        types (List[str]): Custom labels to categorize the process.

        comments (bool): Whether to display comments about the process when the code runs.

        has_wrong_params (bool): Whether the corresponding process in the stochastic library has parameters in an unexpected format.

        custom (bool): Whether the process is a custom process.

        simulate_with_differential (bool): Whether the process is simulated using differential methods.

        output_dir (str): The directory where the process results are saved.

        increment_process (bool): Whether the process is an increment process.

        memory (int): The memory of the process.
    """
    def __init__(self, name: str, multiplicative: bool, independent: bool, ito: bool, process_class: Type[Any] = None, **kwargs):
        """
        Initialize the Process class.

        :param name: The name of the process
        :type name: str
        :param multiplicative: Whether the process is multiplicative
        :type multiplicative: bool
        :param independent: Whether the process increments are independent
        :type independent: bool
        :param ito: Whether the process is an Ito process
        :type ito: bool
        :param process_class: To what class in the stochastic library the process corresponds, if any
        :type process_class: Type[Any]
        :param kwargs: Process-specific parameters
        """
        if name is None:
            name = "Process"
        if multiplicative is None:
            multiplicative = False
        if independent is None:
            independent = True
        if multiplicative is True:
            independent = False
        if ito is None:
            ito = False
        self._types = ["default"]
        self._name = name
        self._multiplicative = multiplicative
        self._independent = independent
        self._ito = ito
        self._process_class = process_class
        self._types = ["default"]
        self._comments = default_comments
        self._has_wrong_params = False
        self._custom = False
        self._simulate_with_differential = True
        self._dims = 1
        self._external_simulator = True
        self._output_dir = f"{self.name}_main"
        self._increment_process = False
        self._memory = 0

        if not os.path.exists(self._output_dir) and self._increment_process is False:
            os.makedirs(self._output_dir)

    @property
    def name(self) -> str:
        """
        Get the name of the process.

        :return: The name of the process
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, new_name: str):
        """
        Set the name of the process.

        :param new_name: The new name of the process
        :type new_name: str
        :return: None
        :rtype: None
        """
        self._name = new_name

    @property
    def multiplicative(self) -> bool:
        """
        Get the multiplicative property of the process.
        It shows if the process is multiplicative.

        :return: The multiplicative property of the process
        :rtype: bool
        """
        return self._multiplicative

    @property
    def independent(self) -> bool:
        """
        Get the independent property of the process.
        It shows if the increments of the process are independent.

        :return: The independent property of the process
        :rtype: bool
        """
        return self._independent

    @property
    def ito(self) -> bool:
        """
        Get the Ito property of the process.
        It shows if the process is an Ito process/

        :return: The Ito property of the process
        :rtype: bool
        """
        return self._ito

    @property
    def comments(self) -> bool:
        """
        Get the comments property of the process.
        It is used to display comments about the process when the code runs.

        :return: The comments property of the process.
        :rtype: bool
        """
        return self._comments

    @comments.setter
    def comments(self, value: bool):
        """
        Set the comments property of the process.
        It is used to display comments about the process when the code runs.

        :param value: The new value of the comments property
        :type value: bool
        :return: None
        :rtype: None
        :raises ValueError: If the value is not a boolean
        """
        if value is not bool:
            raise ValueError("Comments must be a boolean value.")
        self._comments = value

    @classmethod
    def get_comments(cls):
        """
        Get the comments property of the process.
        This is a class method. It means that it is a method that is called on the class itself, not on an instance of the class.

        :return: The comments property of the process
        :rtype: bool
        """
        return cls._comments

    @property
    def process_class(self) -> Type[Any]:
        """
        Get the process class of the process.
        Here, the process class is the class in the stochastic library that corresponds to the process.
        It has nothing to do with the Process class itself (inside the Ergodicity Library).

        :return: The process class of the process
        :rtype: Type[Any]
        """
        return self._process_class

    @property
    def has_wrong_params(self) -> bool:
        """
        Get the has_wrong_params property of the process.
        It shows if the corresponding process in the stochastic library has parameters in an unexpected format that must be corrected.

        :return: The has_wrong_params property of the process
        :rtype: bool
        """
        return self._has_wrong_params

    @has_wrong_params.setter
    def has_wrong_params(self, value: bool):
        """
        Set the has_wrong_params property of the process.

        :param value: Whether the process has wrong parameters
        :type value: bool
        :return: None
        :rtype: None
        """
        self._has_wrong_params = value

    @property
    def custom(self) -> bool:
        """
        Get the custom property of the process.
        It shows if the process is a custom process (and not a standard process from the library).

        :return: The custom property of the process
        :rtype: bool
        """
        return self._custom

    @property
    def simulate_with_differential(self) -> bool:
        """
        Get the simulate_with_differential property of the process.
        It sets if the process is simulated using differential methods.
        If it is False, the process is simulated using relevant probability distribution.

        :return: The simulate_with_differential property of the process
        :rtype: bool
        """
        return self._simulate_with_differential

    @simulate_with_differential.setter
    def simulate_with_differential(self, value: bool):
        """
        Set the simulate_with_differential property of the process.

        :param value: Whether the process is simulated using differential methods
        :type value: bool
        :return: None
        :rtype: None
        """
        self._simulate_with_differential = value

    @property
    def output_dir(self) -> str:
        """
        Get the output directory of the process.
        It is the directory where the process results are saved.

        :return: The output directory of the process
        :rtype: str
        """
        return self._output_dir

    @output_dir.setter
    def output_dir(self, new_output_dir: str):
        """
        Set the output directory of the process.

        :param new_output_dir: The new output directory of the process
        :type new_output_dir: str
        :return: None
        :rtype: None
        """
        self._output_dir = new_output_dir

    @property
    def types(self) -> List[str]:
        """
        Get the types of the process.
        The types are used to categorize the process. They are custom labels that can be used to identify the process.

        :return: The types of the process
        :rtype: List[str]
        """
        return self._types

    @types.setter
    def types(self, new_types: List[str]):
        """
        Set the types of the process.

        :param new_types:  The new types of the process
        :type new_types: List[str]
        :return: None
        :rtype: None
        :raises ValueError: If the types are not a list of strings
        """
        if not isinstance(new_types, list):
            raise ValueError("Types must be a list of strings.")
        self._types = new_types

    def add_type(self, new_type: str):
        """
        Add a new type to the types list if it is not already present.

        :param new_type: The new type to add
        :type new_type: str
        :return: None
        :rtype: None
        """
        if new_type not in self._types:
            self._types.append(new_type)
        else:
            print(f"The type '{new_type}' is already present in the list.")

    always_present_keys = ['self', 'name', 'multiplicative', 'ito', 'types', 'process_class']

    def get_params(self):
        """
        Retrieves parameters specific to the current process class, excluding always_present_keys.
        This is usually needed when creating methods and functions than need to work for an arbitrary process class.
        For the end user, the method is usually not needed.

        :return: The process-specific parameters of the process
        :rtype: dict
        """
        signature = inspect.signature(self.__init__)
        params = {}
        for name in signature.parameters:
            if name not in self.always_present_keys and hasattr(self, f"_{name}"):
                params[name] = getattr(self, f"_{name}")

        return params


    def correct_params(self, params, t):
        """
        Correct the parameters for the process.
        This method fixes some unintended behaviour of the stochastic library which is present for some classes.
        It is not needed for the end user.

        :param params: The parameters to correct
        :type params: dict
        :param t: The total time for the simulation
        :type t: float
        :return: The corrected parameters
        :rtype: dict
        """
        if self._has_wrong_params:
            if 'drift' in params:
                params['drift'] = params['drift'] * t
                self._has_wrong_params = False
                self._drift = params['drift']

        return params

    def data_for_simulation(self, t: float = t_default, timestep: float = timestep_default, num_instances: int = num_instances_default) -> Any:
        """
        Prepare the data for simulation.
        This is an intermediate method that is further used in the simulate method which is already used by the user.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param num_instances: Number of instances to simulate
        :type num_instances: int
        :return: The number of steps, the times, and the data
        :rtype: Any
        """
        num_steps = int(t / timestep)
        times = np.linspace(0, t, num_steps)
        data = np.zeros((num_instances, num_steps))

        return num_steps, times, data

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Custom increment function for the process.
        This method simulates a discrete approximation of the increment (differential) of the process.

        :param X: The current value of the process
        :type X: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :return: The increment of the process
        :rtype: Any
        """
        return None


    def separate(self, data):
        """
        Separate the discrete time moments and corresponding process data in the dataset typically generated by many methods in the Ergodicity Library.
        It is usually needed if you need to extract simulated data without time moments, or just time moments, if you need to plot the data, or if you need to pass the data to a custom method.
        It returns the times and the data as arrays.

        :param data: The data to separate
        :type data: Any
        :return: The times and the data
        :rtype: Any
        """
        times = data[0]
        data = data[1:]
        return times, data

    def memory_update(self, step):
        """
        Update the memory of the process. Used in the construction of processes with memory.

        :param step: The current step
        :type step: int
        :return: The updated memory
        :rtype: Any
        """
        pass

    def custom_simulate_raw(self, t: float = t_default, timestep: float = timestep_default,
                    num_instances: int = num_instances_default, X0 : float = None) -> Any:

        """
        Simulate the process using a custom simulation method. It is an intermediate method that is further used in the simulate method which is already used by the user.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param num_instances: Number of instances to simulate
        :type num_instances: int
        :param X0: Initial value for the process
        :type X0: float
        :return: The simulated data of shape (num_instances + 1, num_steps)
        :rtype: NumPy array of shape (num_instances + 1, num_steps)
        """

        num_steps, times, data = self.data_for_simulation(t, timestep, num_instances)
        X = 0
        if self.custom_increment(X, timestep) is None:
            return None
        else:
            for i in range(num_instances):
                if X0 is None:
                    if self._multiplicative is False:
                        X = 0
                    else:
                        X = 1
                else:
                    X = X0
                    if self._multiplicative and X0<=0:
                        warnings.warn("You have selected a negative or zero initial value for a multiplicative process. It is likely that it is done by mistake. We recommend to change the initial value to a positive value.", KnowWhatYouDoWarning)
                for step in range(num_steps):
                    data[i, step] = X
                    dX = self.custom_increment(X, timestep)
                    self._memory = self.memory_update(step)
                    X = X + dX
                    if self._multiplicative is True:
                        if X < 0:
                            X = 2**(-1000)
                            warnings.warn("The process has reached a negative value. The simulation will continue with a very small positive value. It may impact the results of the simulation in an unexpected or unintended way. We recommend to decrease the timestep to avoid this issue.", UserWarning)
                    if verbose and step % 1000 == 0:
                        print(f"Simulating instance {i}, step {step}, X = {X}")

            data = np.concatenate((times.reshape(1, -1), data), axis=0)

            return data

    def simulate_until(self, timestep: float = timestep_default, num_instances: float = num_instances_default, X0: float = None, condition: Callable[..., bool] = None, save=False, plot=True) -> Any:
        """
        Simulate the process until a certain condition is reached.

        :param timestep: Time step for the simulation
        :type timestep: float
        :param num_instances: Number of instances to simulate
        :type num_instances: int
        :param X0: Initial value for the process
        :type X0: float
        :param condition: The condition to reach
        :type condition: Callable[..., bool]
        :param save: Whether to save the simulation results
        :type save: bool
        :param plot: Whether to plot the simulation results
        :type plot: bool
        :return: The simulated data
        :rtype: NumPy array of shape (num_instances + 1, num_steps)
        """
        data = np.zeros((num_instances+1, int(0)))
        X = 0
        if self.custom_increment(X, timestep) is None:
            return None
        else:
            for i in range(num_instances):
                if X0 is None:
                    if self._multiplicative is False:
                        X = 0
                    else:
                        X = 1
                else:
                    X = X0
                    if self._multiplicative and X0<=0:
                        warnings.warn("You have selected a negative or zero initial value for a multiplicative process. It is likely that it is done by mistake. We recommend to change the initial value to a positive value.", KnowWhatYouDoWarning)
                step = 0
                while not condition(X):
                    if step >= data.shape[1]:
                        data = np.hstack((data, np.zeros((data.shape[0], step + 1 - data.shape[1]))))
                    # if step >= data.shape[1]:
                    #     new_shape = (data.shape[0], step + 1)
                    #     data = np.resize(data, new_shape)

                    data[i+1, step] = X
                    if i == 0:
                        data[0, step] = step*timestep

                    dX = self.custom_increment(X, timestep)
                    self._memory = self.memory_update(step)
                    X = X + dX
                    if self._multiplicative is True:
                        if X < 0:
                            X = 2 ** (-1000)
                            warnings.warn(
                                "The process has reached a negative value. The simulation will continue with a very small positive value. It may impact the results of the simulation in an unexpected or unintended way. We recommend to decrease the timestep to avoid this issue.",
                                UserWarning)
                    if verbose and step % 1000 == 0:
                        print(f"Simulating instance {i}, step {step}, X = {X}")
                    step += 1
            # data_raw, times = self.separate(data)
            self.plot(data_full=data, num_instances=num_instances, save=save, plot=plot)

            return data

    def save_to_file(self, data, file_name: str, save: bool = False):
        """
        Save the data to a file.

        :param data: The data to save
        :type data: Any
        :param file_name: The name of the file
        :type file_name: str
        :param save: Whether to save the file
        :type save: bool
        :return: None
        :rtype: None
        :exception Exception: If the file is created but empty
        """
        # print(f"Debug: Entering save_to_file method")
        # print(f"Debug: save parameter: {save}")
        # print(f"Debug: file_name: {file_name}")
        # print(f"Debug: data shape: {data.shape}")

        if save:
            full_path = os.path.join(self._output_dir, f"{self._name}_{file_name}")
            # print(f"Debug: Attempting to save file to: {full_path}")
            try:
                # First, try to save using numpy's savetxt
                np.savetxt(full_path, data, delimiter=",")
                # print(f"Data saved to {full_path} using np.savetxt")

                # Verify that the file is not empty
                if os.path.getsize(full_path) == 0:
                    raise Exception("File was created but is empty")

            except Exception as e:
                # print(f"Error saving file with np.savetxt: {str(e)}")
                # print("Attempting to save with csv writer...")

                try:
                    # If numpy's savetxt fails, try using csv writer
                    with open(full_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')
                        for row in data:
                            writer.writerow(row)
                    # print(f"Data saved to {full_path} using csv writer")

                    # Verify that the file is not empty
                    if os.path.getsize(full_path) == 0:
                        raise Exception("File was created but is empty")

                except Exception as e:
                    print(f"Error saving file with csv writer: {str(e)}")
                    print("Debug: Printing first few rows of data:")
                    print(data[:5])  # Print first 5 rows of data

            # finally:
            #     # Check if the file exists and its size
            #     if os.path.exists(full_path):
            #         print(f"File size: {os.path.getsize(full_path)} bytes")
            #     else:
            #         print("File does not exist after save attempt")
        else:
            # print("Debug: save is False, not saving file")
            pass
        # print("Debug: Exiting save_to_file method")

    def plot(self, data_full, num_instances: int, save: bool = False, plot: bool = False, average_and_max: bool = False,
             plotlog: bool = False):
        """
        Plot the simulation results.

        :param data_full: The full data to plot
        :type data_full: Any
        :param num_instances: The number of instances to plot
        :type num_instances: int
        :param save: Whether to save the plot
        :type save: bool
        :param plot: Whether to plot the data
        :type plot: bool
        :param average_and_max: Whether to plot the average and max values
        :type average_and_max: bool
        :param plotlog: Whether to plot the data on a logarithmic scale
        :type plotlog: bool
        :return: None
        :rtype: None
        """
        if plot:
            times, data = self.separate(data_full)
            t = times[-1]
            timestep = times[1] - times[0]

            if not average_and_max:
                # Visualization
                plt.figure(figsize=(10, 6))

                for i in range(num_instances):
                    plt.plot(times, data[i, :], lw=0.5)

                plt.title(f'Simulation of {self.name}')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.grid(True)  # Change the color and thickness of the grid lines

                if plotlog:
                    plt.yscale('log')  # Apply logarithmic scale for y-axis if plotlog is True

                if save:
                    plt.savefig(os.path.join(self._output_dir,
                                             f'{self.name}_{self.get_params()}_t:{t}_timestep:{timestep}_num_instances:{num_instances}_process_simulation.png'))

                plt.show()

            else:
                # Calculate average and max values at each time point
                avg_data = np.mean(data, axis=0)
                max_data = np.max(data, axis=0)

                # Visualization
                plt.figure(figsize=(10, 6))

                # Plot individual instances with faint lines
                for i in range(num_instances):
                    plt.plot(times, data[i, :], lw=0.5, alpha=0.3, color='gray')  # Fainter color and lower alpha

                # Plot average and max values with stronger visibility
                plt.plot(times, avg_data, lw=2, label='Average', color='blue')  # Thicker line for average
                plt.plot(times, max_data, lw=2, label='Max', color='red')  # Thicker line for max

                plt.title(f'Simulation of {self.name}')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.grid(True)
                plt.legend()

                if plotlog:
                    plt.yscale('log')  # Apply logarithmic scale for y-axis if plotlog is True

                if save:
                    plt.savefig(os.path.join(self._output_dir,
                                             f'{self.name}_{self.get_params()}_t:{t}_timestep:{timestep}_num_instances:{num_instances}_process_simulation.png'))

                plt.show()

    def simulate(self, t: float = t_default, timestep: float = timestep_default, num_instances: int = num_instances_default, save: bool = False, plot: bool = False, average_and_max: bool = False, plotlog: bool = False, X0: float = None) -> Any:
        """
        Simulate the process using the given stochastic process class.
        A general and widely used method that is used to simulate the process.
        It uses several intermediate methods defined before.
        Many functions and methods rely on this method.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param num_instances: Number of instances to simulate
        :type num_instances: int
        :param save: Whether to save the results to a file
        :type save: bool
        :param plot: Whether to plot the results
        :type plot: bool
        :param average_and_max: Whether to plot the average and max values
        :type average_and_max: bool
        :param plotlog: Whether to plot the data on a logarithmic scale
        :type plotlog: bool
        :param X0: Initial value for the process
        :type X0: float
        :return: Simulated data array of shape (num_instances + 1, num_steps) where the first row is time and the rest are process values
        :rtype: NumPy array of shape (num_instances + 1, num_steps)
        """
        params = self.correct_params(self.get_params(), t)

        if ((self._process_class is not None) and (self._external_simulator is True)):

            if X0 is not None:
                raise InDevelopmentWarning('Initial value X0 is not supported for external simulators.'
                                           'Please either use an internal simulator or do not use an initial value.'
                                           'For external simulators, the initial value is set 0 for additivive process and 1 for multiplicative processes, and cannot be changed.')

            num_steps, times, data = self.data_for_simulation(t, timestep, num_instances)

            process = self._process_class(t=t, **params)
            for i in range(num_instances):
                data[i, :] = process.sample(num_steps - 1)
                if verbose == True and i % 1000000 == 0:
                    print(f"Simulating instance {i + 1} of {num_instances}...")

            # add times as the first row
            data_full = np.concatenate((times.reshape(1, -1), data), axis=0)

        else:
            custom_simulate_value = self.custom_simulate_raw(t, timestep, num_instances, X0=X0)

            if custom_simulate_value is not None:
                self._has_wrong_params = False

                data_full = custom_simulate_value

            else:
                raise ValueError("The process must be simulated using either an external simulator or a custom simulation method. For this process, neither is available.")

        self.save_to_file(data_full,
                          f"process_simulation_{self.get_params()}, t:{t}, timestep:{timestep}, num_instances:{num_instances}.csv",
                          save)

        self.plot(data_full, num_instances, save, plot, average_and_max=average_and_max, plotlog=plotlog)

        return data_full

    def simulate_ensembles(self, t: float = t_default, timestep: float = timestep_default,
                           num_instances: int = num_instances_default, num_ensembles: int = num_ensembles_default,
                           save: bool = False, plot: bool = False, plot_separate_ensembles: bool = True, X0: float = None) -> Any:
        """
        Simulate multiple ensembles of the process and visualize the results.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param num_instances: Number of instances to simulate
        :type num_instances: int
        :param num_ensembles: Number of ensembles to simulate
        :type num_ensembles: int
        :param save: Whether to save the results to a file
        :type save: bool
        :param plot: Whether to plot the results
        :type plot: bool
        :param plot_separate_ensembles: Whether to plot the separate ensembles
        :type plot_separate_ensembles: bool
        :param X0: Initial value for the process
        :type X0: float
        :return: The ensemble simulation data and the total average
        :rtype: Any
        """
        num_steps = int(t / timestep)
        ensemble = np.empty((num_ensembles + 1, num_steps))
        times = np.linspace(0, t, num_steps)
        ensemble[0, :] = times

        for i in range(num_ensembles):
            instance = self.simulate(t, timestep, num_instances, plot=False, X0=X0)
            averages = average(instance, visualize=False)

            # Check the shape of averages and adjust if necessary
            if averages.shape == (2, num_steps):
                # If averages has shape (2, num_steps), take the second row
                averages = averages[1, :]
            elif averages.shape != (num_steps,):
                raise ValueError(
                    f"Unexpected shape of averages: {averages.shape}. Expected ({num_steps},) or (2, {num_steps})")

            ensemble[i + 1, :] = averages

        total_average = np.mean(ensemble[1:], axis=0)
        if plot:
            # visualize all paths of averages as well as total average
            plt.figure(figsize=(10, 6))
            if plot_separate_ensembles:
                for i in range(num_ensembles):
                    plt.plot(times, ensemble[i+1, :], lw=0.5)
            plt.plot(times, total_average, 'k--', lw=2)
            plt.title(f'Ensemble Simulation of {self.name}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.grid(True)
            if save:
                plt.savefig(os.path.join(self._output_dir, f'{self.name}_{self.get_params()}_t:{t}_timestep:{timestep}_num_instances:{num_instances}_num_ensembles:{num_ensembles}_process_simulation.png'))
            plt.show()
        return ensemble, total_average

    def simulate_weights(self, t: float = t_default, timestep: float = timestep_default,
                         num_instances: int = num_instances_default, save: bool = False,
                         plot: bool = False) -> np.ndarray:
        """
        Simulate the weights (relative shares) of the process instances in the ensemble.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param num_instances: Number of instances to simulate
        :type num_instances: int
        :param save: Whether to save the results to a file
        :type save: bool
        :param plot: Whether to plot the results
        :type plot: bool
        :return: Array of simulated weights
        :rtype: NumPy array of shape (num_instances + 1, num_steps)
        """
        # Simulate the process
        data_full = self.simulate(t, timestep, num_instances, save=False, plot=False)

        # Extract times and process values
        times = data_full[0, :]
        process_values = data_full[1:, :]

        if self._multiplicative:
            # For multiplicative processes, weights are directly proportional to the process values
            weights = process_values / np.sum(process_values, axis=0)
        else:
            # For non-multiplicative processes, we need to handle potential negative values
            # We'll use the softmax function to ensure positive weights
            exp_values = np.exp(process_values)
            weights = exp_values / np.sum(exp_values, axis=0)

        # Combine times and weights
        weight_data = np.vstack((times, weights))

        if save:
            filename = f"weights_simulation_{self.get_params()}, t:{t}, timestep:{timestep}, num_instances:{num_instances}.csv"
            header = ','.join(['time'] + [f'weight_{i}' for i in range(num_instances)])
            np.savetxt(filename, weight_data.T, delimiter=',', header=header, comments='')
            print(f"Weights saved to {filename}")

        if plot:
            self.plot_weights(times, weights, save)

        return weight_data

    def plot_weights(self, times, weights, save):
        """
        Plot the simulated weights.

        :param times: Array of time values
        :type times: np.ndarray
        :param weights: Array of simulated weights
        :type weights: np.ndarray
        :param save: Whether to save the plot
        :type save: bool
        :return: None
        :rtype: None
        """
        plt.figure(figsize=(10, 6))
        for i in range(weights.shape[0]):
            plt.plot(times, weights[i, :], label=f'Weight {i + 1}')
        plt.xlabel('Time')
        plt.ylabel('Weight')
        plt.title('Simulated Weights')
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(f"weights_plot_{self.get_params()}.png")
        plt.show()

    def simulate_2d(self, t: float = t_default, timestep: float = timestep_default,
                    num_instances: int = num_instances_default, save: bool = False, plot: bool = False) -> Any:
        """
        Simulate a 2D process by combining two 1D simulations.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param num_instances: Number of instances to simulate
        :type num_instances: int
        :param save: Whether to save the simulation results
        :type save: bool
        :param plot: Whether to plot the simulation results
        :type plot: bool
        :return: Simulated 2D data array
        :rtype: NumPy array of shape (2 * num_instances + 1, num_steps)
        """
        # Simulate two 1D processes
        data_x = self.simulate(t, timestep, num_instances, save=False, plot=False)
        data_y = self.simulate(t, timestep, num_instances, save=False, plot=False)

        # Extract time values
        times = data_x[0, :]

        # Combine the two 1D simulations into a 2D simulation
        data_2d = np.zeros((2 * num_instances + 1, len(times)))
        data_2d[0, :] = times  # First row is time
        data_2d[1:num_instances + 1, :] = data_x[1:, :]  # X dimension
        data_2d[num_instances + 1:, :] = data_y[1:, :]  # Y dimension

        filename = f"process_simulation_2d_{self.get_params()}, t:{t}, timestep:{timestep}, num_instances:{num_instances}.csv"
        self.save_to_file(data_2d, filename, save)

        self.plot_2d(data_2d, num_instances, save, plot)
        self.plot_2dt(data_2d, num_instances, save, plot)

        return data_2d

    def plot_2d(self, data_2d: np.ndarray, num_instances: int, save: bool = False, plot: bool = True):
        """
        Plot the 2D simulation results.

        :param data_2d: 2D simulation data
        :type data_2d: np.ndarray
        :param num_instances: Number of instances simulated
        :type num_instances: int
        :param save: Whether to save the plot
        :type save: bool
        :param plot: Whether to display the plot
        :type plot: bool
        :return: None
        :rtype: None
        """
        if not plot:
            return

        times = data_2d[0, :]
        x_data = data_2d[1:num_instances + 1, :]
        y_data = data_2d[num_instances + 1:, :]

        plt.figure(figsize=(10, 8))
        for i in range(num_instances):
            plt.plot(x_data[i, :], y_data[i, :], alpha=0.5)

        plt.title(f"2D Process Simulation (t={times[-1]}, timestep={times[1] - times[0]}, instances={num_instances})")
        plt.xlabel("X dimension")
        plt.ylabel("Y dimension")

        if save:
            plt.savefig(f"2d_simulation_plot_{self.get_params()}.png")

        if plot:
            plt.show()

    def plot_2dt(self, data_2d: np.ndarray, num_instances: int, save: bool = False, plot: bool = True):
        """
        Plot the 2D simulation results in a 3D graph with time as the third dimension.

        :param data_2d: 2D simulation data
        :type data_2d: np.ndarray
        :param num_instances: Number of instances simulated
        :type num_instances: int
        :param save: Whether to save the plot
        :type save: bool
        :param plot: Whether to display the plot
        :type plot: bool
        :return: None
        :rtype: None
        """
        if not plot:
            return None

        times = data_2d[0, :]
        x_data = data_2d[1:num_instances + 1, :]
        y_data = data_2d[num_instances + 1:, :]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(num_instances):
            ax.plot(x_data[i, :], y_data[i, :], times, alpha=0.5)

        ax.set_xlabel('X dimension')
        ax.set_ylabel('Y dimension')
        ax.set_zlabel('Time')
        ax.set_title(
            f"3D Visualization of 2D Process Simulation\n(t={times[-1]}, timestep={times[1] - times[0]}, instances={num_instances})")

        if save:
            plt.savefig(f"3d_simulation_plot_{self.get_params()}.png")

        if plot:
            plt.show()

    def simulate_3d(self, t: float = t_default, timestep: float = timestep_default,
                    num_instances: int = num_instances_default, save: bool = False, plot: bool = False) -> Any:
        """
        Simulate a 3D process by combining three 1D simulations.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param num_instances: Number of instances to simulate
        :type num_instances: int
        :param save: Whether to save the simulation results
        :type save: bool
        :param plot: Whether to plot the simulation results
        :type plot: bool
        :return: Simulated 3D data array
        :rtype: NumPy array of shape (3 * num_instances + 1, num_steps)
        """
        # Simulate three 1D processes
        data_x = self.simulate(t, timestep, num_instances, save=False, plot=False)
        data_y = self.simulate(t, timestep, num_instances, save=False, plot=False)
        data_z = self.simulate(t, timestep, num_instances, save=False, plot=False)

        # Extract time values
        times = data_x[0, :]

        # Combine the three 1D simulations into a 3D simulation
        data_3d = np.zeros((3 * num_instances + 1, len(times)))
        data_3d[0, :] = times  # First row is time
        data_3d[1:num_instances + 1, :] = data_x[1:, :]  # X dimension
        data_3d[num_instances + 1:2 * num_instances + 1, :] = data_y[1:, :]  # Y dimension
        data_3d[2 * num_instances + 1:, :] = data_z[1:, :]  # Z dimension

        filename = f"process_simulation_3d_{self.get_params()}, t:{t}, timestep:{timestep}, num_instances:{num_instances}.csv"
        self.save_to_file(data_3d, filename, save)

        self.plot_3d(data_3d, num_instances, save, plot)

        return data_3d

    def plot_3d(self, data_3d: np.ndarray, num_instances: int, save: bool = False, plot: bool = True):
        """
        Plot the 3D simulation results.

        :param data_3d: 3D simulation data
        :type data_3d: np.ndarray
        :param num_instances: Number of instances simulated
        :type num_instances: int
        :param save: Whether to save the plot
        :type save: bool
        :param plot: Whether to display the plot
        :type plot: bool
        :return: None
        :rtype: None
        """
        if not plot:
            return

        times = data_3d[0, :]
        x_data = data_3d[1:num_instances + 1, :]
        y_data = data_3d[num_instances + 1:2 * num_instances + 1, :]
        z_data = data_3d[2 * num_instances + 1:, :]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(num_instances):
            ax.plot(x_data[i, :], y_data[i, :], z_data[i, :], alpha=0.5)

        ax.set_xlabel('X dimension')
        ax.set_ylabel('Y dimension')
        ax.set_zlabel('Z dimension')
        ax.set_title(
            f"3D Process Simulation\n(t={times[-1]}, timestep={times[1] - times[0]}, instances={num_instances})")

        if save:
            plt.savefig(f"3d_simulation_plot_{self.get_params()}.png")

        if plot:
            plt.show()

    def get_data_for_moments(self, data, t_m=10, timestep_m=0.01, num_instances_m=10):
        """
        Get the data for moments calculation. If the data was already pre-computed and passed to the method, it will be used.
        Else, the data will be simulated using the specified parameters.
        Note that if you pass the data and simultaneously define the parameters, the parameters will be ignored, and the data will be used.

        :param data: The data array of the process.
        :type data: Any
        :param t_m: Total time for the simulation.
        :type t_m: float
        :param timestep_m: Time step for the simulation.
        :type timestep_m: float
        :param num_instances_m: Number of instances to simulate.
        :type num_instances_m: int
        :return: The data array for moments calculation.
        :rtype: NumPy array
        """
        if data is not None:
            data = data
        else:
            data = self.simulate(t=t_m, timestep=timestep_m, num_instances=num_instances_m)
        return data

    def moments(self, data=None, save: bool = False, t = 10, timestep = 0.01, num_instances = 10) -> Any:
        """
        Calculate the cumulative moments of the process increments up to every point in time using an optimized iterative approach.
        If the process is multiplicative, the increments are calculated for the logarithm of the data.
        The moments can be calculated either based on the already simulated data (using simulate method or exporting data) or by simulating the data using the specified parameters.

        :param data: The data array of the process.
        :type data: Any
        :param save: Whether to save the results to a file.
        :type save: bool
        :param t: Total time for the simulation.
        :type t: float
        :param timestep: Time step for the simulation.
        :type timestep: float
        :param num_instances: Number of instances to simulate.
        :type num_instances: int
        :return: A tuple of times and the calculated moments (mean, variance, skewness, kurtosis, etc.
        :rtype: Tuple
        """
        if data is None:
            data = self.get_data_for_moments(data, t_m=t, timestep_m=timestep, num_instances_m=num_instances)

        times, data_raw = self.separate(data)

        if self._multiplicative:
            data_raw = np.log(data_raw)

        data = np.diff(data_raw, axis=1)

        n_paths, n_times = data.shape

        timestep = times[1] - times[0]

        cum_mean = np.zeros((n_paths, n_times))
        cum_var = np.zeros((n_paths, n_times))
        cum_m3 = np.zeros((n_paths, n_times))
        cum_m4 = np.zeros((n_paths, n_times))
        cum_mad = np.zeros((n_paths, n_times))

        for i in range(n_paths):
            path = data[i]
            cum_mean[i, 0] = path[0]

            for t in range(1, n_times):
                current_value = path[t]
                delta1 = 1 / t
                delta2 = (t - 1) / t

                cum_mean[i, t] = cum_mean[i, t - 1] * delta2 + current_value * delta1
                cum_var[i, t] = cum_var[i, t - 1] * delta2 + ((current_value - cum_mean[i, t]) ** 2) * delta1
                cum_m3[i, t] = cum_m3[i, t - 1] * delta2 + ((current_value - cum_mean[i, t]) ** 3 ) * delta1
                cum_m4[i, t] = cum_m4[i, t - 1] * delta2 + ((current_value - cum_mean[i, t]) ** 4) * delta1
                cum_mad[i, t] = cum_mad[i, t - 1] * delta2 + np.abs(current_value - cum_mean[i, t]) * delta1

        mean = np.mean(cum_mean, axis=0) / timestep
        variance = np.mean(cum_var, axis=0) / timestep
        skewness = np.mean(cum_m3 / np.power(cum_var, 1.5), axis=0)
        kurtosis = np.mean(cum_m4 / np.square(cum_var), axis=0) - 3
        mad = np.mean(cum_mad, axis=0) / timestep ** 0.5

        data_full = np.vstack([times[1:], mean, variance, skewness, kurtosis, mad])

        self.save_to_file(data_full,
                          f"moments_simulation_{self.get_params()}.csv",
                          save)

        return times, mean, variance, skewness, kurtosis, mad

    def k_moments(self, data=None, order: int = 4, save: bool = False, t: float = 10, timestep: float = 0.01,
                          num_instances: int = 10, visualize: bool = False) -> Any:
        """
        Calculate the cumulative moments of the process up to a specified order for every point in time
        using an optimized iterative approach.

        :param data: The data array of the process.
        :type data: Any
        :param order: The maximum order of moments to calculate.
        :type order: int
        :param save: Whether to save the results to a file.
        :type save: bool
        :param t: Total time for the simulation.
        :type t: float
        :param timestep: Time step for the simulation.
        :type timestep: float
        :param num_instances: Number of instances to simulate.
        :type num_instances: int
        :return: A tuple of times and the calculated moments (mean, variance, skewness, kurtosis, etc.).
        :rtype: Tuple
        """
        if data is None:
            data = self.get_data_for_moments(data, t_m=t, timestep_m=timestep, num_instances_m=num_instances)

        times, data_raw = self.separate(data)

        if self._multiplicative:
            data_raw = np.log(data_raw)

        data = np.diff(data_raw, axis=1)

        n_paths, n_times = data.shape

        timestep = times[1] - times[0]

        # Initialize arrays to store cumulative moments
        cum_moments = [np.zeros((n_paths, n_times)) for _ in range(order)]

        # First moment (mean)
        cum_moments[0][:, 0] = data[:, 0]

        for i in range(n_paths):
            path = data[i]

            for t in range(1, n_times):
                current_value = path[t]
                delta1 = 1 / t
                delta2 = (t - 1) / t

                for k in range(order):
                    if k == 0:
                        cum_moments[k][i, t] = cum_moments[k][i, t - 1] * delta2 + current_value * delta1
                    else:
                        cum_moments[k][i, t] = cum_moments[k][i, t - 1] * delta2 + (
                                    (current_value - cum_moments[0][i, t]) ** (k + 1)) * delta1

        # Convert cumulative moments to actual moments
        moments = [np.mean(cum_moments[k], axis=0) / timestep ** (k if k > 0 else 1) for k in range(order)]

        # Adjust moments if necessary (e.g., skewness, kurtosis)
        if order >= 3:
            moments[2] = moments[2] / np.power(moments[1], 1.5)  # Skewness
        if order >= 4:
            moments[3] = moments[3] / np.square(moments[1]) - 3  # Kurtosis

        # Compile data into a single array
        data_full = np.vstack([times[1:]] + moments)

        if save:
            self.save_to_file(data_full,
                              f"moments_simulation_order_{order}_{self.get_params()}.csv",
                              save)
        # remove the first element of times
        times = times[1:]

        if visualize:
            for i in range(order):
                self.plot_moment(times, moments[i], f"moment_{i + 1}", mask=mask_default, save=save)

        return (times, *moments)

    def plot_moment(self, times, moment, label, mask, save: bool = False):
        """
        Visualize a given moment of the process - a helper function for the next methods.

        :param times: The times at which the process is sampled
        :type times: np.ndarray
        :param moment: The moment to plot
        :type moment: np.ndarray
        :param label: The label of the moment
        :type label: str
        :param mask: The mask to apply to the data
        :type mask: int
        :param save: Whether to save the plot
        :type save: bool
        :return: None
        :rtype: None
        """

        times = times[mask:]
        moment = moment[mask:]

        plt.figure(figsize=(10, 6))
        plt.plot(times, moment, label=label)
        plt.title(f'{label} of {self.name}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(os.path.join(self._output_dir, f'{label}_of_{self.name}_simulation.png'))
        plt.show()
        plt.close()

        return None

    def moments_dict(self, data, save, t = 10, timestep = 0.01, num_instances = 10):
        """
        Create a dictionary to further calculate the first, the second, the third, and the fourth moments of the process.

        :param data: The data array of the process
        :type data: NumPy array
        :param save: Whether to save the results to a file
        :type save: bool
        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param num_instances: Number of instances to simulate
        :type num_instances: int
        :return: A dictionary of moments where the keys are the moment names and the values are the calculated moments
        :rtype: Dict
        """
        times, mean, variance, skewness, kurtosis, mad = self.moments(data, save=save, t=t, timestep=timestep, num_instances=num_instances)
        return {"Mean": mean, "Variance": variance, "Skewness": skewness, "Kurtosis": kurtosis, "Mean Absolute Deviation": mad}

    def visualize_moments(self, data=None, mask=mask_default, save: bool = False, t = 10, timestep = 0.01, num_instances = 10):
        """
        Visualize the first, the second, the third, and the fourth moments of the process.

        :param times: The times at which the process is sampled
        :type times: np.ndarray
        :param data: The data array of the process
        :type data: NumPy array
        :param mask: The mask to apply to the data
        :type mask: int
        :param save: Whether to save the plot
        :type save: bool
        :return: None
        :rtype: None
        """
        if data is None:
            data = self.get_data_for_moments(data, t_m=t, timestep_m=timestep, num_instances_m=num_instances)

        times, data_raw = self.separate(data)
        moments = self.moments_dict(data, save=save, t=t, timestep=timestep, num_instances=num_instances)
        times = times[1:]
        for moment in moments:
            self.plot_moment(times, moments[moment], moment, mask, save=save)

        self.save_to_file(moments,
                          f"moments_simulation_{self.get_params()}.csv",
                          save)

    def visualize_moment(self, data=None, label="Mean", mask=mask_default, save: bool = False, t = 10, timestep = 0.01, num_instances = 10):
        """
        Visualize a given moment of the process - a helper function for the next methods.

        :param data: The data array of the process
        :type data: NumPy array
        :param label: The label of the moment
        :type label: str
        :param mask: The mask to apply to the data
        :type mask: int
        :param save: Whether to save the plot
        :type save: bool
        :return: None
        :rtype: None
        """
        if data is None:
            data = self.get_data_for_moments(data, t_m=t, timestep_m=timestep, num_instances_m=num_instances)

        times, data_raw = self.separate(data)
        times = times[1:]
        moments = self.moments_dict(data, save=save, t=t, timestep=timestep, num_instances=num_instances)
        self.plot_moment(times, moments[label], label, mask, save=save)

        self.save_to_file(moments[label],
                          f"{label}_simulation_{self.get_params()}.csv",
                          save)
        return moments[label]

    def simulate_live(self, t: float = t_default, timestep: float = timestep_default, num_instances: int = num_instances_default, save: bool = False, speed: float = 1) -> Any:
        """
        Simulate the process live and save as a video file.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param num_instances: Number of instances to simulate
        :type num_instances: int
        :param save: Whether to save the simulation data
        :type save: bool
        :param speed: Speed multiplier for the video (default is 1.0, higher values make the video faster)
        :type speed: float
        :return: Video file of the simulation
        :rtype: str
        """

        data = self.simulate(t, timestep, num_instances)
        times, data = self.separate(data)

        num_steps = data.shape[1]

        fig, ax = plt.subplots()
        lines = [ax.plot([], [], lw=0.5)[0] for _ in range(num_instances)]
        ax.set_xlim(0, t)
        ax.set_ylim(np.min(data), np.max(data))
        ax.set_title(f'Simulation of {self.name}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True)

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(frame):
            for i in range(num_instances):
                lines[i].set_data(times[:frame + 1], data[i, :frame + 1])
            return lines

        # Calculate fps based on the desired speed
        fps = (1 / timestep) * speed
        interval = 1000 / fps  # interval in milliseconds

        ani = animation.FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=interval)
        ani.save(f'{self.name}_simulation_with_parameters_{self.get_params()}_t:{t}_timestep:{timestep}_num_instances:{num_instances}.mp4', writer='ffmpeg')

        plt.close(fig)

        self.save_to_file(data, "liv_process_simulation_with_parameters_{self.get_params()}_t:{t}_timestep:{timestep}_num_instances:{num_instances}.csv", save)

        return f'{self.name}_simulation.mp4'

    def simulate_live_2d(self, t: float = t_default, timestep: float = timestep_default,
                         num_instances: int = num_instances_default, save: bool = False,
                         speed: float = 1.0) -> str:
        """
        Simulate the 2D process live and save as a video file.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param num_instances: Number of instances to simulate
        :type num_instances: int
        :param save: Whether to save the simulation data
        :type save: bool
        :param speed: Speed multiplier for the video (default is 1.0, higher values make the video faster)
        :type speed: float
        :return: Video file name of the simulation
        :rtype: str
        :exception: CalledProcessError if the video cannot be saved
        """
        data_2d = self.simulate_2d(t, timestep, num_instances)
        times = data_2d[0, :]
        x_data = data_2d[1:num_instances + 1, :]
        y_data = data_2d[num_instances + 1:, :]

        num_steps = data_2d.shape[1]

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        lines = [ax.plot([], [], lw=0.5)[0] for _ in range(num_instances)]
        ax.set_xlim(np.min(x_data), np.max(x_data))
        ax.set_ylim(np.min(y_data), np.max(y_data))
        ax.set_title(f'2D Simulation of {self.name}')
        ax.set_xlabel('X dimension')
        ax.set_ylabel('Y dimension')
        ax.grid(True)

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(frame):
            for i in range(num_instances):
                lines[i].set_data(x_data[i, :frame + 1], y_data[i, :frame + 1])
            return lines

        # Calculate fps based on the desired speed
        fps = (1 / timestep) * speed
        interval = 1000 / fps  # interval in milliseconds

        ani = animation.FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=interval)
        video_filename = f'{self.name}_2d_simulation_with_parameters_{self.get_params()}_t:{t}_timestep:{timestep}_num_instances:{num_instances}.mp4'

        try:
            ani.save(video_filename, writer='ffmpeg', fps=fps, dpi=100, codec='libx264', bitrate=-1,
                     extra_args=['-pix_fmt', 'yuv420p'])
            print(f"2D simulation video saved as {video_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error saving 2D simulation video: {e}")
        finally:
            plt.close(fig)

        self.save_to_file(data_2d, f"{self.name}_2d_simulation_with_parameters_{self.get_params()}_t:{t}_timestep:{timestep}_num_instances:{num_instances}.csv", save)

        return video_filename

    def simulate_live_3d(self, t: float = t_default, timestep: float = timestep_default,
                         num_instances: int = num_instances_default, save: bool = False,
                         speed: float = 1.0) -> str:
        """
        Simulate the 3D process live and save as a video file.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param num_instances: Number of instances to simulate
        :type num_instances: int
        :param save: Whether to save the simulation data
        :type save: bool
        :param speed: Speed multiplier for the video (default is 1.0, higher values make the video faster)
        :type speed: float
        :return: Video file name of the simulation
        :rtype: str
        :exception: CalledProcessError if the video cannot be saved
        """
        data_3d = self.simulate_3d(t, timestep, num_instances)
        times = data_3d[0, :]
        x_data = data_3d[1:num_instances + 1, :]
        y_data = data_3d[num_instances + 1:2 * num_instances + 1, :]
        z_data = data_3d[2 * num_instances + 1:, :]

        num_steps = data_3d.shape[1]

        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        lines = [ax.plot([], [], [], lw=0.5)[0] for _ in range(num_instances)]
        ax.set_xlim(np.min(x_data), np.max(x_data))
        ax.set_ylim(np.min(y_data), np.max(y_data))
        ax.set_zlim(np.min(z_data), np.max(z_data))
        ax.set_title(f'3D Simulation of {self.name}')
        ax.set_xlabel('X dimension')
        ax.set_ylabel('Y dimension')
        ax.set_zlabel('Z dimension')

        ax.view_init(elev=20, azim=45)

        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return lines

        # Calculate fps based on the desired speed
        fps = (1 / timestep) * speed
        interval = 1000 / fps  # interval in milliseconds

        def update(frame):
            for i in range(num_instances):
                lines[i].set_data(x_data[i, :frame + 1], y_data[i, :frame + 1])
                lines[i].set_3d_properties(z_data[i, :frame + 1])
            # ax.view_init(30, 0.3 * frame * speed)  # Rotate view for 3D effect, adjusted for speed
            return lines

        ani = animation.FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=False, interval=interval)
        video_filename = f'{self.name}_3d_simulation_with_parameters_{self.get_params()}_t:{t}_timestep:{timestep}_num_instances:{num_instances}.mp4'

        try:
            ani.save(video_filename, writer='ffmpeg', fps=fps, dpi=100, codec='libx264', bitrate=-1,
                     extra_args=['-pix_fmt', 'yuv420p'])
            print(f"3D simulation video saved as {video_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error saving 3D simulation video: {e}")
        finally:
            plt.close(fig)

        self.save_to_file(data_3d, f"{self.name}_3d_simulation_with_parameters_{self.get_params()}_t:{t}_timestep:{timestep}_num_instances:{num_instances}.csv", save)

        # Create Plotly Scatter3D trace for the simulation path
        trace = go.Scatter3d(x=x_data.flatten(), y=y_data.flatten(), z=z_data.flatten(),
                             mode='lines', line=dict(width=3, color='blue'))

        # Create Plotly figure
        fig = go.Figure()

        # Create a colormap to assign colors to each instance
        cmap = plt.get_cmap('viridis')  # Or any other colormap you prefer

        # Add a Scatter3D trace for each instance with unique color
        for i in range(num_instances):
            color = cmap(i / num_instances)  # Get a color from the colormap
            fig.add_trace(go.Scatter3d(
                x=x_data[i],
                y=y_data[i],
                z=z_data[i],
                mode='lines',
                line=dict(width=3, color=f'rgb{tuple(int(255 * c) for c in color[:3])}'),  # Convert color to RGB string
                name=f'Instance {i + 1}'  # Add a name for the legend
            ))

        # Show figure (optional)
        fig.show()

        # Save as interactive HTML file
        object_filename = f'{self.name}_3d_object_{self.get_params()}_t:{t}_timestep:{timestep}.html'
        fig.write_html(object_filename)
        print(f"3D object saved as {object_filename}")

        return video_filename, object_filename

    def simulate_live_2dt(self, t: float = t_default, timestep: float = timestep_default,
                                   num_instances: int = num_instances_default, save: bool = False,
                                   speed: float = 1.0) -> tuple[str, str]:
        """
        Simulate the 2D process live with time as the third dimension and save as a video file and interactive plot.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param num_instances: Number of instances to simulate (default is 2)
        :type num_instances: int
        :param save: Whether to save the simulation data
        :type save: bool
        :param speed: Speed multiplier for the video (default is 1.0, higher values make the video faster)
        :type speed: float
        :return: Tuple of video file name and interactive plot file name
        :rtype: Tuple
        :exception: CalledProcessError if the video cannot be saved
        """
        data_2d = self.simulate_2d(t, timestep, num_instances)
        times = data_2d[0, :]
        x_data = data_2d[1:num_instances + 1, :]
        y_data = data_2d[num_instances + 1:, :]

        num_steps = data_2d.shape[1]

        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        lines = [ax.plot([], [], [], lw=2)[0] for _ in range(num_instances)]
        ax.set_xlim(np.min(x_data), np.max(x_data))
        ax.set_ylim(np.min(y_data), np.max(y_data))
        ax.set_zlim(np.min(times), np.max(times))
        ax.set_title(f'2D Simulation of {self.name} with Time')
        ax.set_xlabel('X dimension')
        ax.set_ylabel('Y dimension')
        ax.set_zlabel('Time')

        ax.view_init(elev=20, azim=45)

        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return lines

        fps = (1 / timestep) * speed
        interval = 1000 / fps  # interval in milliseconds

        def update(frame):
            for i in range(num_instances):
                lines[i].set_data(x_data[i, :frame + 1], y_data[i, :frame + 1])
                lines[i].set_3d_properties(times[:frame + 1])
            return lines

        ani = animation.FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=False, interval=interval)
        video_filename = f'{self.name}_2d_time_simulation_{self.get_params()}_t:{t}_timestep:{timestep}_num_instances:{num_instances}.mp4'

        try:
            ani.save(video_filename, writer='ffmpeg', fps=fps, dpi=100, codec='libx264', bitrate=-1,
                     extra_args=['-pix_fmt', 'yuv420p'])
            print(f"2D with time simulation video saved as {video_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error saving 2D with time simulation video: {e}")
        finally:
            plt.close(fig)

        self.save_to_file(data_2d,
                          f"{self.name}_2d_time_simulation_{self.get_params()}_t:{t}_timestep:{timestep}_num_instances:{num_instances}.csv",
                          save)

        # Create Plotly figure
        fig = go.Figure()

        # Create a colormap to assign colors to each instance
        cmap = plt.get_cmap('viridis')  # Or any other colormap you prefer

        # Add a Scatter3d trace for each instance with unique color
        for i in range(num_instances):
            color = cmap(i / num_instances)  # Get a color from the colormap
            fig.add_trace(go.Scatter3d(
                x=x_data[i],
                y=y_data[i],
                z=times,
                mode='lines',
                line=dict(width=3, color=f'rgb{tuple(int(255 * c) for c in color[:3])}'),  # Convert color to RGB string
                name=f'Instance {i + 1}'  # Add a name for the legend
            ))

        # Update layout for better visibility
        fig.update_layout(
            scene=dict(
                xaxis_title='X dimension',
                yaxis_title='Y dimension',
                zaxis_title='Time',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            title=f'2D Simulation of {self.name} with Time'
        )

        # Show figure (optional)
        fig.show()

        # Save as interactive HTML file
        object_filename = f'{self.name}_2d_time_object_{self.get_params()}_t:{t}_timestep:{timestep}.html'
        fig.write_html(object_filename)
        print(f"2D with time object saved as {object_filename}")

        return video_filename, object_filename

    def increment_intermediate(self, timestep_increment: float = timestep_default) -> float:
        """
        Calculate the increment of the process for a given timestep increment.
        This method is used as an intermediate step for the increment method.

        :param timestep_increment: The timestep increment for the process
        :type timestep_increment: float
        :return: The increment of the process
        :rtype: float
        """
        if self._process_class is None:
            increment = self.custom_increment(1, timestep_increment)
        else:
            process = self._process_class(t=timestep_increment, **self.get_params())
            increment = process.sample(1)[1]
        return increment

    def increment(self, timestep_increment: float = timestep_default) -> float:
        """
        Calculate the increment of the process for a given timestep increment.

        :param timestep_increment: The timestep increment for the process
        :type timestep_increment: float
        :return: The increment of the process
        :rtype: float
        """
        if self.multiplicative or not self.independent:
            warnings.warn(
                "The increments are not independent for this process. Constructing a process from such increments will not lead to a valid process. This method will return an increment for the default (initial) state of the process.",
                UserWarning
            )
        increment = self.increment_intermediate(timestep_increment)
        return increment

    def ensemble_average(self, num_instances: int = 1000000, timestep: float = 0.01, save: bool = False) -> Any:
        """
        Calculate the finite ensemble average of the process.

        :param num_instances: The number of instances to simulate
        :type num_instances: int
        :param timestep: The time step for the simulation
        :type timestep: float
        :param save: Whether to save the results to a file
        :type save: bool
        :return: The ensemble average of the process
        :rtype: float
        """

        data = self.simulate(t=1, timestep=timestep, num_instances=num_instances)

        times, data = self.separate(data)
        # select the last value of each instance
        data = data[:, -1]
        # calculate the average
        ensemble_average = np.mean(data)

        self.save_to_file(ensemble_average,
                          f"ensemble_average_simulation_{self.get_params()}, timestep:{timestep}, num_instances:{num_instances}.csv",
                          save)

        return ensemble_average

    def time_average(self, t: float = 1000000, timestep: float = 0.01, save: bool = False) -> Any:
        """
        Calculate the time average approximation of the process in a direct way (using one process instance and long simulation time).

        :param t: The time to simulate
        :type t: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :param save: Whether to save the results to a file
        :type save: bool
        :return: The time average of the process
        :rtype: float
        """
        data = self.simulate(t=t, timestep=timestep, num_instances=1)
        times, data = self.separate(data)

        time_average = (np.mean(data))/timestep

        self.save_to_file(time_average,
                          f"time_average_simulation_{self.get_params()}, t:{t}, timestep:{timestep}.csv",
                          save)

        return time_average

    # methods to calculate ensemble and time averages of growth rates

    def growth_rate_ensemble_average(self, num_instances: int = 1000000, timestep: float = 0.01, save: bool = False) -> Any:
        """
        Calculate the ensemble average of the growth rates of the process.

        :param num_instances: The number of instances to simulate
        :type num_instances: int
        :param timestep: The time step for the simulation
        :type timestep: float
        :param save: Whether to save the results to a file
        :type save: bool
        :return: The ensemble average of the growth rates of the process
        :rtype: float
        """
        if self._multiplicative is True:

            data = self.simulate(t=1, timestep=timestep, num_instances=num_instances)

            times, data = self.separate(data)

            # calculate mean of the last column in data
            ensemble_average = np.mean(data[:, -1])

            # Calculate the growth rate for each instance
            ensemble_average_growth_rate = np.log(ensemble_average)

            self.save_to_file(ensemble_average_growth_rate,
                              f"growth_rate_ensemble_average_simulation_{self.get_params()}, timestep:{timestep}, num_instances:{num_instances}.csv",
                              save)

            return ensemble_average_growth_rate

        else:
            raise ValueError("The ensemble average of growth rates can now only be calculated for multiplicative processes.")


    def growth_rate_time_average(self, t: float = 1000000, timestep: float = 0.01, save: bool = False) -> Any:
        """
        Calculate the time average of the growth rates of the process in a direct way (using one process instance and long simulation time).

        :param t: The time to simulate
        :type t: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :param save: Whether to save the results to a file
        :type save: bool
        :return: The time average of the growth rates of the process
        :rtype: float
        """
        if self._multiplicative is True:

            data = self.simulate(t=t, timestep=timestep, num_instances=1)

            times, data = self.separate(data)

            last_element = data[0, -1]
            growth_rate = np.log(last_element)
            growth_rate_over_time = growth_rate / t

            self.save_to_file(growth_rate_over_time,
                              f"growth_rate_time_average_simulation_{self.get_params()}, t:{t}, timestep:{timestep}.csv",
                              save)

            return growth_rate_over_time

        else:
            raise ValueError("The time average of growth rates can now only be calculated for multiplicative processes.")

    def plot_growth_rate_of_average_3d(self, instance_range, time_range, instance_step, time_step, simulation_timestep=timestep_default,
                                    step_type='linear', filename='growth_rate_of_average_function', save_html=True):
        """
        Draw a 3D graph of average growth rate as a function of number of instances and time.

        :param instance_range: Tuple of (min_instances, max_instances)
        :type instance_range: Tuple
        :param time_range: Tuple of (min_time, max_time)
        :type time_range: Tuple
        :param instance_step: Step size for instances
        :type instance_step: float
        :param time_step: Step size for time
        :type time_step: float
        :param simulation_timestep: Time step for the simulation (used in the simulate method)
        :type simulation_timestep: float
        :param step_type: 'linear' or 'logarithmic'
        :type step_type: str
        :param filename: Path to save the results (if None, results won't be saved)
        :type filename: str
        :param save_html: Whether to save the interactive 3D plot as an HTML file
        :type save_html: bool
        :return: None
        :rtype: None
        """
        if step_type == 'linear':
            instances = np.arange(*instance_range, instance_step)
            times = np.arange(*time_range, time_step)
        elif step_type == 'logarithmic':
            instances = np.logspace(np.log10(instance_range[0]), np.log10(instance_range[1]),
                                    num=int((instance_range[1] - instance_range[0]) / instance_step))
            times = np.logspace(np.log10(time_range[0]), np.log10(time_range[1]),
                                num=int((time_range[1] - time_range[0]) / time_step))
        else:
            raise ValueError("step_type must be either 'linear' or 'logarithmic'")

        results = np.zeros((len(instances), len(times)))

        for i, num_instances in enumerate(instances):
            for j, t in enumerate(times):
                data = self.simulate(t=t, timestep=simulation_timestep, num_instances=int(num_instances))
                # print('simulation finished')
                results[i, j] = growth_rate_of_average_per_time(data)

        # Create the 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(times, instances)
        surf = ax.plot_surface(X, Y, results, cmap='viridis')

        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Instances')
        ax.set_zlabel('Average Growth Rate')
        ax.set_title('Average Growth Rate vs. Number of Instances and Time')

        fig.colorbar(surf, shrink=0.5, aspect=5)

        if filename:
            save_path = os.path.join(self._output_dir, filename)
            plt.savefig(save_path)
            np.save(save_path.replace('.png', '.npy'), results)

        plt.show()

        # Create and save interactive 3D plot using Plotly
        if save_html:
            fig_plotly = go.Figure(data=[go.Surface(z=results, x=times, y=instances)])
            fig_plotly.update_layout(title='Average Growth Rate vs. Number of Instances and Time',
                                     scene=dict(xaxis_title='Time',
                                                yaxis_title='Number of Instances',
                                                zaxis_title='Average Growth Rate'))

            html_filename = f'{self.name}_3d_growth_rate_{self.get_params()}_t:{time_range}_instances:{instance_range}.html'
            html_filename = os.path.join(self._output_dir, html_filename)
            fig_plotly.write_html(html_filename)
            print(f"3D interactive graph saved as {html_filename}")

    def simulate_distribution(self, num_instances: int = 100000, t: float = 1, timestep: float = 0.01, save: bool = False, plot: bool = True) -> Any:
        """
        Simulate the probability distribution corresponding to the process.

        :param num_instances: The number of instances to simulate
        :type num_instances: int
        :param t: The time to simulate
        :type t: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :param save: Whether to save the results to a file
        :type save: bool
        :param plot: Whether to plot the distribution
        :type plot: bool
        :return: The distribution of the process represented as a histogram
        :rtype: Tuple
        """
        data = self.simulate(t=t, timestep=timestep, num_instances=num_instances)

        times, data = self.separate(data)

        # Take the last value of each instance
        data = data[:, -1]

        # Calculate the distribution
        distribution = np.histogram(data, bins=100, density=True)

        # plot the distribution
        if plot:
            plt.figure(figsize=(10, 6))
            plt.hist(data, bins=100, density=True, alpha=0.6, color='g')
            plt.title(f'Distribution of {self.name}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.grid(True)
            if save:
                plt.savefig(f"distribution_simulation_{self.get_params()}, t:{t}, timestep:{timestep}, num_instances:{num_instances}.png")
            plt.show()
            plt.close()

        self.save_to_file(distribution,
                          f"distribution_simulation_{self.get_params()}, t:{t}, timestep:{timestep}, num_instances:{num_instances}.csv",
                          save)

        return distribution

    def relative_variance_pea(self, num_instances: int = num_instances_default, t: float = t_default, timestep: float = timestep_default, n: int = 1000000, save: bool = False) -> float:
        """
        Calculate the relative variance of the process using the PEA method.
        It means that we simulate the process for a given time and calculate the relative variance of the process.

        :param num_instances: The number of instances of the process to use in the pea estimation
        :type num_instances: int
        :param t: The time to simulate
        :type t: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :param n: the number of instances to use in the estimation of the ensemble average as approximation of ensemble average (should be rather large)
        :type n: int
        :param save: Whether to save the results to a file
        :type save: bool
        :return: The relative variance of the process
        """
        data = self.simulate(t=t, timestep=timestep, num_instances=n)

        times, data = self.separate(data)

        # Take the last value of each instance
        data = data[:, -1]

        # Calculate the relative variance
        relative_variance = ((1/num_instances) * np.var(data)) / np.mean(data) ** 2

        self.save_to_file(relative_variance,
                          f"relative_variance_pea_simulation_{self.get_params()}, t:{t}, timestep:{timestep}, num_instances:{num_instances}.csv",
                          save)

        return relative_variance

    def self_averaging_time(self, num_instances: int = num_instances_default, t: float = t_default, timestep: float = timestep_default, n: int =1000, plot: bool = True) -> float:
        """
        Calculate the self-averaging time of the process.
        Self-averaging time is the time after which the process ensemble starts to behave as a time average and stops behave as an ensemble average (so stops self-averaging).
        This method will work properly only if you select t that is long enough.
        We do not know any formal method to select t that is long enough for arbitrary process, so you have to check it by yourself.
        The method works only for multiplicative processes.

        :param num_instances: The number of instances of the process to use in the pea estimation
        :type num_instances: int
        :param t: The time to simulate
        :type t: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :param n: the number of instances to use in the estimation of the ensemble average as approximation of ensemble average (should be rather large)
        :type n: int
        :param plot: Whether to plot the increments
        :type plot: bool
        :return: The empirical estimation of self-averaging time of the process
        :rtype: float
        """
        if self._multiplicative is False:
            raise ValueError("The self-averaging time can now only be calculated for multiplicative processes.")

        else:
            times = np.arange(0, t, timestep)
            ensemble, total_average = self.simulate_ensembles(t=t, timestep=timestep, num_instances=num_instances,
                                                              num_ensembles=n)

            print(f"Shape of total_average: {total_average.shape}")
            print(f"Shape of times: {times.shape}")

            data = np.log(total_average)
            index = int(1/timestep)
            ensemble_average = data[index]

            print(f'Ensemble average: {ensemble_average}')

            time_average_rate = (data[-1] - data[0]) / t
            print(f"Time average rate: {time_average_rate}")

            # Ensure data and times have the same length
            min_length = min(len(data), len(times))
            data = data[:min_length]
            times = times[:min_length]

            time_avg_diff = np.cumsum(np.abs(data - time_average_rate * times))
            ensemble_avg_diff = np.cumsum(np.abs(data - ensemble_average * times))

            self_averaging_time = 0

            for i in range(len(data)):
                if time_avg_diff[i] < ensemble_avg_diff[i]:
                    self_averaging_time = times[i]
                    break

            print(f"Self-averaging time: {self_averaging_time}")

            if plot:
                plt.figure(figsize=(12, 8))
                plt.plot(times, data, label='Process')
                plt.plot(times, time_average_rate * times, label='Time Average Projection', linestyle='--')
                plt.plot(times, ensemble_average * times, label='Ensemble Average Projection', linestyle='--')
                nearest_index = np.argmin(np.abs(times - self_averaging_time))
                plt.plot(self_averaging_time, data[nearest_index], 'ro', markersize=10, label='Self-averaging point')

                plt.title(f'Self-averaging Analysis of {self.name}')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                plt.show()

            return self_averaging_time

    def empirical_properties(self):
        """
        Calculate the empirical statistical properties of the process and compare them with the theoretical properties (parameters).
        Check if they are equal.

        :return: None
        :rtype: None
        """
        pass

    def differential(self) -> str:
        """
        Calculate the differential of the process.

        :return: The differential of the process
        :rtype: str
        """
        pass

    def closed_formula(self) -> str:
        """
        Calculate the closed formula for the process using symbolic calculations and Ito calculus when possible.

        :return: The closed formula for the process
        :rtype: str
        """
        print('The closed formula for this process is not yet implemented')
        pass

    def pdf_evolution(self, t=t_default, timestep=timestep_default, num_instances=num_instances_default) -> Any:
        """
        Calculate the evolution of the probability density function of the process.

        :param t: The time to simulate
        :type t: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :param num_instances: The number of instances to simulate
        :type num_instances: int
        :return: The evolution of the probability density function of the process
        :rtype: Numpy array
        """
        pass

    def p_measure(self) -> Any:
        """
        Return the probability density function corresponding to the objective probability measure used in the given process.

        :return: The probability density function corresponding to the objective probability measure
        :rtype: Any
        """
        pass

    def time_average_expression(self) -> str:
        """
        Return the symbolic expression for the time average of the process if possible.

        :return: The symbolic expression for the time average of the process
        :rtype: str
        """
        pass

    def eternal_simulator(self, timestep: float = 0.01, k: int = 100000) -> None:
        """
        Save the simulated images for the process indefinitely every k steps.

        :param timestep: Time step for the simulation.
        :type timestep: float
        :param num_instances: Number of instances to simulate.
        :type num_instances: int
        :param k: Number of steps after which to update the plot.
        :type k: int
        :return: None
        :rtype: None
        :exception ValueError: If the eternal simulator is not available for the process
        """
        if self._process_class is None or self._external_simulator is False:

            num_instances = 1
            output_dir = os.path.join(self._output_dir, f"{self._name}_simulation")

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            csv_file_path = os.path.join(output_dir,
                                         "simulation_data.csv")  # Add this line to set the path for CSV file
            with open(csv_file_path, mode='w',
                      newline='') as csv_file:  # Add this line to open the CSV file for writing
                csv_writer = csv.writer(csv_file)  # Add this line to create a CSV writer object

                fig, ax = plt.subplots(figsize=(12, 7))
                lines = [ax.plot([], [], lw=0.5)[0] for _ in range(num_instances)]
                ax.set_xlim(0, 1)
                ax.set_ylim(0.9, 1.1)
                ax.set_title(f'Eternal Simulation of {self.name}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.grid(True)

                times = []
                data = [[] for _ in range(num_instances)]

                t = 0
                img_counter = 0

                def update_plot():
                    ax.set_xlim(times[0], max(times[-1], 1))
                    all_data = [item for sublist in data for item in sublist]
                    if all_data:
                        ymin, ymax = min(all_data), max(all_data)
                        yrange = max(ymax - ymin, 0.1)
                        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
                    for i, line in enumerate(lines):
                        line.set_data(times, data[i])
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    fig.savefig(os.path.join(output_dir, "simulation.png"))

                def simulate():
                    nonlocal t, img_counter
                    X = np.ones(num_instances)
                    try:
                        while True:
                            for _ in range(k):
                                t += timestep
                                times.append(t)
                                dX = self.custom_increment(X, timestep)
                                X += dX
                                for i in range(num_instances):
                                    data[i].append(X[i])
                            img_counter += 1
                            update_plot()
                            row = [t] + [data[i][-1] for i in range(num_instances)]
                            csv_writer.writerow(row)
                            csv_file.flush()
                    except Exception as e:
                        print(f"Error in simulation thread: {str(e)}")

                simulation_thread = threading.Thread(target=simulate)
                simulation_thread.start()

                try:
                    while True:
                        plt.pause(0.1)
                except KeyboardInterrupt:
                    print("Simulation stopped by user.")
                finally:
                    plt.ioff()
                    simulation_thread.join(timeout=5)
                    if simulation_thread.is_alive():
                        print("Warning: Simulation thread did not exit cleanly.")
                    plt.close(fig)

        else:
            raise ValueError(
                "The eternal simulator is not available for this process because an external simulator is used for it. To use this method, an internal simulator must be implemented.")

from ergodicity.tools.solve import solve, time_average, ergodicity_transform
from ergodicity.tools.compute import solve_fokker_planck_numerically

class ItoProcess(Process, ABC):
    """
    Abstract class representing an Ito process.
    Ito calculus can be applied to Ito processes.
    All Ito processes have drift and stochastic term.

    Attributes:

        drift_term: The drift term of the process

        stochastic_term: The stochastic term of the process
    """
    def __init__(self, name: str, process_class: Type[Any], drift_term: float, stochastic_term: float):
        """
        Initialize the Ito process with the given name, process class, drift term, and stochastic term.

        :param name: The name of the process
        :type name: str
        :param process_class: The class of the process
        :type process_class: Type[Any]
        :param drift_term: The drift term of the process
        :type drift_term: float
        :param stochastic_term: The stochastic term of the process
        :type stochastic_term: float
        :raises ValueError: If the stochastic term is negative
        """
        super().__init__(name, multiplicative=False, independent=True, ito=True, process_class=process_class)
        self._drift_term = drift_term
        if stochastic_term >= 0:
            self._stochastic_term = stochastic_term
        else:
            raise ValueError("The stochastic term must be greater than or equal to zero.")
        self._drift_term_sympy = None
        self._stochastic_term_sympy = None

    @property
    def drift_term(self) -> float:
        """
        Get the drift term of the process.

        :return: The drift term of the process
        :rtype: float
        """
        return self._drift_term

    @drift_term.setter
    def drift_term(self, drift_term: float):
        """
        Set the drift term of the process.

        :param drift_term: The drift term of the process
        :type drift_term: float
        """
        self._drift_term = drift_term

    @property
    def stochastic_term(self) -> float:
        """
        Get the stochastic term of the process.

        :return: The stochastic term of the process
        :rtype: float
        """
        return self._stochastic_term

    @stochastic_term.setter
    def stochastic_term(self, stochastic_term: float):
        """
        Set the stochastic term of the process.

        :param stochastic_term: The stochastic term of the process
        :type stochastic_term: float
        """
        if stochastic_term >= 0:
            self._stochastic_term = stochastic_term
        else:
            raise ValueError("The stochastic term must be greater than or equal to zero.")

    def closed_formula(self):
        """
        Find the analytical solution for the given Ito process using Ito calculus.

        :return: The analytical solution for the Ito process
        :rtype: Sympy expression
        """
        x = sp.symbols('x')
        t = sp.symbols('t')
        solution = solve(mu=self._drift_term_sympy, sigma=self._stochastic_term_sympy, x=x, t=t)
        print(f'The closed formula for the process is: {solution}')
        return solution

    def ergodicity_transform(self):
        """
        Find the ergodicity transformation for the given Ito process using Ito calculus.

        :return: The ergodicity transformation for the Ito process
        :rtype: Sympy expression
        """
        transform = ergodicity_transform(self._drift_term_sympy, self._stochastic_term_sympy, x=sp.symbols('x'), t=sp.symbols('t'))
        return transform

    def differential(self) -> str:
        """
        Calculate the differential of the process.

        :return: The differential of the process
        :rtype: str
        """
        # print(f'dX = {self._drift_term_sympy} * dt + {self._stochastic_term_sympy} * dW')
        dt = sp.symbols('dt')
        dW = sp.symbols('dW(t)')
        dx = self._drift_term_sympy * dt + self._stochastic_term_sympy * dW
        return dx

    def time_average_expression(self) -> str:
        """
        Return the symbolic expression for the time average of the process if possible using Ito calculus.

        :return: The symbolic expression for the time average of the process
        :rtype: Sympy expression
        """
        x = sp.symbols('x')
        t = sp.symbols('t')

        solution = time_average(mu=self._drift_term_sympy, sigma=self._stochastic_term_sympy, x=x, t=t)

        return solution

    def expected_value_expression(self, initial_condition=1) -> str:
        """
        Return the symbolic expression for the expected value of the process if possible using conventional calculus.

        :return: The symbolic expression for the expected value of the process
        :rtype: Sympy expression
        """
        t = sp.symbols('t')
        x = sp.Function('x')(t)
        mu = self._drift_term_sympy

        # replace x symbol with x function in mu:
        mu = mu.subs(sp.Symbol('x'), x)

        # Define the differential equation
        diffeq = sp.Eq(sp.diff(x, t), mu)

        # Solve the differential equation
        solution = sp.dsolve(diffeq, x)

        # Substitute the initial condition C1
        solution = solution.subs(sp.Symbol('C1'), initial_condition)

        print("Differential equation:")
        print(diffeq)
        print("\nSolution:")
        print(solution)

        return solution

class NonItoProcess(Process, ABC):
    """
    Abstract class representing a non-Ito process.
    """
    def __init__(self, name: str, process_class: Type[Any]):
        """
        Initialize the non-Ito process with the given name and process class.

        :param name: The name of the process
        :type name: str
        :param process_class: The class of the process
        :type process_class: Type[Any]
        """
        super().__init__(name, multiplicative=False, independent = True, ito=False, process_class=process_class)

class CustomProcess(Process, ABC):
    """
    Abstract class representing a custom process.
    This class can be used by user to create custom processes in an easy way.
    """
    def __init__(self, name: str):
        """
        Initialize the custom process with the given name.

        :param name:
        :type name: str
        """
        super().__init__(name, multiplicative=False, independent=True, ito=False, process_class=None)
        self._custom = True


def simulation_decorator(simulate_func: Callable) -> Callable:
    """
    Decorator for simulation methods to add verbose option.

    :param simulate_func: The simulation method to decorate
    :type simulate_func: Callable
    :return: The decorated simulation method
    :rtype: Callable
    """
    def wrapper(self, t: float, timestep: float, num_instances: int) -> Any:
        """
        Wrapper function for the simulation method.

        :param t: The time to simulate
        :type t: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :param num_instances: The number of instances to simulate
        :type num_instances: int
        :return: The simulation data
        :rtype: NumPy array
        """
        num_steps, times, data = self.data_for_simulation(t, timestep, num_instances)
        for i in range(num_instances):
            X=1
            if verbose == True and num_instances % 100 == 0:
                print(f"Simulating instance {i}...")
            for step in range(num_steps):
                data[i, step] = X
                dX = simulate_func(self, X, timestep)
                X = X + dX
                if verbose == True and num_steps>1000000 and step % 1000000 == 0 and step != 0:
                    print(f"Simulating instance {i}, step {step}, X = {X}")

        data = np.concatenate((times.reshape(1, -1), data), axis=0)

        return data

    return wrapper

def check_simulate_with_differential(self):
    """
    Check if the simulate method uses the differential method.

    :param self: The process object
    :type self: Process
    :return: True if the simulate method uses the differential method, False otherwise
    :rtype: bool
    """
    return self._simulate_with_differential


def create_correlation_matrix(size, correlation):
    """
    Create a correlation matrix with given correlations between all elements.

    :param size: The size of the correlation matrix
    :type size: int
    :param correlation: The correlation value for all off-diagonal elements
    :type correlation: float
    :return: The correlation matrix
    :rtype: np.ndarray
    """
    if correlation >= 1 or correlation <= -1:
        raise ValueError("Correlation must be between -1 and 1, exclusive.")

    # Initialize the correlation matrix with the correlation value
    corr_matrix = np.full((size, size), correlation)

    # Set the diagonal elements to 1 (self-correlation)
    np.fill_diagonal(corr_matrix, 1)

    return corr_matrix

def correlation_to_covariance(correlation_matrix, std_devs):
    """
    Transform a correlation matrix into a variance-covariance matrix.

    :param correlation_matrix: The correlation matrix
    :type correlation_matrix: np.ndarray
    :param std_devs: The standard deviations of the variables
    :type std_devs: np.ndarray
    :return: The variance-covariance matrix
    :rtype: np.ndarray
    """
    std_devs = np.array(std_devs)
    covariance_matrix = correlation_matrix * np.outer(std_devs, std_devs)
    return covariance_matrix


