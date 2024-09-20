"""
multiprocessing Submodule

The `multiprocessing` submodule provides tools and functions to execute tasks in parallel, optimizing the performance of stochastic process simulations and other computationally intensive tasks. The submodule leverages Python's `concurrent.futures` and `multiprocessing` libraries to distribute tasks across multiple CPU cores.

Key Features:

1. **Parallel Task Execution**:

   - `ParallelExecutionManager`: Manages the execution of tasks in parallel using `ProcessPoolExecutor`. It supports both standalone functions and methods within objects.

   - `run_task_parallel`: Executes individual tasks either as standalone functions or as methods of objects.

2. **Task Wrappers**:

   - `task_wrapper`: A decorator that facilitates parallel execution and saves results to disk. It can be applied to functions that create tasks for parallel execution.

   - `initial_tasks_wrapper` and `secondary_tasks_wrapper`: Decorators for wrapping tasks related to simulations and visualizations. These wrappers handle the creation and submission of tasks for parallel execution.

   - `growth_rate_wrapper`: A special-purpose decorator for wrapping functions related to growth rate calculations.

3. **Parallel Processing Pipelines**:

   - `multi_simulations`: Executes multiple simulations in parallel, each with a different parameter configuration.

   - `multi_visualize_moments`: Executes parallel tasks for visualizing moments of the simulation results.

   - `multi_growth_rate_time_average` and `multi_growth_rate_ensemble_average`: Executes growth rate calculations for time average and ensemble average in parallel.

4. **Parallel Execution of Arbitrary Functions**:

   - `parallel_processor`: A decorator that wraps any function, allowing it to process multiple datasets in parallel. It submits each dataset as a separate task to the `ProcessPoolExecutor`.

   - `parallel`: A higher-level function that runs any library function in parallel with multiple sets of arguments.

5. **Task Result Management**:

   - The submodule handles task results by saving them to files, ensuring that they can be reloaded or inspected later.

   - The results are organized and filtered to remove invalid or failed tasks before further analysis or visualization.

6. **Example Pipelines**:

   - `general_pipeline`: Runs a full pipeline that includes simulations, visualizations, and growth rate calculations.

   - `ensemble_vs_time_average_pipeline`: Compares the ensemble average and time average growth rates in parallel.

7. **Parallel Function Wrapping**:

   - `Parallel.wrapp`: A class method that converts any function into a parallel processing function, automatically distributing its workload across available CPU cores.

Example Usage:

if __name__ == "__main__":

    import ergodicity.process.multiplicative as ep

    # Define parameter ranges for a Geometric Lévy process

    param_ranges = {
        'alpha': [1.5, 1.6, 1.7],
        'loc': [0.004, 0.005, 0.006],
        'scale': [0.002, 0.003, 0.004],
        'beta': [0]
    }

    # Run a general pipeline that includes simulations, visualizations, and growth rate calculations

    results = general_pipeline(ep.GeometricLevyProcess, param_ranges)

    # Parallel processing example using the 'average' function

    data_arrays = [np.random.rand(10, 100) for _ in range(5)]

    results1 = parallel(average, data_arrays, visualize=False, name="multi_average")

    print(results1)
"""
import concurrent.futures
import inspect
import json
from ergodicity.process.default_values import *
from itertools import product
import numpy as np
from ergodicity.tools.helper import save_to_file
from ergodicity.configurations import *
from functools import wraps
from ergodicity.tools.compute import *
from typing import Callable, Any, List
from warnings import warn
import functools
from multiprocessing import freeze_support
from concurrent.futures import ProcessPoolExecutor

def get_object_name(obj):
    """
    Get the name of an object from the local variables in the calling frame.

    :param obj: The object to get the name of
    :type obj: Any
    :return: The name of the object
    :rtype: str
    """
    frame = inspect.currentframe().f_back
    names = {id(v): k for k, v in frame.f_locals.items()}
    return names.get(id(obj), 'unknown')

class ParallelExecutionManager:
    """
    A class to manage the parallel execution of tasks using ProcessPoolExecutor.

    :param object: The object to get the name of
    :type object: Any
    :return: The name of the object
    :rtype: str
    """
    def __init__(self):
        pass

    def run_task_parallel(self, task):
        """
        Run a task in parallel using ProcessPoolExecutor.

        :param task: The task to run
        :type task: dict
        :return: The result of the task
        :rtype: Any
        """
        # Determine if the task is a method of an object or a standalone function
        if 'object' in task:
            # If it's a method, get the method from the object
            method = getattr(task['object'], task['name'])
            # Call the method with kwargs
            return method(**task.get('arguments', {}))
        else:
            # If it's a standalone function, call the function
            function = task['function']
            # Call the function with kwargs
            return function(**task.get('arguments', {}))

    def execute(self, tasks):
        """
        Execute multiple tasks in parallel using ProcessPoolExecutor.

        :param tasks: A list of tasks to execute
        :type tasks: List[dict]
        :return: A dictionary of results for each task
        :rtype: dict
        :exception Exception: If an error occurs during task execution
        """
        results = {}
        # Use ProcessPoolExecutor to manage parallel execution
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit each task to the executor
            future_to_task = {}
            for task in tasks:
                if 'object' in task:
                    object_name = get_object_name(task['object'])
                    task_key = f"{object_name}.{task['name']}({task['arguments']})"
                else:
                    task_key = f"{task['name']}({task['arguments']})"
                future_to_task[executor.submit(self.run_task_parallel, task)] = task_key

            for future in concurrent.futures.as_completed(future_to_task):
                task_key = future_to_task[future]
                try:
                    results[task_key] = future.result()
                except Exception as exc:
                    results[task_key] = f'Generated an exception: {exc}'
        return results

def create_task_key(params):
    """"
    Create a unique key for a task based on its parameters.

    :param params: The parameters of the task
    :type params: dict
    :return: A string key representing the task
    :rtype: str
    """
    sorted_params = {k: str(params[k]) for k in sorted(params)}
    return f"simulate({json.dumps(sorted_params)})"


def task_wrapper(save: bool = pipeline_parameters['save'],
                 output_dir=pipeline_parameters['output_dir'],
                 print_debug=pipeline_parameters['print_debug']):
    """
    A decorator for functions that create tasks for parallel execution.

    :param save: Whether to save the results to disk
    :type save: bool
    :param output_dir: The directory to save the results
    :type output_dir: str
    :param print_debug: Whether to print debug information
    :type print_debug: bool
    :return: The decorator function
    :rtype: Callable
    """
    def decorator(func):
        """
        Decorator function that wraps the task creation function.

        :param func: The function to wrap
        :type func: Callable
        :return: The wrapped function
        :rtype: Callable
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper function that executes the original function and manages the parallel execution of tasks.

            :param args: The arguments to the original function
            :param kwargs: The keyword arguments to the original function
            :return: The results of the parallel execution
            :rtype: Any
            """
            tasks, param_to_task = func(*args, **kwargs)

            manager = ParallelExecutionManager()
            results = manager.execute(tasks)

            if print_debug:
                print(f"\nTask results:")
                print(f"Number of tasks: {len(tasks)}")
                print(f"Number of results: {len(results)}")
                for task in tasks:
                    print(f"Task: {task['name']}, Object: {task['object']}")

            # Use the original function name, falling back to func.__name__ if __wrapped__ is not available
            original_func_name = getattr(wrapper, '__wrapped__', func).__name__

            # Include the function name in the file names
            results_file_name = f'{original_func_name}_results_data.npy'
            params_file_name = f'params_to_task_data.npy'

            save_to_file(data=results, output_dir=output_dir, file_name=results_file_name, save=save)
            save_to_file(data=param_to_task, output_dir=output_dir, file_name=params_file_name, save=save)

            return results, param_to_task
        return wrapper
    return decorator

def initial_tasks_wrapper(task_name, specific_args):
    """
    A decorator for functions that create initial tasks for parallel execution.
    Initial tasks are the tasks that generate the data to be used in subsequent tasks.
    They do not require any input data.

    :param task_name: The name of the task
    :type task_name: str
    :param specific_args: Specific arguments for the task
    :type specific_args: dict
    :return: The decorator function
    :rtype: Callable
    """
    def decorator(func):
        """
        Decorator function that wraps the task creation function.

        :param func: The function to wrap
        :type func: Callable
        :return: The wrapped function
        :rtype: Callable
        """
        @wraps(func)
        @task_wrapper()
        def wrapper(process_class, param_ranges, **kwargs):
            """
            Wrapper function that executes the original function and manages the parallel execution of tasks.

            :param process_class: The class of the stochastic process
            :type process_class: type
            :param param_ranges: The ranges of parameters for the process
            :type param_ranges: dict
            :param kwargs: Additional keyword arguments
            :return: The results of the parallel execution
            :rtype: Any
            """
            tasks = []
            param_to_task = {}
            param_combinations = list(product(*param_ranges.values()))

            for param_comb in param_combinations:
                params = {key: value for key, value in zip(param_ranges.keys(), param_comb)}
                process_obj = process_class(**params)
                task_key = create_task_key(params)

                # Use provided kwargs or default values from specific_args
                task_args = {
                    param: kwargs.get(param, default)
                    for param, default in specific_args.items()
                }
                task_args['save'] = True

                task = {
                    'name': task_name,
                    'object': process_obj,
                    'arguments': task_args
                }
                tasks.append(task)
                param_to_task[task_key] = {'object': process_obj}

            return tasks, param_to_task
        return wrapper
    return decorator

def secondary_tasks_wrapper(task_name, specific_args):
    """
    A decorator for functions that create secondary tasks for parallel execution.
    Secondary tasks are the tasks that process the data generated by the initial tasks.
    They require input data from the initial tasks.

    :param task_name: The name of the task
    :type task_name: str
    :param specific_args: Specific arguments for the task
    :type specific_args: dict
    :return: The decorator function
    :rtype: Callable
    """
    def decorator(func):
        """
        Decorator function that wraps the task creation function.

        :param func: The function to wrap
        :type func: Callable
        :return: The wrapped function
        :rtype: Callable
        """
        @wraps(func)
        @task_wrapper()
        def wrapper(results, param_to_task, **kwargs):
            """
            Wrapper function that executes the original function and manages the parallel execution of tasks.

            :param results: The results of the initial tasks
            :type results: Any
            :param param_to_task: The mapping of parameters to tasks
            :type param_to_task: dict
            :param kwargs: Additional keyword arguments
            :return: The results of the parallel execution
            :rtype: Any
            """
            tasks = []

            # Extract the single result if there's only one
            if len(results) == 1:
                single_result_key = list(results.keys())[0]
                single_result = results[single_result_key]

                # Create tasks for each parameter combination
                for task_key, task in param_to_task.items():
                    process_obj = task['object']

                    # Prepare arguments
                    task_args = {
                        arg: kwargs.get(arg, default)
                        for arg, default in specific_args.items()
                    }
                    task_args['data'] = single_result
                    task_args['save'] = True

                    tasks.append({
                        'name': task_name,
                        'object': process_obj,
                        'arguments': task_args
                    })
            else:
                print(f"Error: Expected a single result, but got {len(results)}.")

            return tasks, param_to_task

        return wrapper

    return decorator

def parallel_processor(func: Callable):
    """
    A decorator that wraps any function, allowing it to process multiple datasets in parallel.

    :param func: The function to wrap
    :type func: Callable
    :return: The wrapped function
    :rtype: Callable
    """
    def multi_data_processor(data_list: List[np.ndarray], *args, **kwargs):
        """
        Wrapper function that executes the original function and manages the parallel execution of tasks.

        :param data_list: The list of data arrays to process
        :type data_list: List[np.ndarray]
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        :return: The results of the parallel execution
        :rtype: Any
        """
        tasks = []

        for i, data in enumerate(data_list):
            task = {
                'name': f'{func.__name__}_{i}',
                'object': func,
                'arguments': {'data': data, **kwargs}
            }
            tasks.append(task)
            print(f"Created task: {task['name']}")

        print(f"Total tasks created: {len(tasks)}")

        manager = ParallelExecutionManager()
        print("Executing tasks...")
        results = manager.execute(tasks)
        print(f"Execution completed. Results: {results}")

        ordered_results = []
        for i in range(len(data_list)):
            result_key = next((key for key in results.keys() if f'{func.__name__}_{i}' in key), None)
            if result_key:
                ordered_results.append(results[result_key])
                print(f"Result found for task {func.__name__}_{i}: {results[result_key]}")
            else:
                print(f"Warning: No result found for task {func.__name__}_{i}")
                ordered_results.append(None)

        warn("This pipeline will work correctly if the total number of parameter combinations"
             "in the pipeline is less than 10000. To increase this limit, update the range in the for loop in the growth_rate_wrapper.")

        return ordered_results

    return multi_data_processor

@initial_tasks_wrapper('simulate', {'t': t_default, 'timestep': timestep_default, 'num_instances': num_instances_default, 'plot': True})
def multi_simulations(process_class, param_ranges):
    """
    Run multiple simulations of stochastic processes in parallel for different parameter combinations.

    :param process_class: The stochastic process class
    :type process_class: type
    :param param_ranges: Dictionary of process parameters ranges
    :type param_ranges: dict
    :return: A tuple of tasks and a mapping of parameters to tasks
    :rtype: tuple
    """
    pass

@secondary_tasks_wrapper('visualize_moments', {'mask': mask_default, 'save': True})
def multi_visualize_moments(simulation_results, param_to_task):
    """
    Run multiple tasks in parallel to visualize moments of the simulation results.

    :param simulation_results: The results of the simulations
    :type simulation_results: Any
    :param param_to_task: The mapping of parameters to tasks
    :return: A tuple of tasks and a mapping of parameters to tasks
    :rtype: tuple
    """
    pass

@initial_tasks_wrapper('growth_rate_time_average', {
    't': 1000000,
    'timestep': 0.0001
})
def multi_growth_rate_time_average(process_class, param_ranges):
    """
    Run multiple growth rate calculations for time average in parallel.

    :param process_class: The stochastic process class
    :type process_class: type
    :param param_ranges: Dictionary of process parameters ranges
    :type param_ranges: dict
    :return: A tuple of tasks and a mapping of parameters to tasks
    :rtype: tuple
    """
    pass


@initial_tasks_wrapper('growth_rate_ensemble_average', {
    'num_instances': 1000000
})
def multi_growth_rate_ensemble_average(process_class, param_ranges):
    """
    Run multiple growth rate calculations for ensemble average in parallel.

    :param process_class: The stochastic process class
    :type process_class: type
    :param param_ranges: Dictionary of process parameters ranges
    :type param_ranges: dict
    :return: A tuple of tasks and a mapping of parameters to tasks
    :rtype: tuple
    """
    pass

@secondary_tasks_wrapper('visualize_moment', {'moment': 'mean', 'mask': mask_default, 'save': True})
def multi_visualize_moment(simulation_results, param_to_task):
    """
    Run multiple tasks in parallel to visualize a specific moment of the simulation results.

    :param simulation_results: The results of the simulations
    :type simulation_results: Any
    :param param_to_task: The mapping of parameters to tasks
    :type param_to_task: dict
    :return: A tuple of tasks and a mapping of parameters to tasks
    :rtype: tuple
    """
    pass


def growth_rate_wrapper(func):
    """
    A decorator that wraps functions related to growth rate calculations.

    :param func: The function to wrap
    :type func: Callable
    :return: The wrapped function
    :rtype: Callable
    """
    @wraps(func)
    def wrapper(**kwargs):
        """
        Wrapper function that executes the original function and calculates the growth rate.

        :param kwargs: Additional keyword arguments
        :return: The growth rate
        :rtype: float
        """
        data = kwargs.get('data')
        if data is not None:
            return func(data)
        else:
            raise ValueError("No data provided to the wrapped function")

    # Add attributes expected by the parallel processor
    for i in range(10000):  # Adjust this range based on your maximum expected number of tasks
        setattr(wrapper, f'{func.__name__}_{i}', wrapper)

    return wrapper

@growth_rate_wrapper
def simulate_and_calculate_growth_rate(data: np.ndarray):
    """
    Simulate a stochastic process and calculate the growth rate of the average.

    :param data: The data array for the simulation
    :type data: np.ndarray
    :return: The growth rate of the average
    :rtype: float
    :exception Exception: If an error occurs during the simulation or calculation
    """
    process_class, params, t, timestep, num_instances = data
    try:
        process = process_class(**params)
        simulated_data = process.simulate(t=t, timestep=timestep, num_instances=num_instances)
        result = growth_rate_of_average_per_time(simulated_data)
        print(f"Simulation result for t={t}, num_instances={num_instances}: {result}")
        return result
    except Exception as e:
        print(f"Error in simulation for t={t}, num_instances={num_instances}: {e}")
        return None

# Update the multi_growth_rate_processor to use the wrapped function
def multi_growth_rate_processor(process_class, params, t_range, num_instances_range,
                                timestep=pipeline_parameters['timestep']):
    """
    Run the growth rate calculation for multiple parameter combinations in parallel.

    :param process_class: The stochastic process class
    :type process_class: type
    :param params: Dictionary of process parameters
    :type params: dict
    :param t_range: List of time values
    :type t_range: list
    :param num_instances_range: List of number of instances
    :type num_instances_range: list
    :param timestep: Time step for the simulation
    :type timestep: float
    :return: The growth rates for each parameter combination
    :rtype: dict
    :exception Exception: If an error occurs during the simulation or calculation
    """

    data_list = [np.array([process_class, params, t, timestep, n], dtype=object)
                 for t, n in product(t_range, num_instances_range)]

    multi_processor = parallel_processor(simulate_and_calculate_growth_rate)
    results = multi_processor(data_list)

    print("Raw results from parallel_processor:")
    for i, result in enumerate(results):
        print(f"Task {i}: {result}")

    organized_results = {}
    for (t, n), result in zip(product(t_range, num_instances_range), results):
        print(f"Processing result for t={t}, num_instances={n}: {result}")
        if result is not None and not isinstance(result, str):
            try:
                float_result = float(result)
                organized_results[(t, n)] = float_result
                print(f"Valid result added: t={t}, num_instances={n}, result={float_result}")
            except ValueError:
                print(f"Invalid result skipped: t={t}, num_instances={n}, result={result}")
        else:
            print(f"Invalid result skipped: t={t}, num_instances={n}, result={result}")

    if organized_results:
        print(f"Number of valid results: {len(organized_results)}")
        visualize_growth_rates(organized_results, process_class, params)
    else:
        print("No valid results to visualize.")

    return organized_results

def visualize_growth_rates(results, process_class, params):
    """
    Visualize the growth rates for multiple parameter combinations.

    :param results: The growth rates for each parameter combination
    :type results: dict
    :param process_class: The stochastic process class
    :type process_class: type
    :param params: Dictionary of process parameters
    :type params: dict
    :return: None
    :rtype: None
    """
    t_values, num_instances_values, growth_rates = zip(*[(t, n, rate)
                                                         for (t, n), rate in results.items()])

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(t_values, num_instances_values, growth_rates, c=growth_rates, cmap='viridis')

    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Number of Instances')
    ax.set_zlabel('Growth Rate')
    ax.set_title(f'Growth Rate vs Time and Number of Instances for {process_class.__name__} with parameters {params}')

    fig.colorbar(scatter, label='Growth Rate')

    plt.show()

def general_pipeline(process_class, param_ranges, t=pipeline_parameters['t'], timestep=pipeline_parameters['timestep'], num_instances=pipeline_parameters['num_instances'], time_average_time=pipeline_parameters['time_average_time']):
    """
    Run an example of general research pipeline that includes simulations, visualizations, and growth rate calculations.

    :param process_class: The stochastic process class
    :type process_class: type
    :param param_ranges: Dictionary of process parameters ranges
    :type param_ranges: dict
    :param t: Total time for the simulation
    :type t: float
    :param timestep: Time step for the simulation
    :type timestep: float
    :param num_instances: Number of instances to simulate
    :type num_instances: int
    :param time_average_time: Time for time average calculation
    :type time_average_time: float
    :return: A dictionary of results
    :rtype: dict
    """
    simulation_results, param_to_task = multi_simulations(process_class, param_ranges, t=t, timestep=timestep, num_instances=num_instances)
    visualization_results = multi_visualize_moments(simulation_results, param_to_task)
    growth_rate_results = multi_growth_rate_time_average(process_class=process_class, param_ranges=param_ranges, time_average_time=time_average_time, timestep=timestep)

    return {
        'simulation_results': simulation_results,
        'visualization_results': visualization_results,
        'growth_rate_results': growth_rate_results
    }

def ensemble_vs_time_average_pipeline(process_class, param_ranges, timestep=pipeline_parameters['timestep'], num_instances=pipeline_parameters['num_instances'], time_average_time=pipeline_parameters['time_average_time']):
    """
       Compare the ensemble average and time average growth rates for multiple parameter combinations.

       :param process_class: The stochastic process class
       :type process_class: type
       :param param_ranges: Dictionary of process parameters ranges
       :type param_ranges: dict
       :param timestep: Time step for the simulation
       :type timestep: float
       :param num_instances: Number of instances to simulate
       :type num_instances: int
       :param time_average_time: Time for time average calculation
       :type time_average_time: float
       :return: A dictionary of results
       :rtype: dict
       """
    ensemble_results = multi_growth_rate_ensemble_average(process_class=process_class, param_ranges=param_ranges, num_instances=num_instances)
    time_average_results = multi_growth_rate_time_average(process_class=process_class, param_ranges=param_ranges, time_average_time=time_average_time, timestep=timestep)

    return {
        'ensemble_results': ensemble_results,
        'time_average_results': time_average_results
    }


def growth_rate_of_average_pipeline(process_class, process_params, t, timestep):
    """
    Compute the growth rate of the average of multiple instances of a stochastic process.
    When num_instances = 1 and time is large, this function returns a time average approximation.
    When num_instance -> infinity and time is relatively small, this function returns an ensemble average approximation.
    The idea is to look at what exactly happens when num_instances is in between 1 and infinity.

    :param process_class: The stochastic process class
    :type process_class: type
    :param process_params: Dictionary of process parameters
    :type process_params: dict
    :param t: Total time for the simulation
    :type t: float
    :param timestep: Time step for the simulation
    :type timestep: float
    :return: The growth rate of the average of multiple instances
    :rtype: float
    """
    num_instances_range = np.logspace(2, 5, 10, dtype=int)

    results = multi_growth_rate_processor(process_class, process_params, t, num_instances_range, timestep=timestep)

    return results

# Create a new parallel processor function to take functions as input and return outputs of these functions that are run in parallel

class Parallel:
    """
    A class that provides tools for parallel processing of functions.
    """
    @staticmethod
    def run_function(func, kwargs):
        """
        Helper function to run the target function with given arguments.

        :param func: The target function to run
        :type func: Callable
        :param kwargs: The keyword arguments to pass to the function
        :type kwargs: dict
        :return: The result of the function
        :rtype: Any
        """
        return func(**kwargs)

    @classmethod
    def wrapp(cls, func: Callable) -> Callable:
        """
        Wrapper function that transforms a given function into a parallel processing function.

        :param func: The original function to be parallelized
        :type func: Callable
        :return: A new function that runs parallel instances of the original function
        :rtype: Callable
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper function that executes the original function in parallel with multiple argument sets.

            :param args: The arguments to the original function
            :param kwargs: Additional keyword arguments
            :return: The results of the parallel execution
            :rtype: Any
            """
            if not args or not isinstance(args[0], list):
                raise ValueError("The first argument must be a list of argument sets")

            arg_sets = args[0]
            other_args = args[1:]

            tasks = []
            for arg_set in arg_sets:
                if isinstance(arg_set, dict):
                    task_kwargs = arg_set
                elif isinstance(arg_set, (list, tuple)):
                    task_kwargs = dict(zip(func.__code__.co_varnames, arg_set + other_args))
                else:
                    raise ValueError("Each argument set must be a dictionary, list, or tuple")

                task_kwargs.update(kwargs)
                tasks.append((func, task_kwargs))

            with ProcessPoolExecutor() as executor:
                results = list(executor.map(cls.run_function, [func for _ in tasks], [task[1] for task in tasks]))

            return results

        return wrapper


def parallel(func: Callable, data, **kwargs) -> int:
    """
    Run any library function in parallel with multiple sets of arguments.

    :param func: The function to be run in parallel
    :type func: Callable
    :param data: The list of arguments to be passed to the function
    :type data: list
    :param kwargs: Additional keyword arguments to be passed to the function
    :type kwargs: dict
    :return: The results of the function run in parallel
    :rtype: int
    """

    parallel_function = Parallel.wrapp(func)
    arguments = []
    for i in range(len(data)):
        b = data[i]
        if kwargs=={}:
            a = {'data': b}
            arguments.append(a)
        else:
            a = {'data': b} | kwargs
            arguments.append(a)

    # print(arguments)
    results = parallel_function(arguments)
    return results

def parallel_example():
    """
    Example function to demonstrate the parallel processing capabilities.

    :return: The results of the parallel execution
    :rtype: Any
    """
    data_arrays = [np.random.rand(4, 5) for _ in range(5)]  # Create 5 random data arrays
    results = parallel(average, data_arrays, name="multi_average")
    # print(results)
    return results

# Example usage:
if __name__ == "__main__":
    import ergodicity.process.multiplicative as ep

    alphas = [1.5, 1.6, 1.7]
    locs = [0.004, 0.005, 0.006]
    scales = [0.002, 0.003, 0.004]
    betas = [0]

    param_ranges = {
        'alpha': alphas,
        'loc': locs,
        'scale': scales,
        'beta': betas
    }

    results = general_pipeline(ep.GeometricLevyProcess, param_ranges)

    multi_average = parallel_processor(average)

    # Usage
    data_arrays = [np.random.rand(10, 100) for _ in range(5)]  # Create 5 random data arrays
    results1 = multi_average(data_arrays, visualize=False, name="multi_average")

    for key, value in results1.items():
        print(f"Result for {key}:")
        print(value)










