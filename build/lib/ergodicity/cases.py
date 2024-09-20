"""
cases Module Overview

The **`cases`** module showcases a selection of illustrative and practical examples that demonstrate how the library can be used in a variety of contexts. These examples highlight the key functionalities of the library and serve as a guide for users to explore different features in action. Whether you are a beginner seeking to understand the basic processes or an advanced user exploring complex models, this module provides valuable insights through real-world applications.

Key Purposes of this Module:

1. **Demonstrate Library Capabilities**:

   - The cases are designed to show the power and flexibility of the library by showcasing its application in different domains, such as stochastic processes, evolutionary neural networks, and utility function fitting.

2. **Educational and Illustrative**:

   - Each case walks through a specific feature or combination of features, offering a hands-on way to learn how the library can be applied to real-world problems.

3. **Complete, Ready-to-Run Examples**:

   - All cases are self-contained, meaning you can execute them as they are to explore various processes, agents, and tools provided by the library.

4. **Broad Spectrum of Examples**:

   - The examples cover simple simulations, advanced stochastic modeling, neural networks, agent-based modeling, utility function fitting, and more. These cases provide a broad perspective on how the library can be leveraged in different scenarios.

Structure of the Module:

1. **`IntroCase`**:

   - A basic introduction to using the library with the `GeometricBrownianMotion` process. The case involves simulating data, visualizing moments, and comparing averages.

2. **`UtilityFitting_case`**:

   - Demonstrates utility function fitting by showcasing how multiple utility functions can be fitted using agent choices and different stochastic processes.

3. **`EvolutionaryNN_case`**:

   - Explores the use of evolutionary neural networks with agents making decisions based on encoded processes. The case covers process generation, neural network mutation, cloning, and evolutionary training of agents.

4. **`StochasticHeatEquation_case`**:

   - Simulates a stochastic partial differential equation (PDE), specifically the stochastic heat equation, showcasing advanced simulation techniques and visualizations like 3D plotting and animations.

5. **`BasicUtilityAgent_case`**:

   - Illustrates the use of basic utility agents interacting with `GeometricBrownianMotion`, comparing symbolic and numerical expected utilities, and running evolutionary algorithms to optimize agent behavior.

6. **`TimeAverageDynamicsGBM_case`**:

   - Focuses on time-average dynamics in a Geometric Brownian Motion process and demonstrates the ergodicity transformation.

7. **`GeometricLevyProcess_case`**:

   - Simulates and visualizes the Geometric Levy Process, showcasing how ensemble and time averages can be computed and compared.

8. **`VariousSimulations_case`**:

   - A collection of simulations involving various stochastic processes such as the Bessel process, Brownian bridge, Cauchy process, and more, illustrating the library's capabilities across multiple process types.

9. **`MultivariateGeometricBrownianMotion_case`**:

   - Demonstrates how to simulate and visualize multivariate Geometric Brownian Motion with a specified correlation matrix.

10. **`GeometricBrownianMotion_case`**:

   - Uses parallel execution to simulate the Geometric Brownian Motion process efficiently across multiple settings.

11. **`ItoLemmaApplication`**:

   - Applies Ito's Lemma to a given stochastic differential equation (SDE), providing insight into how symbolic manipulation can be used for process analysis.

Use Cases:

- **Educational Use**: The module provides learning materials for users who are new to stochastic processes, agent-based modeling, or neural networks.

- **Advanced Experimentation**: For experienced users, the cases demonstrate advanced features such as process encoding, utility fitting, evolutionary strategies, and stochastic PDEs.

- **Library Exploration**: Users can explore different parts of the library by running these cases and understanding how each component functions in practice.

## Important Notes:

- **Illustrative Nature**: While these cases serve as valuable educational tools, they also illustrate how the library can be applied to complex scenarios in a meaningful way.

- **Self-Contained**: Each case is self-contained and can be run independently to experiment with specific functionalities of the library.

- **Comprehensive Coverage**: The examples in this module cover a wide range of topics, from basic stochastic processes to advanced neural networks, making it a versatile resource for learning.

"""

import ergodicity.process as ep
from ergodicity.process import multiplicative as em
import ergodicity.agents as ea
import ergodicity.integrations as ei
from ergodicity import developer_tools as dt
import numpy as np
from ergodicity.process.default_values import *
from ergodicity.configurations import *
from ergodicity.tools.compute import *
import sympy as sp
from ergodicity.tools.solve import *
from ergodicity.process.multiplicative import GeometricBrownianMotion
from ergodicity.process.basic import BrownianMotion
from ergodicity.agents.agents import *
from ergodicity.tools.partial_sde import PSDESimulator
from ergodicity.agents.evolutionary_nn import *
from ergodicity.agents.evaluation import *


def GBM_Properties_case():
    """
    This case demonstrates how to access and modify properties of the GeometricBrownianMotion process.

    :return: None
    """
    from ergodicity.process.multiplicative import GeometricBrownianMotion

    gbm = GeometricBrownianMotion(drift=0.05, volatility=0.2)

    print(f"Process name: {gbm.name}")
    print(f"Is multiplicative: {gbm.multiplicative}")
    print(f"Has independent increments: {gbm.independent}")
    print(f"Is an Ito process: {gbm.ito}")

    gbm.name = "My Custom GBM"
    gbm.output_dir = "custom_gbm_results"

    print(f"Updated process name: {gbm.name}")
    print(f"New output directory: {gbm.output_dir}")

    if gbm.custom:
        print("This is a custom process")
    else:
        print("This is a standard library process")

    if gbm.simulate_with_differential:
        print("This process uses differential methods for simulation")
    else:
        print("This process uses probability distributions for simulation")

    print(f"Process types: {gbm.types}")

    # Add a new type
    gbm.types.append("financial")

    # Check if the process is of a certain type
    if "geometric" in gbm.types:
        print("This is a geometric process")

    from ergodicity.process.basic import BrownianMotion

    processes = [
        GeometricBrownianMotion(drift=0.05, volatility=0.2),
        BrownianMotion(drift=0, scale=1)
    ]

    for process in processes:
        print(f"{process.name}:")
        print(f"  Multiplicative: {process.multiplicative}")
        print(f"  Independent increments: {process.independent}")
        print(f"  Types: {', '.join(process.types)}")
        print()

def IntroCase():
    """
    This is an introductory case that demonstrates the basic usage of the library with the GeometricBrownianMotion process.

    :return: simulated_data: The simulated data from the GeometricBrownianMotion process.
    :rtype: np.ndarray
    """
    from ergodicity.process.multiplicative import GeometricBrownianMotion as GBM
    gbm = GBM(drift=0.02, volatility=0.3)
    simulated_data = gbm.simulate(t=10, timestep=0.01, num_instances=10, save=True, plot=True)
    moments = gbm.visualize_moments(simulated_data)
    from ergodicity.tools.compute import compare_averages
    averages = compare_averages(simulated_data)

    return simulated_data, moments, averages

def UtilityFitting_case(model_path):
    """
    This case demonstrates the functionality for the empirical utility function fitting using the UtilityFunctionInference class.

    :param model_path:
    :type model_path: str
    :return: None
    """
    processes = [
        {'type': 'BrownianMotion'},
        {'type': 'GeometricBrownianMotion'},
        # Add more process types here as needed
    ]

    param_ranges = {
        'BrownianMotion': {'drift': (0, 0.5), 'scale': (0.1, 0.5)},
        'GeometricBrownianMotion': {'drift': (0, 0.5), 'volatility': (0.1, 0.5)},
        # Add parameter ranges for other process types here
    }

    # Initialize the UtilityFunctionInference with the path to your trained model
    ufi = UtilityFunctionInference(model_path, param_ranges)

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

    # Generate dataset
    dataset = ufi.generate_dataset(processes=processes, num_instances=2, n_simulations=2)

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


def EvolutionaryNN_case():
    """
    This case demonstrates the use of evolutionary neural networks for agent-based modeling with encoded processes.
    It covers process generation, neural network mutation, cloning, and evolutionary training of agents.
    The goal is to optimize agent behavior based on encoded processes and fitness evaluation.

    :return: final_agents: The final agents after the evolutionary training.
    :rtype: list
    """
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
        output_dir='output_nn/3',
    )

    population, history = trainer.train(n_steps=300, save_interval=50)
    best_agent = max(population, key=lambda agent: agent.accumulated_wealth)
    print(f"Best agent accumulated wealth: {best_agent.accumulated_wealth}")


def StochasticHeatEquation_case():
    """
    This case demonstrates the simulation of a stochastic partial differential equation (PDE), specifically the stochastic heat equation.

    :return: None
    """
    # Define the PSDE components
    def drift(t, x, u, u_x, u_xx):
        return 0.01 * u_xx  # Heat equation term

    def diffusion(t, x, u):
        return 0.2 * np.ones_like(x)  # Constant noise

    def initial_condition(x):
        return np.sin(np.pi * x)  # Initial sine wave

    def boundary_condition(t, x):
        return 0  # Zero at boundaries

    # Set up the simulation
    simulator = PSDESimulator(
        drift=drift,
        diffusion=diffusion,
        initial_condition=initial_condition,
        x_range=(0, 10),
        t_range=(0, 50),
        nx=1000,
        nt=5000,
        boundary_condition=("neumann", boundary_condition)
    )

    # Run the simulation
    simulator.simulate()

    # Plot the results
    simulator.plot_results()

    # Create 3D plot
    simulator.plot_3d()

    # Create animation
    simulator.create_animation()

    print("Simulation and visualization complete.")

def BasicUtilityAgent_case():
    """
    This case demonstrates the use of basic utility agents interacting with GeometricBrownianMotion processes.

    :return: final_agents: The final agents after the evolutionary training.
    :rtype: list
    """
    drift=0.4
    volatility=0.3

    process = GeometricBrownianMotion(drift=drift, volatility=volatility)
    process_dict = process_to_dict(process)
    print(process_dict)
    params = np.array([2, 0.4, 0.5, 0.7, 0])
    t = 1.0


    Agent_utility.compare_numerical_and_symbolic_expected_utility(process_dict, params, GeometricBrownianMotion, t=1.0)

    print(stochastic_process_class.closed_formula())

    # Example usage with multiple processes
    def symbolic_process(t, W, mu, sigma):
        return sp.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)

    def generate_processes(num_processes, drift_range=(-0.2, 0.2), volatility_range=(0.01, 0.5)):
        processes = []
        for i in range(num_processes):
            drift = np.random.uniform(*drift_range)
            volatility = np.random.uniform(*volatility_range)

            if i % 2 == 0:
                # Create a dictionary-based process
                processes.append({
                    'symbolic': lambda t, W, drift, volatility: sp.exp(
                        (drift - 0.5 * volatility ** 2) * t + volatility * W),
                    'drift': drift,
                    'volatility': volatility
                })
            else:
                # Create a GeometricBrownianMotion instance
                processes.append(GeometricBrownianMotion(drift=drift, volatility=volatility))

        return processes

    n_agents = 10
    n_steps = 20
    save_interval = 10
    removal_percentage = 0.5
    removal_interval = 5
    process_selection_share: float = 1
    keep_top_n = 10
    evolution_steps = 5

    processes = generate_processes(10)
    # print(processes)

    parameter = 2

    param_means = np.array([parameter, parameter, parameter, parameter, 0])
    param_stds = [parameter] * 5
    param_stds = np.array(param_stds)
    mutation_rate = 0.05

    # final_agents, history = Agent_utility.evolutionary_algorithm(
    #     n_agents, n_steps, save_interval, processes,
    #     param_means, param_stds, mutation_rate,
    #     stochastic_process_class=GeometricBrownianMotion,
    #     keep_top_n=keep_top_n,  # Specify the number of top agents to keep
    #     removal_interval=removal_interval,
    #     process_selection_share=process_selection_share,
    #     numeric_utilities=True
    # )

    final_agents, history = Agent_utility.evolutionary_algorithm_with_exchange(n_agents=n_agents, n_steps=n_steps,
                                                                               save_interval=save_interval,
                                                                               processes=processes,
                                                                               param_means=param_means,
                                                                               param_stds=param_stds,
                                                                               noise_std=mutation_rate,
                                                                               top_k=5,
                                                                               stochastic_process_class=GeometricBrownianMotion,
                                                                               process_selection_share=process_selection_share,
                                                                               output_dir='evolution_results',
                                                                               s=evolution_steps,
                                                                               numeric_utilities=False)

    print('Final agents:')
    print(final_agents, history)

    # Print some results
    print(f"Number of agents at the end: {len(final_agents)}")
    print(f"Average wealth at the end: {np.mean([agent.wealth for agent in final_agents]):.2f}")
    best_agent = max(final_agents, key=lambda a: a.total_accumulated_wealth)
    print(f"Best agent's wealth: {best_agent.wealth:.2f}")
    print(f"Best agent's total accumulated wealth: {best_agent.total_accumulated_wealth:.2f}")
    print(f"Best agent's parameters: {best_agent.params}")

    visualize_agent_evolution(history)
    analyze_utility_function_trends(history)

def TimeAverageDynamicsGBM_case():
    """
    This case focuses on time-average dynamics in a Geometric Brownian Motion process and demonstrates the ergodicity transformation.

    :return: None
    """
    # Define symbols
    x, t = sp.symbols('x t')

    # Example: Geometric Brownian Motion
    mu = 0.1 * x
    sigma = 0.3 * x

    # Call ergodicity_transform function (assuming it's defined elsewhere)
    is_consistent, u, a_u, b_u = ergodicity_transform(mu, sigma, x, t)

    if is_consistent:
        # Calculate time average dynamics
        time_avg_dynamics = calculate_time_average_dynamics(u, a_u, x, t)

        if time_avg_dynamics:
            print("\nTime average dynamics for Geometric Brownian Motion:")
            print(f"x(t) = {time_avg_dynamics}")

            # Optional: Simplify the result
            simplified_dynamics = sp.simplify(time_avg_dynamics)
            print(f"Simplified time average dynamics: x(t) = {simplified_dynamics}")
    else:
        print("The process is not ergodic, cannot calculate time average dynamics.")

# Simulate Geometric Levy Process
def GeometricLevyProcess_case():
    """
    This case simulates and visualizes the Geometric Levy Process, showcasing how ensemble and time averages can be computed and compared.
    It is a very general and frequently used process in many areas and the library puts a strong emphasis on it.

     :return: None
    """
    variance = 0.004
    mean = 0.0036
    correct_scale = (scale_default)*variance**0.5
    glp = ep.multiplicative.GeometricLevyProcess(alpha = 1.5, beta=-0.04, loc = mean, scale = correct_scale)
    data = glp.simulate(t=1, timestep=0.001, num_instances=1, plot=True)
    moments = glp.visualize_moments(data, save=True, mask=1000)
    e = glp.growth_rate_ensemble_average(num_instances=10)
    print(e)
    ta = glp.growth_rate_time_average(timestep = 0.01, t=10)
    print(ta)
    simulated_data_glp = (glp.simulate(t=10, timestep=0.0001, num_instances=1, plot=True))
    c = compare_averages(simulated_data_glp)
    print(c)
    ge = glp.growth_rate_ensemble_average(num_instances=10)
    print(ge)
    gt = glp.growth_rate_time_average(t=10)
    # print(gt)

    return None

def VariousSimulations_case():
    """
    This case demonstrates a collection of simulations involving various stochastic processes, showcasing the library's capabilities across multiple process types.

    :return: data: The simulated data for each process.
    :rtype: list
    """
    data = []
    moments_all = []

    # Simulate Bessel Process
    bp = ep.basic.StandardBesselProcess()
    simulated_data_bp = bp.simulate(t=10.0, timestep=0.01, num_instances=10, plot=True)
    moments = bp.moments(simulated_data_bp, save = True)
    data.append(simulated_data_bp)
    moments_all.append(moments)

    # Simulate Brownian Bridge
    bb = ep.basic.StandardBrownianBridge(b=1)
    simulated_data_bb = bb.simulate_raw(plot=True)
    moments = bb.moments(simulated_data_bb, save = True)
    data.append(simulated_data_bb)

    # Simulate Brownian Excursion
    be = ep.basic.StandardBrownianExcursion()
    simulated_data_be = be.simulate()
    moments = be.moments(simulated_data_be, save = True)
    data.append(simulated_data_be)
    moments.app.append(moments)

    # Simulate Brownian Meander
    bm = ep.basic.StandardBrownianMeander()
    simulated_data_bm = bm.simulate()
    moments = bm.moments(simulated_data_bm, save = True)
    data.append(simulated_data_bm)
    moments_all.append(moments)

    # Simulate Cauchy Process
    cp = ep.basic.CauchyProcess()
    simulated_data_cp = cp.simulate()
    moments = cp.moments(simulated_data_cp, save = True)
    data.append(simulated_data_cp)
    moments_all.append(moments)

    # Simulate Fractional Brownian Motion
    fbm = ep.basic.StandardFractionalBrownianMotion(hurst=0.5)
    simulated_data_fbm = fbm.simulate()
    moments = fbm.moments(simulated_data_fbm, save = True)
    data.append(simulated_data_fbm)
    moments_all.append(moments)

    # Simulate Gamma Process
    gp = ep.basic.GammaProcess(rate=2.0)
    simulated_data_gp = gp.simulate()
    moments = gp.moments(simulated_data_gp, save = True)
    data.append(simulated_data_gp)
    moments_all.append(moments)

    # Simulate Inverse Gaussian Process
    igp = ep.basic.InverseGaussianProcess()
    simulated_data_igp = igp.simulate()
    moments = igp.moments(simulated_data_igp, save = True)
    data.append(simulated_data_igp)
    moments_all.append(moments)

    # Simulate StandardMultifractionalBrownianMotion
    smbm = ep.basic.StandardMultifractionalBrownianMotion(hurst = lambda t: 0.1)
    simulated_data_smbm = smbm.simulate()
    moments = smbm.moments(simulated_data_smbm, save = True)
    data.append(simulated_data_smbm)
    moments_all.append(moments)

    # Simulate Wiener Process
    pp = ep.basic.WienerProcess()
    simulated_data_pp = pp.simulate()
    moments = pp.moments(simulated_data_pp, save = True)
    data.append(simulated_data_pp)
    moments_all.append(moments)

    # Simulate Poisson Process
    pp = ep.basic.PoissonProcess()
    simulated_data_pp = pp.simulate_live()
    moments = pp.moments(simulated_data_pp, save = True)
    data.append(simulated_data_pp)
    moments_all.append(moments)

    # Simulate Levy Stable Process
    lsp = ep.basic.LevyStableProcess(alpha = 2, loc=0.01, scale=(scale_default)*0.02**0.5)
    simulated_data_lsp = lsp.simulate_raw(t = 10, timestep=0.01, num_instances = 100, plot=True)
    moments = lsp.visualize_moments(simulated_data_lsp)
    data.append(simulated_data_lsp)
    moments_all.append(moments)

    return data, moments_all


# Simulate Multivariate Brownian Motion
def MultivariateGeometricBrownianMotion_case():
    """
    This case demonstrates how to simulate and visualize multivariate Geometric Brownian Motion with a specified correlation matrix.

    :return: None
    """
    from ergodicity.process.definitions import correlation_to_covariance
    from ergodicity.process.definitions import create_correlation_matrix
    size = 500
    correlation = 0.1
    correlation_matrix = create_correlation_matrix(size, correlation)

    standard_devs = [0.1**0.5] * size

    covariance_matrix = correlation_to_covariance(correlation_matrix, standard_devs)

    print(covariance_matrix)

    mbm = ep.multiplicative.MultivariateGeometricBrownianMotion(drift = [0.04]*size, scale = covariance_matrix)
    # simulated_data_mbm = mbm.simulate(t=100, timestep=0.1, plot=True)
    mbm.simulate_ensemble(n=50, t=40, timestep=0.1, save=True)

    return None

# Simulate Geometric Brownian Motion using parallel execution

def GeometricBrownianMotion_case():
    """
    This case uses parallel execution to simulate the Geometric Brownian Motion process efficiently across multiple settings.
    Geometric Brownian Motion is a fundamental process in stochastic calculus and financial mathematics.

    :return: results: The results of the parallel execution.
    :rtype: list
    """
    gbm = ep.multiplicative.GeometricBrownianMotion(drift=0.0011, volatility=0.002**0.5)

    from ergodicity.tools.multiprocessing import ParallelExecutionManager

    manager = ParallelExecutionManager()

    tasks_to_run = [
        {'name': 'simulate_raw', 'object': gbm, 'arguments': {'t': 10, 'timestep': 0.01, 'num_instances': 1000, 'save': True}},
        {'name': 'simulate_raw', 'object': gbm, 'arguments': {'t': 10, 'timestep': 0.001, 'num_instances': 100, 'save': True}},
        {'name': 'simulate_raw', 'object': gbm, 'arguments': {'t': 10, 'timestep': 0.0001, 'num_instances': 10, 'save': True}},
    ]

    results = manager.execute(tasks_to_run)

    return results

def ItoLemmaApplication():
    """
    This case applies Ito's Lemma to a given stochastic differential equation (SDE), providing insight into how symbolic manipulation can be used for process analysis.

    :return: result: The result of applying Ito's Lemma to the given SDE.
    :rtype: sympy.Expr
    """
    # Define symbols
    x, t = sp.symbols('x t')
    mu1 = sp.symbols('mu1')
    sigma1 = sp.symbols('sigma1')

    # Define example functions
    f = sp.log(x)
    mu = mu1*x
    sigma = sigma1*x

    # Apply Ito's Lemma
    result = ito_differential(f, mu, sigma, x, t)

    print("Ito's Lemma applied to f(x,t) = x^2 * e^t with dx = 2x*dt + x*dW:")
    print(result)

    return result



