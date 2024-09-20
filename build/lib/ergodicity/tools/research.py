"""
research Submodule

This submodule provides predefined high-level functions to run massive parallel simulations and compute various statistical measures for different stochastic processes. The functions are organized into pipelines, each designed to handle a specific process type with customizable parameter ranges. These pipelines return simulation results in the form of dictionaries, allowing for easy analysis and visualization.

Key Functions:

1. **LevyMicro_case**:

    - Simulates and analyzes Geometric Lévy processes for specific values of the stability parameter (α), location (loc), scale, and skewness (β).

2. **TestingGeometricLevyPipeline_case**:

    - Runs a similar pipeline as `LevyMicro_case`, but for a broader range of α values.

3. **GeometricLevyPipeline_case**:

    - Simulates Geometric Lévy processes across a larger set of parameter ranges, including more granular values for α, location, scale, and β.

4. **GBMPipeline_case**:

    - Simulates Multivariate Geometric Brownian Motion (GBM) processes for a range of drift and scale parameters.

5. **FractionalGBMPipeline_case**:

    - Runs simulations of Geometric Fractional Brownian Motion (fBM) for different values of drift, scale, and Hurst exponent (H).

6. **LevyPipeline_case**:

    - Simulates Lévy stable processes using different combinations of α, loc, scale, and β parameters.

7. **Live 3D Visualization Pipelines**:

    - `live_3d_Levy_pipeline_case`, `live_3d_Brownian_pipeline_case`, and `live_3d_geometricLevy_pipeline_case` are used to generate live 3D visualizations for Lévy, Brownian Motion, and Geometric Lévy processes respectively.

8. **TINstancesPipeline_case**:

    - This pipeline is designed to analyze the growth rate of the average over time for multiple instances of the Geometric Lévy process.

9. **AverageRateVsRateAveragePipeline_case**:

    - Compares the average of growth rates over multiple instances with the rate of the overall average for Geometric Lévy processes.

Combining Pipelines:

- **live_3d_meta_pipeline_case**:

    - Combines the 3D visualization pipelines for Lévy, Brownian Motion, and Geometric Lévy processes into a single meta pipeline.

Example Usage:

if __name__ == "__main__":

    # Example: Running the Geometric Lévy process pipeline with a variety of parameters

    results = GeometricLevyPipeline_case()

    print("Geometric Lévy process simulation results:", results)

    # Example: Running the Brownian Motion process pipeline

    results_gbm = GBMPipeline_case()

    print("GBM process simulation results:", results_gbm)

    # Example: Running a live 3D visualization for Geometric Lévy processes

    live_3d_geometricLevy_pipeline_case()
"""

from ergodicity.tools.multiprocessing import *
# from ergodicity.process.multiplicative import *
# from ergodicity.process.basic import *
from ergodicity.tools.automate import *

def LevyMicro_case():
    """
    This function runs a small pipeline for Geometric Lévy processes with specific parameter values.

    :return: results (dict): Simulation results for the Geometric Lévy process
    :rtype: dict
    """
    alphas = [2, 1]
    locs = [0.001, 0.005]
    scales = [0.001, 0.05]
    betas = [0]

    param_ranges = {
        'alpha': alphas,
        'loc': locs,
        'scale': scales,
        'beta': betas}

    results = general_pipeline(GeometricLevyProcess, param_ranges)

    return results

def TestingGeometricLevyPipeline_case():
    """
    This function runs a testing pipeline for Geometric Lévy processes with a broader range of α values.

    :return: results (dict): Simulation results for the Geometric Lévy process
    :rtype: dict
    """
    alphas = [2, 1.9, 1.5, 1.0]
    locs = [0.001, 0.005]
    scales = [0.001, 0.05]
    betas = [0]

    param_ranges = {
        'alpha': alphas,
        'loc': locs,
        'scale': scales,
        'beta': betas}

    results = general_pipeline(GeometricLevyProcess, param_ranges)

    return results

def GeometricLevyPipeline_case():
    """
    This function runs a large full-fledged pipeline for Geometric Lévy processes with a larger set of parameter ranges.

    :return: results (dict): Simulation results for the Geometric Lévy process
    :rtype: dict
    """
    alphas = [2, 1.95, 1.9, 1.7, 1.5, 1.2, 1.0, 0.5]
    locs = [0.001, 0.002, 0.005, 0.01]
    scales = [0.001, 0.002, 0.005, 0.01, 0.02]
    betas = [-1, -0.5, 0, 0.5, 1]

    param_ranges = {
        'alpha': alphas,
        'loc': locs,
        'scale': scales,
        'beta': betas}

    results = general_pipeline(GeometricLevyProcess, param_ranges)

    return results

def GBMPipeline_case():
    """
    This function runs a pipeline for Multivariate Geometric Brownian Motion processes with a range of drift and scale parameters.

    :return: results (dict): Simulation results for the Multivariate GBM process
    :rtype: dict
    """

    drifts = [0.001, 0.002, 0.005, 0.01]
    scales = [0.001, 0.002, 0.005, 0.01, 0.02]

    param_ranges = {
        'drift': drifts,
        'scale': scales}

    results = general_pipeline(MultivariateGeometricBrownianMotion, param_ranges)

    return results

def FractionalGBMPipeline_case():
    """
    This function runs a pipeline for Geometric Fractional Brownian Motion processes with different values of drift, scale, and Hurst exponent.

    :return: results (dict): Simulation results for the Geometric fBM process
    :rtype: dict
    """

    drifts = [0.001, 0.002, 0.005, 0.01]
    scales = [0.001, 0.002, 0.005, 0.01, 0.02]
    hursts = [0.1, 0.3, 0.5, 0.7, 0.9]

    param_ranges = {
        'drift': drifts,
        'scale': scales,
        'hurst': hursts}

    results = general_pipeline(GeometricFractionalBrownianMotion, param_ranges)

    return results

def LevyPipeline_case():
    """
    This function runs a pipeline for Lévy stable processes with different combinations of α, loc, scale, and β parameters.

    :return: results (dict): Simulation results for the Lévy stable process
    :rtype: dict
    """
    alphas = [2, 1.95, 1.9, 1.7, 1.5, 1.2, 1.0, 0.5]
    locs = [0.001, 0.002, 0.005, 0.01]
    scales = [0.001, 0.002, 0.005, 0.01, 0.02]
    betas = [-1, -0.5, 0, 0.5, 1]

    param_ranges = {
        'alpha': alphas,
        'loc': locs,
        'scale': scales,
        'beta': betas}

    results = general_pipeline(LevyStableProcess, param_ranges)

    return results

def live_3d_Levy_pipeline_case():
    """
    This function generates a live 3D visualization for Lévy stable processes.

    :return: None
    """
    alphas = [2, 1.95, 1.9, 1.7, 1.5, 1.2, 1.0, 0.5]
    locs = [0.001, 0.002, 0.005, 0.01]
    scales = [0.001, 0.002, 0.005, 0.01, 0.02]
    betas = [-1, -0.5, 0, 0.5, 1]

    param_ranges = {
        'alpha': alphas,
        'loc': locs,
        'scale': scales,
        'beta': betas}

    automated_live_visualization(3, LevyStableProcess, param_ranges, t=10, timestep=0.00001, num_instances=1, speed=1.0)

def live_3d_Brownian_pipeline_case():
    """
    This function generates a live 3D visualization for Brownian Motion processes.

    :return: None
    """
    drifts = [0.001, 0.002, 0.005, 0.01]
    scales = [0.001, 0.002, 0.005, 0.01, 0.02]

    param_ranges = {
        'drift': drifts,
        'scale': scales}

    automated_live_visualization(3, BrownianMotion, param_ranges, t=10, timestep=0.00001, num_instances=1, speed=1.0)

def live_3d_geometricLevy_pipeline_case():
    """
    This function generates a live 3D visualization for Geometric Lévy processes.

    :return: None
    """
    alphas = [2, 1.95, 1.9, 1.7, 1.5, 1.2, 1.0, 0.5]
    locs = [0.001, 0.002, 0.005, 0.01]
    scales = [0.001, 0.002, 0.005, 0.01, 0.02]
    betas = [-1, -0.5, 0, 0.5, 1]

    param_ranges = {
        'alpha': alphas,
        'loc': locs,
        'scale': scales,
        'beta': betas}

    automated_live_visualization(3, GeometricLevyProcess, param_ranges, t=10, timestep=0.00001, num_instances=1, speed=1.0)

#combine 3 pipelines into one
def live_3d_meta_pipeline_case():
    """
    This function combines the 3D visualization pipelines for Lévy, Brownian Motion, and Geometric Lévy processes into a single meta pipeline.

    :return: None
    """
    live_3d_Levy_pipeline_case()
    live_3d_Brownian_pipeline_case()
    live_3d_geometricLevy_pipeline_case()


def TINstancesPipelineTest_case():
    """
    This function runs a test pipeline for analyzing the growth rate of the average over time for multiple instances of the Geometric Lévy process.

    :return: results (dict): Simulation results for the growth rate analysis
    :rtype: dict
    """
    process_params = {'alpha': 1.5, 'beta': 0, 'loc': 0.01, 'scale': 0.1}
    t_range = [1,2]
    num_instances_range = [1,2]

    results = multi_growth_rate_processor(GeometricLevyProcess, process_params, t_range, num_instances_range, timestep=0.001)
    return results


def TINstancesPipeline_case():
    """
    This function runs a pipeline for analyzing the growth rate of the average over time for multiple instances of the Geometric Lévy process.

    :return: results (dict): Simulation results for the growth rate analysis
    :rtype: dict
    """
    process_params = {'alpha': 1.5, 'beta': 0, 'loc': 0.01, 'scale': 0.1}
    t_range = np.linspace(1, 10, 10)
    num_instances_range = np.logspace(2, 5, 10, dtype=int)

    results = multi_growth_rate_processor(GeometricLevyProcess, process_params, t_range, num_instances_range)
    return results

def AverageRateVsRateAveragePipeline_case():
    """
    This function compares the average of growth rates over multiple instances with the rate of the overall average for Geometric Lévy processes.

    :return: results (dict): Simulation results for the average rate comparison
    :rtype: dict
    """
    alphas = [2]
    locs = [0.001, 0.002]
    scales = [0.001]
    betas = [0]

    param_ranges = {
        'alpha': alphas,
        'loc': locs,
        'scale': scales,
        'beta': betas}

    results = average_rate_vs_rate_average_pipeline(GeometricLevyProcess, param_ranges)

    return results
