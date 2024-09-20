"""
This submodule provides a comprehensive suite of stochastic differential equation (SDE) simulations
for various financial and mathematical models. It includes custom decorators for both individual
and system-wide simulations, allowing for flexible and efficient modeling of complex stochastic processes.

Key components:

Decorator functions:

custom_simulate: Wraps individual simulation functions
custom_simulate_system: Wraps system simulation functions

Individual process simulations, including but not limited to:

Ornstein-Uhlenbeck

Generalized Diffusion

Constant Elasticity of Variance

Cox-Ingersoll-Ross

Vasicek

Lévy processes

Geometric Brownian Motion

Heston model

Jump Diffusion models

Stochastic Volatility models

Fractional Brownian Motion

System simulations:

TestSystemSimulation: A system of SDEs including Ornstein-Uhlenbeck and Geometric Brownian Motion

LotkaVolterraSimulation: Stochastic version of the Lotka-Volterra predator-prey model


Each simulation function is decorated to handle time stepping, data storage, and optional plotting.
"""
import ergodicity.process.definitions as definitions
from ergodicity.process.increments import WP, LP15, LP05, LP15, LPL, LPC
from ergodicity.configurations import *
from ergodicity.process.definitions import simulation_decorator
from typing import List, Any, Type, Callable
import numpy as np
from ergodicity.tools.helper import plot
from ergodicity.tools.helper import plot_system
from ergodicity.configurations import *

params = {'t': 10, 'timestep': 0.01, 'num_instances': 1, 'save': True, 'plot': True}

def custom_simulate(simulate_func: Callable) -> Callable:
    """
    Decorator for individual simulation functions. Handles time stepping, data storage, and optional plotting.

    :param simulate_func: Simulation function to be wrapped
    :type simulate_func: Callable
    :return: Wrapper function for simulation
    :rtype: Callable
    """
    def wrapper(verbose: bool = False, **kwargs) -> Any:
        """
        Wrapper function for simulation. Handles time stepping, data storage, and optional plotting.

        :param verbose: Print simulation progress if True
        :type verbose: bool
        :param kwargs: Custom parameters for simulation
        :return: Simulation data
        :rtype: Any
        """
        # Use custom values if provided, otherwise use defaults from params
        t = kwargs.pop('t', params['t'])
        timestep = kwargs.pop('timestep', params['timestep'])
        num_instances = kwargs.pop('num_instances', params['num_instances'])

        num_steps = int(t / timestep)
        times = np.linspace(0, t, num_steps)

        data = np.zeros((num_instances, num_steps))
        dt = timestep

        for i in range(num_instances):
            X = 1
            for step in range(num_steps):
                data[i, step] = X
                dX = simulate_func(X, dt, **kwargs)  # Ensure X is not in **params
                X = X + dX
                if verbose and step % 1000000 == 0:
                    print(f"Simulating instance {i}, step {step}, X = {X}")

        data = np.concatenate((times.reshape(1, -1), data), axis=0)
        if lib_plot:
            plot(data, num_instances, save=params['save'], plot=params['plot'])
        return data

    return wrapper

def custom_simulate_system(simulate_func: Callable) -> Callable:
    """
    Decorator for system simulation functions. Handles time stepping, data storage, and optional plotting.

    :param simulate_func: Simulation function to be wrapped
    :type simulate_func: Callable
    :return: Wrapper function for system simulation
    :rtype: Callable
    """
    def wrapper(verbose: bool = False, **kwargs) -> Any:
        """
        Wrapper function for system simulation. Handles time stepping, data storage, and optional plotting.

        :param verbose: Print simulation progress if True
        :type verbose: bool
        :param kwargs: Custom parameters for simulation
        :return: Simulation data
        :rtype: Any
        """
        # Use custom values if provided, otherwise use defaults from params
        t = kwargs.pop('t', params['t'])
        timestep = kwargs.pop('timestep', params['timestep'])
        num_instances = kwargs.pop('num_instances', params['num_instances'])

        num_steps = int(t / timestep)
        times = np.linspace(0, t, num_steps)

        initial_X = kwargs.get('initial_X', [1, 1])
        num_equations = len(initial_X)

        data = np.zeros((num_instances, num_equations, num_steps))

        for i in range(num_instances):
            X = np.array(initial_X, dtype=float)
            for step in range(num_steps):
                data[i, :, step] = X
                dX = simulate_func(X, timestep, **kwargs)
                X = X + dX
                if verbose and step % 100000 == 0:
                    print(f"Simulating instance {i}, step {step}, X = {X}")

        # Reshape data to include time as the first row
        data_with_time = np.vstack((times, data.reshape(-1, num_steps)))

        plot_system(data_with_time, num_instances, num_equations,
                    save=kwargs.get('save', params['save']),
                    plot=kwargs.get('plot', params['plot']))

        return data_with_time

    return wrapper

@custom_simulate_system
def TestSystemSimulation(X: np.ndarray, dt: float, **kwargs) -> np.ndarray:
    """
    System of SDEs for testing purposes. Can be customized as needed.

    :param X: Process variables
    :type X: np.ndarray
    :param dt: Time step
    :type dt: float
    :param kwargs: Custom parameters for simulation
    :type kwargs: dict
    :return: Increments for system of SDEs
    :rtype: np.ndarray
    """
    # Example system of SDEs (can be customized as needed)

    dW = WP.increment()
    dX = np.zeros_like(X)

    # SDE 1: Ornstein-Uhlenbeck process
    dX[0] = -kwargs.get('theta', 0.5) * X[0] * dt + kwargs.get('sigma', 0.5) * dW

    # SDE 2: Geometric Brownian Motion
    dW = WP.increment()
    dX[1] = kwargs.get('mu', 0.1) * X[1] * dt + kwargs.get('sigma_gbm', 0.2) * X[1] * dW

    dW = WP.increment()
    dX[2] = kwargs.get('theta', 0.5) * X[1] * dt + kwargs.get('sigma', 0.5) * X[1] * dW

    # Add more equations as needed

    return dX

@custom_simulate_system
def LotkaVolterraSimulation(X: np.ndarray, dt: float, **kwargs) -> np.ndarray:
    """
    Stochastic version of the Lotka-Volterra predator-prey model.

    :param X: Process variables for prey and predator populations
    :type X: np.ndarray
    :param dt: Time step
    :type dt: float
    :param kwargs: Custom parameters for simulation
    :return: Increments for prey and predator populations
    :rtype: np.ndarray
    """
    prey, predator = X

    dX = np.zeros_like(X)

    # Extract parameters
    a = kwargs.get('a', 1.0)  # Prey growth rate
    b = kwargs.get('b', 0.1)  # Predation rate
    c = kwargs.get('c', 1.5)  # Predator death rate
    d = kwargs.get('d', 0.075)  # Predator growth rate due to predation
    sigma1 = kwargs.get('sigma1', 0.5)  # Noise intensity for prey
    sigma2 = kwargs.get('sigma2', 0.5)  # Noise intensity for predator

    # Deterministic terms
    d_prey = a * prey * dt - b * prey * predator * dt + sigma1 * prey * WP.increment()
    d_predator = -c * predator * dt + d * prey * predator * dt + sigma2 * predator * WP.increment()

    dX[0] = d_prey
    dX[1] = d_predator

    return dX

@custom_simulate
def OrnsteinUhlenbeckSimulation(X: float, dt: float, theta: float = 0.5, sigma: float = 0.5) -> float:
    """
    Simulates an Ornstein-Uhlenbeck process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param theta: Mean reversion parameter
    :type theta: float
    :param sigma: Volatility parameter
    :type sigma: float
    :return: Increment for Ornstein-Uhlenbeck process
    :rtype: float
    """
    dW = WP.increment()
    dX = - theta * X * dt + sigma * dW
    return dX

@custom_simulate
def GeneralizedDiffusionSimulation(X: float, dt: float, theta: float = 0.5, mu: float = 1.0, sigma: float = 0.5, gamma: float = 1.0) -> float:
    """
    Simulates a generalized diffusion process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param theta: Mean reversion parameter
    :type theta: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :param gamma: Diffusion parameter
    :type gamma: float
    :return: Increment for generalized diffusion process
    :rtype: float
    """
    dW = WP.increment()
    dX = theta * mu * dt - theta * X * dt + sigma * X ** gamma * dW
    return dX

@custom_simulate
def ConstantElasticityOfVarianceSimulation(X: float, dt: float, theta: float = 0.5, mu: float = 1.0, sigma: float = 0.5, gamma: float = 1.0) -> float:
    """
    Simulates a constant elasticity of variance process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param theta: Mean reversion parameter
    :type theta: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :param gamma: Diffusion parameter
    :type gamma: float
    :return: Increment for constant elasticity of variance process
    :rtype: float
    """
    dW = WP.increment()
    dX = theta * mu * X * dt - sigma * X ** gamma * dW
    return dX

@custom_simulate
def CoxIngersollRossSimulation(X: float, dt: float, X_mean: float = 1, i: int = 0, theta: float = 0.5, mu: float = 1.0, sigma: float = 0.5) -> float:
    """
    Simulates a Cox-Ingersoll-Ross process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param X_mean: Mean of X
    :type X_mean: float
    :param i: Current iteration
    :type i: int
    :param theta: Mean reversion parameter
    :type theta: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :return: Increment for Cox-Ingersoll-Ross process
    :rtype: float
    """
    dW = WP.increment()
    k = i+1
    X_mean = X_mean * ((k-1)/k) + X * (1/k)
    dX = theta * (mu - X) * dt + sigma * X_mean ** 0.5 * dW
    return dX

@custom_simulate
def VasicekSimulation(X: float, dt: float, theta: float = 0.5, mu: float = 1.0, sigma: float = 0.5) -> float:
    """
    Simulates a Vasicek process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param theta: Mean reversion parameter
    :type theta: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :return: Increment for Vasicek process
    :rtype: float
    """
    dW = WP.increment()
    dX = theta * (mu - X) * dt + sigma * dW
    return dX

@custom_simulate
def CauchyOrnsteinUlehnbeckSimulation(X: float, dt: float, theta: float = 0.5, mu: float = 1.0, sigma: float = 0.5) -> float:
    """
    Simulates a Cauchy Ornstein-Uhlenbeck process. This is a version of the Ornstein-Uhlenbeck process with Cauchy noise.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param theta: Mean reversion parameter
    :type theta: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :return: Increment for Cauchy Ornstein-Uhlenbeck process
    :rtype: float
    """
    dL = LPC.increment()
    dX = - theta * X * dt + sigma * dL
    return dX

@custom_simulate
def LevyStableOrnsteinUlehnbeckSimulation(X: float, dt: float, theta: float = 0.5, mu: float = 1.0, sigma: float = 0.5) -> float:
    """
    Simulates a Levy-stable Ornstein-Uhlenbeck process. This is a version of the Ornstein-Uhlenbeck process with Levy-stable noise.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param theta: Mean reversion parameter
    :type theta: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :return: Increment for Levy-stable Ornstein-Uhlenbeck process
    :rtype: float
    """
    dL = LPL.increment()
    dX = - theta * X * dt + sigma * dL
    return dX

@custom_simulate
def Levy05OrnsteinUlehnbeckSimulation(X: float, dt: float, theta: float = 0.5, mu: float = 1.0, sigma: float = 0.5) -> float:
    """
    Simulates a Levy 0.5 Ornstein-Uhlenbeck process. This is a version of the Ornstein-Uhlenbeck process with Levy 0.5 noise.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param theta: Mean reversion parameter
    :type theta: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :return: Increment for Levy 0.5 Ornstein-Uhlenbeck process
    :rtype: float
    """
    dL = LP05.increment()
    dX = - theta * X * dt + sigma * dL
    return dX

@custom_simulate
def Levy15OrnsteinUlehnbeckSimulation(X: float, dt: float, theta: float = 0.5, mu: float = 1.0, sigma: float = 0.5) -> float:
    """
    Simulates a Levy 1.5 Ornstein-Uhlenbeck process. This is a version of the Ornstein-Uhlenbeck process with Levy 1.5 noise.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param theta: Mean reversion parameter
    :type theta: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :return: Increment for Levy 1.5 Ornstein-Uhlenbeck process
    :rtype: float
    """
    dL = LP15.increment()
    dX = - theta * X * dt + sigma * dL
    return dX

@custom_simulate
def GeometricBrownianMotionSimulation(X: float, dt: float, mu: float = 0.1, sigma: float = 0.2) -> float:
    """
    Simulates a geometric Brownian motion process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :return: Increment for geometric Brownian motion process
    :rtype: float
    """
    dW = np.random.normal(0, np.sqrt(dt))
    dX = mu * X * dt + sigma * X * dW
    return dX

@custom_simulate
def HestonSimulation(X: float, dt: float, mu: float = 0.1, v0: float = 0.1, kappa: float = 1.0, theta: float = 0.2, xi: float = 0.3, rho: float = -0.7) -> float:
    """
    Simulates a Heston model.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param mu: Drift parameter
    :type mu: float
    :param v0: Initial variance
    :type v0: float
    :param kappa: Mean reversion parameter
    :type kappa: float
    :param theta: Long-term variance
    :type theta: float
    :param xi: Volatility parameter
    :type xi: float
    :param rho: Correlation parameter
    :type rho: float
    :return: Increment for Heston model
    :rtype: float
    """
    dW1 = np.random.normal(0, np.sqrt(dt))
    dW2 = np.random.normal(0, np.sqrt(dt))
    dV = kappa * (theta - v0) * dt + xi * np.sqrt(v0 * dt) * (rho * dW1 + np.sqrt(1 - rho**2) * dW2)
    dX = mu * X * dt + np.sqrt(v0) * X * dW1
    v0 = max(v0 + dV, 0)
    return dX

@custom_simulate
def JumpDiffusionSimulation(X: float, dt: float, mu: float = 0.1, sigma: float = 0.2, jump_intensity: float = 0.1, jump_mean: float = 0.05, jump_std: float = 0.1) -> float:
    """
    Simulates a jump diffusion process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :param jump_intensity: Jump intensity
    :type jump_intensity: float
    :param jump_mean: Jump mean
    :type jump_mean: float
    :param jump_std: Jump standard deviation
    :type jump_std: float
    :return: Increment for jump diffusion process
    :rtype: float
    """
    dW = np.random.normal(0, np.sqrt(dt))
    jump = np.random.poisson(jump_intensity * dt) * np.random.normal(jump_mean, jump_std)
    dX = mu * X * dt + sigma * X * dW + jump
    return dX

@custom_simulate
def MeanRevertingSquareRootDiffusionSimulation(X: float, dt: float, theta: float = 0.5, mu: float = 1.0, sigma: float = 0.5) -> float:
    """
    Simulates a mean-reverting square root diffusion process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param theta: Mean reversion parameter
    :type theta: float
    :param mu: Drift parameter\
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :return: Increment for mean-reverting square root diffusion process
    :rtype: float
    """
    dW = np.random.normal(0, np.sqrt(dt))
    dX = theta * (mu - X) * dt + sigma * np.sqrt(X) * dW
    return dX

@custom_simulate
def StochasticVolatilityModelSimulation(X: float, dt: float, mu: float = 0.1, sigma: float = 0.2, v0: float = 0.1, kappa: float = 1.0, theta: float = 0.2, xi: float = 0.3, rho: float = -0.7) -> float:
    """
    Simulates a stochastic volatility model.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :param v0: Initial variance
    :type v0: float
    :param kappa: Mean reversion parameter
    :type kappa: float
    :param theta: Long-term variance
    :type theta: float
    :param xi: Volatility parameter
    :type xi: float
    :param rho: Correlation parameter
    :type rho: float
    :return: Increment for stochastic volatility model
    :rtype: float
    """
    dW1 = np.random.normal(0, np.sqrt(dt))
    dW2 = np.random.normal(0, np.sqrt(dt))
    dV = kappa * (theta - v0) * dt + xi * np.sqrt(v0 * dt) * (rho * dW1 + np.sqrt(1 - rho**2) * dW2)
    dX = mu * X * dt + np.sqrt(v0) * X * dW1
    v0 = max(v0 + dV, 0)
    return dX

@custom_simulate
def GeometricOrnsteinUhlenbeckSimulation(X: float, dt: float, theta: float = 0.5, mu: float = 1.0, sigma: float = 0.5) -> float:
    """
    Simulates a geometric (multiplicative) Ornstein-Uhlenbeck process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param theta: Mean reversion parameter
    :type theta: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :return: Increment for geometric Ornstein-Uhlenbeck process
    :rtype: float
    """
    dW = np.random.normal(0, np.sqrt(dt))
    dX = (mu - theta * np.log(X)) * X * dt + sigma * X * dW
    return dX

@custom_simulate
def HullWhiteSimulation(X: float, dt: float, theta: float = 0.5, mu: float = 1.0, sigma: float = 0.5) -> float:
    """
    Simulates a Hull-White process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param theta: Mean reversion parameter
    :type theta: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :return: Increment for Hull-White process
    :rtype: float
    """
    dW = np.random.normal(0, np.sqrt(dt))
    dX = theta * (mu - X) * dt + sigma * dW
    return dX

@custom_simulate
def VarianceGammaSimulation(X: float, dt: float, theta: float = 0.5, sigma: float = 0.5, nu: float = 0.2) -> float:
    """
    Simulates a variance gamma process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param theta: Mean reversion parameter
    :type theta: float
    :param sigma: Volatility parameter
    :type sigma: float
    :param nu: Jump intensity parameter
    :type nu: float
    :return: Increment for variance gamma process
    :rtype: float
    """
    dW = np.random.normal(0, np.sqrt(dt))
    dG = np.random.gamma(shape=dt/nu, scale=nu)
    dX = theta * dG + sigma * dW
    return dX

@custom_simulate
def KouJumpDiffusionSimulation(X: float, dt: float, mu: float = 0.1, sigma: float = 0.2, lambda_p: float = 0.1, lambda_m: float = 0.1, p: float = 0.5, eta1: float = 0.05, eta2: float = 0.05) -> float:
    """
    Simulates a Kou jump diffusion process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :param lambda_p: Jump intensity parameter for positive jumps
    :type lambda_p: float
    :param lambda_m: Jump intensity parameter for negative jumps
    :type lambda_m: float
    :param p: Probability of positive jump
    :type p: float
    :param eta1: Mean of positive jump
    :type eta1: float
    :param eta2: Mean of negative jump
    :type eta2: float
    :return: Increment for Kou jump diffusion process
    :rtype: float
    """
    dW = np.random.normal(0, np.sqrt(dt))
    N = np.random.poisson((lambda_p + lambda_m) * dt)
    Y = np.sum(np.random.choice([np.random.exponential(eta1), -np.random.exponential(eta2)], size=N, p=[p, 1-p]))
    dX = mu * X * dt + sigma * X * dW + Y
    return dX

@custom_simulate
def DoubleExponentialJumpDiffusionSimulation(X: float, dt: float, mu: float = 0.1, sigma: float = 0.2, lambda_p: float = 0.1, lambda_m: float = 0.1, p: float = 0.5, eta1: float = 0.05, eta2: float = 0.05) -> float:
    """
    Simulates a double exponential jump diffusion process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :param lambda_p: Jump intensity parameter for positive jumps
    :type lambda_p: float
    :param lambda_m: Jump intensity parameter for negative jumps
    :type lambda_m: float
    :param p: Probability of positive jump
    :type p: float
    :param eta1: Mean of positive jump
    :type eta1: float
    :param eta2: Mean of negative jump
    :type eta2: float
    :return: Increment for double exponential jump diffusion process
    :rtype: float
    """
    dW = np.random.normal(0, np.sqrt(dt))
    N = np.random.poisson((lambda_p + lambda_m) * dt)
    Y = np.sum(np.random.choice([np.random.exponential(eta1), -np.random.exponential(eta2)], size=N, p=[p, 1-p]))
    dX = mu * X * dt + sigma * X * dW + Y
    return dX

@custom_simulate
def SubordinatorSimulation(X: float, dt: float, alpha: float = 0.5, beta: float = 0.5, gamma: float = 0.5) -> float:
    """
    Simulates a subordinator process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param alpha: Controls the heaviness of the tails of the distribution (alpha)
    :type alpha: float
    :param beta: Drift component of the process (beta)
    :type beta: float
    :param gamma: Scaling parameter for the size of jumps in the process (gamma)
    :type gamma: float
    :return: Increment for subordinator process
    :rtype: float
    """
    dL = np.random.exponential(scale=gamma) ** (1/alpha)
    dX = (beta * X * dt) + (gamma * X * dL)
    return dX

@custom_simulate
def StochasticAlphaBetaRhoSimulation(X: float, dt: float, alpha: float = 0.5, beta: float = 0.5, rho: float = 0.5, sigma: float = 0.5) -> float:
    """
    Simulates a stochastic alpha beta rho process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param alpha: Drift parameter
    :type alpha: float
    :param beta: Volatility parameter
    :type beta: float
    :param rho: Correlation parameter
    :type rho: float
    :param sigma: Volatility parameter
    :type sigma: float
    :return: Increment for stochastic alpha beta rho process
    :rtype: float
    """
    dW = np.random.normal(0, np.sqrt(dt))
    dB = np.random.normal(0, np.sqrt(dt))
    dX = alpha * X * dt + beta * dW + rho * dB
    return dX

@custom_simulate
def MertonJumpDiffusionSimulation(X: float, dt: float, mu: float = 0.1, sigma: float = 0.2, lambda_: float = 0.1, m: float = 0.05, s: float = 0.1) -> float:
    """
    Simulates a Merton jump diffusion process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param mu: Drift parameter
    :type mu: float
    :param sigma: Volatility parameter
    :type sigma: float
    :param lambda_: Jump intensity parameter
    :type lambda_: float
    :param m: Mean of jump
    :type m: float
    :param s: Standard deviation of jump
    :type s: float
    :return: Increment for Merton jump diffusion process
    :rtype: float
    """
    dW = np.random.normal(0, np.sqrt(dt))
    N = np.random.poisson(lambda_ * dt)
    J = np.sum(np.random.normal(m, s, N))
    dX = mu * X * dt + sigma * X * dW + J
    return dX

from ergodicity.process.basic import StandardFractionalBrownianMotion

@custom_simulate
def FractionalBrownianMotionSimulation(X: float, dt: float, mean: float = 0, hurst: float = 0.5) -> float:
    """
    Simulates a fractional Brownian motion process.

    :param X: Process variable
    :type X: float
    :param dt: Time step
    :type dt: float
    :param mean: Mean of the process
    :type mean: float
    :param hurst: Hurst parameter
    :type hurst: float
    :return: Increment for fractional Brownian motion process
    :rtype: float
    """
    FBM = StandardFractionalBrownianMotion(hurst=hurst)
    dW = FBM.increment()
    dX = mean*dt + dt**hurst * dW
    return dX

@custom_simulate
def GeometricFractionalBrownianMotionSimulation(X: float, dt: float, mean: float = 0, hurst: float = 0.5) -> float:
    """
    Simulates a geometric (multiplicative) fractional Brownian motion process.

    :param X: Process variable
    :type X: float
    :param dt: Time stzep
    :type dt: float
    :param mean: Mean of the process
    :type mean: float
    :param hurst: Hurst parameter
    :type hurst: float
    :return: Increment for geometric fractional Brownian motion process
    :rtype: float
    """
    FBM = StandardFractionalBrownianMotion(hurst=hurst)
    dW = FBM.increment()
    dX = mean * X * dt + dt**hurst * X * dW
    return dX
