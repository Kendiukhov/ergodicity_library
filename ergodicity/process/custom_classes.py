"""
custom_classes Submodule

The `Custom Classes` submodule provides a framework for implementing specialized, non-standard stochastic processes. These classes extend the `CustomProcess` class and can be used to represent more complex processes that are not typically covered by standard Ito or Non-Ito frameworks. The submodule encourages the definition of processes with unique dynamics that can be customized to fit specific modeling needs.

Key Features:

1. **User-Defined Stochastic Processes**:

    - This submodule is designed for developers and researchers who need to implement non-standard processes for simulations. Users can create their own stochastic processes by extending the `CustomProcess` base class and implementing a custom increment function that defines the process's dynamics.

2. **Flexibility in Process Definition**:

    - The submodule supports defining processes with state-dependent volatility, complex feedback dynamics, or other advanced stochastic behaviors. It can handle cases where the standard Ito or Non-Ito frameworks may not be sufficient for representing the desired phenomena.

3. **Multiplicative or Additive Processes**:

    - The custom processes can be either multiplicative (changes are proportional to the current value, like in Geometric Brownian Motion) or additive (changes are independent of the current value). This allows for modeling a wide range of phenomena such as financial asset prices, population growth, or physical processes with varying volatility.

Example Class: Constant Elasticity of Variance Process (CEV)

The **Constant Elasticity of Variance Process** (CEV) is a custom stochastic process that extends the Geometric Brownian Motion model by introducing state-dependent volatility. It is particularly useful for modeling phenomena where volatility increases or decreases with the process level. This class serves as an example of how to implement custom processes using this submodule.

Key attributes of the **CEV Process**:

- **State-dependent volatility**: The volatility changes depending on the current level of the process, allowing for more realistic modeling of real-world phenomena.

- **Elasticity parameter**: A crucial parameter that determines how volatility behaves in relation to the process level. It can produce different types of dynamics:

    - γ = 1: Reduces to Geometric Brownian Motion.

    - γ > 1: Volatility increases with the process level (leverage effect).

    - γ < 1: Volatility decreases with the process level (inverse leverage effect).

- **Mean-reversion**: The process has a mean-reverting behavior controlled by the mean reversion rate θ. This feature makes it valuable for modeling financial instruments or other time-dependent quantities that tend to stabilize over time.

Applications of CEV:

1. **Financial Markets**: Used in option pricing models to describe the behavior of asset prices with level-dependent volatility.

2. **Population Dynamics**: Models population growth where randomness depends on the population size.

3. **Physics**: Used to model diffusion in non-homogeneous media, where variance depends on the concentration of the diffusing substance.

Workflow for Creating Custom Processes:

1. **Define Parameters**: The process should have its key parameters (drift, volatility, etc.) defined in the class's constructor.

2. **Override `custom_increment`**: The core of the custom process lies in the `custom_increment` method, which calculates the state changes based on the current process value and other parameters.

3. **Handle Dynamics**: Processes may have state-dependent dynamics, feedback loops, or other complex behaviors that are implemented in this method.

4. **Integrate with Simulation Tools**: Once defined, the custom process can be integrated into the larger simulation framework to model and analyze specific scenarios.
"""

import ergodicity.process.definitions as definitions
from ergodicity.process.default_values import *
from typing import List, Any, Type, Callable
import numpy as np

class ConstantElasticityOfVarianceProcess(definitions.CustomProcess):
    """
    ConstantElasticityOfVarianceProcess (CEV) represents a sophisticated stochastic process that extends
    the concept of geometric Brownian motion by allowing the volatility to depend on the current level
    of the process. This continuous-time process, denoted as (S_t)_{t≥0}, is defined by the stochastic
    differential equation:

    dS_t = θ * μ * S_t * dt + σ * S_t^γ * dW_t

    where:

    - μ: Long-term mean level or drift

    - σ: Volatility scale parameter

    - γ: Elasticity parameter, determining how volatility changes with the process level

    - θ: Mean reversion rate

    - W_t: Standard Brownian motion

    Key parameters:

    1. mu (μ): Influences the overall trend of the process.

    2. sigma (σ): Base level of volatility.

    3. gamma (γ): Elasticity of variance, crucial in determining the process's behavior.

    4. theta (θ): Rate of mean reversion, controlling how quickly the process tends towards its long-term mean.

    Key properties:

    1. State-dependent volatility: Volatility changes with the level of the process, allowing for more
       realistic modeling of various phenomena.

    2. Multiplicative nature: The process is inherently multiplicative, suitable for modeling quantities
       that cannot become negative (e.g., prices, populations).

    3. Flexible behavior: Depending on the value of γ, the process can exhibit different characteristics:

       - γ = 1: Reduces to geometric Brownian motion

       - γ < 1: Volatility decreases as the process level increases (inverse leverage effect)

       - γ > 1: Volatility increases as the process level increases (leverage effect)

    4. Mean reversion: The process tends to revert to a long-term mean level, with the speed determined by θ.

    This implementation extends the CustomProcess class, providing a specialized increment function
    that captures the unique dynamics of the CEV process. The process is explicitly set as multiplicative
    (_multiplicative = True), reflecting its nature in modeling proportional changes.

    Applications span various fields:

    - Financial modeling: Asset prices with state-dependent volatility, particularly useful in option pricing.

    - Population dynamics: Species growth with density-dependent randomness.

    - Physics: Diffusion processes in non-homogeneous media.

    - Economics: Interest rate models with level-dependent volatility.

    Researchers and practitioners should be aware of several important considerations:

    1. Parameter estimation: The interplay between parameters, especially γ and σ, can make estimation challenging.

    2. Numerical stability: Care must be taken in simulation, particularly when γ < 0.5, to avoid numerical issues.

    3. Analytical tractability: Closed-form solutions are not always available, necessitating numerical methods.

    4. Regime-dependent behavior: The process can exhibit significantly different characteristics in different
       ranges, requiring careful interpretation.

    The ConstantElasticityOfVarianceProcess offers a powerful and flexible framework for modeling phenomena
    with state-dependent volatility. Its ability to capture a wide range of behaviors makes it valuable in
    many applications, but also requires careful consideration in parameter selection, simulation techniques,
    and result interpretation. The process's rich dynamics provide opportunities for sophisticated modeling
    but also demand a thorough understanding of its properties and limitations.
    """
    def __init__(self, name = 'CEV Process', mu = 0.1, sigma = 0.5, gamma = 0.5, theta = 0.1):
        """
        Initialize the process with the given parameters.

        :param name: Name of the process
        :type name: str
        :param mu: Long-term mean level or drift
        :type mu: float
        :param sigma: Volatility scale parameter
        :type sigma: float
        :param gamma: Elasticity parameter
        :type gamma: float
        :param theta: Mean reversion rate
        :type theta: float
        """
        super().__init__(name)
        self._mu = mu
        self._sigma = sigma
        self._gamma = gamma
        self._theta = theta
        self._multiplicative = True

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Compute the increment of the process at the given state X and timestep.

        :param X: Current state of the process
        :type X: float
        :param timestep: Timestep for the increment
        :type timestep: float
        :return: Increment of the process at the given state and timestep
        :rtype: Any
        """
        dX = self._theta * self._mu * X * timestep - self._sigma * X ** self._gamma * timestep ** 0.5 * np.random.normal()
        return dX
