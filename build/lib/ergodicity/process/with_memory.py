"""
with_memory Submodule

The `With Memory` Submodule focuses on stochastic processes that retain and utilize historical information to influence their future behavior. These processes deviate from classical Markovian models, which rely solely on the current state, by incorporating **memory mechanisms** that adjust their dynamics based on past states or increments. This submodule provides a framework for modeling **non-Markovian processes** with varying types of memory effects.

Key Features:

1. **Non-Markovian Dynamics**:

    - Unlike Markovian processes where future behavior is independent of the past (given the present), processes in this submodule leverage historical data to influence their future states. This makes them suitable for modeling phenomena with long-range dependence or adaptive behavior.

2. **Adaptive Drift and Volatility**:

    - The processes typically feature **adaptive drift** or volatility, which changes based on the process's past trajectory. This allows for more complex and realistic modeling of systems where trends evolve over time, such as financial markets, physical systems, or biological processes.

3. **Memory Update Mechanism**:

    - A core aspect of these processes is the **memory update mechanism**, which adjusts key parameters like drift or volatility based on historical increments or states. This can lead to a variety of interesting behaviors, such as mean reversion, long-term memory, or even self-learning dynamics.

4. **Wide Applications**:

    - Processes with memory are particularly useful in areas where past behavior significantly impacts the future, including:

      - **Financial markets**: Modeling asset prices with trends influenced by historical performance.

      - **Control systems**: Adapting control mechanisms based on past errors or deviations.

      - **Environmental science**: Modeling systems with long-term dependencies, such as climate data.

      - **Machine learning**: Adaptive stochastic optimization methods that incorporate past performance into their future decisions.

## Illustrative Example: Brownian Motion With Mean Memory

The **BrownianMotionWithMeanMemory** class provides a concrete example of a process with memory, where the drift term dynamically adjusts based on the process's history. This process evolves according to the following dynamics:

\[ dX_t = \mu_t dt + \left( \frac{\sigma}{\mu_t} \right) dW_t \]

Where:

- \( \mu_t \) is the **time-varying drift** that updates based on the process's history.

- \( \sigma \) is a **scale parameter** controlling the magnitude of random fluctuations.

- \( W_t \) is a standard **Brownian motion**.

Key Characteristics:

1. **Adaptive Drift**: The drift term \( \mu_t \) is adjusted based on past increments, allowing the process to learn from its own behavior.

2. **Memory Mechanism**: A **memory update function** dynamically modifies the drift using an exponential moving average of the past increments.

3. **Scale Modulation**: The volatility is inversely proportional to the drift, introducing a unique coupling between the random and deterministic parts of the process.

Code Example:

class BrownianMotionWithMeanMemory(NonItoProcess):

    def __init__(self, name: str = "Brownian Motion With Mean Memory", process_class: Type[Any] = None,
                 drift: float = drift_term_default, scale: float = stochastic_term_default):

        super().__init__(name, process_class)

        self._memory = drift

        self._drift = drift

        self._scale = scale if scale > 0 else ValueError("The scale parameter must be positive.")

        self._dx = 0

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:

        dX = timestep * self._drift + (timestep ** 0.5) * self._scale * np.random.normal(0, 1) / self._memory

        self._dx = dX

        return dX

    def memory_update(self, step):

        step += 1

        delta1, delta2 = 1 / step, (step - 1) / step

        new_memory = self._memory * delta2 + delta1 * self._dx

        return new_memory
"""

from ergodicity.process.basic import *
from ergodicity.process.definitions import *
from ergodicity.process.multiplicative import *

class BrownianMotionWithMeanMemory(NonItoProcess):
    """
    BrownianMotionWithMeanMemory is an illustrative example of a process with memory which represents an extension of standard Brownian motion,
    incorporating a dynamic, self-adjusting drift based on the process's history. This continuous-time
    stochastic process, denoted as (X_t)_{t≥0}, evolves according to the following dynamics:

    dX_t = μ_t dt + (σ / μ_t) dW_t

    where:

    - μ_t is the time-varying drift, updated based on the process's history

    - σ is the scale parameter, controlling the magnitude of random fluctuations

    - W_t is a standard Brownian motion

    Key features:

    1. Adaptive Drift: The drift term μ_t is dynamically updated, reflecting the process's mean behavior
       over time. This adaptation allows the process to "learn" from its past trajectory.

    2. Memory Mechanism: The process maintains a memory of its increments, used to adjust the drift.
       This feature introduces a form of long-range dependence not present in standard Brownian motion.

    3. Scale Modulation: The stochastic term is modulated by the inverse of the current drift, creating
       a unique interplay between the deterministic and random components.

    The process is initialized with a name, optional process class, initial drift, and scale parameters.
    It inherits the core functionality of BrownianMotion while implementing custom increment generation
    and memory update mechanisms.

    Key methods:

    1. custom_increment: Generates the next increment of the process, incorporating the memory-adjusted
       drift and scale modulation.

    2. memory_update: Updates the memory (drift) based on the most recent increment, using an exponential
       moving average approach.

    Researchers and practitioners should note several important considerations:

    1. Non-Markovian nature: The dependence on history makes this process non-Markovian, requiring
       specialized analysis techniques.

    2. Parameter sensitivity: The interplay between drift updates and scale modulation can lead to
       complex dynamics, necessitating careful parameter calibration.

    3. Computational considerations: The continuous updating of the drift parameter may increase
       computational overhead in simulations.

    4. Theoretical implications: The process's unique structure may require the development of new
       mathematical tools for rigorous analysis.

    While BrownianMotionWithMeanMemory offers a novel approach to modeling adaptive stochastic processes,
    its use should be carefully considered in the context of specific applications. The memory mechanism
    introduces a form of "learning" into the process, potentially capturing more complex behaviors than
    standard Brownian motion, but also introducing additional complexity in analysis and interpretation.
    """
    def __init__(self, name: str = "Brownian Motion With Mean Memory", process_class: Type[Any] = None,
                 drift: float = drift_term_default, scale: float = stochastic_term_default):
        """
        Initialize the Brownian Motion With Mean Memory class.

        :param name: The name of the process
        :type name: str
        :param process_class: The class of the process
        :type process_class: Type[Any]
        :param drift: The initial drift term
        :type drift: float
        :param scale: The scale parameter for the stochastic term
        :type scale: float
        :raises ValueError: If the scale parameter is non-positive
        """
        # Call the parent class constructor to initialize inherited attributes
        super().__init__(name, process_class)
        self._memory = drift
        self._drift = drift
        if scale <= 0:
            raise ValueError("The scale parameter must be positive.")
        else:
            self._scale = scale
        self._dx = 0

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Generate the next increment of the process, incorporating memory-adjusted drift and scale modulation.

        :param X: The current value of the process
        :type X: float
        :param timestep: The time step for the increment generation
        :type timestep: float
        :return: The next increment of the process
        :rtype: Any
        """
        dX = timestep * self._drift + (timestep ** 0.5) * self._scale * np.random.normal(0, 1)/(self._memory)
        self._dx = dX
        return dX

    def memory_update(self, step):
        """
        Update the memory based on the most recent increment.

        :param step: The current step number
        :type step: int
        :return: The updated memory value
        :rtype: float
        """
        step = step + 1
        delta1 = 1 / step
        delta2 = (step - 1) / step
        new_memory = self._memory * delta2 + delta1 * self._dx
        return new_memory

class OrnsteinUhlenbeckWithAdaptiveRate(NonItoProcess):
    """
    OrnsteinUhlenbeckWithAdaptiveRate is a process with memory that extends the standard Ornstein-Uhlenbeck process.
    It incorporates an adaptive mean-reversion rate based on the process's history. This continuous-time
    stochastic process, denoted as (X_t)_{t≥0}, evolves according to the following dynamics:

    dX_t = θ_t (μ - X_t) dt + σ dW_t

    where:

    - θ_t is the time-varying mean-reversion rate, updated based on the process's history
    - μ is the long-term mean level
    - σ is the volatility parameter
    - W_t is a standard Brownian motion

    Key features:

    1. Adaptive Mean-Reversion: The mean-reversion rate θ_t is dynamically updated, reflecting the
       process's tendency to return to its mean over time. This adaptation allows the process to
       "learn" from its past trajectory and adjust its mean-reversion speed.

    2. Memory Mechanism: The process maintains a memory of its past states, used to adjust the
       mean-reversion rate. This feature introduces a form of long-range dependence not present
       in the standard Ornstein-Uhlenbeck process.

    3. Constant Volatility: Unlike the mean-reversion rate, the volatility σ remains constant,
       providing a stable measure of random fluctuations.

    The process is initialized with a name, optional process class, long-term mean, initial
    mean-reversion rate, and volatility parameters. It inherits the core functionality of NonItoProcess
    while implementing custom increment generation and memory update mechanisms.

    Key methods:

    1. custom_increment: Generates the next increment of the process, incorporating the memory-adjusted
       mean-reversion rate.

    2. memory_update: Updates the memory (mean-reversion rate) based on the most recent state and increment,
       using an exponential moving average approach.

    Researchers and practitioners should note several important considerations:

    1. Non-Markovian nature: The dependence on history makes this process non-Markovian, requiring
       specialized analysis techniques.

    2. Parameter sensitivity: The adaptive mean-reversion rate can lead to complex dynamics,
       necessitating careful parameter calibration.

    3. Computational considerations: The continuous updating of the mean-reversion rate parameter may
       increase computational overhead in simulations.

    4. Theoretical implications: The process's unique structure may require the development of new
       mathematical tools for rigorous analysis, particularly in understanding the long-term behavior
       and stationary distribution (if it exists).

    While OrnsteinUhlenbeckWithAdaptiveRate offers a novel approach to modeling adaptive mean-reverting
    processes, its use should be carefully considered in the context of specific applications. The memory
    mechanism introduces a form of "learning" into the process, potentially capturing more complex behaviors
    than the standard Ornstein-Uhlenbeck process, but also introducing additional complexity in analysis
    and interpretation.
    """
    def __init__(self, name: str = "Ornstein-Uhlenbeck With Adaptive Rate", process_class: Type[Any] = None,
                 mean: float = 0.0, initial_rate: float = 0.1, volatility: float = stochastic_term_default):
        """
        Initialize the Ornstein-Uhlenbeck With Adaptive Rate class.

        :param name: The name of the process
        :type name: str
        :param process_class: The class of the process
        :type process_class: Type[Any]
        :param mean: The long-term mean level
        :type mean: float
        :param initial_rate: The initial mean-reversion rate
        :type initial_rate: float
        :param volatility: The volatility parameter
        :type volatility: float
        :raises ValueError: If the volatility parameter is non-positive
        """
        super().__init__(name, process_class)
        self._mean = mean
        self._memory = {'rate': initial_rate, 'state_history': []}
        if volatility <= 0:
            raise ValueError("The volatility parameter must be positive.")
        else:
            self._volatility = volatility

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Generate the next increment of the process, incorporating memory-adjusted mean-reversion rate.

        :param X: The current value of the process
        :type X: float
        :param timestep: The time step for the increment generation
        :type timestep: float
        :return: The next increment of the process
        :rtype: Any
        """
        dX = self._memory['rate'] * (self._mean - X) * timestep + self._volatility * np.sqrt(timestep) * np.random.normal(0, 1)
        return dX

    def memory_update(self, step):
        """
        Update the memory based on the most recent state and increment.

        :param step: The current step number
        :type step: int
        :return: The updated memory value
        :rtype: dict
        """
        step += 1
        current_state = self._memory['state_history'][-1] if self._memory['state_history'] else self._mean
        delta = 1 / step
        new_rate = self._memory['rate'] * (1 - delta) + delta * abs(self._mean - current_state)
        self._memory['rate'] = new_rate
        self._memory['state_history'].append(current_state)
        if len(self._memory['state_history']) > 100:  # Keep only last 100 states
            self._memory['state_history'] = self._memory['state_history'][-100:]
        return self._memory

class GeometricBrownianMotionWithVolatilityMemory(NonItoProcess):
    """
    GeometricBrownianMotionWithVolatilityMemory is a process with memory that extends the standard Geometric Brownian Motion.
    It incorporates an adaptive volatility parameter based on the process's recent history. This continuous-time
    stochastic process, denoted as (X_t)_{t≥0}, evolves according to the following dynamics:

    dX_t = μ X_t dt + σ_t X_t dW_t

    where:

    - μ is the constant drift parameter
    - σ_t is the time-varying volatility, updated based on the process's recent historical volatility
    - W_t is a standard Brownian motion

    Key features:

    1. Adaptive Volatility: The volatility σ_t is dynamically updated, reflecting the process's recent
       historical volatility. This adaptation allows the process to adjust its randomness based on
       recent market conditions or system behavior.

    2. Memory Mechanism: The process maintains a memory of its recent logarithmic returns, used to
       estimate and adjust the current volatility. This feature introduces a form of short-term
       dependence not present in the standard Geometric Brownian Motion.

    3. Constant Drift: Unlike the volatility, the drift μ remains constant, providing a stable
       long-term growth rate.

    The process is initialized with a name, optional process class, drift, initial volatility, and
    memory length parameters. It inherits the core functionality of NonItoProcess while implementing
    custom increment generation and memory update mechanisms.

    Key methods:

    1. custom_increment: Generates the next increment of the process, incorporating the memory-adjusted
       volatility.

    2. memory_update: Updates the memory (recent returns and current volatility estimate) based on
       the most recent increment, using an exponential moving average approach for volatility estimation.

    Researchers and practitioners should note several important considerations:

    1. Non-Markovian nature: The dependence on recent history makes this process non-Markovian, requiring
       specialized analysis techniques.

    2. Parameter sensitivity: The adaptive volatility can lead to complex dynamics, potentially
       exhibiting volatility clustering similar to observed in financial markets.

    3. Computational considerations: The continuous updating of the volatility parameter and
       maintenance of recent returns may increase computational overhead in simulations.

    4. Theoretical implications: The process's unique structure may require the development of new
       mathematical tools for rigorous analysis, particularly in understanding the long-term behavior
       and moments of the process.

    While GeometricBrownianMotionWithVolatilityMemory offers a novel approach to modeling adaptive volatility
    in growth processes, its use should be carefully considered in the context of specific applications.
    The memory mechanism introduces a form of "market feedback" into the process, potentially capturing
    more realistic behaviors than the standard Geometric Brownian Motion, but also introducing additional
    complexity in analysis and interpretation.
    """

    def __init__(self, name: str = "Geometric Brownian Motion With Volatility Memory", process_class: Type[Any] = None,
                 drift: float = drift_term_default, initial_volatility: float = stochastic_term_default,
                 memory_length: int = 20):
        """
        Initialize the Geometric Brownian Motion With Volatility Memory class.

        :param name: The name of the process
        :type name: str
        :param process_class: The class of the process
        :type process_class: Type[Any]
        :param drift: The constant drift parameter
        :type drift: float
        :param initial_volatility: The initial volatility parameter
        :type initial_volatility: float
        :param memory_length: The number of recent returns to consider for volatility estimation
        :type memory_length: int
        :raises ValueError: If the initial volatility parameter is non-positive or memory_length is less than 2
        """
        super().__init__(name, process_class)
        self._drift = drift
        if initial_volatility <= 0:
            raise ValueError("The initial volatility parameter must be positive.")
        if memory_length < 2:
            raise ValueError("The memory length must be at least 2.")
        self._memory = {
            'volatility': initial_volatility,
            'returns': [],
            'length': memory_length
        }

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Generate the next increment of the process, incorporating memory-adjusted volatility.

        :param X: The current value of the process
        :type X: float
        :param timestep: The time step for the increment generation
        :type timestep: float
        :return: The next increment of the process
        :rtype: Any
        """
        dX = self._drift * X * timestep + self._memory['volatility'] * X * np.sqrt(timestep) * np.random.normal(0, 1)
        return dX

    def memory_update(self, step):
        """
        Update the memory based on the most recent increment, adjusting the volatility estimate.

        :param step: The current step number
        :type step: int
        :return: The updated memory value
        :rtype: dict
        """
        if step > 0:
            current_return = np.log(1 + self._dx / self._X)  # Calculate log return
            self._memory['returns'].append(current_return)

            if len(self._memory['returns']) > self._memory['length']:
                self._memory['returns'].pop(0)  # Remove oldest return if we exceed memory length

            # Update volatility estimate using exponential moving average
            lambda_param = 2 / (len(self._memory['returns']) + 1)  # Smoothing factor
            new_volatility = np.sqrt(np.mean(np.array(self._memory['returns']) ** 2) / timestep)
            self._memory['volatility'] = (1 - lambda_param) * self._memory['volatility'] + lambda_param * new_volatility

        return self._memory

class LevyStableProcessWithMemory(LevyStableProcess):
    """
    LevyStableProcessWithMemory extends the LevyStableProcess by incorporating a memory mechanism
    that adjusts the scale parameter based on recent process behavior. This adaptation allows
    the process to exhibit time-varying volatility while maintaining the heavy-tailed characteristics
    of Lévy stable processes.

    The process evolves according to the following dynamics:

    dX_t = μ dt + σ_t^(1/α) dL_t^(α,β)

    where:
    - μ is the location parameter (drift)
    - σ_t is the time-varying scale parameter, updated based on recent process increments
    - α is the stability parameter (0 < α ≤ 2)
    - β is the skewness parameter (-1 ≤ β ≤ 1)
    - L_t^(α,β) is a standard α-stable Lévy process

    The memory mechanism adjusts σ_t based on an exponential moving average of recent
    absolute increments, allowing the process to adapt its scale to recent volatility levels.

    Key features:
    1. Adaptive Scale: The scale parameter σ_t is dynamically updated, reflecting recent
       volatility in the process.
    2. Memory Mechanism: The process maintains a memory of recent absolute increments,
       used to adjust the scale parameter.
    3. Lévy Stable Properties: Inherits the heavy-tailed and potentially skewed nature of
       Lévy stable processes, while incorporating adaptive behavior.

    This process can be particularly useful in modeling systems with varying levels of
    volatility or risk, such as financial markets during periods of calm and turbulence,
    or physical systems with regime changes in their random fluctuations.
    """

    def __init__(self, name: str = "Levy Stable Process with Memory", process_class: Type[Any] = None,
                 alpha: float = alpha_default, beta: float = beta_default, scale: float = scale_default,
                 loc: float = loc_default, memory_length: int = 20, **kwargs):
        super().__init__(name, process_class, alpha, beta, scale, loc, **kwargs)
        self._memory = {
            'scale': scale,
            'increments': [],
            'length': memory_length
        }

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Generate the next increment of the process, using the memory-adjusted scale parameter.

        :param X: The current value of the process
        :type X: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :return: The increment value
        :rtype: float
        """
        if not self._tempered:
            dX = self._loc * timestep + (timestep ** (1 / self._alpha)) * levy_stable.rvs(
                alpha=self._alpha, beta=self._beta, loc=0, scale=self._memory['scale'])
        else:
            dX = self._loc * timestep + self.tempered_stable_rvs(timestep)

        if self._truncated:
            dX = self.truncate(dX)

        return dX

    def memory_update(self, step):
        """
        Update the memory based on the most recent increment, adjusting the scale parameter.

        :param step: The current step number
        :type step: int
        :return: The updated memory value
        :rtype: dict
        """
        if step > 0:
            # Store the absolute value of the most recent increment
            self._memory['increments'].append(abs(self._dx))

            if len(self._memory['increments']) > self._memory['length']:
                self._memory['increments'].pop(0)  # Remove oldest increment if we exceed memory length

            # Update scale parameter using exponential moving average
            lambda_param = 2 / (len(self._memory['increments']) + 1)  # Smoothing factor
            new_scale = np.mean(self._memory['increments']) / (timestep ** (1 / self._alpha))
            self._memory['scale'] = (1 - lambda_param) * self._memory['scale'] + lambda_param * new_scale

        return self._memory

class LevyStableProcessWithAdaptiveSkewness(LevyStableProcess):
    """
    LevyStableProcessWithAdaptiveSkewness extends the LevyStableProcess by incorporating a memory mechanism
    that adjusts the skewness parameter (beta) based on recent process behavior. This adaptation allows
    the process to exhibit time-varying asymmetry while maintaining the heavy-tailed characteristics
    of Lévy stable processes.

    The process evolves according to the following dynamics:

    dX_t = μ dt + σ^(1/α) dL_t^(α,β_t)

    where:
    - μ is the location parameter (drift)
    - σ is the scale parameter
    - α is the stability parameter (0 < α ≤ 2)
    - β_t is the time-varying skewness parameter (-1 ≤ β_t ≤ 1), updated based on recent process increments
    - L_t^(α,β_t) is a standard α-stable Lévy process with time-varying skewness

    The memory mechanism adjusts β_t based on an exponential moving average of recent
    increments, allowing the process to adapt its asymmetry to recent trends in the data.

    Key features:
    1. Adaptive Skewness: The skewness parameter β_t is dynamically updated, reflecting recent
       trends in the process.
    2. Memory Mechanism: The process maintains a memory of recent increments, used to adjust
       the skewness parameter.
    3. Lévy Stable Properties: Inherits the heavy-tailed nature of Lévy stable processes,
       while incorporating adaptive asymmetry.

    This process can be particularly useful in modeling systems with varying levels of
    asymmetry, such as financial markets during bull and bear periods, or physical systems
    with changing directional biases in their random fluctuations.
    """

    def __init__(self, name: str = "Levy Stable Process with Adaptive Skewness", process_class: Type[Any] = None,
                 alpha: float = alpha_default, beta: float = beta_default, scale: float = scale_default,
                 loc: float = loc_default, memory_length: int = 50, **kwargs):
        super().__init__(name, process_class, alpha, beta, scale, loc, **kwargs)
        self._memory = {
            'beta': beta,
            'increments': [],
            'length': memory_length
        }

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Generate the next increment of the process, using the memory-adjusted skewness parameter.

        :param X: The current value of the process
        :type X: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :return: The increment value
        :rtype: float
        """
        if not self._tempered:
            dX = self._loc * timestep + (timestep ** (1 / self._alpha)) * levy_stable.rvs(
                alpha=self._alpha, beta=self._memory['beta'], loc=0, scale=self._scale)
        else:
            # For tempered version, we use the memory-adjusted beta
            levy_sample = self._loc * timestep + (timestep ** (1 / self._alpha)) * levy_stable.rvs(
                alpha=self._alpha, beta=self._memory['beta'], loc=0, scale=self._scale)
            dX = levy_sample * np.exp(-self._tempering * abs(levy_sample))

        if self._truncated:
            dX = self.truncate(dX)

        return dX

    def memory_update(self, step):
        """
        Update the memory based on the most recent increment, adjusting the skewness parameter.

        :param step: The current step number
        :type step: int
        :return: The updated memory value
        :rtype: dict
        """
        if step > 0:
            # Store the normalized increment
            normalized_increment = self._dx / (self._scale * (timestep ** (1 / self._alpha)))
            self._memory['increments'].append(normalized_increment)

            if len(self._memory['increments']) > self._memory['length']:
                self._memory['increments'].pop(0)  # Remove oldest increment if we exceed memory length

            # Update beta parameter using exponential moving average of the sign of increments
            lambda_param = 2 / (len(self._memory['increments']) + 1)  # Smoothing factor
            avg_sign = np.mean(np.sign(self._memory['increments']))
            new_beta = np.clip(avg_sign, -1, 1)  # Ensure beta stays in [-1, 1]
            self._memory['beta'] = (1 - lambda_param) * self._memory['beta'] + lambda_param * new_beta

        return self._memory

    def differential(self) -> str:
        """
        Express the Levy process with adaptive skewness as a differential equation.

        :return: The differential equation of the process
        :rtype: str
        """
        truncation_str = f" with truncation level {self._truncation_level} ({self._truncation_type})" if self._truncated else ""
        tempering_str = f" tempered by exp(-{self._tempering} * |X|)" if self._tempered else ""
        return f"dX(t) = {self._loc} * dt + dt^(1/{self._alpha}) * {self._scale} * levy_stable({self._alpha}, β_t, 0, 1){tempering_str}{truncation_str}, where β_t is adaptive"

class MultivariateGBMWithAdaptiveDrift(MultivariateGeometricBrownianMotion):
    """
    MultivariateGBMWithAdaptiveDrift extends the MultivariateGeometricBrownianMotion by incorporating
    a memory mechanism that adjusts the drift vector based on recent process behavior. This adaptation
    allows the process to exhibit time-varying growth rates while maintaining the core properties of
    a multivariate geometric Brownian motion.

    The process evolves according to the following dynamics:

    dS_i(t) = μ_i(t) S_i(t) dt + Σ_ij S_i(t) dW_j(t)  for i = 1, ..., n

    where:
    - μ_i(t) is the time-varying drift for the i-th component
    - Σ_ij and W_j(t) remain as in the parent class

    The memory mechanism adjusts μ(t) based on an exponential moving average of recent returns,
    allowing the process to adapt its growth rates to recent trends in the data.

    Key features:
    1. Adaptive Drift: The drift vector μ(t) is dynamically updated, reflecting recent
       growth patterns in each component of the process.
    2. Memory Mechanism: The process maintains a memory of recent returns for each component,
       used to adjust the drift vector.
    3. Preserved Correlation Structure: The scale matrix Σ remains constant, preserving the
       correlation structure between components while allowing for adaptive growth rates.
    """

    def __init__(self, name: str = "Multivariate GBM with Adaptive Drift",
                 drift: List[float] = mean_list_default,
                 scale: List[List[float]] = variance_matrix_default,
                 memory_length: int = 50):
        super().__init__(name, drift, scale)
        self._memory = {
            'drift': np.array(drift),
            'returns': [],
            'length': memory_length
        }

    def custom_increment(self, X: List[float], timestep: float = timestep_default) -> Any:
        dW = np.random.multivariate_normal(mean=[0] * len(self._drift), cov=self._scale * timestep)
        dX = np.array(X) * timestep * self._memory['drift'] + np.array(X) * dW
        return dX

    def memory_update(self, step):
        if step > 0:
            returns = np.log(self._X / self._X_prev)
            self._memory['returns'].append(returns)

            if len(self._memory['returns']) > self._memory['length']:
                self._memory['returns'].pop(0)

            lambda_param = 2 / (len(self._memory['returns']) + 1)
            avg_returns = np.mean(self._memory['returns'], axis=0)
            new_drift = avg_returns / timestep_default
            self._memory['drift'] = (1 - lambda_param) * self._memory['drift'] + lambda_param * new_drift

        self._X_prev = np.copy(self._X)
        return self._memory


class MultivariateGBMWithAdaptiveCorrelation(MultivariateGeometricBrownianMotion):
    """
    MultivariateGBMWithAdaptiveCorrelation extends the MultivariateGeometricBrownianMotion by incorporating
    a memory mechanism that adjusts the correlation structure (and thus the scale matrix) based on recent
    process behavior. This adaptation allows the process to exhibit time-varying correlations between
    components while maintaining the core properties of a multivariate geometric Brownian motion.

    The process evolves according to the following dynamics:

    dS_i(t) = μ_i S_i(t) dt + Σ_ij(t) S_i(t) dW_j(t)  for i = 1, ..., n

    where:
    - μ_i remains constant as in the parent class
    - Σ_ij(t) is the time-varying scale matrix, reflecting changing correlations
    - W_j(t) remains as in the parent class

    The memory mechanism adjusts Σ(t) based on an exponential moving average of recent cross-products
    of returns, allowing the process to adapt its correlation structure to recent patterns in the data.

    Key features:
    1. Adaptive Correlation: The scale matrix Σ(t) is dynamically updated, reflecting recent
       correlation patterns between components of the process.
    2. Memory Mechanism: The process maintains a memory of recent returns for each component,
       used to adjust the scale matrix.
    3. Preserved Drift: The drift vector μ remains constant, preserving the average growth rates
       while allowing for adaptive correlations.
    """

    def __init__(self, name: str = "Multivariate GBM with Adaptive Correlation",
                 drift: List[float] = mean_list_default,
                 scale: List[List[float]] = variance_matrix_default,
                 memory_length: int = 50):
        super().__init__(name, drift, scale)
        self._memory = {
            'scale': np.array(scale),
            'returns': [],
            'length': memory_length
        }

    def custom_increment(self, X: List[float], timestep: float = timestep_default) -> Any:
        dW = np.random.multivariate_normal(mean=[0] * len(self._drift), cov=self._memory['scale'] * timestep)
        dX = np.array(X) * timestep * np.array(self._drift) + np.array(X) * dW
        return dX

    def memory_update(self, step):
        if step > 0:
            returns = np.log(self._X / self._X_prev)
            self._memory['returns'].append(returns)

            if len(self._memory['returns']) > self._memory['length']:
                self._memory['returns'].pop(0)

            lambda_param = 2 / (len(self._memory['returns']) + 1)
            returns_array = np.array(self._memory['returns'])
            new_cov = np.cov(returns_array.T)
            self._memory['scale'] = (1 - lambda_param) * self._memory['scale'] + lambda_param * new_cov

        self._X_prev = np.copy(self._X)
        return self._memory

class MultivariateGeometricLevyWithAdaptiveAlpha(MultivariateGeometricLevy):
    """
    MultivariateGeometricLevyWithAdaptiveAlpha extends the MultivariateGeometricLevy by incorporating
    a memory mechanism that adjusts the stability parameter (alpha) based on recent process behavior.
    This adaptation allows the process to exhibit time-varying tail behavior while maintaining the
    core properties of a multivariate geometric Lévy process.

    The process evolves similarly to its parent class, but with a time-varying alpha:

    S_i(t) = S_i(0) * exp(X_i(t))  for i = 1, ..., n

    where X(t) = (X_1(t), ..., X_n(t)) is a multivariate Lévy stable process with time-varying α(t).

    Key features:
    1. Adaptive Stability: The stability parameter α(t) is dynamically updated, reflecting recent
       extreme value behavior in the process.
    2. Memory Mechanism: The process maintains a memory of recent increments, used to adjust the
       stability parameter.
    3. Preserved Correlation Structure: Other parameters (β, scale, correlation) remain constant,
       preserving the overall structure while allowing for adaptive tail behavior.
    """

    def __init__(self, name: str = "Multivariate Geometric Levy with Adaptive Alpha",
                 alpha: float = 1.5, beta: float = 0, scale: float = 1,
                 loc: np.ndarray = None, correlation_matrix: np.ndarray = None,
                 pseudovariances: np.ndarray = None, memory_length: int = 50):
        super().__init__(name, alpha, beta, scale, loc, correlation_matrix, pseudovariances)
        self._memory = {
            'alpha': alpha,
            'increments': [],
            'length': memory_length
        }

    def custom_increment(self, X: np.ndarray, timestep: float = 1.0) -> np.ndarray:
        # Generate Levy increments with adaptive alpha
        dL = levy_stable.rvs(alpha=self._memory['alpha'], beta=self._beta,
                             loc=0, scale=self._scale, size=self._dims)
        dL = np.dot(self._A, dL.reshape(-1, 1)).flatten() * timestep ** (1 / self._memory['alpha'])

        drift_term = self._loc * timestep
        variance_term = 0.5 * np.diag(np.dot(self._A, self._A.T)) * timestep

        dX = X * np.exp(drift_term - variance_term + dL) - X
        return dX

    def memory_update(self, step):
        if step > 0:
            increment = np.log(self._X / self._X_prev)
            self._memory['increments'].append(increment)

            if len(self._memory['increments']) > self._memory['length']:
                self._memory['increments'].pop(0)

            # Estimate new alpha based on the kurtosis of recent increments
            increments_array = np.array(self._memory['increments'])
            kurtosis = np.mean(stats.kurtosis(increments_array, axis=0))
            new_alpha = min(2, max(0.1, 2 / (1 + np.sqrt(kurtosis / 3))))

            lambda_param = 2 / (len(self._memory['increments']) + 1)
            self._memory['alpha'] = (1 - lambda_param) * self._memory['alpha'] + lambda_param * new_alpha

        self._X_prev = np.copy(self._X)
        return self._memory


class MultivariateGeometricLevyWithAdaptiveCorrelation(MultivariateGeometricLevy):
    """
    MultivariateGeometricLevyWithAdaptiveCorrelation extends the MultivariateGeometricLevy by incorporating
    a memory mechanism that adjusts the correlation structure based on recent process behavior. This
    adaptation allows the process to exhibit time-varying dependencies between components while
    maintaining the core properties of a multivariate geometric Lévy process.

    The process evolves similarly to its parent class, but with a time-varying correlation structure:

    S_i(t) = S_i(0) * exp(X_i(t))  for i = 1, ..., n

    where X(t) = (X_1(t), ..., X_n(t)) is a multivariate Lévy stable process with time-varying
    correlation matrix R(t).

    Key features:
    1. Adaptive Correlation: The correlation matrix R(t) is dynamically updated, reflecting recent
       dependency patterns between components of the process.
    2. Memory Mechanism: The process maintains a memory of recent increments, used to adjust the
       correlation structure.
    3. Preserved Marginal Behavior: Other parameters (α, β, scale) remain constant, preserving the
       marginal distributions while allowing for adaptive dependencies.
    """

    def __init__(self, name: str = "Multivariate Geometric Levy with Adaptive Correlation",
                 alpha: float = 1.5, beta: float = 0, scale: float = 1,
                 loc: np.ndarray = None, correlation_matrix: np.ndarray = None,
                 pseudovariances: np.ndarray = None, memory_length: int = 50):
        super().__init__(name, alpha, beta, scale, loc, correlation_matrix, pseudovariances)
        self._memory = {
            'correlation': correlation_matrix,
            'increments': [],
            'length': memory_length
        }

    def custom_increment(self, X: np.ndarray, timestep: float = 1.0) -> np.ndarray:
        # Recalculate A matrix based on updated correlation
        L = np.linalg.cholesky(self._memory['correlation'])
        self._A = np.dot(np.diag(np.sqrt(self._pseudovariances)), L)

        dL = levy_stable.rvs(alpha=self._alpha, beta=self._beta,
                             loc=0, scale=self._scale, size=self._dims)
        dL = np.dot(self._A, dL.reshape(-1, 1)).flatten() * timestep ** (1 / self._alpha)

        drift_term = self._loc * timestep
        variance_term = 0.5 * np.diag(np.dot(self._A, self._A.T)) * timestep

        dX = X * np.exp(drift_term - variance_term + dL) - X
        return dX

    def memory_update(self, step):
        if step > 0:
            increment = np.log(self._X / self._X_prev)
            self._memory['increments'].append(increment)

            if len(self._memory['increments']) > self._memory['length']:
                self._memory['increments'].pop(0)

            # Estimate new correlation based on recent increments
            increments_array = np.array(self._memory['increments'])
            new_correlation = np.corrcoef(increments_array.T)

            lambda_param = 2 / (len(self._memory['increments']) + 1)
            self._memory['correlation'] = (1 - lambda_param) * self._memory[
                'correlation'] + lambda_param * new_correlation

        self._X_prev = np.copy(self._X)
        return self._memory

class MultivariateGeometricLevyWithAdaptiveScale(MultivariateGeometricLevy):
    """
    MultivariateGeometricLevyWithAdaptiveScale extends the MultivariateGeometricLevy by incorporating
    a memory mechanism that adjusts the scale parameter based on recent process volatility. This
    adaptation allows the process to exhibit time-varying volatility while maintaining the core
    properties of a multivariate geometric Lévy process.

    The process evolves similarly to its parent class, but with a time-varying scale:

    S_i(t) = S_i(0) * exp(X_i(t))  for i = 1, ..., n

    where X(t) = (X_1(t), ..., X_n(t)) is a multivariate Lévy stable process with time-varying
    scale parameter σ(t).

    Key features:
    1. Adaptive Scale: The scale parameter σ(t) is dynamically updated, reflecting recent
       volatility in the process.
    2. Memory Mechanism: The process maintains a memory of recent absolute increments, used to
       adjust the scale parameter.
    3. Preserved Distribution Shape: Other parameters (α, β, correlation) remain constant,
       preserving the overall shape of the distribution while allowing for adaptive volatility.
    """

    def __init__(self, name: str = "Multivariate Geometric Levy with Adaptive Scale",
                 alpha: float = 1.5, beta: float = 0, scale: float = 1,
                 loc: np.ndarray = None, correlation_matrix: np.ndarray = None,
                 pseudovariances: np.ndarray = None, memory_length: int = 50):
        super().__init__(name, alpha, beta, scale, loc, correlation_matrix, pseudovariances)
        self._memory = {
            'scale': scale,
            'increments': [],
            'length': memory_length
        }

    def custom_increment(self, X: np.ndarray, timestep: float = 1.0) -> np.ndarray:
        dL = levy_stable.rvs(alpha=self._alpha, beta=self._beta,
                             loc=0, scale=self._memory['scale'], size=self._dims)
        dL = np.dot(self._A, dL.reshape(-1, 1)).flatten() * timestep ** (1 / self._alpha)

        drift_term = self._loc * timestep
        variance_term = 0.5 * np.diag(np.dot(self._A, self._A.T)) * timestep

        dX = X * np.exp(drift_term - variance_term + dL) - X
        return dX

    def memory_update(self, step):
        if step > 0:
            increment = np.abs(np.log(self._X / self._X_prev))
            self._memory['increments'].append(increment)

            if len(self._memory['increments']) > self._memory['length']:
                self._memory['increments'].pop(0)

            # Estimate new scale based on recent absolute increments
            new_scale = np.mean(self._memory['increments']) / timestep_default ** (1 / self._alpha)

            lambda_param = 2 / (len(self._memory['increments']) + 1)
            self._memory['scale'] = (1 - lambda_param) * self._memory['scale'] + lambda_param * new_scale

        self._X_prev = np.copy(self._X)
        return self._memory


class MultivariateGeometricLevyWithAdaptiveBeta(MultivariateGeometricLevy):
    """
    MultivariateGeometricLevyWithAdaptiveBeta extends the MultivariateGeometricLevy by incorporating
    a memory mechanism that adjusts the skewness parameter (beta) based on recent process behavior.
    This adaptation allows the process to exhibit time-varying asymmetry while maintaining the core
    properties of a multivariate geometric Lévy process.

    The process evolves similarly to its parent class, but with a time-varying beta:

    S_i(t) = S_i(0) * exp(X_i(t))  for i = 1, ..., n

    where X(t) = (X_1(t), ..., X_n(t)) is a multivariate Lévy stable process with time-varying
    skewness parameter β(t).

    Key features:
    1. Adaptive Skewness: The skewness parameter β(t) is dynamically updated, reflecting recent
       asymmetry in the process increments.
    2. Memory Mechanism: The process maintains a memory of recent increments, used to adjust the
       skewness parameter.
    3. Preserved Stability and Scale: Other parameters (α, σ, correlation) remain constant,
       preserving the overall stability and scale while allowing for adaptive asymmetry.
    """

    def __init__(self, name: str = "Multivariate Geometric Levy with Adaptive Beta",
                 alpha: float = 1.5, beta: float = 0, scale: float = 1,
                 loc: np.ndarray = None, correlation_matrix: np.ndarray = None,
                 pseudovariances: np.ndarray = None, memory_length: int = 50):
        super().__init__(name, alpha, beta, scale, loc, correlation_matrix, pseudovariances)
        self._memory = {
            'beta': beta,
            'increments': [],
            'length': memory_length
        }

    def custom_increment(self, X: np.ndarray, timestep: float = 1.0) -> np.ndarray:
        dL = levy_stable.rvs(alpha=self._alpha, beta=self._memory['beta'],
                             loc=0, scale=self._scale, size=self._dims)
        dL = np.dot(self._A, dL.reshape(-1, 1)).flatten() * timestep ** (1 / self._alpha)

        drift_term = self._loc * timestep
        variance_term = 0.5 * np.diag(np.dot(self._A, self._A.T)) * timestep

        dX = X * np.exp(drift_term - variance_term + dL) - X
        return dX

    def memory_update(self, step):
        if step > 0:
            increment = np.log(self._X / self._X_prev)
            self._memory['increments'].append(increment)

            if len(self._memory['increments']) > self._memory['length']:
                self._memory['increments'].pop(0)

            # Estimate new beta based on the skewness of recent increments
            increments_array = np.array(self._memory['increments'])
            skewness = np.mean(stats.skew(increments_array, axis=0))
            new_beta = np.clip(skewness, -1, 1)  # Ensure beta stays in [-1, 1]

            lambda_param = 2 / (len(self._memory['increments']) + 1)
            self._memory['beta'] = (1 - lambda_param) * self._memory['beta'] + lambda_param * new_beta

        self._X_prev = np.copy(self._X)
        return self._memory
