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
        # print(f'memory: {self._memory}')
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
