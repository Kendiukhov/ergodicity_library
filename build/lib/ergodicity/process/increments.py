"""
increments Submodule

The `Increments Submodule` provides a comprehensive framework for generating and managing increments of different stochastic processes. These increments represent the discrete changes in the state of a process over small time steps, and they are the building blocks for simulating various types of continuous-time stochastic processes. The submodule supports a wide range of processes, including standard Wiener processes, Lévy processes, and Fractional Brownian Motions, each of which can exhibit different types of random behavior.

Key Features:

1. **Wide Range of Increments**:

    - The submodule supports increments for standard Wiener processes (Brownian motion), Lévy stable processes (including special cases like the Cauchy process), and Fractional Brownian Motion (FBM) with different Hurst parameters.

    - Increments are tailored to ensure that they have **unit variance**, making them suitable for integration in stochastic differential equations (SDEs) with appropriate scaling.

2. **Handling Non-Gaussian Processes**:

    - Lévy stable processes, which generalize Brownian motion by allowing heavy-tailed distributions and non-Gaussian behavior, are included in this submodule. These processes can be used to model extreme events or phenomena with jumps, offering a more flexible framework than Gaussian-based processes.

    - Special cases like the **Cauchy Process** (Lévy stable with α = 1) are explicitly supported.

3. **Flexible Scaling and Time-Stepping**:

    - Each increment takes into account the **timestep** of the process, ensuring that the increments are appropriately scaled. This allows users to simulate processes over arbitrary time intervals while maintaining correct statistical properties.

    - Time-stepping is critical, especially for non-independent processes like **Fractional Brownian Motion**, where increments at different times can be correlated depending on the Hurst parameter.

4. **Variance-Controlled Increments**:

    - All increments are scaled to have a variance of 1 by default, making them suitable for various applications, including SDEs and Monte Carlo simulations, without requiring further normalization.

    - This ensures consistency across different processes and simplifies the simulation of processes with different characteristics.

Increments Overview:

1. **Wiener Process (Brownian Motion)**:

    - A standard Wiener process, also known as Brownian motion, is a continuous-time stochastic process with independent, normally distributed increments. This is the canonical process used in SDEs and financial models.

    - The increments \( dW \) are generated using a standard normal distribution with mean zero and variance equal to the timestep \( dt \).

    WP = WienerProcess()

    dW = WP.increment()
    ```

2. **Lévy Stable Process**:

    - Lévy stable processes generalize Brownian motion by introducing heavy-tailed distributions, characterized by the stability parameter \( \alpha \). For \( \alpha = 2 \), the Lévy process reduces to a standard Wiener process.

    - The submodule provides increments for various values of \( \alpha \), including special cases like the **Cauchy Process** (\( \alpha = 1 \)).

    # Increment for a Lévy process with α = 1.5

    LP15 = LevyStableProcess(alpha=1.5)

    dL_15 = LP15.increment()

    # Cauchy Process (α = 1)

    LPC = LevyStableProcess(alpha=1)

    dCauchy = LPC.increment()

    # Sub-Gaussian Lévy process (α = 0.5)

    LP05 = LevyStableProcess(alpha=0.5)

    dL_05 = LP05.increment()
    ```

3. **Fractional Brownian Motion**:

    - Fractional Brownian Motion (FBM) is a generalization of the Wiener process where the increments are not independent. The degree of correlation between increments is controlled by the **Hurst parameter** \( H \).

    - The submodule provides FBM increments for different values of \( H \), allowing the user to model long-range dependent processes (for \( H > 0.5 \)) or short-range dependent processes (for \( H < 0.5 \)).

    # Fractional Brownian Motion with H = 0.5 (standard Brownian motion)

    FBM = StandardFractionalBrownianMotion(hurst=0.5)

    dFBM = FBM.increment()

    # FBM with Hurst parameter H = 0.2 (short-range dependence)

    FBM02 = StandardFractionalBrownianMotion(hurst=0.2)

    dFBM02 = FBM02.increment()

    # FBM with Hurst parameter H = 0.8 (long-range dependence)

    FBM08 = StandardFractionalBrownianMotion(hurst=0.8)

    dFBM08 = FBM08.increment()
    ```

Ensuring Correct Time-Stepping:

The submodule ensures that the **timestep** used for generating increments matches the timestep of the underlying process. This consistency is critical, especially when simulating processes over non-uniform time grids or with adaptive time-stepping algorithms.

Researching Non-Independent Processes:

The submodule lays the groundwork for producing increments for **non-independent processes**, such as processes with memory or feedback. In particular, the behavior of Fractional Brownian Motion highlights the complexities involved in generating increments for correlated processes. Future extensions will explore other processes with complex dependence structures, such as **Ornstein-Uhlenbeck processes** or **mean-reverting processes** with stochastic volatility.

Usage Example:

```python

from increments import WienerProcess, LevyStableProcess, StandardFractionalBrownianMotion

# Generate Wiener process increment

WP = WienerProcess()

dW = WP.increment()

# Generate Lévy process increment (α = 1.5)

LP15 = LevyStableProcess(alpha=1.5)

dL_15 = LP15.increment()

# Generate Fractional Brownian Motion increment (H = 0.8)

FBM08 = StandardFractionalBrownianMotion(hurst=0.8)

dFBM08 = FBM08.increment()
"""
from dataclasses import dataclass

from .basic import WienerProcess
from .basic import LevyStableProcess
from .basic import StandardFractionalBrownianMotion

# Wiener Process increment
WP = WienerProcess()
WP._increment_process = True
dW = WP.increment()

# # Levy Stable Process increment for alpha = 1.5
LP15 = LevyStableProcess(alpha=1.5, comments=False)
dL_15 = LP15.increment()

# Cauchy increment
LPC = LevyStableProcess(alpha=1, comments=False)
dL_1 = LPC.increment()
dCauchy = dL_1

# Levy Process increment
LPL = LevyStableProcess(alpha=0.5, beta=1, comments=False)
dLP = LPL.increment()

# Levy Stable Process increment for alpha = 0.5
LP05 = LevyStableProcess(alpha=0.5)
dL_05 = LP05.increment()

def dependent_increments():
    # Fractional Brownian Motion increment
    FBM = StandardFractionalBrownianMotion(hurst=0.5)
    dFBM = FBM.increment()

    FBM02 = StandardFractionalBrownianMotion(hurst=0.2)
    dFBM02 = FBM02.increment()

    FBM08 = StandardFractionalBrownianMotion(hurst=0.8)
    dFBM08 = FBM08.increment()

    return dFBM, dFBM02, dFBM08

# Add increments for all possible probability distributions. Provide increments in such form that they have variance 1.
# Check that timesteps in increments are equal to the timesteps in the process.
# Research how to produce increments for non-independent processes.

