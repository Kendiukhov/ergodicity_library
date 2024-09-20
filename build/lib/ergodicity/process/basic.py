"""
basic.py

This module provides foundational stochastic processes used in simulations, particularly focusing on
Itô and non-Itô processes. These processes are employed to model various types of continuous-time
stochastic behaviors, including Brownian motion, Bessel processes, and more complex, specialized
stochastic processes like the Brownian bridge, Brownian meander, and fractional Brownian motion.

Classes:

    - EmptyProcess: A process that remains constant at 0 or 1, used for placeholder or null-action purposes.

    - StandardBesselProcess: Represents a standard Bessel process, modeling the Euclidean distance of a Brownian motion from the origin.

    - StandardBrownianBridge: Models a Brownian motion constrained to start and end at specified points over a fixed time interval.

    - StandardBrownianExcursion: Models a Brownian motion conditioned to stay positive and to start and end at zero over a fixed time interval.

    - StandardBrownianMeander: Models a Brownian motion conditioned to stay positive with an unconstrained endpoint.

    - BrownianMotion: Represents standard Brownian motion (Wiener process), a fundamental stochastic process in various fields.

    - CauchyProcess: Models random motion with heavy-tailed distributions following a Cauchy distribution.

    - StandardFractionalBrownianMotion: Models fractional Brownian motion with long-range dependence and self-similarity, governed by the Hurst parameter.

    - FractionalBrownianMotion: Extends the fractional Brownian motion to include a deterministic trend (mean).

    - GammaProcess: Models a process with independent, stationary increments following a gamma distribution.

    - InverseGaussianProcess: Models independent, stationary increments following an inverse Gaussian distribution.

    - StandardMultifractionalBrownianMotion: Represents a multifractional Brownian motion with a time-varying Hurst parameter.

    - SquaredBesselProcess: Models the square of the Euclidean norm of a d-dimensional Brownian motion.

    - VarianceGammaProcess: Represents a variance-gamma process with a mix of Gaussian and gamma process characteristics.

    - WienerProcess: A standard implementation of Brownian motion, a cornerstone of stochastic models.

    - PoissonProcess: Models the occurrence of random events at a constant average rate, a pure jump process.

    - LevyStableProcess: Generalizes the Gaussian distribution to allow for heavy tails and skewness.

    - LevyStableStandardProcess: A standardized version of the Lévy stable process.

    - MultivariateBrownianMotion: Models correlated Brownian motion in multiple dimensions.

    - MultivariateLevy: Extends the Lévy stable process to multiple dimensions, allowing for complex, correlated phenomena.

    - GeneralizedHyperbolicProcess: A versatile process encompassing a wide range of distributions like variance-gamma and normal-inverse Gaussian.

    - ParetoProcess: Represents a process based on the Pareto distribution, modeling heavy-tailed phenomena.

Dependencies:

    - math, numpy, matplotlib, plotly: Libraries used for mathematical operations and visualization.

    - scipy.stats: Statistical functions used to model different distributions.

    - stochastic: Provides the base stochastic processes extended by this module.

    - aiohttp.client_exceptions: Used for exception handling in certain client processes.

This module is essential for defining different stochastic processes used throughout the library, including basic and advanced processes for financial modeling, physics, biology, and more.
"""

import math
from typing import List, Any, Type, Callable, Union

from aiohttp.client_exceptions import ssl_error_bases

from .definitions import ItoProcess
from .definitions import NonItoProcess
from .definitions import Process
from .default_values import *
import numpy as np
from .definitions import simulation_decorator
from ergodicity.configurations import *
import os
import plotly.graph_objects as go
from scipy.stats import norm
import matplotlib.pyplot as plt
from ergodicity.custom_warnings import *
from ..tools.helper import covariance_to_correlation


class EmptyProcess(ItoProcess):
    """
    A process that is always zero or one.
    It may be used as a placeholder, for testing purposes, or often in the agents module as a null-action.

    :param name: The name of the process
    :type name: str
    :param zero_or_one: The value of the process (0 or 1)
    :type zero_or_one: float
    """
    def __init__(self, name: str = "Empty Process", zero_or_one: float = 1):
        """
        Constructor method for the EmptyProcess class.

        :param name: The name of the process
        :type name: str
        :param zero_or_one: The value of the process (0 or 1)
        :type zero_or_one: float
        :raises ValueError: If zero_or_one is not 0 or 1
        """
        super().__init__(name, process_class=None, drift_term=0, stochastic_term=0)
        self.types = ["empty"]
        if zero_or_one not in [0, 1]:
            raise ValueError("zero_or_one must be 0 or 1.")
        self._zero_or_one = zero_or_one
        self._independent = True
        self._multiplicative = False

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Custom increment function for the process.

        :param X: The current value of the process
        :type X: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :return: The increment value, which is always 0 in this case
        :rtype: float
        """
        return 0

    def simulate(self, t: float = t_default, timestep: float = timestep_default, num_instances: int = num_instances_default, save: bool = False, plot: bool = False) -> Any:
        """
        Simulate the process.

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
        :return: The simulated data
        :rtype: np.ndarray
        """
        if self._zero_or_one == 0:
            return np.zeros((num_instances, int(t / timestep) + 1))
        else:
            return np.ones((num_instances, int(t / timestep) + 1))

from stochastic.processes.continuous import BesselProcess as StochasticBesselProcess
class StandardBesselProcess(ItoProcess):
    """
    A standard Bessel process.
    A StandardBesselProcess represents a continuous-time stochastic process that models the Euclidean distance of a
    Brownian motion from its starting point. As an Itô process, it follows the rules of stochastic calculus and is
    defined mathematically as R_t = ||B_t||, where (B_t)_{t≥0} is a d-dimensional Brownian motion and ||·|| denotes
    the Euclidean norm. This process is characterized by its dimension (d), which influences its behavior, including
    non-negativity, martingale properties (for d=2), and recurrence/transience (recurrent for d≤2, transient for d>2).
    The Bessel process maintains continuous sample paths and finds applications in mathematical finance for interest
    rate modeling, statistical physics for particle diffusion studies, and probability theory as a fundamental
    process. It is initialized with a name, process class, and dimension, using default drift
    and stochastic terms inherited from the ItoProcess parent class. The StandardBesselProcess is categorized as both
    a "bessel" and "standard" process type, reflecting its nature and standardized implementation.
    """
    def __init__(self, name: str = "Bessel Process", process_class: Type[Any] = StochasticBesselProcess, dim: int = dim_default):
        """
        Constructor method for the StandardBesselProcess class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param dim: The dimension of the Bessel process
        :type dim: int
        """
        super().__init__(name, process_class, drift_term=drift_term_default, stochastic_term=stochastic_term_default)
        self.types = ["bessel", "standard"]
        self._dim = dim

from stochastic.processes.continuous import BrownianBridge as StochasticBrownianBridge
class StandardBrownianBridge(ItoProcess):
    """
    A StandardBrownianBridge represents a continuous-time stochastic process that models a Brownian motion constrained
    to start and end at specified points, typically 0 and b, over a fixed time interval [0, 1]. This process, denoted
    as (B_t)_{0≤t≤1}, is defined by B_t = W_t - tW_1, where (W_t) is a standard Brownian motion. The bridge process
    is characterized by its non-independent increments and its "tied-down" nature at the endpoints. It exhibits
    several key properties: it's a Gaussian process, has continuous sample paths, and maintains a covariance structure
    of min(s,t) - st. The StandardBrownianBridge finds applications in statistical inference, particularly in
    goodness-of-fit tests, as well as in finance for interest rate modeling and in biology for modeling evolutionary
    processes. As an Itô process, it adheres to the principles of stochastic calculus. It is
    initialized with a name, process class, and an endpoint value b, using default drift and stochastic terms
    inherited from the ItoProcess parent class. The StandardBrownianBridge is explicitly categorized as both a
    "bridge" and "standard" process type, reflecting its nature as a standard implementation of the Brownian bridge
    concept. The _independent attribute is set to False, highlighting the process's non-independent increment
    property, which distinguishes it from standard Brownian motion.
    """
    def __init__(self, name: str = "Standard Brownian Bridge", process_class: Type[Any] = StochasticBrownianBridge, b: float = 0.0):
        """
        Constructor method for the StandardBrownianBridge class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param b: The endpoint value for the bridge process
        :type b: float
        """
        self.types = ["bridge", "standard"]
        super().__init__(name, process_class, drift_term=drift_term_default, stochastic_term=stochastic_term_default)
        self._b = b
        self._independent = False

from stochastic.processes.continuous import BrownianExcursion as StochasticBrownianExcursion
class StandardBrownianExcursion(ItoProcess):
    """
    A StandardBrownianExcursion represents a continuous-time stochastic process that models a Brownian motion
    conditioned to be positive and to start and end at zero over a fixed time interval, typically [0, 1]. This
    process, denoted as (E_t)_{0≤t≤1}, can be conceptualized as the absolute value of a Brownian bridge scaled to
    have a maximum of 1. Mathematically, it's related to the Brownian bridge (B_t) by E_t = |B_t| / max(|B_t|).
    The excursion process is characterized by its non-negative paths, non-independent increments, and its constrained
    behavior at the endpoints. It exhibits several key properties: it's a non-Markovian process, has continuous
    sample paths, and its probability density at time t is related to the Airy function. The StandardBrownianExcursion
    finds applications in various fields, including queueing theory, where it models busy periods, in statistical
    physics for studying polymer chains, and in probability theory as a fundamental object related to Brownian motion.
    As an Itô process, it adheres to the principles of stochastic calculus, although its specific dynamics are more
    complex due to its constrained nature. It is initialized with a name and process
    class, using default drift and stochastic terms inherited from the ItoProcess parent class. The
    StandardBrownianExcursion is explicitly categorized as both an "excursion" and "standard" process type, reflecting
    its nature as a standard implementation of the Brownian excursion concept. The _independent attribute is set to
    False, emphasizing the process's non-independent increment property, which is a crucial characteristic
    distinguishing it from standard Brownian motion and highlighting its unique constrained behavior.
    """
    def __init__(self, name: str = "Standard Brownian Excursion", process_class: Type[Any] = StochasticBrownianExcursion):
        """
        Constructor method for the StandardBrownianExcursion class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        """
        self.types = ["excursion", "standard"]
        super().__init__(name, process_class, drift_term=drift_term_default, stochastic_term=stochastic_term_default)
        self._independent = False

from stochastic.processes.continuous import BrownianMeander as StochasticBrownianMeander
class StandardBrownianMeander(ItoProcess):
    """
    A StandardBrownianMeander represents a continuous-time stochastic process that models a Brownian motion
    conditioned to stay positive over a fixed time interval, typically [0, 1], with an unconstrained endpoint.
    This process, denoted as (M_t)_{0≤t≤1}, can be constructed from a standard Brownian motion (B_t) by
    M_t = |B_t| / √(1-t) for 0 ≤ t < 1, with a specific limiting distribution at t = 1. The meander process
    is characterized by its non-negative paths, non-independent increments, and its free endpoint behavior.
    It exhibits several key properties: it's a non-Markovian process, has continuous sample paths, and its
    transition probability density is related to the heat kernel on the half-line with absorbing boundary
    conditions. The StandardBrownianMeander finds applications in various fields, including queuing theory
    for modeling busy periods with unfinished work, in financial mathematics for studying asset prices
    conditioned on positivity, and in probability theory as a fundamental object related to Brownian motion
    and its local time. As an Itô process, it adheres to the principles of stochastic calculus, although its
    specific dynamics are more complex due to its constrained nature. It is initialized with a name and process
    class, using default drift and stochastic terms inherited from the ItoProcess parent class. The
    StandardBrownianMeander is explicitly categorized as both a "meander" and "standard" process type,
    reflecting its nature as a standard implementation of the Brownian meander concept. The _independent
    attribute is set to False, emphasizing the process's non-independent increment property, which is a
    crucial characteristic distinguishing it from standard Brownian motion and highlighting its unique
    constrained behavior while allowing for endpoint flexibility.
    """
    def __init__(self, name: str = "Standard Brownian Meander", process_class: Type[Any] = StochasticBrownianMeander):
        """
        Constructor method for the StandardBrownianMeander class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        """
        self.types = ["meander", "standard"]
        super().__init__(name, process_class, drift_term=drift_term_default, stochastic_term=stochastic_term_default)
        self._independent = False
        
from stochastic.processes.continuous import BrownianMotion as StochasticBrownianMotion
class BrownianMotion(ItoProcess):
    """
    BrownianMotion represents a fundamental continuous-time stochastic process, also known as Wiener process,
    which models random motion observed in particles suspended in a fluid. This process, denoted as (W_t)_{t≥0},
    is characterized by its independent increments, continuous paths, and Gaussian distribution. Mathematically,
    for 0 ≤ s < t, the increment W_t - W_s follows a normal distribution N(μ(t-s), σ²(t-s)), where μ is the drift
    and σ is the scale parameter. Key properties include: stationary and independent increments, continuous sample
    paths (almost surely), and self-similarity. The process starts at 0 (W_0 = 0) and has an expected value of
    E[W_t] = μt and variance Var(W_t) = σ²t. As an Itô process, Brownian motion is fundamental in stochastic
    calculus and serves as a building block for more complex stochastic processes. It finds widespread applications
    in various fields, including physics (particle diffusion), finance (stock price modeling), biology (population
    dynamics), and engineering (noise in electronic systems). This implementation allows for both standard
    (μ = 0, σ = 1) and generalized Brownian motion. The class is initialized with customizable drift and scale
    parameters, defaulting to standard values. It's categorized under the "brownian" type, reflecting its nature.
    The _has_wrong_params attribute is set to True, indicating that the parameters might need adjustment or
    special handling in certain contexts, particularly when integrating this process into larger systems or
    when transitioning between different time scales.
    """
    def __init__(self, name: str = "Standard Brownian Motion", process_class: Type[Any] = StochasticBrownianMotion, drift: float = drift_term_default, scale: float = stochastic_term_default):
        """
        Constructor method for the BrownianMotion class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param drift: The drift term for the process
        :type drift: float
        :param scale: The stochastic term for the process
        :type scale: float
        """
        self.types = ["brownian"]
        super().__init__(name, process_class, drift_term=drift, stochastic_term=scale)
        self._drift = drift
        self._scale = scale
        self._drift_term = self._drift
        self._has_wrong_params = True
        if drift == 0 and scale == 1:
            print(f'Congratulations! You have created a standard Brownian motion process (Wiener process).')
            self.add_type("standard")
        self._drift_term_sympy = self._drift
        self._stochastic_term_sympy = self._scale

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Custom increment function for the process.

        :param X: The current value of the process
        :type X: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :return: The increment value
        :rtype: float
        """
        dX = timestep * self._drift + (timestep ** 0.5) * self._scale * np.random.normal(0, 1)
        return dX

from stochastic.processes.continuous import CauchyProcess as StochasticCauchyProcess
class CauchyProcess(NonItoProcess):
    """
    CauchyProcess represents a continuous-time stochastic process that models random motion with heavy-tailed
    distributions. This process, denoted as (C_t)_{t≥0}, is characterized by its stable distribution with index
    α = 1, making it a special case of Lévy processes. Unlike Brownian motion, the Cauchy process has undefined
    moments beyond the first order, including an undefined mean and infinite variance. It exhibits several key
    properties: stationary and independent increments, self-similarity, and sample paths that are continuous but
    highly irregular with frequent large jumps. For any time interval [s, t], the increment C_t - C_s follows a
    Cauchy distribution with location parameter 0 and scale parameter |t-s|. The process is non-Gaussian and
    does not satisfy the conditions of the central limit theorem, leading to its classification as a NonItoProcess.
    CauchyProcess finds applications in various fields, including physics (modeling resonance phenomena), finance
    (risk assessment in markets with extreme events), and signal processing (robust statistical methods). It's
    particularly useful in scenarios where extreme events or outliers play a significant role. This implementation
    is initialized with a name and process class, and is categorized under the "cauchy" type. The lack of defined
    drift and stochastic terms reflects the process's unique nature, where traditional moment-based analysis does
    not apply. Researchers and practitioners should be aware of the challenges in working with Cauchy processes,
    including the inapplicability of standard statistical tools that rely on finite moments.
    """
    def __init__(self, name: str = "Cauchy Process", process_class: Type[Any] = StochasticCauchyProcess):
        """
        Constructor method for the CauchyProcess class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        """
        self.types = ["cauchy"]
        super().__init__(name, process_class)

from stochastic.processes.continuous import FractionalBrownianMotion as StochasticFractionalBrownianMotion
class StandardFractionalBrownianMotion(NonItoProcess):
    """
    StandardFractionalBrownianMotion (fBm) represents a generalization of classical Brownian motion, characterized
    by long-range dependence and self-similarity. This continuous-time Gaussian process, denoted as (B^H_t)_{t≥0},
    is uniquely determined by its Hurst parameter H ∈ (0,1), which governs its correlation structure and path
    properties. The process exhibits several key features: stationary increments, self-similarity with parameter H,
    and long-range dependence for H > 0.5. Its covariance function is given by E[B^H_t B^H_s] = 0.5(|t|^2H + |s|^2H -
    |t-s|^2H). When H = 0.5, fBm reduces to standard Brownian motion; for H > 0.5, it shows persistent behavior,
    while for H < 0.5, it displays anti-persistent behavior. Unlike standard Brownian motion, fBm is not a
    semimartingale for H ≠ 0.5 and thus does not fit into the classical Itô calculus framework, hence its
    classification as a NonItoProcess. The process finds wide applications in various fields: in finance for
    modeling long-term dependencies in asset returns, in network traffic analysis for capturing self-similar
    patterns, in hydrology for studying long-term correlations in river flows, and in biophysics for analyzing
    anomalous diffusion phenomena. This implementation is initialized with a name, process class, and Hurst
    parameter (defaulting to 0.5), and is categorized as both "standard" and "fractional". The _independent
    attribute is set to False, reflecting the process's inherent long-range dependence. Researchers and
    practitioners should be aware of the unique challenges in working with fBm, including the need for specialized
    stochastic calculus tools and careful interpretation of its long-range dependence properties in practical
    applications.
    """
    def __init__(self, name: str = "Standard Fractional Brownian Motion", process_class: Type[Any] = StochasticFractionalBrownianMotion, hurst: float = 0.5):
        """
        Constructor method for the StandardFractionalBrownianMotion class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param hurst: The Hurst parameter for the fractional Brownian motion
        :type hurst: float
        :raises ValueError: If the Hurst parameter is outside the range (0, 1)
        """
        self.types = ['standard', "fractional"]
        super().__init__(name, process_class)
        if 0 < hurst < 1:
            self._hurst = hurst
        else:
            raise ValueError("The Hurst parameter must be in the range (0, 1).")
        self._independent = False

class FractionalBrownianMotion(StandardFractionalBrownianMotion):
    """
    FractionalBrownianMotion extends the StandardFractionalBrownianMotion, offering a more generalized implementation
    with additional parametric flexibility. This process, denoted as (X^H_t)_{t≥0}, is a continuous-time Gaussian
    process characterized by its Hurst parameter H and a constant mean μ. It is defined as X^H_t = μt + B^H_t, where
    B^H_t is the standard fractional Brownian motion. The process inherits key properties from fBm, including
    self-similarity with parameter H, stationary increments, and long-range dependence for H > 0.5. Its covariance
    structure is given by Cov(X^H_t, X^H_s) = 0.5(|t|^2H + |s|^2H - |t-s|^2H), independent of μ. The mean parameter
    allows for modeling scenarios with deterministic trends superimposed on the fractal behavior of fBm. This
    implementation is particularly useful in fields where both long-term correlations and underlying trends are
    significant, such as in financial econometrics for modeling asset returns with both momentum and mean-reversion,
    in climate science for analyzing temperature anomalies with long-term trends, and in telecommunications for
    studying network traffic with evolving baselines. The class is initialized with a name, optional process class,
    mean (defaulting to the standard drift term), and Hurst parameter (defaulting to a predefined value). It's
    categorized specifically under the "fractional" type, emphasizing its nature as a fractional process. Researchers
    and practitioners should note that while the added mean parameter enhances modeling flexibility, it does not
    affect the fundamental fractal properties governed by the Hurst parameter. Care should be taken in estimation
    and interpretation, particularly in distinguishing between the effects of the mean trend and the intrinsic
    long-range dependence of the process.
    """
    def __init__(self, name: str = "Fractional Brownian Motion", process_class: Type[Any] = None, mean: float = drift_term_default, scale: float = stochastic_term_default, hurst: float = hurst_default):
        """
        Constructor method for the FractionalBrownianMotion class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param mean: The mean value for the process
        :type mean: float
        :param scale: The scale parameter for the process
        :type scale: float
        :param hurst: The Hurst parameter for the fractional Brownian motion
        :type hurst: float
        :raises ValueError: If the scale parameter is negative
        """
        super().__init__(name, process_class, hurst)
        self.types = ["fractional"]
        self._mean = mean
        if scale >= 0:
            self._scale = scale
        else:
            raise ValueError("The scale parameter must be non-negative.")

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Custom increment function for the process.

        :param X: The current value of the process
        :type X: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :return: The increment value
        :rtype: float
        """
        FBM = StandardFractionalBrownianMotion(hurst=self._hurst)
        dW = FBM.increment(timestep_increment=timestep)
        dX = self._mean * timestep + (timestep ** self._hurst) * dW
        return dX

from stochastic.processes.continuous import GammaProcess as StochasticGammaProcess
class GammaProcess(NonItoProcess):
    """
    GammaProcess represents a continuous-time stochastic process with independent, stationary increments following
    a gamma distribution. This process, denoted as (G_t)_{t≥0}, is characterized by its rate parameter α > 0 and
    scale parameter θ > 0. For any time interval [s, t], the increment G_t - G_s follows a gamma distribution with
    shape α(t-s) and scale θ. Key properties include: strictly increasing sample paths (making it suitable for
    modeling cumulative processes), infinite divisibility, and self-similarity. The process has expected value
    E[G_t] = αθt and variance Var(G_t) = αθ²t. As a Lévy process, it possesses jumps and is not a semimartingale,
    hence its classification as a NonItoProcess. GammaProcess finds diverse applications: in finance for modeling
    aggregate claims in insurance or cumulative losses, in reliability theory for describing degradation processes,
    and in physics for studying certain types of particle emissions. It's particularly useful in scenarios requiring
    non-negative, increasing processes with possible jumps. This implementation is initialized with a name, process
    class, rate (α, defaulting to 1.0), and scale (θ, defaulting to a predefined stochastic term). It's categorized
    under the "gamma" type. The separate rate and scale parameters offer flexibility in modeling, allowing for
    fine-tuning of both the frequency (via rate) and magnitude (via scale) of the increments. Practitioners should
    be aware of the process's non-Gaussian nature and the implications for statistical analysis and risk assessment,
    particularly in heavy-tailed scenarios where the gamma process can provide a more realistic model than
    Gaussian-based alternatives.
    """
    def __init__(self, name: str = "Gamma Process", process_class: Type[Any] = StochasticGammaProcess, rate: float = 1.0, scale: float = stochastic_term_default):
        """
        Constructor method for the GammaProcess class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param rate: The rate parameter for the gamma process
        :type rate: float
        :param scale: The scale parameter for the gamma process
        :type scale: float
        :raises ValueError: If the rate parameter is non-positive
        :raises ValueError: If the scale parameter is negative
        """
        self.types = ["gamma"]
        super().__init__(name, process_class)
        if rate <= 0:
            raise ValueError("The rate parameter must be positive.")
        else:
            self._rate = rate
        if scale >= 0:
            self._scale = scale
        else:
            raise ValueError("The scale parameter must be non-negative.")

from stochastic.processes.continuous import InverseGaussianProcess as StochasticInverseGaussianProcess
class InverseGaussianProcess(NonItoProcess):
    """
    InverseGaussianProcess represents a continuous-time stochastic process with independent, stationary increments
    following an inverse Gaussian distribution. This process, denoted as (IG_t)_{t≥0}, is characterized by its
    mean function μ(t) and scale parameter λ > 0. For any time interval [s, t], the increment IG_t - IG_s follows
    an inverse Gaussian distribution with mean μ(t) - μ(s) and shape parameter λ(t-s). Key properties include:
    strictly increasing sample paths, infinite divisibility, and a more complex self-similarity structure compared
    to the Gamma process. The process has expected value E[IG_t] = μ(t) and variance Var(IG_t) = μ(t)³/λ. As a
    Lévy process, it exhibits jumps and is not a semimartingale, hence its classification as a NonItoProcess.
    InverseGaussianProcess finds applications in various fields: in finance for modeling first passage times in
    diffusion processes or asset returns with asymmetric distributions, in hydrology for describing particle
    transport in porous media, and in reliability theory for modeling degradation processes with a natural barrier.
    It's particularly useful in scenarios requiring non-negative, increasing processes with possible jumps and
    where the relationship between mean and variance is non-linear. This implementation is initialized with a name,
    process class, mean function (defaulting to the identity function λ(t) = t), and scale parameter (defaulting
    to a predefined stochastic term). It's categorized under both "inverse" and "gaussian" types, reflecting its
    nature as an inverse Gaussian process. The flexible mean function allows for modeling time-varying trends,
    while the scale parameter controls the variability of the process. Practitioners should be aware of the
    process's unique distributional properties, particularly its skewness and heavy right tail, which can be
    advantageous in modeling phenomena with occasional large positive deviations.
    """
    def __init__(self, name: str = "Inverse Gaussian Process", process_class: Type[Any] = StochasticInverseGaussianProcess, mean: Callable[[float], float] = lambda t: t, scale: float = stochastic_term_default):
        """
        Constructor method for the InverseGaussianProcess class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param mean: The mean function for the process
        :type mean: Callable[[float], float]
        :param scale: The scale parameter for the process
        :type scale: float
        :raises ValueError: If the scale parameter is negative
        """
        super().__init__(name, process_class)
        self._mean = mean
        if scale >= 0:
            self._scale = scale
        else:
            raise ValueError("The scale parameter must be non-negative.")
        self.types = ["inverse", "gaussian"]

from stochastic.processes.continuous import MultifractionalBrownianMotion as StochasticMultifractionalBrownianMotion
class StandardMultifractionalBrownianMotion(NonItoProcess):
    """
    StandardMultifractionalBrownianMotion (mBm) represents a generalization of fractional Brownian motion, allowing
    for a time-varying Hurst parameter. This continuous-time Gaussian process, denoted as (B^H(t)_t)_{t≥0}, is
    characterized by its Hurst function H(t) : [0,∞) → (0,1), which governs its local regularity and correlation
    structure. The process exhibits several key features: non-stationary increments, local self-similarity, and
    variable long-range dependence. Its covariance structure is complex, approximated by E[B^H(t)_t B^H(s)_s] ≈
    0.5(t^(H(t)+H(s)) + s^(H(t)+H(s)) - |t-s|^(H(t)+H(s))). When H(t) is constant, mBm reduces to fractional
    Brownian motion. The process allows for modeling phenomena with time-varying fractal behavior, where the local
    regularity evolves over time. As a non-stationary and generally non-Markovian process, it is classified as a
    NonItoProcess, requiring specialized stochastic calculus techniques. StandardMultifractionalBrownianMotion
    finds wide applications in various fields: in finance for modeling assets with time-varying volatility and
    long-range dependence, in image processing for texture analysis with varying local regularity, in geophysics
    for studying seismic data with evolving fractal characteristics, and in network traffic analysis for capturing
    time-dependent self-similar patterns. This implementation is initialized with a name, process class, and a
    Hurst function (defaulting to a constant function H(t) = 0.5, which corresponds to standard Brownian motion).
    It's categorized under "multifractional", "fractional", "standard", and "brownian" types, reflecting its nature
    as a generalized Brownian motion. The _independent attribute is set to False, emphasizing the process's complex
    dependence structure. Researchers and practitioners should be aware of the challenges in working with mBm,
    including parameter estimation of the Hurst function, interpretation of local and global properties, and the
    need for advanced numerical methods for simulation and analysis.
    """
    def __init__(self, name: str = "Multifractional Brownian Motion", process_class: Type[Any] = StochasticMultifractionalBrownianMotion, hurst: Callable[[float], float] = lambda t: 0.5):
        """
        Constructor method for the StandardMultifractionalBrownianMotion class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param hurst: The Hurst parameter function for the multifractional Brownian motion
        :type hurst: Callable[[float], float]
        """
        self.types = ["multifractional", "fractional", 'standard', 'brownian']
        super().__init__(name, process_class)
        self._hurst = hurst
        self._independent = False

from stochastic.processes.continuous import SquaredBesselProcess as StochasticSquaredBesselProcess
class SquaredBesselProcess(ItoProcess):
    """
    SquaredBesselProcess represents a continuous-time stochastic process that models the square of the Euclidean
    norm of a d-dimensional Brownian motion. This process, denoted as (R²_t)_{t≥0}, is characterized by its
    dimension parameter d > 0, which need not be an integer. For a d-dimensional Brownian motion (B_t), the squared
    Bessel process is defined as R²_t = ||B_t||². It satisfies the stochastic differential equation dR²_t = d dt +
    2√(R²_t) dW_t, where W_t is a standard Brownian motion. Key properties include: non-negativity, the dimension
    parameter d determining its behavior (recurrent for 0 < d < 2, transient for d ≥ 2), and its role in the
    Pitman-Yor process for d = 0. The process exhibits different characteristics based on d: for d ≥ 2, it never
    reaches zero; for 0 < d < 2, it touches zero but immediately rebounds; for d = 0, it is absorbed at zero.
    SquaredBesselProcess finds applications in various fields: in financial mathematics for modeling interest rates
    and volatility (particularly in the Cox-Ingersoll-Ross model), in population genetics for describing the
    evolution of genetic diversity, and in queueing theory for analyzing busy periods in certain queue models. As
    an Itô process, it follows the rules of Itô calculus, making it amenable to standard stochastic calculus
    techniques. This implementation is initialized with a name, process class, and dimension parameter (defaulting
    to a predefined value). It's categorized under both "squared" and "bessel" types, reflecting its nature as the
    square of a Bessel process. The drift and stochastic terms are set to default values, with the actual dynamics
    governed by the dimension parameter. Researchers and practitioners should be aware of the process's unique
    properties, particularly its dimension-dependent behavior, which can be crucial in accurately modeling and
    analyzing phenomena in various applications.
    """
    def __init__(self, name: str = "Squared Bessel Process", process_class: Type[Any] = StochasticSquaredBesselProcess, dim: int = dim_default):
        """
        Constructor method for the SquaredBesselProcess class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param dim: The dimension of the squared Bessel process
        :type dim: int
        """
        self.types = ["squared", "bessel"]
        super().__init__(name, process_class, drift_term=drift_term_default, stochastic_term=stochastic_term_default)
        self._dim = dim

from stochastic.processes.continuous import VarianceGammaProcess as StochasticVarianceGammaProcess
class VarianceGammaProcess(NonItoProcess):
    """
    GeneralizedHyperbolicProcess represents a highly flexible class of continuous-time stochastic processes that
    encompasses a wide range of distributions, including normal, Student's t, variance-gamma, and normal-inverse
    Gaussian as special or limiting cases. This process, denoted as (X_t)_{t≥0}, is characterized by five parameters:
    α (tail heaviness), β (asymmetry), μ (location), δ (scale), and λ (a shape parameter, often denoted as 'a' in
    the implementation). The process is defined through its increments, which follow a generalized hyperbolic
    distribution. Key properties include: semi-heavy tails (heavier than Gaussian but lighter than power-law),
    ability to model skewness, and a complex autocorrelation structure. The process allows for both large jumps
    and continuous movements, making it highly adaptable to various phenomena. It's particularly noted for its
    capacity to capture both the central behavior and the extreme events in a unified framework.

    Parameter restrictions are crucial for the proper definition of the process:

    - α > 0: Controls the tail heaviness, with larger values leading to lighter tails.

    - |β| < α: Determines the skewness, with β = 0 yielding symmetric distributions.

    - δ > 0: Acts as a scaling factor.

    - μ can take any real value.

    - λ (if provided via kwargs) can be any real number, affecting the shape of the distribution.

    The GeneralizedHyperbolicProcess finds extensive applications in finance for modeling asset returns, particularly
    in markets exhibiting skewness and kurtosis; in risk management for more accurate tail risk assessment; in
    physics for describing particle movements in heterogeneous media; and in signal processing for modeling
    non-Gaussian noise. This implementation is initialized with parameters α, β, μ (loc), and δ (scale), with
    additional parameters possible through kwargs. It's categorized under both "generalized" and "hyperbolic" types,
    reflecting its nature as a broad, hyperbolic-based process. The class uses a custom increment function,
    indicated by the _external_simulator flag set to False. This allows for precise control over the generation
    of process increments, crucial for accurately representing the complex distribution. Researchers and
    practitioners should be aware of the computational challenges in parameter estimation and simulation,
    particularly in high-dimensional settings or with extreme parameter values. The flexibility of the generalized
    hyperbolic process comes with increased model complexity, requiring careful consideration in application and
    interpretation. Its ability to nest simpler models allows for sophisticated hypothesis testing and model
    selection in empirical studies.
    """
    def __init__(self, name: str = "Variance Gamma Process", process_class: Type[Any] = StochasticVarianceGammaProcess, drift: float = drift_term_default, variance: float = 1, scale: float = stochastic_term_default):
        """
        Constructor method for the VarianceGammaProcess class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param drift: The drift term for the process
        :type drift: float
        :param variance: The variance parameter for the process
        :type variance: float
        :param scale: The scale parameter for the process
        :type scale: float
        :raises ValueError: If the scale parameter is negative
        :raises ValueError: If the variance parameter is non-positive
        """
        self.types = ["variance", "gamma"]
        super().__init__(name, process_class)
        self._drift = drift
        if scale >= 0:
            self._scale = scale
        else:
            raise ValueError("The scale parameter must be non-negative.")
        if variance > 0:
            self._variance = variance
        else:
            raise ValueError("The variance parameter must be positive.")

from stochastic.processes.continuous import WienerProcess as StochasticWienerProcess
class WienerProcess(ItoProcess):
    """
    WienerProcess represents the fundamental continuous-time stochastic process, also known as standard Brownian
    motion, which forms the cornerstone of many stochastic models in science and finance. This process, denoted as
    (W_t)_{t≥0}, is characterized by its properties of independent increments, continuous paths, and Gaussian
    distribution. For any time interval [s,t], the increment W_t - W_s follows a normal distribution N(0, t-s).
    Key properties include: almost surely continuous sample paths, non-differentiability at any point, self-similarity,
    and the strong Markov property. The process starts at 0 (W_0 = 0) and has an expected value of E[W_t] = 0 and
    variance Var(W_t) = t. As the quintessential Itô process, it serves as the building block for more complex
    stochastic differential equations and is central to Itô calculus. WienerProcess finds ubiquitous applications
    across various fields: in physics for modeling Brownian motion and diffusion processes, in financial mathematics
    for describing stock price movements and as a basis for the Black-Scholes model, in signal processing for
    representing white noise, and in control theory for modeling disturbances in dynamical systems. This implementation
    is initialized with a name and process class, with drift term fixed at 0 and stochastic term at 1, adhering to
    the standard definition. It's categorized under both "wiener" and "standard" types, emphasizing its nature as
    the canonical continuous-time stochastic process. The simplicity of its parameter-free definition belies the
    complexity and richness of its behavior, making it a versatile tool in stochastic modeling. Researchers and
    practitioners should be aware of both its power as a modeling tool and its limitations, particularly in capturing
    more complex real-world phenomena that may require extensions or generalizations of the basic Wiener process.
    """
    def __init__(self, name: str = "Wiener Process", process_class: Type[Any] = StochasticWienerProcess):
        """
        Constructor method for the WienerProcess class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        """
        self.types = ["wiener", 'standard']
        super().__init__(name, process_class, drift_term=0, stochastic_term=1)
        self._drift_term_sympy = 0
        self._stochastic_term_sympy = 1

from stochastic.processes.continuous import PoissonProcess as StochasticPoissonProcess
class PoissonProcess(NonItoProcess):
    """
    PoissonProcess represents a fundamental continuous-time stochastic process that models the occurrence of
    random events or arrivals at a constant average rate. This process, denoted as (N_t)_{t≥0}, is characterized
    by its rate parameter λ > 0, which represents the average number of events per unit time. For any time
    interval [s,t], the number of events N_t - N_s follows a Poisson distribution with parameter λ(t-s). Key
    properties include: independent increments, stationary increments, right-continuous step function sample paths,
    and the memoryless property. The process starts at 0 (N_0 = 0) and has an expected value of E[N_t] = λt and
    variance Var(N_t) = λt. As a pure jump process, it is classified as a NonItoProcess, distinct from continuous
    processes like Brownian motion. PoissonProcess finds extensive applications across various fields: in queueing
    theory for modeling arrival processes, in reliability theory for describing failure occurrences, in insurance
    for modeling claim arrivals, in neuroscience for representing neuronal firing patterns, and in physics for
    modeling radioactive decay. This implementation is initialized with a name, process class, and a rate parameter
    (defaulting to 2.0), allowing for flexible modeling of event frequencies. It's categorized under the "poisson"
    type, reflecting its nature as a Poisson process. The simplicity of its single-parameter definition belies its
    powerful modeling capabilities, particularly for discrete events in continuous time. Researchers and
    practitioners should be aware of both its strengths in modeling random occurrences and its limitations,
    such as the assumption of constant rate and independence between events, which may not hold in all real-world
    scenarios. Extensions like non-homogeneous Poisson processes or compound Poisson processes can address some
    of these limitations for more complex modeling needs.
    """
    def __init__(self, name: str = "Poisson Process", process_class: Type[Any] = StochasticPoissonProcess, rate: float = 2.0):
        """
        Constructor method for the PoissonProcess class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param rate: The rate parameter for the Poisson process
        :type rate: float
        :raises ValueError: If the rate parameter is non-positive
        """
        self.types = ["poisson", "jump", "increasing"]
        super().__init__(name, process_class)
        if rate > 0:
            self._rate = rate
        else:
            raise ValueError("The rate parameter must be positive.")

    def simulate(self, t: float = t_default, timestep: float = timestep_default, num_instances: int = num_instances_default, save: bool = False, plot: bool = False) -> Any:
        """
        Simulate the Poisson process, plot, and save it.

        :param t: The time horizon for the simulation
        :type t: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :param num_instances: The number of process instances to simulate
        :type num_instances: int
        :param save: Whether to save the simulation results
        :type save: bool
        :param plot: Whether to plot the simulation results
        :type plot: bool
        :return: The simulated Poisson process dataset as a NumPy array of shape (num_instances+1, num_steps)
        :rtype: Any
        """
        num_steps, times, data = self.data_for_simulation(t, timestep, num_instances)

        process = self._process_class(**self.get_params())
        for i in range(num_instances):
            data[i, :] = process.sample(num_steps - 1)

        data_full = np.concatenate((times.reshape(1, -1), data), axis=0)

        self.plot(data_full, num_instances, save, plot)

        return data_full

from scipy.stats import levy_stable
from scipy.stats import levy_stable
import numpy as np
from .definitions import NonItoProcess

class LevyStableProcess(NonItoProcess):
    """
    LevyStableProcess represents a versatile class of continuous-time stochastic processes that generalize the
    Gaussian distribution to allow for heavy tails and skewness. This process, denoted as (X_t)_{t≥0}, is
    characterized by four parameters: α (stability), β (skewness), σ (scale), and μ (location). The α parameter,
    ranging from 0 to 2, determines the tail heaviness, with α = 2 corresponding to Gaussian behavior. Key
    properties include: stable distribution of increments, self-similarity, and, for α < 2, infinite variance
    and potential for large jumps. The process offers remarkable flexibility, encompassing Gaussian (α = 2),
    Cauchy (α = 1, β = 0), and Lévy (α = 0.5, β = 1) processes as special cases. This implementation extends
    the basic Lévy stable process with options for tempering and truncation, allowing for more nuanced modeling
    of extreme events. Tempering introduces exponential decay in the tails, while truncation (either 'hard' or
    'soft') limits the maximum jump size. These modifications can be crucial in financial modeling to ensure
    finite moments or in physical systems with natural limits. The process finds wide applications in finance
    for modeling asset returns and risk, in physics for describing anomalous diffusion, in telecommunications
    for network traffic analysis, and in geophysics for modeling natural phenomena. The class includes built-in
    validity checks and informative comments about special cases. It allows for scaled parameterization and
    provides methods for generating increments, including tempered and truncated variants. The differential
    and elementary expressions offer insights into the process's structure. Researchers and practitioners
    should be aware of the computational challenges in simulating and estimating Lévy stable processes,
    particularly for small α values, and the interpretative complexities introduced by tempering and truncation.
    This implementation strikes a balance between the theoretical richness of Lévy stable processes and the
    practical needs of modeling real-world phenomena with potentially bounded extreme events.
    """
    def __init__(self, name: str = "Levy Stable Process", process_class: Type[Any] = None, alpha: float = alpha_default, beta: float = beta_default, scale: float = scale_default, loc: float = loc_default, comments: bool = default_comments, tempering: float = 0, truncation_level: float = None, truncation_type: str = 'hard', scaled_scale: bool = False):
        """
        Constructor method for the LevyStableProcess class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param alpha: The stability parameter (0 < α ≤ 2)
        :type alpha: float
        :param beta: The skewness parameter (-1 ≤ β ≤ 1)
        :type beta: float
        :param scale: The scale parameter (σ > 0)
        :type scale: float
        :param loc: The location parameter (μ)
        :type loc: float
        :param default_comments: Whether to display default comments
        :type default_comments: bool
        :param tempering: The tempering parameter (λ)
        :type tempering: float
        :param truncation_level: The truncation level for the process
        :type truncation_level: float
        :param truncation_type: The type of truncation ('hard' or 'soft')
        :type truncation_type: str
        :param scaled_scale: Whether to use the scaled scale parameter. If True, the scale parameter is equivalent to the normal process standard deviation.
        :type scaled_scale: bool
        :raises ValueError: If the stability parameter is outside the range (0, 2]
        :raises ValueError: If the skewness parameter is outside the range [-1, 1]
        :raises ValueError: If the scale parameter is non-positive
        :raises ValueError: If the tempering parameter is negative
        :raises ValueError: If the truncation level is negative
        """
        self.types = ["stable", "levy stable"]
        super().__init__(name, process_class)
        if alpha <= 0 or alpha > 2:
            raise ValueError("The stability parameter alpha must be in the interval (0, 2].")
        else:
            self._alpha = alpha
        if beta < -1 or beta > 1:
            raise ValueError("The skewness parameter beta must be in the interval [-1, 1].")
        else:
            self._beta = beta
        if scale <= 0:
            raise ValueError("The scale parameter sigma must be positive.")
        else:
            self._scale = scale
        self._loc = loc
        self._loc_scaled = loc / 10
        self._comments = comments
        self._external_simulator = False
        if tempering < 0:
            raise ValueError("The tempering parameter must be non-negative.")
        else:
            self._tempering = tempering
        if truncation_level is not None and truncation_level < 0:
            raise ValueError("The truncation level must be non-negative.")
        else:
            self._truncation_level = truncation_level
        self._truncation_type = truncation_type.lower() if truncation_level is not None else None

        # Determine if the process is tempered
        if self._tempering == 0:
            self._tempered = False
        else:
            self._tempered = True

        # Determine if truncation is applied
        self._truncated = truncation_level is not None

        # Set this true to use Wiener process variance equivalent as the scale parameter.
        self._scaled_scale = scaled_scale
        if self._scaled_scale:
            self._scale = (1/2)**0.5 * self._scale**0.5

        # Validity checks
        if self._scale <= 0:
            raise ValueError('Scale parameter must be positive.')
        if self._alpha <= 0 or self._alpha > 2:
            raise ValueError('Alpha parameter must be in the interval (0, 2].')
        if self._truncation_level is not None and self._truncation_level <= 0:
            raise ValueError('Truncation level must be positive.')

        if self._comments:
            self._generate_comments()

    def _generate_comments(self):
        """
        Generate default comments based on the process parameters.
        It provides information on the type of process created.
        This is helpfull if you need to know the specifics of the process you created.

        :return: None
        :rtype: None
        """
        if self._alpha == 2 and self._beta == 0:
            print(f'Congratulations! With your parameters, you created a Brownian Motion Process with drift = {self._loc} and scale = {self._scale * (2 ** 0.5)}.')
            self.add_type('brownian')
            if self._loc == 0:
                print('Even more, you created a Standard Brownian Motion Process!')
                self.add_type('standard')
                if self._scale * (2 ** 0.5) < 1.00001 and self._scale * (2 ** 0.5) > 0.99999:
                    print('Even more, you created a Wiener Process!')
                    self.add_type('wiener')
        elif self._alpha == 1 and self._beta == 0:
            print(f'Congratulations! With your parameters, you created a Cauchy Process with drift = {self._loc} and scale = {self._scale}.')
            self.add_type('cauchy')
        elif self._alpha == 0.5 and self._beta == 1:
            print(f'Congratulations! With your parameters, you created a Levy Process with drift = {self._loc} and scale = {self._scale}.')
            self.add_type('levy')
            if self._loc == 0:
                print(f'Even more, you created a Levy Smirnov process with scale = {self._scale}.')
        elif self._alpha == 1.5 and self._beta == 0:
            print(f'Congratulations! With your parameters, you created a Holtsmark process with drift = {self._loc} and scale = {self._scale}.')
            self.add_type('holtsmark')
        elif self._alpha == 1 and self._beta == 1:
            print(f'Congratulations! With your parameters, you created a Landau process with drift = {self._loc} and scale = {self._scale}.')
            self.add_type('landau')
        elif self._alpha <= 0.01:
            print(f'Congratulations! With your parameters, you created an inverse gamma process approximation with beta = {self._beta}, drift = {self._loc} and scale = {self._scale}.')
            print('This is but an approximation of the inverse gamma process, for the true process is is achieved when alpha approaches zero.')
            self.add_type('inverse gamma')

    @property
    def alpha(self):
        """
        The stability parameter (0 < α ≤ 2).

        :return: The stability parameter
        :rtype: float
        """
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float):
        """
        Setter method for the stability parameter.

        :param alpha: The stability parameter
        :type alpha: float
        :return: None
        :rtype: None
        """
        if alpha <= 0 or alpha > 2:
            raise ValueError("The stability parameter alpha must be in the interval (0, 2].")
        self._alpha = alpha

    @property
    def beta(self):
        """
        The skewness parameter (-1 ≤ β ≤ 1).

        :return: The skewness parameter
        :rtype: float
        """
        return self._beta

    @beta.setter
    def beta(self, beta: float):
        """
        Setter method for the skewness parameter.

        :param beta: The skewness parameter
        :type beta: float
        :return: None
        :rtype: None
        """
        if beta < -1 or beta > 1:
            raise ValueError("The skewness parameter beta must be in the interval [-1, 1].")
        self._beta = beta

    @property
    def scale(self):
        """
        The scale parameter (σ > 0).
        :return: The scale parameter
        :rtype: float
        """
        return self._scale

    @scale.setter
    def scale(self, scale: float):
        """
        Setter method for the scale parameter.

        :param scale: The scale parameter
        :type scale: float
        :return: None
        :rtype: None
        """
        if scale <= 0:
            raise ValueError("The scale parameter sigma must be positive.")
        self._scale = scale

    @property
    def loc(self):
        """
        The location parameter (μ).

        :return: The location parameter
        :rtype: float
        """
        return self._loc

    @loc.setter
    def loc(self, loc: float):
        """
        Setter method for the location parameter.

        :param loc: The location parameter
        :type loc: float
        :return: None
        :rtype: None
        """
        self._loc = loc
        self._loc_scaled = loc / 10

    @property
    def tempering(self):
        """
        The tempering parameter (λ).
        :return: The tempering parameter
        :rtype: float
        """
        return self._tempering

    @tempering.setter
    def tempering(self, tempering: float):
        """
        Setter method for the tempering parameter.

        :param tempering: The tempering parameter
        :type tempering: float
        :return: None
        :rtype: None
        """
        if tempering < 0:
            raise ValueError("The tempering parameter must be non-negative.")
        self._tempering = tempering

    @property
    def truncation_level(self):
        """
        The truncation level for the process.

        :return: The truncation level
        :rtype: float
        """
        return self._truncation_level

    @truncation_level.setter
    def truncation_level(self, truncation_level: float):
        """
        Setter method for the truncation level.

        :param truncation_level: The truncation level
        :type truncation_level: float
        :return: None
        :rtype: None
        """
        if truncation_level < 0:
            raise ValueError("The truncation level must be non-negative.")
        self._truncation_level = truncation_level

    @property
    def truncation_type(self):
        """
        The type of truncation ('hard' or 'soft').

        :return: The type of truncation
        :rtype: str
        """
        return self._truncation_type

    @truncation_type.setter
    def truncation_type(self, truncation_type: str):
        """
        Setter method for the truncation type.

        :param truncation_type: The type of truncation
        :type truncation_type: str
        :return: None
        :rtype: None
        """
        self._truncation_type = truncation_type.lower()

    @property
    def tempered(self):
        """
        Whether the process is tempered.

        :return: Whether the process is tempered
        :rtype: bool
        """
        return self._tempered

    @property
    def truncated(self):
        """
        Whether the process is truncated.

        :return: Whether the process is truncated
        :rtype: bool
        """
        return self._truncated

    @property
    def scaled_scale(self):
        """
        This getter method is needed because the scale parameter has a different scale in the library which is used for the simulation.

        :return: The scaled scale parameter
        :rtype: bool
        """
        return self._scaled_scale

    @scaled_scale.setter
    def scaled_scale(self, scaled_scale: bool):
        """
        This setter method is needed because the scale parameter has a different scale in the library which is used for the simulation.

        :param scaled_scale: The scaled scale parameter
        :type scaled_scale: bool
        :return: None
        :rtype: None
        """
        if scaled_scale is not bool:
            raise ValueError('scaled_scale must be a boolean.')
        self._scaled_scale = scaled_scale

    def tempered_stable_rvs(self, timestep):
        """
        Generate a tempered stable random variable.
        Tempering is applied by multiplying the Levy stable random variable with an exponential factor.
        Tempering parameter must be non-negative.

        :param timestep: The time step for the simulation
        :type timestep: float
        :return: A tempered stable random variable
        :rtype: float
        """
        # Generate a sample from a Levy stable distribution
        levy_sample = self._loc*timestep + (timestep**(1/self._alpha))*levy_stable.rvs(alpha=self._alpha, beta=self._beta, loc=0, scale=self._scale)
        # Apply the exponential tempering
        tempered_sample = levy_sample * np.exp(-self._tempering * abs(levy_sample))
        return tempered_sample

    def truncate(self, value: float) -> float:
        """
        Truncate the value based on the truncation level and type.
        Truncation can be 'hard' (capping the value) or 'soft' (applying exponential tempering).
        Truncation is a common technique to limit the impact of extreme values in a process.
        It may be needed to apply for a Levy process because of its heavy tails, especially for small alpha.
        Truncation parameter must be positive.

        :param value: The value to truncate
        :type value: float
        :return: The truncated value
        :rtype: float
        :raises ValueError: If the truncation type is invalid (not 'hard' or 'soft')
        """
        if self._truncation_type == 'hard':
            # Hard truncation: cap the value at the truncation level
            return np.clip(value, -self._truncation_level, self._truncation_level)
        elif self._truncation_type == 'soft':
            # Soft truncation: apply exponential tempering
            return value * np.exp(-abs(value) / self._truncation_level)
        else:
            raise ValueError("Truncation type must be either 'hard' or 'soft'.")

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Custom increment function for the process.

        :param X: The current value of the process
        :type X: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :return: The increment value
        :rtype: float
        """
        if not self._tempered:
            dX = self._loc*timestep + (timestep ** (1 / self._alpha)) * levy_stable.rvs(alpha=self._alpha, beta=self._beta, loc=0, scale=self._scale)
        else:
            dX = self.tempered_stable_rvs(timestep)

        if self._truncated:
            dX = self.truncate(dX)

        return dX

    def differential(self) -> str:
        """
        Express the Levy process as a differential equation.

        :return: The differential equation of the process
        :rtype: str
        """
        truncation_str = f" with truncation level {self._truncation_level} ({self._truncation_type})" if self._truncated else ""
        tempering_str = f" tempered by exp(-{self._tempering} * |X|)" if self._tempered else ""
        return f"dX(t) = {self._loc} * dt + dt^(1/{self._alpha}) * {self._scale} * levy_stable({self._alpha}, {self._beta}, 0, 1){tempering_str}{truncation_str}"

    def express_as_elementary(self) -> str:
        """
        Express a given Levy process as a function of an elementary Levy process.

        :return: The Levy process expressed in terms of elementary Levy processes
        :rtype: str
        """
        return f"X(t) = {self._loc} * t + {self._scale} * LevyStableProcess(alpha={self._alpha}, beta={self._beta}, truncation={self._truncation_level}, type={self._truncation_type}).increment()"

    def characteristic_function(self) -> str:
        """
        Express the characteristic function of the Lévy stable distribution corresponding to the process.

        :return: The characteristic function of the process
        :rtype: str
        """
        if self._alpha != 1:
            phi = f"tan(pi * {self._alpha} / 2)"
        else:
            phi = f"-(2/pi) * log(|t|)"

        char_function = f"exp(i * {self._loc} * t - |{self._scale} * t|^{self._alpha} * (1 - i * {self._beta} * sign(t) * {phi}))"
        return char_function

class LevyStableStandardProcess(LevyStableProcess):
    """
    LevyStableStandardProcess represents a standardized version of the Lévy stable process, a class of
    continuous-time stochastic processes known for their ability to model heavy-tailed distributions and
    asymmetry. This process, denoted as (X_t)_{t≥0}, is a special case of the general Lévy stable process
    with fixed scale and location parameters (set to 1/2**0.5 and 0 correspondingly). It is primarily characterized by two parameters:
    Scale is set to 1/2**0.5 because it corresponds to the standard deviation of the process when the process is Gaussian.

    1. α (alpha): The stability parameter, where 0 < α ≤ 2. This parameter determines the tail heaviness
       of the distribution. As α approaches 2, the process behaves more like Brownian motion, while
       smaller values lead to heavier tails and more extreme jumps.

    2. β (beta): The skewness parameter, where -1 ≤ β ≤ 1. This parameter controls the asymmetry of the
       distribution. When β = 0, the process is symmetric.

    The process is standardized with a scale parameter of 1/√2 and a location parameter of 0. This
    standardization allows for easier comparison and analysis across different α and β combinations.

    Key properties of the LevyStableStandardProcess include:

    - Self-similarity: The distribution of the process at any time t is the same as that at time 1,
      up to a scaling factor.

    - Stable distributions: The sum of independent copies of the process follows the same distribution,
      up to scaling and shifting.

    - Potential for infinite variance: For α < 2, the process has infinite variance, capturing extreme
      events more effectively than Gaussian processes.

    The 'standard' type is added to the process classification.
    This allows for more straightforward theoretical analysis and comparison between different
    parameterizations of the Lévy stable family.

    Researchers and practitioners should be aware that while this standardized form offers analytical
    advantages, it may require rescaling and shifting for practical applications. The process's rich
    behavior, especially for α < 2, necessitates careful interpretation and often specialized numerical
    methods for simulation and statistical inference.
    """
    def __init__(self, name: str = "Standard Levy Stable Process", process_class: Type[Any] = None, alpha: float = 1, beta: float = 0):
        """
        Constructor method for the LevyStableStandardProcess class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param alpha: The stability parameter (0 < α ≤ 2)
        :type alpha: float
        :param beta: The skewness parameter (-1 ≤ β ≤ 1)
        :type beta: float
        """
        super().__init__(name, process_class, alpha, beta, 1/2**0.5, 0)
        self.add_type('standard')

# https://pylevy.readthedocs.io/en/latest/levy.html#classes

import matplotlib.pyplot as plt
import matplotlib.animation as animation
class MultivariateBrownianMotion(ItoProcess):
    """
    MultivariateBrownianMotion represents a generalization of the standard Brownian motion to multiple dimensions,
    providing a powerful tool for modeling correlated random processes in various fields. This continuous-time
    stochastic process, denoted as (X_t)_{t≥0} where X_t is a vector in R^n, is characterized by its drift vector
    μ ∈ R^n and a positive semi-definite covariance matrix Σ. For any time interval [s,t], the increment
    X_t - X_s follows a multivariate normal distribution N(μ(t-s), Σ(t-s)). Key properties include: independent
    and stationary increments, continuous sample paths in each dimension, and the preservation of the Markov
    property. The process starts at 0 (X_0 = 0) and has an expected value of E[X_t] = μt and covariance matrix
    Cov(X_t) = Σt. As a multivariate Itô process, it extends the mathematical framework of stochastic calculus
    to vector-valued processes, enabling the modeling of complex, interrelated phenomena. MultivariateBrownianMotion
    finds extensive applications across various domains: in finance for modeling correlated asset prices and risk
    factors, in physics for describing the motion of particles in multiple dimensions, in biology for analyzing
    the joint evolution of different species or genes, and in engineering for simulating multi-dimensional noise
    in control systems. This implementation is initialized with a name, optional process class, drift vector, and
    scale matrix (representing Σ), allowing for flexible specification of the process's statistical properties.
    It's categorized under both "multivariate" and "brownian" types, reflecting its nature as a vector-valued
    extension of Brownian motion. The class handles the dimensionality automatically based on the input drift
    vector, and stores the state in the _X attribute. The _external_simulator flag is set to False, indicating
    that the simulation is handled internally. Researchers and practitioners should be aware of the increased
    complexity in simulating and analyzing multivariate processes, particularly in high dimensions, and the
    importance of ensuring the positive semi-definiteness of the scale matrix for valid covariance structures.
    """
    def __init__(self, name: str = "Multivariate Brownian Motion", process_class: Type[Any] = None, drift: List[float] = mean_list_default, scale: List[List[float]] = variance_matrix_default):
        """
        Constructor method for the MultivariateBrownianMotion class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param drift: The drift term for the process: a vector of mean values
        :type drift: List[float]
        :param scale: The scale parameter for the process: a matrix of variances and covariances
        :type scale: List[List[float]]
        :raises ValueError: If the scale matrix is not positive definite
        :raises ValueError: If the scale matrix does not match the dimension of the drift vector
        """
        self.types = ["multivariate", "brownian"]
        super().__init__(name, process_class, drift_term=0, stochastic_term=0)
        self._dims = len(drift)
        self._drift = np.array(drift)
        if not np.all(np.linalg.eigvals(scale) > 0):
            raise ValueError("The scale matrix must be positive definite.")
        if np.shape(scale) != (self._dims, self._dims):
            raise ValueError("The scale matrix must be square and match the dimension of the drift vector.")
        self._scale = np.array(scale)
        self._X = np.zeros(len(drift))
        self._external_simulator = False

    def custom_increment(self, X: List[float], timestep: float = timestep_default) -> Any:
        """
        Custom increment function for the process.

        :param X: The current value of the process
        :type X: List[float]
        :param timestep: The time step for the simulation
        :type timestep: float
        :return: The increment value
        :rtype: List[float]
        """
        dX = np.random.multivariate_normal(mean=self._drift * timestep, cov=self._scale * timestep)
        return dX

    def simulate(self, t: float = t_default, timestep: float = timestep_default, save: bool = False, plot: bool = False) -> Any:
        """
        Simulate the Multivariate Brownian Motion process, plot, and save it.

        :param t: The time horizon for the simulation
        :type t: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :param save: Whether to save the simulation results
        :type save: bool
        :param plot: Whether to plot the simulation results
        :type plot: bool
        :return: The simulated process dataset as a NumPy array of shape (num_instances+1, num_steps)
        :rtype: Any
        """
        num_instances = self._dims
        num_steps, times, data = self.data_for_simulation(t, timestep, num_instances)
        data = np.zeros((num_instances, num_steps))
        X = self._X
        for step in range(num_steps):
            data[:, step] = X
            dX = self.custom_increment(X, timestep)
            X = X + dX
            if verbose and step % 1000 == 0:
                print(f"Simulating step {step}, X = {X}")

        data = np.concatenate((times.reshape(1, -1), data), axis=0)

        self.save_to_file(data,
                          f"process_simulation_{self.get_params()}, t:{t}, timestep:{timestep}, num_instances:{self._dims}.csv",
                          save)

        self.plot(data, num_instances, save, plot)

        return data

    def simulate_weights(self, t: float = t_default, timestep: float = timestep_default, save: bool = True,
                                  plot: bool = False) -> np.ndarray:
        """
        Simulate the weights (relative shares) of the instances of a Multivariate Brownian Motion process.

        :param t: The time horizon for the simulation
        :type t: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :param save: Whether to save the simulation results
        :type save: bool
        :param plot: Whether to plot the simulation results
        :type plot: bool
        :return: The simulated weights dataset as a NumPy array of shape (num_instances+1, num_steps)
        :rtype: np.ndarray
        """
        # Simulate the process
        data = self.simulate(t, timestep, save=False, plot=False)

        # Extract the simulated values (exclude the time column)
        simulated_values = data[1:, :]

        # Ensure non-negative values by exponentiating
        exp_values = np.exp(simulated_values)

        # Calculate weights (shares)
        total = np.sum(exp_values, axis=0)
        weights = exp_values / total

        # Prepare data for saving
        times = data[0, :]
        weight_data = np.vstack((times, weights))

        self.save_to_file(data,
                          f"weights_simulation_{self.get_params()}, t:{t}, timestep:{timestep}, num_instances:{self._dims}.csv",
                          save)

        if plot:
            plt.figure(figsize=(10, 6))
            for i in range(self._dims):
                plt.plot(times, weights[i, :], label=f'Weight {i}')
            plt.xlabel('Time')
            plt.ylabel('Weight')
            plt.title('Simulated Weights of Multivariate Brownian Motion')
            plt.legend()
            plt.grid(True)
            if save:
                plt.savefig(f"weights_plot_{self.get_params()}, t:{t}, timestep:{timestep}.png")
            plt.show()

        return weight_data

    def simulate_live(self, t: float = t_default, timestep: float = timestep_default) -> Any:
        """
        Simulate the process live and save as a video file.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :return: Video file name of the simulation
        :rtype: str
        """
        data = self.simulate(t, timestep)
        times, data = self.separate(data)
        num_instances = self._dims

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

        ani = animation.FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True)
        ani.save(f'{self.name}_simulation.mp4', writer='ffmpeg')

        plt.close(fig)
        return f'{self.name}_simulation.mp4'

    def simulate_2d(self, t: float = t_default, timestep: float = timestep_default,
                    save: bool = False, plot: bool = False) -> Any:
        """
        Simulate a 2D Multivariate Brownian Motion.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param save: Whether to save the simulation results
        :type save: bool
        :param plot: Whether to plot the simulation results
        :type plot: bool
        :raises ValueError: If the process is not 2D
        :return: Simulated 2D data array
        :rtype: Any
        """
        if self._dims != 2:
            raise ValueError("This method is only for 2D Multivariate Brownian Motion")

        num_steps = int(t / timestep)
        times = np.linspace(0, t, num_steps)

        data_2d = np.zeros((3, num_steps))  # 3 rows: time, x, y
        data_2d[0, :] = times

        X = self._X[:2]  # Use only first two dimensions
        for step in range(num_steps):
            data_2d[1:, step] = X
            dX = self.custom_increment(X, timestep)[:2]  # Use only first two dimensions
            X = X + dX

        filename = f"process_simulation_2d_{self.get_params()}, t:{t}, timestep:{timestep}.csv"
        self.save_to_file(data_2d, filename, save)

        if plot:
            self.plot_2d(data_2d, save, plot)

        return data_2d

    def simulate_live_2d(self, t: float = t_default, timestep: float = timestep_default,
                         save: bool = False, speed: float = 1.0) -> str:
        """
        Simulate the 2D process live and save as a video file.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param save: Whether to save the simulation data
        :type save: bool
        :param speed: Speed multiplier for the video
        :type speed: float
        :return: Video file name of the simulation
        :rtype: str
        """
        data_2d = self.simulate_2d(t, timestep)
        times = data_2d[0, :]
        x_data = data_2d[1, :]
        y_data = data_2d[2, :]

        num_steps = data_2d.shape[1]

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        line, = ax.plot([], [], lw=0.5)
        ax.set_xlim(np.min(x_data), np.max(x_data))
        ax.set_ylim(np.min(y_data), np.max(y_data))
        ax.set_title(f'2D Simulation of {self.name}')
        ax.set_xlabel('X dimension')
        ax.set_ylabel('Y dimension')
        ax.grid(True)

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            line.set_data(x_data[:frame + 1], y_data[:frame + 1])
            return line,

        fps = (1 / timestep) * speed
        interval = 1000 / fps  # interval in milliseconds

        ani = animation.FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=interval)
        video_filename = f'{self.name}_2d_simulation_t:{t}_timestep:{timestep}.mp4'

        ani.save(video_filename, writer='ffmpeg', fps=fps)
        plt.close(fig)

        if save:
            np.savetxt(f'{self.name}_2d_simulation_data_t:{t}_timestep:{timestep}.csv', data_2d, delimiter=',')

        return video_filename

    def simulate_3d(self, t: float = t_default, timestep: float = timestep_default,
                    save: bool = False, plot: bool = False) -> Any:
        """
        Simulate a 3D Multivariate Brownian Motion.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param save: Whether to save the simulation results
        :type save: bool
        :param plot: Whether to plot the simulation results
        :type plot: bool
        :raises ValueError: If the process is not 3D
        :return: Simulated 3D data array
        :rtype: Any
        """
        if self._dims != 3:
            raise ValueError("This method is only for 3D Multivariate Brownian Motion")

        num_steps = int(t / timestep)
        times = np.linspace(0, t, num_steps)

        data_3d = np.zeros((4, num_steps))  # 4 rows: time, x, y, z
        data_3d[0, :] = times

        X = self._X  # Use all three dimensions
        for step in range(num_steps):
            data_3d[1:, step] = X
            dX = self.custom_increment(X, timestep)
            X = X + dX

        filename = f"process_simulation_3d_{self.get_params()}, t:{t}, timestep:{timestep}.csv"
        self.save_to_file(data_3d, filename, save)

        if plot:
            self.plot_3d(data_3d, save, plot)

        return data_3d

    def simulate_live_3d(self, t: float = t_default, timestep: float = timestep_default,
                         save: bool = False, speed: float = 1.0) -> str:
        """
        Simulate the 3D process live and save as a video file.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param save: Whether to save the simulation data
        :type save: bool
        :param speed: Speed multiplier for the video
        :type speed: float
        :return: Video file name of the simulation
        :rtype: str
        """
        data_3d = self.simulate_3d(t, timestep)
        times = data_3d[0, :]
        x_data = data_3d[1, :]
        y_data = data_3d[2, :]
        z_data = data_3d[3, :]

        num_steps = data_3d.shape[1]

        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        line, = ax.plot([], [], [], lw=0.5)
        ax.set_xlim(np.min(x_data), np.max(x_data))
        ax.set_ylim(np.min(y_data), np.max(y_data))
        ax.set_zlim(np.min(z_data), np.max(z_data))
        ax.set_title(f'3D Simulation of {self.name}')
        ax.set_xlabel('X dimension')
        ax.set_ylabel('Y dimension')
        ax.set_zlabel('Z dimension')

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            return line,

        def update(frame):
            line.set_data(x_data[:frame + 1], y_data[:frame + 1])
            line.set_3d_properties(z_data[:frame + 1])
            ax.view_init(30, 0.3 * frame)  # Rotate view for 3D effect
            return line,

        fps = (1 / timestep) * speed
        interval = 1000 / fps  # interval in milliseconds

        ani = animation.FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=False, interval=interval)
        video_filename = f'{self.name}_3d_simulation_t:{t}_timestep:{timestep}.mp4'

        ani.save(video_filename, writer='ffmpeg', fps=fps)
        plt.close(fig)

        if save:
            np.savetxt(f'{self.name}_3d_simulation_data_t:{t}_timestep:{timestep}.csv', data_3d, delimiter=',')

        return video_filename

    def simulate_live_2dt(self, t: float = t_default, timestep: float = timestep_default,
                          save: bool = False, speed: float = 1.0) -> tuple[str, str]:
        """
        Simulate the 2D process live with time as the third dimension and save as a video file and interactive plot.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param save: Whether to save the simulation data
        :type save: bool
        :param speed: Speed multiplier for the video (default is 1.0, higher values make the video faster)
        :type speed: float
        :return: Tuple of video file name and interactive plot file name
        :rtype: tuple[str, str]
        """
        data_2d = self.simulate_2d(t, timestep)
        times = data_2d[0, :]
        x_data = data_2d[1, :]
        y_data = data_2d[2, :]

        num_steps = data_2d.shape[1]

        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        line, = ax.plot([], [], [], lw=2)
        ax.set_xlim(np.min(x_data), np.max(x_data))
        ax.set_ylim(np.min(y_data), np.max(y_data))
        ax.set_zlim(np.min(times), np.max(times))
        ax.set_title(f'2D Simulation of {self.name} with Time')
        ax.set_xlabel('X dimension')
        ax.set_ylabel('Y dimension')
        ax.set_zlabel('Time')

        ax.view_init(elev=20, azim=45)

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            return line,

        fps = (1 / timestep) * speed
        interval = 1000 / fps  # interval in milliseconds

        def update(frame):
            line.set_data(x_data[:frame + 1], y_data[:frame + 1])
            line.set_3d_properties(times[:frame + 1])
            return line,

        ani = animation.FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=False, interval=interval)
        video_filename = f'{self.name}_2d_time_simulation_{self.get_params()}_t:{t}_timestep:{timestep}.mp4'

        try:
            ani.save(video_filename, writer='ffmpeg', fps=fps, dpi=100, codec='libx264', bitrate=-1,
                     extra_args=['-pix_fmt', 'yuv420p'])
            print(f"2D with time simulation video saved as {video_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error saving 2D with time simulation video: {e}")
        finally:
            plt.close(fig)

        if save:
            np.savetxt(f"{self.name}_2d_time_simulation_{self.get_params()}_t:{t}_timestep:{timestep}.csv",
                       data_2d, delimiter=',')

        # Create Plotly figure
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=x_data,
            y=y_data,
            z=times,
            mode='lines',
            line=dict(width=3, color='blue'),
            name='2D Brownian Motion'
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

from scipy.stats import genhyperbolic
class GeneralizedHyperbolicProcess(NonItoProcess):
    """
    GeneralizedHyperbolicProcess represents a highly flexible class of continuous-time stochastic processes that
    encompasses a wide range of distributions, including normal, Student's t, variance-gamma, and normal-inverse
    Gaussian as special or limiting cases. This process, denoted as (X_t)_{t≥0}, is characterized by five parameters:
    α (tail heaviness), β (asymmetry), μ (location), δ (scale), and λ (a shape parameter, often denoted as 'a' in
    the implementation). The process is defined through its increments, which follow a generalized hyperbolic
    distribution. Key properties include: semi-heavy tails (heavier than Gaussian but lighter than power-law),
    ability to model skewness, and a complex autocorrelation structure. The process allows for both large jumps
    and continuous movements, making it highly adaptable to various phenomena. It's particularly noted for its
    capacity to capture both the central behavior and the extreme events in a unified framework. The
    GeneralizedHyperbolicProcess finds extensive applications in finance for modeling asset returns, particularly
    in markets exhibiting skewness and kurtosis; in risk management for more accurate tail risk assessment; in
    physics for describing particle movements in heterogeneous media; and in signal processing for modeling
    non-Gaussian noise. This implementation is initialized with parameters α, β, μ (loc), and δ (scale), with
    additional parameters possible through kwargs. It's categorized under both "generalized" and "hyperbolic" types,
    reflecting its nature as a broad, hyperbolic-based process. The class uses a custom increment function,
    indicated by the _external_simulator flag set to False. This allows for precise control over the generation
    of process increments, crucial for accurately representing the complex distribution. Researchers and
    practitioners should be aware of the computational challenges in parameter estimation and simulation,
    particularly in high-dimensional settings or with extreme parameter values. The flexibility of the generalized
    hyperbolic process comes with increased model complexity, requiring careful consideration in application and
    interpretation. Its ability to nest simpler models allows for sophisticated hypothesis testing and model
    selection in empirical studies.

    Attention! The class must be used with caution. The generalized hyperbolic distribution is not convolution-closed.
    It means that the sum of two generalized hyperbolic random variables may be not a generalized hyperbolic random variable.
    This makes the simulation of the process using finite differential problematic.
    A related problem is that it is unclear how to scale time increments dt in the simulations. The process may not be self-similar.
    Or if we make it self-similar by design, many options are possible which lead to different processes.
    """
    def __init__(self, name: str = "Generalized Hyperbolic Process", process_class: Type[Any] = None, plambda: float = 0, alpha: float = 1.7, beta: float = 0, loc: float = 0.0005, delta: float = 0.01, t_scaling: Callable[[float], float] = lambda t: t**0.5, **kwargs):
        """
        Constructor method for the GeneralizedHyperbolicProcess class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param alpha: The shape parameter (α)
        :type alpha: float
        :param beta: The skewness parameter (β)
        :type beta: float
        :param loc: The location parameter (μ)
        :type loc: float
        :param plambda: The shape parameter (λ)
        :type plambda: float
        :param t_scaling: The scaling function for time increments
        :type t_scaling: Callable[[float], float]
        :param kwargs: Additional keyword arguments for the process
        :raises ValueError: If the scale parameter is non-positive
        :raises ValueError: If the shape parameter (a) is non-positive
        :raises KnowWhatYouDoWarning: The class is under development and there are nuances with simulation. Please use with caution.
        """
        self.types = ["generalized", "hyperbolic"]
        super().__init__(name, process_class)
        self._alpha = alpha
        self._beta = beta
        self._loc = loc
        self._delta = delta
        self._plambda = plambda
        self._external_simulator = False  # Using custom increment function
        self._t_scaling = t_scaling

        # Compute 'a' based on alpha and beta if not provided directly
        # if 'a' in kwargs:
        #     self._a = kwargs['a']
        # else:
        #     self._a = (self._beta ** 2 + self._alpha ** 2) ** 0.5

        # Validity checks
        if self._delta <= 0:
            raise ValueError("Parameter 'delta' must be positive.")

        # Parametetrization to use with scipy
        self._a = self._alpha*self._delta
        self._b = self._beta*self._delta
        self._scale = self._delta
        self._p = self._plambda

        warnings.warn("The GeneralizedHyperbolicProcess class is still under development and may not be fully functional."
                                   "Its usage comes with potential risks if you do now know what you do exactly."
                                   "Specifically, the parameters in the class define the distribution of the increments used in the simulation (stochastic differential approximation), and the distribution of process instances may be different and unpredictable."
                                   "The generalized hyperbolic distribution is, generally, not convolution-closed."
                                   "It means that the sum of two generalized hyperbolic random variables may be not a generalized hyperbolic random variable."
                                   "This makes the simulation of the process using finite differential problematic."
                                   "A related problem is that it is unclear how to scale time increments dt in the simulations. The process may not be self-similar."
                                   "Or if we make it self-similar by design, many options are possible which lead to different processes.", KnowWhatYouDoWarning)

    @property
    def alpha(self):
        """
        Return the shape parameter (α) of the process.

        :return: The shape parameter (α)
        :rtype: float
        """
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float):
        """
        Set the shape parameter (α) of the process.

        :param alpha: The shape parameter (α)
        :type alpha: float
        """
        self._alpha = alpha

    @property
    def beta(self):
        """
        Return the skewness parameter (β) of the process.

        :return: The skewness parameter (β)
        :rtype: float
        """
        return self._beta

    @beta.setter
    def beta(self, beta: float):
        """
        Set the skewness parameter (β) of the process.

        :param beta: The skewness parameter (β)
        :type beta: float
        """
        self._beta = beta

    @property
    def loc(self):
        """
        Return the location parameter (μ) of the process.

        :return: The location parameter (μ)
        :rtype: float
        """
        return self._loc

    @loc.setter
    def loc(self, loc: float):
        """
        Set the location parameter (μ) of the process.

        :param loc: The location parameter (μ)
        :type loc: float
        """
        self._loc = loc

    @property
    def delta(self):
        """
        Return the scale parameter (σ) of the process.

        :return: The scale parameter (σ)
        :rtype: float
        """
        return self._delta

    @delta.setter
    def delta(self, delta: float):
        """
        Set the scale parameter (σ) of the process.

        :param delta: The scale parameter (σ)
        :type delta: float
        """
        self._delta = delta

    @property
    def a(self):
        """
        Return the shape parameter 'a' of the process, parametrized for the usage with scipy.

        :return: The shape parameter 'a'
        :rtype: float
        """
        return self._a

    @property
    def b(self):
        """
        Return the skewness parameter 'b' of the process, parametrized for the usage with scipy.

        :return: The skewness parameter 'b'
        :rtype: float
        """
        return self._b

    @property
    def scale(self):
        """
        Return the scale parameter 'scale' of the process, parametrized for the usage with scipy.

        :return: The scale parameter 'scale'
        :rtype: float
        """
        return self._scale

    @property
    def plambda(self):
        """
        Return the shape parameter 'λ' of the process.

        :return: The shape parameter 'λ'
        :rtype: float
        """
        return self._plambda

    @plambda.setter
    def plambda(self, plambda: float):
        """
        Set the shape parameter 'λ' of the process.

        :param plambda: The shape parameter 'λ'
        :type plambda: float
        """
        self._plambda = plambda

    @property
    def t_scaling(self):
        """
        Return the time scaling function of the process.

        :return: The time scaling function
        :rtype: Callable[[float], float]
        """
        return self._t_scaling

    @t_scaling.setter
    def t_scaling(self, t_scaling: Callable[[float], float]):
        """
        Set the time scaling function of the process.

        :param t_scaling: The time scaling function
        :type t_scaling: Callable[[float], float]
        """
        self._t_scaling = t_scaling

    def apply_time_scaling(self, t: float):
        """
        Apply the time scaling function to the given time (increment) value.

        :param t: he time (increment) value
        :type t: float
        :return: The scaled time value
        :rtype: float
        """
        return self._t_scaling(t)

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Custom increment function for the process.

        :param X: The current value of the process
        :type X: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :return: The increment value
        :rtype: float
        """
        dX = self._loc*timestep + self.apply_time_scaling(timestep) * genhyperbolic.rvs(p=self._plambda, b=self._b, loc=0, scale=self._scale, a=self._a)
        return dX

    def differential(self) -> str:
        """"
        Return the differential equation of the process.

        :return: The differential equation of the process
        :rtype: str
        """
        return f"dX(t) = {self._loc}*dt + (dt)^0.5 * {self._scale} * GH(alpha={self._alpha}, beta={self._beta}, loc={self._loc}, scale={self._scale}, a={self._a})"

    def express_as_elementary(self) -> str:
        """
        Express a given Generalized Hyperbolic process as a function of an elementary Hyperbolic process.

        :return: The Generalized Hyperbolic process expressed in terms of elementary Hyperbolic processes
        :rtype: str
        """
        return f"X(t) = {self._loc}*t + {self._scale} * GeneralizedHyperbolic(alpha={self._alpha}, beta={self._beta}, a={self._a}).increment()"

from scipy.stats import pareto
class ParetoProcess(NonItoProcess):
    """
    ParetoProcess represents a continuous-time stochastic process based on the Pareto distribution, known for
    modeling phenomena with power-law tail behavior. This process, denoted as (X_t)_{t≥0}, is characterized by
    three parameters: shape (α), scale (σ), and location (μ). The Pareto distribution is renowned for its "80-20
    rule" or "law of the vital few" property, making it particularly suitable for modeling size distributions in
    various natural and social contexts.

    Key parameters:

    - shape (α > 0): Determines the tail behavior of the distribution. Smaller values lead to heavier tails,
      representing more extreme events.

    - scale (σ > 0): Sets the minimum scale of the process, effectively acting as a threshold parameter.

    - loc (μ): Shifts the entire distribution, allowing for flexibility in modeling.

    The process exhibits several important properties:

    1. Heavy-tailed behavior: For α < 2, the process has infinite variance, capturing extreme events more
       effectively than processes based on normal distributions.

    2. Scale invariance: The relative probabilities of large events remain consistent regardless of scale.

    3. Power-law decay: The probability of extreme events decays as a power law, rather than exponentially.

    This implementation uses a custom increment function (_external_simulator = False), allowing for precise
    control over the generation of process increments. The class performs validity checks to ensure that the
    shape and scale parameters are strictly positive, which is crucial for maintaining the integrity of the
    Pareto distribution.

    Researchers and practitioners should be aware of the challenges in parameter estimation, especially for
    small shape values where moments may not exist. The process's heavy-tailed nature can lead to
    counterintuitive results in statistical analyses and requires careful interpretation. While powerful in
    modeling extreme phenomena, the Pareto process should be used judiciously, with consideration of its
    underlying assumptions and their applicability to the system being modeled.
    """
    def __init__(self, name: str = "Pareto Process", process_class: Type[Any] = None, shape: float = 2.0, scale: float = 1.0, loc: float = 0.0):
        """
        Constructor method for the ParetoProcess class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param shape: The shape parameter (α)
        :type shape: float
        :param scale: The scale parameter (x_m)
        :type scale: float
        :param loc: The location parameter (x_0)
        :type loc: float
        :raises ValueError: If the shape parameter is non-positive
        :raises ValueError: If the scale parameter is non-positive
        """
        self.types = ["pareto"]
        super().__init__(name, process_class)
        self._shape = shape
        self._scale = scale
        self._loc = loc
        self._external_simulator = False  # Using custom increment function

        # Validity checks
        if self._shape <= 0:
            raise ValueError("Shape parameter must be positive.")
        if self._scale <= 0:
            raise ValueError("Scale parameter must be positive.")

    @property
    def shape(self):
        """
        Return the shape parameter (α) of the process.

        :return: The shape parameter (α)
        :rtype: float
        """
        return self._shape

    @property
    def scale(self):
        """
        Return the scale parameter (σ) of the process.

        :return: The scale parameter (σ)
        :rtype: float
        """
        return self._scale

    @property
    def loc(self):
        """
        Return the location parameter (μ) of the process.

        :return: The location parameter (μ)
        :rtype: float
        """
        return self._loc

    def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
        """
        Custom increment function for the process.

        :param X: The current value of the process
        :type X: float
        :param timestep: The time step for the simulation
        :type timestep: float
        :return: The increment value
        :rtype: float
        """
        dX = pareto.rvs(b=self._shape, scale=self._scale, loc=self._loc) * timestep
        return dX

    def differential(self) -> str:
        """
        Return the differential equation of the process.

        :return: The differential equation of the process
        :rtype: str
        """
        return f"dX(t) = (dt) * Pareto(shape={self._shape}, scale={self._scale}, loc={self._loc})"

    def express_as_elementary(self) -> str:
        """
        Express a given Pareto process as a function of elementary processes.

        :return: The Pareto process expressed in terms of elementary processes
        :rtype: str
        """
        return f"X(t) = {self._loc}*t + {self._scale} * ParetoProcess(shape={self._shape}, scale={self._scale}).increment()"

class MultivariateLevy(LevyStableProcess):
    """
    MultivariateLevy represents a sophisticated extension of the Lévy stable process to multiple dimensions,
    providing a powerful tool for modeling complex, correlated heavy-tailed phenomena. This continuous-time
    stochastic process, denoted as (X_t)_{t≥0} where X_t is a vector in R^n, inherits the fundamental
    characteristics of Lévy stable distributions while incorporating cross-dimensional dependencies.

    Key parameters:

    - α (alpha): Stability parameter (0 < α ≤ 2), governing tail heaviness across all dimensions.

    - β (beta): Skewness parameter (-1 ≤ β ≤ 1), controlling asymmetry.

    - scale: Global scale parameter for the process.

    - loc: Location vector (μ ∈ R^n), shifting the process in each dimension.

    - correlation_matrix: Specifies the correlation structure between dimensions.

    - pseudovariances: Vector of pseudovariances for each dimension, generalizing the concept of variance.

    Advanced features:

    - Tempering: Optional exponential tempering to ensure finite moments.

    - Truncation: 'Hard' or 'soft' truncation options to limit extreme values.

    The process is constructed using a Cholesky decomposition of the correlation matrix, scaled by
    pseudovariances, ensuring a valid covariance structure. This approach allows for modeling complex
    interdependencies while maintaining the heavy-tailed nature of Lévy stable processes in each dimension.

    Key properties:

    1. Multivariate stability: The sum of independent copies of the process follows the same distribution,
       up to affine transformations.

    2. Heavy tails and potential infinite variance in each dimension for α < 2.

    3. Complex dependency structures captured by the correlation matrix and pseudovariances.

    The class implements custom simulation methods, including a specialized increment function that
    respects the multivariate structure. It includes extensive error checking to ensure the validity of
    input parameters, particularly for the correlation matrix and pseudovariances.

    Researchers and practitioners should be aware of the computational challenges in simulating and
    estimating multivariate Lévy stable processes, especially in high dimensions or with extreme
    parameter values. The interplay between α, the correlation structure, and pseudovariances requires
    careful interpretation. While offering great flexibility in modeling complex, heavy-tailed multivariate
    phenomena, users should exercise caution in parameter selection and model interpretation, particularly
    when dealing with empirical data.
    """
    def __init__(self, name: str = "Multivariate Levy Stable Process", process_class: Type[Any] = None,
                 alpha: float = 1.5, beta: float = 0, loc: np.ndarray = mean_list_default,
                 comments: bool = False, tempering: float = 0, truncation_level: float = None,
                 truncation_type: str = 'hard', pseudocorrelation_matrix: np.ndarray = correlation_matrix_default,
                 pseudovariances: np.ndarray = variances_default):
        """
        Constructor method for the MultivariateLevy class.

        :param name: The name of the process
        :type name: str
        :param process_class: The specific stochastic process class to use for simulation
        :type process_class: Type[Any]
        :param alpha: The stability parameter (0 < α ≤ 2)
        :type alpha: float
        :param beta: The skewness parameter (-1 ≤ β ≤ 1)
        :type beta: float
        :param scale: The scale parameter (σ > 0)
        :type scale: float
        :param loc: The location parameter (μ)
        :type loc: np.ndarray
        :param default_comments: Whether to include default comments in the process description
        :type default_comments: bool
        :param tempering: The tempering parameter (0 ≤ θ ≤ 1)
        :type tempering: float
        :param truncation_level: The truncation level for the process
        :type truncation_level: float
        :param truncation_type: The type of truncation ('hard' or 'soft')
        :type truncation_type: str
        :param pseudocorrelation_matrix: A generalization of the correlation matrix for the fat-tailed processes
        :type pseudocorrelation_matrix: np.ndarray
        :param pseudovariances:The pseudovariances for the multivariate process
        :type pseudovariances: np.ndarray
        """
        # First, initialize the base class
        super().__init__(name, process_class, alpha, beta, scale=1, loc=0, comments=comments,
                         tempering=tempering, truncation_level=truncation_level, truncation_type=truncation_type)

        # Then, initialize MultivariateLevy-specific attributes
        self._initialize_multivariate(loc, pseudocorrelation_matrix, pseudovariances)

    def _initialize_multivariate(self, loc: np.ndarray, pseudocorrelation_matrix: np.ndarray, pseudovariances: np.ndarray):
        """
        Initialize the multivariate parameters for the process.

        :param loc: The location vector (μ)
        :type loc: np.ndarray
        :param pseudocorrelation_matrix: The correlation matrix for the process
        :type pseudocorrelation_matrix: np.ndarray
        :param pseudovariances: The pseudovariances for the process
        :type pseudovariances: np.ndarray
        :raises ValueError: If the correlation matrix is not symmetric
        :raises ValueError: If the correlation matrix is not positive definite
        :raises ValueError: If the correlation matrix is not symmetric
        :raises ValueError: If the correlation matrix shape do not match the dimensions of the process
        :return: None
        :rtype: None
        """
        if pseudocorrelation_matrix is None or pseudovariances is None:
            raise ValueError("Correlation matrix and pseudovariances must be provided.")

        # Ensure pseudovariances is a 1D array
        pseudovariances = np.atleast_1d(pseudovariances)
        self._dims = len(pseudovariances)

        print(f"Debug: pseudovariances shape: {pseudovariances.shape}")
        print(f"Debug: self._dims: {self._dims}")

        # Handle loc as a numpy array
        if loc is None:
            self._loc = np.zeros(self._dims)
        elif isinstance(loc, np.ndarray) and loc.shape == (self._dims,):
            self._loc = loc
        else:
            raise ValueError(f"loc must be a numpy array of shape ({self._dims},)")

        print(f"Debug: correlation_matrix shape: {pseudocorrelation_matrix.shape}")

        if not np.allclose(pseudocorrelation_matrix, pseudocorrelation_matrix.T):
            raise ValueError("Correlation matrix must be symmetric.")

        if np.any(np.linalg.eigvals(pseudocorrelation_matrix) <= 0):
            raise ValueError("Correlation matrix must be positive definite.")

        self._pseudocorrelation_matrix = pseudocorrelation_matrix
        self._pseudovariances = pseudovariances

        # Compute the Cholesky decomposition of the correlation matrix
        L = np.linalg.cholesky(self._pseudocorrelation_matrix)

        # Create the matrix A by scaling L with the standard deviations (square roots of pseudovariances)
        self._A = np.dot(np.diag(np.sqrt(self._pseudovariances)), L)

        print(f"Debug: self._A shape: {self._A.shape}")
        print(f"Debug: Final self._dims: {self._dims}")

        if self._A.shape != (self._dims, self._dims):
            raise ValueError(f"Expected self._A to be a {self._dims}x{self._dims} matrix, but got {self._A.shape}")

        self._X = np.zeros(self._dims)

    @property
    def pseudocorrelation_matrix(self):
        """
        Return the correlation matrix of the process.

        :return: The correlation matrix
        :rtype: np.ndarray
        """
        return self._pseudocorrelation_matrix

    @pseudocorrelation_matrix.setter
    def pseudocorrelation_matrix(self, correlation_matrix: np.ndarray):
        """
        Set the correlation matrix of the process.

        :param correlation_matrix: A new correlation matrix
        :type correlation_matrix: np.ndarray
        :return: None
        :rtype: None
        """
        self._dims = correlation_matrix.shape[0]
        self._pseudocorrelation_matrix = correlation_matrix

    @property
    def pseudovariances(self):
        """
        Return the pseudovariances of the process.

        :return: The pseudovariances
        :rtype: np.ndarray
        """
        return self._pseudovariances

    @pseudovariances.setter
    def pseudovariances(self, pseudovariances: np.ndarray):
        """
        Set the pseudovariances of the process.

        :param pseudovariances: A new set of pseudovariances
        :type pseudovariances: np.ndarray
        :return: None
        :rtype: None
        """
        self._dims = len(pseudovariances)
        self._pseudovariances = pseudovariances

    @property
    def loc(self):
        """
        Return the location vector of the process.

        :return: The location vector
        :rtype: np.ndarray
        """
        return self._loc

    @loc.setter
    def loc(self, loc: np.ndarray):
        """
        Set the location vector of the process.

        :param loc: A new location vector
        :type loc: np.ndarray
        :return: None
        :rtype: None
        """
        self._dims = len(self._loc)
        self._loc = loc

    def custom_increment(self, X: np.ndarray, timestep: float = 1.0) -> np.ndarray:
        """
        Generate a custom increment for the multivariate process.

        :param X: The current value of the process
        :type X: np.ndarray
        :param timestep: The time step for the simulation
        :type timestep: float
        :return: The multivariate increment
        :rtype: np.ndarray
        """
        # Generate independent increments for each component
        dZ = np.array([levy_stable.rvs(alpha=self._alpha, beta=self._beta, loc=0, scale=self._scale)
                       for _ in range(self._dims)])

        # Scale the increments by timestep
        dZ *= timestep ** (1 / self._alpha)

        # Reshape dZ to a column vector
        dZ = dZ.reshape(-1, 1)

        # Apply matrix A to create correlated increments
        dX = np.dot(self._A, dZ).flatten()

        # Add loc * timestep to the increment
        dX += self._loc * timestep

        # Apply truncation and/or tempering if necessary
        if self._tempered:
            dX = np.array([self.tempered_stable_rvs() for _ in dX])

        if self._truncated:
            dX = np.array([self.truncate(dx) for dx in dX])

        return dX

    def simulate(self, t: float = 2.0, timestep: float = 0.1, num_instances: int = 1, save: bool = False,
                 plot: bool = False) -> np.ndarray:
        """
        Simulate the multivariate Lévy process.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param num_instances: Number of simulation instances to generate
        :type num_instances: int
        :param save: Whether to save the simulation data
        :type save: bool
        :param plot: Whether to plot the simulation results
        :type plot: bool
        :return: The simulated data array of shape (dims+1, num_steps)
        :rtype
        """
        num_steps = max(int(t / timestep), 2)  # Ensure at least 2 steps
        times = np.linspace(0, t, num_steps)

        data = np.zeros((num_instances, self._dims, num_steps))

        for instance in range(num_instances):
            X = self._X.copy()
            for step in range(num_steps):
                data[instance, :, step] = X
                dX = self.custom_increment(X, timestep)
                X = X + dX

        if save:
            self.save_to_file(data,
                              f"levy_process_simulation_{self.get_params()}_t:{t}_timestep:{timestep}_num_instances:{num_instances}.csv",
                              save)

        if plot:
            self.plot(times, data, save, plot)

        return data

    def plot(self, times, data, save: bool = False, plot: bool = False):
        """
        Plot the simulation results.

        :param times: The time points for the simulation
        :type times: np.ndarray
        :param data: The simulated data array
        :type data: np.ndarray
        :param save: Whether to save the plot
        :type save: bool
        :param plot: Whether to display the plot
        :type plot: bool
        :raises ValueError: If there are insufficient time points to plot the simulation
        :return: None
        :rtype: None
        """
        if plot:
            if np.isscalar(times) or len(times) < 2:
                raise ValueError(
                    "Insufficient time points to plot the simulation. Ensure there are at least two time steps.")

            t = times[-1]
            timestep = times[1] - times[0]

            num_instances, dimension, _ = data.shape

            plt.figure(figsize=(12, 8))

            for d in range(dimension):
                for i in range(num_instances):
                    plt.plot(times, data[i, d, :], lw=0.5, label=f'Dim {d + 1}, Inst {i + 1}')

            plt.title(f'Simulation of {self.name}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend()

            if save:
                try:
                    plt.savefig(os.path.join(self._output_dir,
                                         f'{self.name}_{self.get_params()}_t:{t}_timestep:{timestep}_num_instances:{num_instances}_process_simulation.png'))
                except:
                    plt.savefig(os.path.join(self._output_dir,
                                         f'{self.name}_simulation.png'))

            plt.tight_layout()
            plt.show()

    def plot_2d(self, data_2d: np.ndarray, save: bool = False, plot: bool = True):
        """
        Plot the 2D simulation results.

        :param data_2d: 2D simulation data
        :type data_2d: np.ndarray
        :param save: Whether to save the plot
        :type save: bool
        :param plot: Whether to display the plot
        :type plot: bool
        :return None
        :rtype None
        """
        if not plot:
            return

        times = data_2d[0, :]
        x_data = data_2d[1, :]
        y_data = data_2d[2, :]

        plt.figure(figsize=(10, 8))
        plt.plot(x_data, y_data, alpha=0.5)

        plt.title(f"2D Process Simulation (t={times[-1]}, timestep={times[1] - times[0]})")
        plt.xlabel("X dimension")
        plt.ylabel("Y dimension")

        if save:
            plt.savefig(f"2d_simulation_plot_{self.get_params()}.png")

        if plot:
            plt.show()

    def plot_2dt(self, data_2d: np.ndarray, save: bool = False, plot: bool = True):
        """
        Plot the 2D simulation results in a 3D graph with time as the third dimension.

        :param data_2d: 2D simulation data
        :type data_2d: np.ndarray
        :param save: Whether to save the plot
        :type save: bool
        :param plot: Whether to display the plot
        :type plot: bool
        :return None
        :rtype None
        """
        if not plot:
            return None

        times = data_2d[0, :]
        x_data = data_2d[1, :]
        y_data = data_2d[2, :]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(x_data, y_data, times, alpha=0.5)

        ax.set_xlabel('X dimension')
        ax.set_ylabel('Y dimension')
        ax.set_zlabel('Time')
        ax.set_title(
            f"3D Visualization of 2D Process Simulation\n(t={times[-1]}, timestep={times[1] - times[0]})")

        if save:
            plt.savefig(f"3d_simulation_plot_{self.get_params()}.png")

        if plot:
            plt.show()

    def plot_3d(self, data_3d: np.ndarray, save: bool = False, plot: bool = True):
        """
        Plot the 3D simulation results.

        :param data_3d: 3D simulation data
        :type data_3d: np.ndarray
        :param save: Whether to save the plot
        :type save: bool
        :param plot: Whether to display the plot
        :type plot: bool
        :return None
        :rtype None
        """
        if not plot:
            return

        times = data_3d[0, :]
        x_data = data_3d[1, :]
        y_data = data_3d[2, :]
        z_data = data_3d[3, :]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(x_data, y_data, z_data, alpha=0.5)

        ax.set_xlabel('X dimension')
        ax.set_ylabel('Y dimension')
        ax.set_zlabel('Z dimension')
        ax.set_title(
            f"3D Process Simulation\n(t={times[-1]}, timestep={times[1] - times[0]})")

        if save:
            plt.savefig(f"3d_simulation_plot_{self.get_params()}.png")

        if plot:
            plt.show()

    def simulate_live(self, t: float = 1.0, timestep: float = 0.01) -> str:
        """
        Simulate the process live and save as a video file.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :return: The filename of the saved video
        :rtype: str
        """
        data = self.simulate(t, timestep)
        times = np.linspace(0, t, data.shape[2])
        num_instances, dimension, _ = data.shape

        fig, ax = plt.subplots(figsize=(12, 8))
        lines = [ax.plot([], [], lw=0.5, label=f'Dimension {d + 1}')[0] for d in range(dimension)]
        ax.set_xlim(0, t)
        ax.set_ylim(np.min(data), np.max(data))
        ax.set_title(f'Simulation of {self.name}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.legend()

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(frame):
            for d in range(dimension):
                lines[d].set_data(times[:frame + 1], data[0, d, :frame + 1])
            return lines

        ani = animation.FuncAnimation(fig, update, frames=len(times), init_func=init, blit=True)
        video_filename = f'{self.name}_simulation.mp4'
        ani.save(video_filename, writer='ffmpeg')

        plt.close(fig)
        return video_filename

    def simulate_2d(self, t: float = 1.0, timestep: float = 0.01, save: bool = False, plot: bool = False) -> np.ndarray:
        """
        Simulate the 2D Multivariate Lévy Process.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param save: Whether to save the simulation data
        :type save: bool
        :param plot: Whether to plot the simulation results
        :type plot: bool
        :raises ValueError: If the process is not 2D
        :return: The simulated 2D data array of shape (3, num_steps)
        :rtype: np.ndarray
        """
        if self._dims != 2:
            raise ValueError("This method is only for 2D Multivariate Lévy Process")

        data = self.simulate(t, timestep)
        times = np.linspace(0, t, data.shape[2])

        data_2d = np.zeros((3, data.shape[2]))  # 3 rows: time, x, y
        data_2d[0, :] = times
        data_2d[1:, :] = data[0, :, :]  # Use the first (and only) instance

        if save:
            self.save_to_file(data_2d, f"levy_process_2d_simulation_{self.get_params()}_t:{t}_timestep:{timestep}.csv",
                              save)

        if plot:
            self.plot_2d(data_2d, save, plot)

        return data_2d

    def simulate_live_2d(self, t: float = 1.0, timestep: float = 0.01, save: bool = False, speed: float = 1.0) -> str:
        """
        Simulate the 2D process live and save as a video file.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: time step for the simulation
        :type timestep: float
        :param save: Whether to save the video
        :type save: bool
        :param speed: Speed of the simulation
        :type speed: float
        :return: The filename of the saved video
        :rtype: str
        """
        data_2d = self.simulate_2d(t, timestep)
        times = data_2d[0, :]
        x_data = data_2d[1, :]
        y_data = data_2d[2, :]

        fig, ax = plt.subplots(figsize=(10, 10))
        line, = ax.plot([], [], lw=0.5)
        ax.set_xlim(np.min(x_data), np.max(x_data))
        ax.set_ylim(np.min(y_data), np.max(y_data))
        ax.set_title(f'2D Simulation of {self.name}')
        ax.set_xlabel('X dimension')
        ax.set_ylabel('Y dimension')
        ax.grid(True)

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            line.set_data(x_data[:frame + 1], y_data[:frame + 1])
            return line,

        fps = (1 / timestep) * speed
        interval = 1000 / fps  # interval in milliseconds

        ani = animation.FuncAnimation(fig, update, frames=len(times), init_func=init, blit=True, interval=interval)
        video_filename = f'{self.name}_2d_simulation.mp4'
        ani.save(video_filename, writer='ffmpeg', fps=fps)

        plt.close(fig)
        return video_filename

    def simulate_3d(self, t: float = 1.0, timestep: float = 0.01, save: bool = False, plot: bool = False) -> np.ndarray:
        """
        Simulate the 3D Multivariate Lévy Process.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param save: Whether to save the simulation data
        :type save: bool
        :param plot: Whether to plot the simulation results
        :type plot: bool
        :raises ValueError: If the process is not 3D
        :return: The simulated 3D data array of shape (4, num_steps)
        :rtype
        """
        if self._dims != 3:
            raise ValueError("This method is only for 3D Multivariate Lévy Process")

        data = self.simulate(t, timestep)
        times = np.linspace(0, t, data.shape[2])

        data_3d = np.zeros((4, data.shape[2]))  # 4 rows: time, x, y, z
        data_3d[0, :] = times
        data_3d[1:, :] = data[0, :, :]  # Use the first (and only) instance

        if save:
            self.save_to_file(data_3d, f"levy_process_3d_simulation_{self.get_params()}_t:{t}_timestep:{timestep}.csv",
                              save)

        if plot:
            self.plot_3d(data_3d, save, plot)

        return data_3d

    def simulate_live_3d(self, t: float = 1.0, timestep: float = 0.01, save: bool = False, speed: float = 1.0) -> str:
        """
        Simulate the 3D process live and save as a video file.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param save: Whether to save the video
        :type save: bool
        :param speed: Speed of the simulation
        :type speed: float
        :return: The filename of the saved video
        :rtype: str
        """
        data_3d = self.simulate_3d(t, timestep)
        times = data_3d[0, :]
        x_data = data_3d[1, :]
        y_data = data_3d[2, :]
        z_data = data_3d[3, :]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        line, = ax.plot([], [], [], lw=0.5)
        ax.set_xlim(np.min(x_data), np.max(x_data))
        ax.set_ylim(np.min(y_data), np.max(y_data))
        ax.set_zlim(np.min(z_data), np.max(z_data))
        ax.set_title(f'3D Simulation of {self.name}')
        ax.set_xlabel('X dimension')
        ax.set_ylabel('Y dimension')
        ax.set_zlabel('Z dimension')

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            return line,

        def update(frame):
            line.set_data(x_data[:frame + 1], y_data[:frame + 1])
            line.set_3d_properties(z_data[:frame + 1])
            ax.view_init(30, 0.3 * frame)  # Rotate view for 3D effect
            return line,

        fps = (1 / timestep) * speed
        interval = 1000 / fps  # interval in milliseconds

        ani = animation.FuncAnimation(fig, update, frames=len(times), init_func=init, blit=False, interval=interval)
        video_filename = f'{self.name}_3d_simulation.mp4'
        ani.save(video_filename, writer='ffmpeg', fps=fps)

        plt.close(fig)
        return video_filename

    def simulate_live_2dt(self, t: float = 1.0, timestep: float = 0.01, save: bool = False, speed: float = 1.0) -> tuple[str, str]:
        """
        Simulate the 2D process live with time and save as a video file.

        :param t: Total time for the simulation
        :type t: float
        :param timestep: Time step for the simulation
        :type timestep: float
        :param save: Whether to save the video
        :type save: bool
        :param speed: Speed of the simulation
        :type speed: float
        :return: The filenames of the saved video and interactive object
        :rtype: tuple[str, str]
        """
        data_2d = self.simulate_2d(t, timestep)
        times = data_2d[0, :]
        x_data = data_2d[1, :]
        y_data = data_2d[2, :]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        line, = ax.plot([], [], [], lw=2)
        ax.set_xlim(np.min(x_data), np.max(x_data))
        ax.set_ylim(np.min(y_data), np.max(y_data))
        ax.set_zlim(np.min(times), np.max(times))
        ax.set_title(f'2D Simulation of {self.name} with Time')
        ax.set_xlabel('X dimension')
        ax.set_ylabel('Y dimension')
        ax.set_zlabel('Time')

        ax.view_init(elev=20, azim=45)

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            return line,

        fps = (1 / timestep) * speed
        interval = 1000 / fps  # interval in milliseconds

        def update(frame):
            line.set_data(x_data[:frame + 1], y_data[:frame + 1])
            line.set_3d_properties(times[:frame + 1])
            return line,

        ani = animation.FuncAnimation(fig, update, frames=len(times), init_func=init, blit=False, interval=interval)
        video_filename = f'{self.name}_2d_time_simulation.mp4'
        ani.save(video_filename, writer='ffmpeg', fps=fps, dpi=100, codec='libx264', bitrate=-1,
                 extra_args=['-pix_fmt', 'yuv420p'])

        plt.close(fig)

        if save:
            np.savetxt(f"{self.name}_2d_time_simulation_{self.get_params()}_t:{t}_timestep:{timestep}.csv", data_2d,
                       delimiter=',')

        # Create Plotly figure
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=x_data,
            y=y_data,
            z=times,
            mode='lines',
            line=dict(width=3, color='blue'),
            name='2D Lévy Process'
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
        object_filename = f'{self.name}_2d_time_object.html'
        fig.write_html(object_filename)

        return video_filename, object_filename
