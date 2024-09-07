# Check if a time series is stationary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sm.s.tsa.stattools import adfuller

def is_stationary(ts, alpha=0.05):
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test.

    Parameters:
    - ts: pd.Series
        The time series to be checked.
    - alpha: float
        The significance level for the test.

    Returns:
    - stationary: bool
        True if the time series is stationary, False otherwise.
    """
    result = adfuller(ts)
    p_value = result[1]
    return p_value < alpha

# Check if the distribution of X(t+dt) - X(t) depends only on dt

# Check how the distribution X(t+dt) - X(t) depends on dt 
