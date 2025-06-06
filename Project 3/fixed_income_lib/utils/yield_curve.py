# fixed_income_lib/utils/yield_curve.py

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def create_flat_yield_curve(base_rate=5.0):
    """
    Generate a flat monthly yield curve from month 1 to 360 (30 years).
    Returns: DataFrame with shape (1, 360), yields in percentage.
    """
    rates = np.full((1, 360), base_rate)
    columns = list(range(1, 361))
    return pd.DataFrame(rates, columns=columns)


def interpolate_yield_curve(yield_curve, months):
    """
    Interpolates the yield curve for non-integer maturities.
    Inputs:
      - yield_curve: DataFrame with shape (1, 360), monthly yields.
      - months: list or array of fractional maturities (in months).
    Returns: numpy array of interpolated yields for each month.
    """
    x = np.arange(1, 361)
    y = yield_curve.iloc[0].values
    cs = CubicSpline(x, y)
    return cs(months)
