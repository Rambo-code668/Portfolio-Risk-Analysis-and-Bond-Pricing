"""
Validation Module for Fixed Income Library

This module provides functions for validating input parameters for bonds,
yield curves, and portfolios. It performs checks such as:
  - Ensuring maturities are positive and in 0.5-year increments.
  - Validating that payment frequency flags are booleans.
  - Verifying that coupon bonds have a proper coupon rate (a non-negative number
    expressed as a decimal fraction, e.g., 0.05 for 5%) and that zero-coupon bonds
    have coupon_rate = None.
  - Confirming that face values and asset positions are numeric (and positive where appropriate).
  - Checking that yield curve inputs are provided in a proper pandas DataFrame,
    with numeric yields and column names convertible to integers.
  - Validating that all portfolio arrays have equal, non-zero lengths.
  
Each function raises a ValueError (or TypeError when applicable) with a clear message
if the input does not meet the validation criteria.
"""

import math
import numpy as np
import pandas as pd


def check_maturity_and_payment(maturity: float, semi_annual_payment: bool) -> None:
    """
    Validate the maturity and coupon payment frequency.

    1. Maturity must be positive.
    2. Maturity should be in 0.5-year increments (e.g., 0.5, 1.0, 1.5, 2.0, ...).
    3. Optionally, if maturity is below 0.5 years, then semi-annual payment must be False.

    Raises:
        ValueError: If any condition is violated.
    """
    if maturity <= 0:
        raise ValueError(f"Maturity must be positive, got {maturity}.")

    # Check for 0.5-year increments: 2 * maturity should be an integer.
    doubled = 2.0 * maturity
    if not math.isclose(doubled, round(doubled), rel_tol=1e-8):
        raise ValueError(f"Maturity {maturity} must be in 0.5-year increments.")

    if maturity < 0.5 and semi_annual_payment:
        raise ValueError(f"Maturity {maturity} is too short for semi-annual payments.")


def check_yield_curve(yield_curve: pd.DataFrame) -> None:
    """
    Validate the yield curve input.

    Checks:
      - Must be a pandas DataFrame.
      - Must not be empty.
      - Column names must be convertible to integers (representing months).
      - All yield values must be numeric and non-negative.

    Raises:
        TypeError or ValueError as applicable.
    """
    if not isinstance(yield_curve, pd.DataFrame):
        raise TypeError("Yield curve must be a pandas DataFrame.")
    if yield_curve.empty:
        raise ValueError("Yield curve is empty; provide at least one row.")
    
    for col in yield_curve.columns:
        try:
            int(col)
        except Exception:
            raise ValueError("All yield curve column names must be convertible to integers (e.g., 1, 2, ..., 360).")
    
    if not np.issubdtype(yield_curve.values.dtype, np.number):
        raise ValueError("Yield curve values must be numeric.")
    
    if (yield_curve < 0).any().any():
        raise ValueError("Yield curve contains negative yield values.")


def check_portfolio_arrays(*arrays) -> None:
    """
    Validate that all provided arrays for portfolio parameters have equal non-zero lengths.

    Parameters:
      *arrays: Array-like objects.

    Raises:
      ValueError if any array is empty or if lengths differ.
    """
    if not arrays:
        raise ValueError("No arrays provided for validation.")
    
    expected_length = len(arrays[0])
    if expected_length == 0:
        raise ValueError("Input arrays must be non-empty.")
    
    for arr in arrays:
        if len(arr) != expected_length:
            raise ValueError("All input arrays must have the same length.")


def check_coupon_rate(coupon_rate, is_zcb: bool) -> None:
    """
    Validate coupon rate in context of bond type.

    For coupon bonds (is_zcb == False):
      - coupon_rate must be provided and convertible to a non-negative float.
      - It should be given as a decimal (e.g., 0.05 for 5%).
    For zero-coupon bonds (is_zcb == True):
      - coupon_rate must be None.

    Raises:
      ValueError if validation fails.
    """
    if is_zcb:
        if coupon_rate is not None:
            raise ValueError("Zero-coupon bonds should have coupon_rate = None.")
    else:
        if coupon_rate is None:
            raise ValueError("Coupon bonds must have a coupon_rate provided.")
        try:
            rate = float(coupon_rate)
        except Exception:
            raise ValueError(f"Coupon rate must be numeric; got {coupon_rate}.")
        if rate < 0:
            raise ValueError("Coupon rate cannot be negative.")
        if rate > 1:
            raise ValueError("Coupon rate appears too high; expected a decimal (e.g., 0.05 for 5%).")


def check_face_value(face_value) -> None:
    """
    Validate face value.

    Must be convertible to float and positive.

    Raises:
      ValueError if invalid.
    """
    try:
        fv = float(face_value)
    except Exception:
        raise ValueError("Face value must be numeric.")
    if fv <= 0:
        raise ValueError("Face value must be positive.")


def check_num_assets(num_assets) -> None:
    """
    Validate number of assets.

    Must be numeric. Negative values may be allowed to indicate short positions.

    Raises:
      ValueError if not numeric.
    """
    try:
        _ = float(num_assets)
    except Exception:
        raise ValueError("Number of assets must be numeric.")


def check_semi_annual_payment(semi_annual_payment) -> None:
    """
    Validate the semi-annual payment flag is boolean.

    Raises:
      ValueError if not a boolean.
    """
    if semi_annual_payment not in [True, False]:
        raise ValueError("Semi-annual payment flag must be a boolean.")


def validate_bond_parameters(maturity, num_assets, face_value, is_zcb, coupon_rate, semi_annual_payment) -> None:
    """
    Validate all individual bond parameters.

    Calls:
      - check_maturity_and_payment
      - check_num_assets
      - check_face_value
      - check_coupon_rate (in context of is_zcb)
      - check_semi_annual_payment

    Raises:
      ValueError if any parameter fails.
    """
    if not isinstance(maturity, (int, float)):
        raise ValueError("Maturity must be numeric.")
    check_maturity_and_payment(maturity, semi_annual_payment)
    check_num_assets(num_assets)
    check_face_value(face_value)
    
    if is_zcb not in [True, False]:
        raise ValueError("is_zcb must be a boolean.")
    check_coupon_rate(coupon_rate, is_zcb)
    check_semi_annual_payment(semi_annual_payment)