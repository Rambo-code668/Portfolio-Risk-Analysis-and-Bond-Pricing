import numpy as np
def validate_returns_single(returns):
    """
    Validates that the returns input for a single fixed-income position is a one-dimensional numpy array of numeric values.
    """
    if not isinstance(returns, np.ndarray):
        raise TypeError("Returns must be a numpy array.")
    if returns.ndim != 1:
        raise ValueError("Returns array must be one-dimensional.")
    if not np.issubdtype(returns.dtype, np.number):
        raise ValueError("Returns array must contain numeric values.")

def validate_returns_portfolio(returns):
    """
    Validates that the returns input for a fixed-income portfolio is a two-dimensional numpy array of numeric values.
    Each column represents the returns of a fixed-income position.
    """
    if not isinstance(returns, np.ndarray):
        raise TypeError("Portfolio returns must be a numpy array.")
    if returns.ndim != 2:
        raise ValueError("Portfolio returns array must be two-dimensional.")
    if not np.issubdtype(returns.dtype, np.number):
        raise ValueError("Portfolio returns array must contain numeric values.")

def check_position_single(position):
    """
    Validates that the position for a single fixed-income security is a numeric value.
    """
    if not isinstance(position, (int, float)):
        raise TypeError("Position must be a numeric type (int or float).")

def validate_alpha(alpha):
    """
    Validates that the significance level alpha is a float between 0 and 1.
    """
    if not isinstance(alpha, float):
        raise TypeError("Alpha must be a float.")
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")

def validate_method(method):
    """
    Validates that the calculation method is a string and either 'quantile' or 'bootstrap'.
    """
    if not isinstance(method, str):
        raise TypeError("Method must be a string.")
    if method not in ["quantile", "bootstrap"]:
        raise ValueError("Method must be either 'quantile' or 'bootstrap'.")

def validate_n_bootstrap_samples(n_bootstrap_samples):
    """
    Validates that the number of bootstrap samples is a positive integer.
    """
    if not isinstance(n_bootstrap_samples, int):
        raise TypeError("n_bootstrap_samples must be an integer.")
    if n_bootstrap_samples <= 0:
        raise ValueError("n_bootstrap_samples must be a positive integer.")

def validate_position(positions, returns):
    """
    Validates that portfolio positions are provided as a one-dimensional list of numeric values,
    and that the length of positions equals the number of securities (i.e., returns.shape[1]).
    """
    # Ensure positions is a list.
    if not isinstance(positions, list):
        raise TypeError("Positions must be provided as a list.")
    
    # Validate that returns is a 2D numpy array.
    if not isinstance(returns, np.ndarray):
        raise TypeError("Returns must be a numpy array.")
    if returns.ndim != 2:
        raise ValueError("Returns array must be two-dimensional.")
    
    # Ensure the length of positions matches the number of securities.
    if len(positions) != returns.shape[1]:
        raise ValueError("Length of positions must equal the number of columns in returns.")
    
    # Validate that each position is numeric.
    for pos in positions:
        if not isinstance(pos, (int, float)):
            raise TypeError("Each position must be numeric (int or float).")