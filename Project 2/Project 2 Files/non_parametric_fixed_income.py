import numpy as np
from scipy.optimize import minimize
from SupportFunctions import (
    compute_quantile_risk, compute_bootstrap_risk, 
    compute_quantile_risk_port, compute_bootstrap_risk_port, 
    neg_log_likelihood
)
from inputsControlFunctions import (
    validate_returns_single,
    validate_returns_portfolio,
    check_position_single,
    validate_alpha,
    validate_method,
    validate_n_bootstrap_samples,
    validate_position
)

class FixedIncomeNprmSingle:
    """
    Models non-parametric risk measures for a single fixed-income position.
    
    Parameters:
    -----------
    returns : np.ndarray
        Historical returns of the fixed-income security.
    position : float
        Quantity held (positive = long, negative = short).
    alpha : float, default=0.05
        Significance level for VaR/ES.
    method : str, default="quantile"
        Calculation method ("quantile" or "bootstrap").
    n_bootstrap_samples : int, default=10000
        Number of bootstrap samples (used if method="bootstrap").
    """
    def __init__(self, returns: np.ndarray, position: float, alpha: float = 0.05,
                 method: str = "quantile", n_bootstrap_samples: int = 10000):
        validate_returns_single(returns)
        check_position_single(position)
        validate_alpha(alpha)
        validate_method(method)
        validate_n_bootstrap_samples(n_bootstrap_samples)
        # Additional validations (e.g., for method and n_bootstrap_samples) can be added here.
        self.returns = returns
        self.position = position
        self.alpha = alpha
        self.method = method
        self.n_bootstrap_samples = n_bootstrap_samples
        self.var = None
        self.es = None
        self.fit()
    
    def fit(self):
        """
        Compute VaR and ES using the specified method.
        """
        if self.method == "quantile":
            self.var, self.es = compute_quantile_risk(self.returns, self.alpha, self.position)
        elif self.method == "bootstrap":
            self.var, self.es = compute_bootstrap_risk(self.returns, self.alpha, self.position,
                                                       n_bootstrap_samples=self.n_bootstrap_samples)
        else:
            raise ValueError("Unsupported method: {}. Choose 'quantile' or 'bootstrap'.".format(self.method))
    
    def summary(self):
        """
        Returns a dictionary of key risk metrics.
        """
        if self.position > 0:
            maxLoss = -self.position * np.min(self.returns)
        else:
            maxLoss = -self.position * np.max(self.returns)
        maxExcessLoss = maxLoss - self.var
        esOverVar = self.es / self.var if self.var != 0 else None
        maxExcessLossOverVar = maxExcessLoss / self.var if self.var != 0 else None
        return {
            "var": self.var,
            "es": self.es,
            "maxLoss": round(maxLoss, 4),
            "maxExcessLoss": round(maxExcessLoss, 4),
            "maxExcessLossOverVar": round(maxExcessLossOverVar, 4) if maxExcessLossOverVar is not None else None,
            "esOverVar": round(esOverVar, 4) if esOverVar is not None else None
        }
    
    def evt(self, quantile_threshold: float = 0.95):
        """
        Estimate VaR/ES using Extreme Value Theory (GPD).
        """
        if self.position > 0:
            losses = -self.returns
        else:
            losses = self.returns
        
        u = np.quantile(losses, quantile_threshold)
        exceedances = losses[losses > u] - u
        n = len(losses)
        n_u = len(exceedances)
        if n_u == 0:
            raise ValueError("No exceedances found beyond the threshold; consider lowering quantile_threshold.")
        
        # Optimize to estimate GPD parameters using the support function.
        initial_guess = [0.1, np.mean(exceedances)]
        result = minimize(neg_log_likelihood, initial_guess, args=(exceedances, n_u),
                          method='L-BFGS-B', bounds=[(-np.inf, np.inf), (1e-6, np.inf)])
        if not result.success:
            raise RuntimeError("GPD parameter optimization did not converge: " + result.message)
        xi_hat, beta_hat = result.x
        
        var_evt = u + (beta_hat / xi_hat) * (((n / n_u) * (1 - self.alpha))**(-xi_hat) - 1)
        es_evt = (var_evt + beta_hat - xi_hat * u) / (1 - xi_hat)
        var_evt = abs(self.position) * var_evt
        es_evt = abs(self.position) * es_evt
        
        return {
            "evt_var": round(max(var_evt, 0), 4),
            "evt_es": round(max(es_evt, 0), 4),
            "xi": xi_hat,
            "beta": beta_hat,
            "u": u,
            "n": n,
            "n_u": n_u
        }
    
    


class FixedIncomeNprmPort:
    """
    Models risk measures for a portfolio of fixed-income positions.
    
    Parameters:
    -----------
    returns : np.ndarray
        Matrix of returns (rows=periods, columns=securities).
    positions : list
        List of positions for each security.
    alpha : float, default=0.05
        Significance level for VaR/ES.
    method : str, default="quantile"
        Calculation method ("quantile" or "bootstrap").
    n_bootstrap_samples : int, default=10000
        Number of bootstrap samples (used if method="bootstrap").
    """
    def __init__(self, returns: np.ndarray, positions: list, alpha: float = 0.05,
                 method: str = "quantile", n_bootstrap_samples: int = 10000):
        validate_returns_portfolio(returns)
        validate_position(positions, returns)
        validate_alpha(alpha)
        validate_method(method)
        validate_n_bootstrap_samples(n_bootstrap_samples)
        if len(positions) != returns.shape[1]:
            raise ValueError("Length of positions must equal the number of columns in returns.")
        self.returns = returns
        self.positions = positions
        self.alpha = alpha
        self.method = method
        self.n_bootstrap_samples = n_bootstrap_samples
        self.portfolio_returns = None  # Aggregated returns for each period.
        self.cum_returns = None        # Cumulative portfolio returns.
        self.var = None
        self.es = None
        self.fit()
    
    def fit(self):
        """
        Compute aggregated portfolio returns and risk measures.
        """
        positions = np.array(self.positions)
        net_position = np.sum(positions)
        self.portfolio_returns = np.sum(self.returns * positions, axis=1)
        self.cum_returns = np.cumprod(1 + self.portfolio_returns) - 1
        # For portfolios, the aggregated returns are already computed; use factor = 1.
        if self.method == "quantile":
            self.var, self.es = compute_quantile_risk_port(self.portfolio_returns, self.alpha, net_position)
        elif self.method == "bootstrap":
            self.var, self.es = compute_bootstrap_risk_port(self.portfolio_returns, self.alpha, net_position, n_bootstrap_samples=self.n_bootstrap_samples)
        else:
            raise ValueError("Unsupported method: {}. Choose 'quantile' or 'bootstrap'.".format(self.method))
    
    def marg_vars(self, scale_factor: float = 0.1):
        """
        Compute Marginal VaR for each position.
        
        """
        original_positions = self.positions.copy()
        original_var = self.var
        marginal_vars = []
        for i in range(len(self.positions)):
            self.positions[i] += scale_factor
            self.fit()
            perturbed_var = self.var
            mvar = (perturbed_var - original_var) / scale_factor
            marginal_vars.append(round(mvar, 4))
            self.positions[i] = original_positions[i]
        self.fit()
        return marginal_vars
    
    def summary(self):
        """
        Return a dictionary of key portfolio risk metrics.
        """
        weights = np.array(self.positions)
        net_position = np.sum(weights)
        if net_position > 0:
            maxLoss = -net_position * np.min(self.portfolio_returns)
        else:
            maxLoss = -net_position * np.max(self.portfolio_returns)
        maxExcessLoss = maxLoss - self.var
        esOverVar = self.es / self.var if self.var != 0 else None
        maxExcessLossOverVar = maxExcessLoss / self.var if self.var != 0 else None
        return {
            "var": self.var,
            "es": self.es,
            "maxLoss": round(maxLoss, 4),
            "maxExcessLoss": round(maxExcessLoss, 4),
            "maxExcessLossOverVar": round(maxExcessLossOverVar, 4) if maxExcessLossOverVar is not None else None,
            "esOverVar": round(esOverVar, 4) if esOverVar is not None else None
        }