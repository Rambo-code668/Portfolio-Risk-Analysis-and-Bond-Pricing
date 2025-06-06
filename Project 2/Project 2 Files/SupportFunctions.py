import numpy as np

def compute_quantile_risk(returns, alpha, position=1):
    """
    Compute VaR and ES using the quantile method for a single asset.
    """
    if position > 0:
        q = np.quantile(returns, alpha)
        var = -position * q
        tail = returns[returns < q]
        es = -position * np.mean(tail) if tail.size > 0 else var
    else:
        q = np.quantile(returns, 1 - alpha)
        var = -position * q
        tail = returns[returns > q]
        es = -position * np.mean(tail) if tail.size > 0 else var
    return round(max(var, 0), 4), round(max(es, 0), 4)


def compute_bootstrap_risk(returns, alpha, position=1, n_bootstrap_samples=10000):
    """
    Compute VaR and ES using the bootstrap method for a single asset.
    """
    boot_var_list = []
    boot_es_list = []
    n = len(returns)
    
    if position > 0:
        for _ in range(n_bootstrap_samples):
            sample = np.random.choice(returns, size=n, replace=True)
            q = np.quantile(sample, alpha)
            boot_var = -position * q
            tail = sample[sample < q]
            boot_es = -position * np.mean(tail) if tail.size > 0 else boot_var
            boot_var_list.append(boot_var)
            boot_es_list.append(boot_es)
    else:
        for _ in range(n_bootstrap_samples):
            sample = np.random.choice(returns, size=n, replace=True)
            q = np.quantile(sample, 1 - alpha)
            boot_var = -position * q
            tail = sample[sample > q]
            boot_es = -position * np.mean(tail) if tail.size > 0 else boot_var
            boot_var_list.append(boot_var)
            boot_es_list.append(boot_es)
            
    var = np.mean(boot_var_list)
    es = np.mean(boot_es_list)
    return round(max(var, 0), 4), round(max(es, 0), 4)


def compute_quantile_risk_port(portfolio_returns, alpha, net_position):
    """
    Compute VaR and ES using the quantile method for a portfolio.
    """
    if net_position > 0:
        quantile_value = np.quantile(portfolio_returns, alpha)
        var = -quantile_value
        tail_losses = portfolio_returns[portfolio_returns < quantile_value]
        es = -np.mean(tail_losses) if tail_losses.size > 0 else var
    else:
        quantile_value = np.quantile(portfolio_returns, 1 - alpha)
        var = -quantile_value
        tail_losses = portfolio_returns[portfolio_returns > quantile_value]
        es = -np.mean(tail_losses) if tail_losses.size > 0 else var
    return round(max(var, 0), 4), round(max(es, 0), 4)


def compute_bootstrap_risk_port(portfolio_returns, alpha, net_position, n_bootstrap_samples):
    """
    Compute VaR and ES using the bootstrap method for a portfolio.
    """
    boot_var_list = []
    boot_es_list = []
    n = len(portfolio_returns)
    
    for _ in range(n_bootstrap_samples):
        sample_indices = np.random.choice(n, size=n, replace=True)
        sample_returns = portfolio_returns[sample_indices]
        
        if net_position > 0:
            q = np.quantile(sample_returns, alpha)
            boot_var = -q
            tail_losses = sample_returns[sample_returns < q]
        else:
            q = np.quantile(sample_returns, 1 - alpha)
            boot_var = -q
            tail_losses = sample_returns[sample_returns > q]
        
        boot_var_list.append(boot_var)
        boot_es_list.append(-np.mean(tail_losses) if tail_losses.size > 0 else boot_var)
    
    var = np.mean(boot_var_list)
    es = np.mean(boot_es_list)
    return round(max(var, 0), 4), round(max(es, 0), 4)


def neg_log_likelihood(params, exceedances, n_u):
    """
    Compute the negative log-likelihood for the Generalized Pareto Distribution (GPD).
    """
    xi, beta = params
    if beta <= 0:
        return np.inf
    if np.any(1 + xi * exceedances / beta <= 0):
        return np.inf
    nll = n_u * np.log(beta) + (1/xi + 1) * np.sum(np.log(1 + xi * exceedances / beta))
    return nll