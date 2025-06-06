# fixed_income_lib/portfolio.py

import numpy as np
import pandas as pd
from scipy.stats import norm

from fixed_income_lib.bond_pricing import FixedIncomePrmSingle
from fixed_income_lib.utils.validation import *

class FixedIncomePrmPort:
    """
    FixedIncomePrmPort (Portfolio Analysis)
    
    Purpose:
      Analyze portfolios of bonds with mixed maturities/types.

    Key Methods:
      - __init__: Store arrays of bond parameters and validate them.
      - fit: Price each bond, compute portfolio metrics, and generate a summary DataFrame.
      - DurationNormal: Compute VaR and ES for the portfolio using a normal approximation 
                       with portfolio-level modified duration.
      - HistoricalSimulation: Compute PnL for all bonds under historical yield shifts 
                             and estimate VaR/ES from the resulting distribution.
    """
    def __init__(self, yield_curve_today, maturity_s, num_assets_s, face_value_s, is_zcb_s,coupon_rate_s=None, semi_annual_payment_s=None):
        """
        Initialize a portfolio of bonds. All input arrays must have the same length.
        
        Parameters:
          yield_curve_today (DataFrame): The current yield curve (columns are months).
          maturity_s (array-like): Array of maturities (in years).
          num_assets_s (array-like): Number of units held for each bond.
          face_value_s (array-like): Face value of each bond.
          is_zcb_s (array-like): Boolean array: True if zero-coupon, False if coupon bond.
          coupon_rate_s (array-like or None): Coupon rates for coupon bonds; None for zero-coupon.
          semi_annual_payment_s (array-like or None): 
             Boolean array specifying semi-annual coupon payment. 
             If None, default is True for coupon bonds, False for zero-coupon.
        """

        check_yield_curve(yield_curve_today)
        check_portfolio_arrays(maturity_s, num_assets_s, face_value_s, is_zcb_s)
        if coupon_rate_s is not None:
            check_portfolio_arrays(coupon_rate_s)
        self.maturity_s = np.array(maturity_s)
        self.num_assets_s = np.array(num_assets_s)
        self.face_value_s = np.array(face_value_s)
        self.is_zcb_s = np.array(is_zcb_s)
        self.yield_curve_today = yield_curve_today

        self.n = len(self.maturity_s)
        
        if coupon_rate_s is not None:
            check_portfolio_arrays(coupon_rate_s)
            self.coupon_rate_s = np.array(coupon_rate_s)
        else:
            self.coupon_rate_s = np.array([None] * self.n)
        
        # Validate semi_annual_payment_s if provided; otherwise, set default.
        if semi_annual_payment_s is not None:
            check_portfolio_arrays(semi_annual_payment_s)
            self.semi_annual_payment_s = np.array(semi_annual_payment_s)
        else:
            default_semi_annual = [False if zcb else True for zcb in self.is_zcb_s]
            self.semi_annual_payment_s = np.array(default_semi_annual)
        
        # Validate each pair (maturity, semi_annual_payment) using check_maturity_and_payment.
        for mat, sa in zip(self.maturity_s, self.semi_annual_payment_s):
            check_maturity_and_payment(mat, sa)
        
        # Prepare to store portfolio data
        self.bonds_data = []
        self.portfolio_value = None
        self.portfolio_weighted_mod_duration = None
        self.summary_df = None

    def fit(self):
        """
        Price each bond using FixedIncomePrmSingle, compute portfolio metrics,
        and generate a summary DataFrame of bond-level info.
        
        Returns:
          (summary_df, portfolio_metrics)
            - summary_df: DataFrame with each row describing a bond
            - portfolio_metrics: dict with keys 'portfolio_value' 
                                and 'portfolio_weighted_modified_duration'
        """
        bonds_list = []
        total_value = 0.0

        # Build each bond and compute its position value
        for i in range(self.n):
            bond = FixedIncomePrmSingle(
                yield_curve_today=self.yield_curve_today,
                maturity=self.maturity_s[i],
                is_zcb=self.is_zcb_s[i],
                coupon_rate=self.coupon_rate_s[i],
                semi_annual_payment=self.semi_annual_payment_s[i],
                face_value=self.face_value_s[i]
            )
            bond.fit()

            position_value = self.num_assets_s[i] * bond.price
            bond_dict = {
                'maturity': self.maturity_s[i],
                'num_assets': self.num_assets_s[i],
                'face_value': self.face_value_s[i],
                'is_zcb': self.is_zcb_s[i],
                'coupon_rate': self.coupon_rate_s[i],
                'semi_annual_payment': self.semi_annual_payment_s[i],
                'price': bond.price,
                'macaulay_duration': bond.macaulay_duration,
                'modified_duration': bond.modified_duration,
                'position_value': position_value,
                'cashflows': bond.cashflows if not self.is_zcb_s[i] else None,
                'cashflow_times': bond.cashflow_times if not self.is_zcb_s[i] else None
            }
            bonds_list.append(bond_dict)
            total_value += position_value

        self.portfolio_value = total_value
        # Compute portfolio-level weighted modified duration
        weighted_mod_duration = 0.0
        for bond_dict in bonds_list:
            # Weight is fraction of total portfolio value
            weight = bond_dict['position_value'] / total_value
            bond_dict['weight'] = weight
            weighted_mod_duration += weight * bond_dict['modified_duration']

        self.portfolio_weighted_mod_duration = weighted_mod_duration
        self.bonds_data = bonds_list
        self.summary_df = pd.DataFrame(bonds_list)

        portfolio_metrics = {
            'portfolio_value': total_value,
            'portfolio_weighted_modified_duration': weighted_mod_duration,
        }
        return self.summary_df, portfolio_metrics

    def DurationNormal(self, historical_yield_curve, volatility_model='simple', interval=1, alpha=0.05, position=1.0):
        """
        Compute VaR and ES for the portfolio using a normal approximation 
        (portfolio-level duration approach).
        
        Steps:
          1. Estimate a 'representative' yield series from historical_yield_curve 
             (e.g., using weighted average maturity or a simpler approach).
          2. Compute yield changes and estimate volatility (simple, ewma, or garch).
          3. Map portfolio sensitivity: mapped_position = portfolio_value * portfolio_weighted_mod_duration.
          4. Compute VaR and ES under a normal assumption.
        
        Parameters:
          historical_yield_curve (DataFrame): Past yield curves (columns = months, index = dates).
          volatility_model (str): 'simple', 'ewma', or 'garch'.
          interval (float): Time interval for scaling volatility, e.g. 1 day or 1 month.
          alpha (float): Confidence level for VaR, e.g. 0.05.
          position (float): Position multiplier, default 1.0.
        
        Returns:
          dict with keys: 'VaR', 'ES', 'mapped_position', 'volatility'.
        """
        if self.portfolio_value is None or self.bonds_data is None:
            raise ValueError("Call fit() before computing portfolio risk metrics.")

        # Approximate a representative maturity
        weighted_maturity = sum(bond['weight'] * bond['maturity'] for bond in self.bonds_data)
        rep_month = int(round(weighted_maturity * 12))
        rep_month = max(1, rep_month)

        yield_series = historical_yield_curve.iloc[:, rep_month - 1]
        delta_yield = yield_series.diff().dropna()

        if volatility_model == 'simple':
            sigma_daily = delta_yield.std()
        elif volatility_model == 'ewma':
            lambda_ = 0.94
            ewma_var = 0.0
            for change in delta_yield:
                ewma_var = lambda_ * ewma_var + (1 - lambda_) * change ** 2
            sigma_daily = np.sqrt(ewma_var)
        elif volatility_model == 'garch':
            sigma_daily = delta_yield.std()  
        else:
            raise ValueError("Invalid volatility model")

        sigma_interval = sigma_daily * np.sqrt(interval)
        mapped_position = self.portfolio_value * self.portfolio_weighted_mod_duration
        quantile = norm.ppf(alpha)

        VaR = abs(position) * mapped_position * sigma_interval * abs(quantile)
        ES = abs(position) * mapped_position * sigma_interval * (norm.pdf(quantile) / alpha)

        return {
            'VaR': VaR,
            'ES': ES,
            'mapped_position': mapped_position,
            'volatility': sigma_interval
        }

    def HistoricalSimulation(self, historical_yield_curve, interval=1, alpha=0.05, position=1.0):
        """
        Perform a historical simulation to compute the portfolio PnL distribution,
        then derive VaR and ES from that distribution.
        
        Steps:
          1. Resample historical_yield_curve to daily (or desired frequency) 
             and linearly interpolate.
          2. For each date shift, re-price every bond using the shifted yields. 
             Sum these bond prices to get the portfolio value for that scenario.
          3. Compute PnL = new_portfolio_value - original_portfolio_value.
          4. From the PnL distribution, compute VaR and ES at the chosen alpha.
        
        Parameters:
          historical_yield_curve (DataFrame): Past yield curves with date index.
          interval (float): The time interval (days or months) used to define yield shifts.
          alpha (float): Confidence level for VaR, e.g. 0.05.
          position (float): Position multiplier.
        
        Returns:
          dict with keys: 'VaR', 'ES', 'pnl_distribution'.
        """
        if self.portfolio_value is None or self.bonds_data is None:
            raise ValueError("Call fit() before running HistoricalSimulation.")

        if not isinstance(historical_yield_curve.index, pd.DatetimeIndex):
            historical_yield_curve.index = pd.to_datetime(historical_yield_curve.index)
        daily_yield_curve = historical_yield_curve.resample('D').interpolate(method='linear')

        weighted_maturity = sum(bond['weight'] * bond['maturity'] for bond in self.bonds_data)
        rep_month = int(round(weighted_maturity * 12))
        rep_month = max(1, rep_month)

        yield_series = daily_yield_curve.iloc[:, rep_month - 1]
        shifts = yield_series.shift(-interval) - yield_series
        shifts = shifts.dropna()

        original_portfolio_value = self.portfolio_value
        pnl = []
        for shift in shifts:
            new_portfolio_value = 0.0
            for bond in self.bonds_data:
                maturity = bond['maturity']
                num_assets = bond['num_assets']
                face_value = bond['face_value']
                is_zcb = bond['is_zcb']
                base_yield_curve = self.yield_curve_today
                if is_zcb:
                    months = int(round(maturity * 12))
                    base_yield = base_yield_curve.iloc[0, months - 1] / 100.0
                    new_yield = base_yield + shift / 100.0
                    new_price = face_value / ((1 + new_yield) ** maturity)
                else:
                    cashflows = bond['cashflows']
                    cashflow_times = bond['cashflow_times']
                    new_price = 0.0
                    for cf, t in zip(cashflows, cashflow_times):
                        t_months = int(round(t * 12))
                        base_yield = base_yield_curve.iloc[0, t_months - 1] / 100.0
                        new_yield = base_yield + shift / 100.0
                        new_price += cf / ((1 + new_yield) ** (t_months / 12.0))
                new_portfolio_value += num_assets * new_price
            pnl.append(new_portfolio_value - original_portfolio_value)

        pnl = np.array(pnl)
        VaR = -np.percentile(pnl, alpha * 100)
        losses = -pnl[pnl <= -VaR]
        ES = losses.mean() if len(losses) > 0 else 0.0

        return {
            'VaR': VaR,
            'ES': ES,
            'pnl_distribution': pnl
        }