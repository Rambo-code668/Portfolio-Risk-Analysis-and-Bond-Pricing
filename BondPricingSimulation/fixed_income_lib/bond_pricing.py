# fixed_income_lib/bond_pricing.py

from scipy.optimize import newton
import numpy as np
from scipy.stats import norm
import pandas as pd

from fixed_income_lib.utils.validation import *
# Helper function 1: Generate czsh flows for a coupon bond
def calculate_cash_flows(face_value, coupon_rate, maturity, semi_annual=True):
    """
    Generate cash flows for a coupon bond.
    Returns a list of tuples (t, cashflow), where:
      - t is the time in years at which the cash flow occurs.
      - cashflow is the coupon payment; the final period includes the face value.
    """
    frequency = 2 if semi_annual else 1
    periods = int(maturity * frequency)
    coupon_payment = (coupon_rate / frequency) * face_value

    cash_flows = []
    for i in range(1, periods + 1):
        t = i / frequency
        cf = coupon_payment
        if i == periods:
            cf += face_value
        cash_flows.append((t, cf))
    return cash_flows

# Helper function 2: Calculate YTM using Newton-Raphson method
def newton_raphson_ytm(cash_flows, price, frequency=2, guess=0.05):
    """
    Solve for the yield to maturity (YTM) using Newton-Raphson such that NPV = Price.
    Input:
      - cash_flows: list of tuples (t, cf) where t is in years.
      - price: observed bond price.
      - frequency: number of payments per year.
      - guess: initial guess for YTM.
    Returns: the estimated YTM (annualized).
    """
    def npv(r):
        total = 0.0
        for t, cf in cash_flows:
            total += cf / ((1 + r / frequency) ** (t * frequency))
        return total - price

    try:
        return newton(npv, guess)
    except RuntimeError:
        return float('nan')

# Helper function 3: Calculate simple volatility (annualized from the basic interval)
def simple_volatility(delta_y, interval):
    """
    Estimate volatility using the simple formula:
      sigma_interval = std(delta_y) * sqrt(interval)
    Inputs:
      - delta_y: a series/array of yield changes.
      - interval: the time interval corresponding to the changes (e.g., 1 day, 1 month).
    Returns: volatility over the specified interval.
    """
    return np.std(delta_y) * np.sqrt(interval)

class FixedIncomePrmSingle:
    def __init__(self, yield_curve_today, maturity, is_zcb, coupon_rate=None, semi_annual_payment=False, face_value=100):
        """
        Initialize a fixed-income instrument.
        
        Parameters:
         - yield_curve_today: DataFrame representing today's yield curve.
                              Columns correspond to months (1 to 360).
         - maturity: bond maturity in years.
         - is_zcb: Boolean indicating if the bond is a zero coupon bond.
         - coupon_rate: Coupon rate (e.g., 0.05 for 5%); only applicable for coupon bonds.
         - semi_annual_payment: Whether coupon payments are semi-annual (otherwise, annual).
         - face_value: The bond's face value (default 100).
        """
        check_yield_curve(yield_curve_today)
        validate_bond_parameters(maturity, 1, face_value, is_zcb, coupon_rate, semi_annual_payment)
        self.yield_curve_today = yield_curve_today
        self.maturity = maturity
        self.is_zcb = is_zcb
        self.coupon_rate = coupon_rate
        self.semi_annual_payment = semi_annual_payment
        self.face_value = face_value
        self.frequency = 2 if semi_annual_payment else 1

        self.price = None
        self.macaulay_duration = None
        self.modified_duration = None
        self.ytm = None
        self.cashflows = None
        self.cashflow_times = None

    def fit(self):
        """
        Price the bond based on today's yield curve and calculate YTM, 
        Macaulay duration, and Modified duration.
        
        For zero coupon bonds, use the direct formula.
        For coupon bonds, generate cash flows using calculate_cash_flows
        and estimate YTM using newton_raphson_ytm.
        """
        if self.is_zcb:
            months = int(self.maturity * 12)
            yield_rate = self.yield_curve_today.iloc[0, months - 1] / 100.0
            self.price = self.face_value / ((1 + yield_rate) ** self.maturity)
            self.macaulay_duration = self.maturity
            self.modified_duration = self.macaulay_duration / (1 + yield_rate)
        else:
            # For coupon bonds, generate cash flows
            self.cashflows = calculate_cash_flows(self.face_value, self.coupon_rate, self.maturity, self.semi_annual_payment)
            self.cashflow_times = [t for (t,cf) in self.cashflows]

            # Price the bond by discounting cash flows using corresponding momthly yields
            price = 0.0
            for (t,cf) in self.cashflows:
                month_index = int(round(t * 12))
                y = self.yield_curve_today.iloc[0, month_index - 1] / 100.0
                price += cf / ((1 + y) ** t)
            self.price = price

            # Calculate YTM using Newton-Raphson method
            self.ytm = newton_raphson_ytm(self.cashflows, self.price, self.frequency)

            # Calculate Macaulay Duration as the weighted average time (using present value weights).
            weighted_sum = 0.0
            for i, (t, cf) in enumerate(self.cashflows, start=1):
                pv = cf / ((1 + self.ytm / self.frequency) ** i)
                weighted_sum += t * pv
            self.macaulay_duration = weighted_sum / self.price
            self.modified_duration = self.macaulay_duration / (1 + self.ytm / self.frequency)
            
    def DurationNormal(self, historical_yield_curve, volatility_model='simple', interval=1, alpha=0.05, position=1.0):
        """
        Calculate duration-based risk metrics (VaR and ES):
         - Extract the historical yield series for the column corresponding to the bond's maturity.
         - Compute yield changes (delta_yield) and then estimate volatility using the selected volatility model.
         - Map the bond's sensitivity (price * modified_duration) to risk measures.
        
        Parameters:
         - historical_yield_curve: DataFrame of historical yield curves (columns correspond to months,
                                   rows are dates).
         - volatility_model: 'simple', 'ewma', or 'garch'
         - interval: Time interval corresponding to the data frequency (e.g., 1 day or 1 month).
         - alpha: Confidence level for VaR (e.g., 0.05).
         - position: Position size multiplier.
         
        Returns: Dictionary with keys 'VaR', 'ES', 'mapped_position', and 'volatility'.
        """
        months = int(self.maturity * 12)
        yield_series = historical_yield_curve.iloc[:, months - 1]
        delta_yield = yield_series.diff().dropna()

        # Estimate volatility based on the selected model
        if volatility_model == 'simple':
            sigma_daily = simple_volatility(delta_yield, interval = 1)
        elif volatility_model == 'ewma':
            lambda_ = 0.94
            ewma_var = 0.0
            for change in delta_yield:
                ewma_var = lambda_ * ewma_var + (1 - lambda_) * change ** 2
            sigma_daily = np.sqrt(ewma_var)
        elif volatility_model == 'garch':
            sigma_daily = delta_yield.std()  # placeholder
        else:
            raise ValueError("Invalid volatility model")

        sigma_interval = sigma_daily * np.sqrt(interval)
        mapped_position = self.price * self.modified_duration
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
        Conduct a historical simulation to compute the bond's profit and loss (PnL) distribution,
        and then estimate VaR and ES.
         - Resample the historical yield curve to daily frequency and use linear interpolation.
         - Compute shifts in the yield curve for the specified interval.
         - For each shift, recalculate the bond's price and obtain the PnL.
        
        Parameters:
         - historical_yield_curve: DataFrame of historical yield curves with date index.
         - interval: The time interval over which the yield changes (in days).
         - alpha: Confidence level for VaR (e.g., 0.05).
         - position: Position size multiplier.
         
        Returns: Dictionary with keys 'VaR', 'ES', and 'pnl_distribution'.
        """
        if not isinstance(historical_yield_curve.index, pd.DatetimeIndex):
            historical_yield_curve.index = pd.to_datetime(historical_yield_curve.index)
        daily_yield_curve = historical_yield_curve.resample('D').interpolate(method='linear')

        months = int(self.maturity * 12)
        yield_series = daily_yield_curve.iloc[:, months - 1]
        shifts = yield_series.shift(-interval) - yield_series
        shifts = shifts.dropna()

        pnl = []
        if self.is_zcb:
            for shift in shifts:
                base_yield = self.yield_curve_today.iloc[0, months - 1] / 100.0
                new_yield = base_yield + shift / 100.0
                new_price = self.face_value / ((1 + new_yield) ** self.maturity)
                pnl.append(new_price - self.price)
        else:
            for shift in shifts:
                new_price = 0.0
                for (t,cf) in self.cashflows:
                    month_index = int(round(t * 12))
                    base_yield = self.yield_curve_today.iloc[0, month_index - 1] / 100.0
                    new_yield = base_yield + shift / 100.0
                    new_price += cf / ((1 + new_yield) ** t)
                pnl.append(new_price - self.price)

        pnl = np.array(pnl)
        VaR = -np.percentile(pnl, alpha * 100)
        losses = -pnl[pnl <= -VaR]
        ES = losses.mean() if len(losses) > 0 else 0.0

        return {
            'VaR': VaR,
            'ES': ES,
            'pnl_distribution': pnl
        }
