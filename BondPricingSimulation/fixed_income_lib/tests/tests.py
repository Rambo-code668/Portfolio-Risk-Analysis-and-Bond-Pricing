import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm

# Import classes and functions using absolute imports.
from fixed_income_lib.bond_pricing import FixedIncomePrmSingle
from fixed_income_lib.portfolio import FixedIncomePrmPort
from fixed_income_lib.utils.validation import (
    check_maturity_and_payment, 
    check_yield_curve,
    check_portfolio_arrays, 
    validate_bond_parameters
)

def generate_complex_yield_curve():
    """
    Generate a more complex yield curve.
    Assumptions:
      - The yield curve has 360 months (30 years).
      - Base yields increase linearly from 2% to 6% over the period.
      - A small normal random noise (mean=0, std=0.1) is added to simulate market variations.
    
    Returns:
      A DataFrame with 1 row and 360 columns, where column names are 1, 2, ..., 360 and yields are in percentage.
    """
    months = np.arange(1, 361)
    # Base yield increases linearly from 2% to 6%
    base_yields = 2 + 4 * (months - 1) / (359)
    # Add small random noise
    noise = np.random.normal(0, 0.1, size=months.shape)
    yields = base_yields + noise
    # Ensure yields are non-negative and round to 2 decimals
    yields = np.clip(yields, 0, None)
    yields = np.round(yields, 2)
    df = pd.DataFrame([yields], columns=months)
    return df

def generate_historical_yield_curve(base_curve, days=20):
    """
    Generate simulated historical yield curve data.
    
    Parameters:
      - base_curve: The current yield curve DataFrame (1 row and 360 columns).
      - days: Number of days of historical data to generate.
    
    Method:
      Each day, the yield curve is perturbed from the base curve by a random fluctuation
      from a normal distribution (mean=0, std=0.05).
    
    Returns:
      A DataFrame of shape (days, 360) with a DatetimeIndex.
    """
    np.random.seed(0)  # Fixed seed for reproducibility
    base = base_curve.values[0]
    data = []
    for i in range(days):
        # Generate a random fluctuation for each month
        daily_noise = np.random.normal(0, 0.05, size=base.shape)
        daily_yields = base + daily_noise
        daily_yields = np.round(np.clip(daily_yields, 0, None), 2)
        data.append(daily_yields)
    df = pd.DataFrame(data, columns=base_curve.columns)
    # Create a datetime index starting at a specific date
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    df.index = pd.to_datetime(dates)
    return df

class TestBondPricing(unittest.TestCase):
    """
    Extended test suite for single-bond pricing (FixedIncomePrmSingle) with complex yield curve.
    """
    def setUp(self):
        # Generate a complex yield curve
        self.yield_curve = generate_complex_yield_curve()

    def test_zcb_pricing(self):
        """
        Test pricing of a zero-coupon bond.
        Expected price = face_value / (1 + yield)^maturity.
        """
        zcb = FixedIncomePrmSingle(
            yield_curve_today=self.yield_curve,
            maturity=5.0,
            is_zcb=True,
            coupon_rate=None,
            semi_annual_payment=False,
            face_value=1000
        )
        zcb.fit()
        # Calculate expected price using yield from the curve at month 60.
        yield_rate = self.yield_curve.iloc[0, 60 - 1] / 100.0  # month 60 is 5 years
        expected_price = 1000 / ((1 + yield_rate) ** 5.0)
        print("Zero-Coupon Bond Price:", zcb.price, "Expected:", expected_price)
        self.assertAlmostEqual(zcb.price, expected_price, delta=2.0)

    def test_negative_yield_curve_validation(self):
        df = generate_complex_yield_curve()
        df.iloc[0, 10] = -0.5
        with self.assertRaises(ValueError):
            check_yield_curve(df)

    def test_coupon_bond_pricing_annual(self):
        """
        Test pricing of an annual coupon bond where coupon rate equals yield.
        Expected price should be near par (1000).
        """
        bond = FixedIncomePrmSingle(
            yield_curve_today=self.yield_curve,
            maturity=5.0,
            is_zcb=False,
            coupon_rate=0.05,
            semi_annual_payment=False,  # Annual payments
            face_value=1000
        )
        bond.fit()
        print("Annual Coupon Bond Price:", bond.price)
        self.assertGreater(bond.ytm, 0)
        self.assertGreater(bond.modified_duration, 0)
        self.assertGreater(bond.macaulay_duration, 0)

    def test_coupon_bond_pricing_semi_annual(self):
        """
        Test pricing of a semi-annual coupon bond.
        """
        bond = FixedIncomePrmSingle(
            yield_curve_today=self.yield_curve,
            maturity=5.0,
            is_zcb=False,
            coupon_rate=0.05,
            semi_annual_payment=True,  # Semi-annual payments
            face_value=1000
        )
        bond.fit()
        print("Semi-Annual Coupon Bond Price:", bond.price)
        self.assertEqual(bond.frequency, 2)
        self.assertGreater(bond.macaulay_duration, 0)
        self.assertGreater(bond.modified_duration, 0)

class TestPortfolio(unittest.TestCase):
    """
    Extended test suite for portfolio-level analysis (FixedIncomePrmPort)
    with more complex (realistic) data.
    """
    def setUp(self):
        # Generate a complex yield curve (with 5% to 6% variation over time).
        self.yield_curve = generate_complex_yield_curve()
        
        # Simulate portfolio arrays for 3 bonds:
        # Bond 1: Zero-Coupon, 5 years
        # Bond 2: Annual coupon bond, 10 years, coupon 6%
        # Bond 3: Semi-annual coupon bond, 7.5 years, coupon 4%
        self.maturity_s = [5.0, 10.0, 7.5]
        self.num_assets_s = [1, 2, 3]
        self.face_value_s = [1000, 1000, 1000]
        self.is_zcb_s = [True, False, False]
        self.coupon_rate_s = [None, 0.06, 0.04]
        self.semi_annual_s = [False, False, True]

        # Also generate a simulated historical yield curve
        self.historical_yield = generate_historical_yield_curve(self.yield_curve, days=20)

    def test_portfolio_fit(self):
        """
        Test that the portfolio can be priced and produce a summary DataFrame and metrics.
        """
        portfolio = FixedIncomePrmPort(
            yield_curve_today=self.yield_curve,
            maturity_s=self.maturity_s,
            num_assets_s=self.num_assets_s,
            face_value_s=self.face_value_s,
            is_zcb_s=self.is_zcb_s,
            coupon_rate_s=self.coupon_rate_s,
            semi_annual_payment_s=self.semi_annual_s
        )
        summary_df, metrics = portfolio.fit()
        print("Portfolio Summary DataFrame:")
        print(summary_df)
        print("Portfolio Metrics:")
        print(metrics)
        self.assertEqual(len(summary_df), 3)
        self.assertIn('portfolio_value', metrics)
        self.assertIn('portfolio_weighted_modified_duration', metrics)
        self.assertGreater(metrics['portfolio_value'], 0)

    def test_duration_normal_and_print(self):
        """
        Test the DurationNormal method and print the resulting risk metrics.
        """
        portfolio = FixedIncomePrmPort(
            yield_curve_today=self.yield_curve,
            maturity_s=self.maturity_s,
            num_assets_s=self.num_assets_s,
            face_value_s=self.face_value_s,
            is_zcb_s=self.is_zcb_s,
            coupon_rate_s=self.coupon_rate_s,
            semi_annual_payment_s=self.semi_annual_s
        )
        portfolio.fit()

        res_normal = portfolio.DurationNormal(self.historical_yield, volatility_model='simple', interval=1, alpha=0.05)
        print("\nDurationNormal Risk Metrics:")
        print(res_normal)
        self.assertIsInstance(res_normal, dict)
        self.assertIn('VaR', res_normal)
        self.assertIn('ES', res_normal)
        self.assertGreaterEqual(res_normal['VaR'], 0)

    def test_historical_simulation_and_print(self):
        """
        Test the HistoricalSimulation method and print the PnL distribution and risk metrics.
        """
        portfolio = FixedIncomePrmPort(
            yield_curve_today=self.yield_curve,
            maturity_s=self.maturity_s,
            num_assets_s=self.num_assets_s,
            face_value_s=self.face_value_s,
            is_zcb_s=self.is_zcb_s,
            coupon_rate_s=self.coupon_rate_s,
            semi_annual_payment_s=self.semi_annual_s
        )
        portfolio.fit()
        
        res_hist = portfolio.HistoricalSimulation(self.historical_yield, interval=1, alpha=0.05)
        print("\nHistorical Simulation Risk Metrics:")
        print(res_hist)
        self.assertIsInstance(res_hist, dict)
        self.assertIn('VaR', res_hist)
        self.assertIn('ES', res_hist)
        self.assertIn('pnl_distribution', res_hist)
        self.assertIsInstance(res_hist['pnl_distribution'], np.ndarray)
        self.assertGreaterEqual(res_hist['VaR'], 0)

class TestValidationEdgeCases(unittest.TestCase):
    def test_invalid_maturity(self):
        with self.assertRaises(ValueError):
            validate_bond_parameters(3.3, 1, 1000, False, 0.05, False)

    def test_negative_face_value(self):
        with self.assertRaises(ValueError):
            validate_bond_parameters(5.0, 1, -1000, False, 0.05, False)

    def test_coupon_rate_missing(self):
        with self.assertRaises(ValueError):
            validate_bond_parameters(5.0, 1, 1000, False, None, False)

    def test_coupon_rate_for_zcb(self):
        with self.assertRaises(ValueError):
            validate_bond_parameters(5.0, 1, 1000, True, 0.05, False)

    def test_coupon_rate_above_one(self):
        with self.assertRaises(ValueError):
            validate_bond_parameters(5.0, 1, 1000, False, 1.5, False)

if __name__ == '__main__':
    unittest.main()