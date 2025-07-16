from typing import Union, List ; import numpy as np ; import pandas as pd


##########################################################################################################################################################################
##########################################################################################################################################################################
## FixedIncomePrmSingle ###############################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class FixedIncomePrmSingle:
    
    """
    Main
    ----
    This class provides the functionalities to price fixed-income securities, specifically bonds, and calculate their 
    associated risk metrics such as duration, yield-to-maturity, Value-at-Risk (VaR), and Expected Shortfall (ES). 
    
    Important
    ---------
    - Due to the constraint in how the 'maturity' parameter is designed, it can only accept either an integer value representing full years or a decimal value
    like 0.5 to indicate semi-annual periods. 
    - As a result of this constraint, the class is designed to assist in evaluating the risk and return of a bond position at the time of the bond's issuance or
    just after a coupon payment in the case of a coupon-bearing bond. For a zero-coupon bond, it is meant to be used either at the time of issuance or
    every six months thereafter.
    The class supports both zero-coupon bonds (ZCB) and coupon-bearing bonds, with the option for annual or semi-annual payments.

    Purpose
    -------
    This class aims to provide a solid foundation for understanding and analyzing fixed-income securities. It is designed to serve two main audiences:

    1. Those who are looking to further develop or integrate more complex calculations and risk management strategies. 
    This class offers a reliable basis upon which additional features and functionalities can be built.
    
    2. Those who intend to use the class within its given boundaries for straightforward pricing and risk assessment of fixed-income securities.
    For this audience, the class provides ready-to-use methods for calculating key metrics like bond price, duration, yield-to-maturity, Value-at-Risk (VaR), and Expected Shortfall (ES).

    - By catering to both of these needs, the class offers a flexible and robust starting point for a range of fixed-income analysis and risk management tasks.

    Initial Parameters
    ------------------
    - yield_curve_today : The DataFrame containing the current yield curve, with terms in months as columns and the yield rates as row values.
    The yield rates should be represented as a percentage, e.g., 3% should be given as 3.

    ```
    | Date      |   1  |   2  |   3  |  4   |  5   | ...  | 355  |  356 |  357 |  358 |  359 |  360 |
    |-----------|------|------|------|------|------|------|------|------|------|------|------|------|
    | 08/29/2023| 5.54 | 5.53 | 5.52 | 5.51 | 5.51 | 4.87 | 4.56 | 4.26 | 4.21 | 4.19 | 4.18 | 4.19 |
    ```

    - maturity: The time to maturity of the bond in years. Must be either an integer value for full years or a decimal value like 0.5 to indicate semi-annual periods.
    Default is 1.
    - num_assets: Number of identical assets/bonds for the calculation.
    Default is 1.
    - face_value: The face value of the bond.
    Default is 1000.
    - is_zcb: Indicator to specify whether the bond is a zero-coupon bond. Default is True.
    - coupon_rate: The annual coupon rate of the bond represented as a percentage (e.g., 3% should be given as 3) (required if is_zcb is set to False).
    Default is None.
    - semi_annual_payment: Indicator to specify if the bond pays coupons semi-annually (required if is_zcb is set to False).
    Default is None.

    Methods
    -------
    - DurationNormal(): Calculates the Value-at-Risk (VaR) and Expected Shortfall (ES) of a bond position using a normal distribution model.
    This method assumes a normal distribution of yield changes and uses the modified duration of the bond
    to approximate the change in bond price for a given change in yield.
    - HistoricalSimulation(): In this approach, historical data is used to generate a distribution of potential outcomes,
    which can be more realistic compared to model-based approaches like the Duration-Normal method.

    Attributes
    ----------
    - bond_price: The price of the bond calculated based on either zero-coupon or coupon-bearing.
    - duration: The Macaulay duration of the bond (in years).
    - modified_duration: The modified duration of the bond (in years).
    - yield_to_maturity: Yield to maturity of the bond (yearly).

    Exceptions
    ----------
    - Raises an exception if the coupon_rate and semi_annual_payment are not provided when is_zcb is set to False.
    - Raises an exception if the maturity provided is beyond the highest maturity available in the yield_curve_today DataFrame.
    - Raises an exception if the maturity provided ends with .5, is_zcb is False and semi_annual_payment is not True.

    Note
    ----
    The yield curve provided should be a monthly yield curve, with maturity ranging from 1 to n months.
    """

    import warnings ; warnings.filterwarnings("ignore")
    
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************
    
    def __init__(self, yield_curve_today : pd.DataFrame,              # not in decimal point. For instance: for 3% it should be 3.
                       maturity: int = 1,                             # in years
                       num_assets: int = 1,
                       face_value: int = 1000,
                       is_zcb: bool = True,     
                       coupon_rate: int = None,                       # not in decimal point. For instance: for 3% it should be 3.
                       semi_annual_payment: bool = None):
        

        from inputsControlFunctions import (check_yield_curve_today, check_maturity, check_num_assets,check_face_value, check_is_zcb,
                                                       check_coupon_rate, check_semi_annual_payment)

        # check procedure
        self.yield_curve_today = check_yield_curve_today(yield_curve_today) ; self.maturity = check_maturity(maturity) ; self.num_assets = check_num_assets(num_assets)
        self.face_value = check_face_value(face_value) ; check_is_zcb(is_zcb) ; self.is_zcb = is_zcb

        # if we have a coupound bond /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if is_zcb == False:
            if coupon_rate is None and semi_annual_payment is None: 
                raise Exception("Both coupon_rate and semi_annual_payment must be provided when is_zcb is False.")
            if coupon_rate is None or semi_annual_payment is None: 
                raise Exception("Either coupon_rate or semi_annual_payment is missing. Both are required when is_zcb is False.")
            if coupon_rate is not None and semi_annual_payment is not None: 
                # check procedure
                self.coupon_rate = check_coupon_rate(coupon_rate) ; check_semi_annual_payment(semi_annual_payment) ; self.semi_annual_payment = semi_annual_payment
            if (self.maturity * 2) % 2 != 0 and not self.semi_annual_payment:
                raise Exception("If maturity is in fractions of .5 years, then semi_annual_payment must be set to True.")
    
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (self.maturity * 12) > self.yield_curve_today.columns[-1]:
            raise Exception(f"""
            The provided maturity for the bond is {maturity} year(s), which is equivalent to {maturity * 12} months.
            This method requires the user to provide a 'yield_curve_today' with monthly frequency.
            The highest maturity available in the 'yield_curve_today' DataFrame is {self.yield_curve_today.columns[-1]} months.
            Therefore, the method does not have the required yield for the given maturity.
            """)

        self.bond_price = None  ; self.duration = None ; self.modified_duration = None ; self.yield_to_maturity = None

        self.fit()  # Automatically call the fit method

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np ; from scipy.optimize import newton ; import pandas as pd

        # -----------------------------------------------------------------------------------------------------------
        # Note:
        # - The yield curve should be a pandas DataFrame with the following characteristics:
        # - The index should represent today's date or the last available date.
        # - Columns should start from 1 and go onward, representing the terms in months.
        # - The DataFrame should contain a single row with interest rates, corresponding element-wise to the columns.
        # - Interest rates should be in the format like 5.56, 5.22, etc., without decimal points.
        # ------------------------------------------------------------------------------------------------------------

        # checking if the bond is a zero-coupon bond
        if self.is_zcb == True:

            # fetching the last yield value from the DataFrame for the given maturity
            exact_yield = self.yield_curve_today.loc[self.yield_curve_today.index[-1], self.maturity * 12]

            # calculating the bond's price using the zero-coupon bond formula
            self.bond_price = self.face_value / ( (1 + (exact_yield/100))  ** self.maturity)

            # setting duration equal to the bond's maturity for a zero-coupon bond
            self.duration = self.maturity 

            # storing the bond's yield to maturity
            self.yield_to_maturity = round(exact_yield/100,6)

            # calculating and storing the modified duration
            self.modified_duration = round(self.duration / (1 + self.yield_to_maturity), 4)

        # checking if the bond is not a zero-coupon bond
        if self.is_zcb == False:

            # checking if the bond pays coupons annually
            if not self.semi_annual_payment:
                paymentSchedules = np.arange(1,(self.maturity + 1),1) * 12 

            # checking if the bond pays coupons semi-annually
            if self.semi_annual_payment:

                # setting the payment schedule for semi-annual payments
                paymentSchedules = np.arange(0.5,(self.maturity + 0.5),0.5) * 12 

                # adjusting the coupon rate for semi-annual payments
                self.coupon_rate = self.coupon_rate / 2 

            # initializing an array to store bond cash flows
            cashFlows = np.zeros(len(paymentSchedules))

            # computing bond cash flows
            for i in range(len(cashFlows)):
                if i != (len(cashFlows)-1):
                    cashFlows[i] = self.face_value * (self.coupon_rate/100)
                else:
                    cashFlows[i] = (self.face_value * (self.coupon_rate/100)) + self.face_value

            # initializing an array to store present values of cash flows
            pvCashFlow = np.zeros(len(paymentSchedules))

            # computing the present value of cash flows
            for index, value in enumerate(paymentSchedules):
                exact_yield = self.yield_curve_today.loc[self.yield_curve_today.index[-1], value]
                pvCashFlow[index] = cashFlows[index] /  ((1+((exact_yield/100)))**(value/12)) 

            # calculating and storing the bond's price
            self.bond_price = np.sum(pvCashFlow)

            # defining a function to compute the difference between NPV and bond price
            def npv_minus_price(irr): 
                cashFlows = np.zeros(len(paymentSchedules))
                for i in range(len(cashFlows)):
                    if i != (len(cashFlows)-1):
                        cashFlows[i] = self.face_value * (self.coupon_rate/100)
                    else:
                        cashFlows[i] = (self.face_value * (self.coupon_rate/100)) + self.face_value
                pvCashFlow = np.zeros(len(paymentSchedules))
                for index, value in enumerate(paymentSchedules):
                    pvCashFlow[index] = cashFlows[index] /  ((1+((irr/100)))**(value/12)) 
                sumPvCashFlow = np.sum(pvCashFlow)
                return sumPvCashFlow - self.bond_price

            # setting an initial guess for the internal rate of return (IRR)
            initial_guess = 5.0

            # computing the IRR using the newton method
            irr_result = newton(npv_minus_price, initial_guess)

            # storing the computed yield to maturity
            self.yield_to_maturity = round(irr_result/100, 6)

            # computing bond's duration
            durationWeights = pvCashFlow / self.bond_price

            # initializing an array to store the product of weights and time
            Weights_times_t = np.zeros(len(durationWeights))

            # computing the product of weights and time
            for index, value in enumerate(durationWeights):
                Weights_times_t[index] = durationWeights[index] * (paymentSchedules[index] / 12)

            # calculating and storing the bond's duration
            self.duration = round(np.sum(Weights_times_t), 4)

            # determining the payment frequency
            if self.semi_annual_payment:
                frequency = 2  
            else:
                frequency = 1  

            # calculating and storing the modified duration
            self.modified_duration = self.duration / (1 + (self.yield_to_maturity / frequency))
            self.modified_duration = round(self.modified_duration, 4)

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # DURATION METHOD <<<<<<<<<<<<<<<

    def DurationNormal(self , yield_curve : pd.DataFrame, vol: str = "simple", interval: int = 1, alpha: float = 0.05, p: int = 1, q: int = 1, lambda_ewma: float = 0.94):

        """
        Main
        -----
        This method calculates the Value at Risk (VaR) and Expected Shortfall (ES) for a given bond position using the Modified Duration mapping approach.

        Parameters
        -----------
        - yield_curve : The yield curve DataFrame with different maturities, with terms in months as columns (int) and the yield rates as row values.

        ```
        | Date      |   1  |   2  |   3  |  4   |  5   | ...  | 355  |  356 |  357 |  358 |  359 |  360 |
        |-----------|------|------|------|------|------|------|------|------|------|------|------|------|
        | 01/03/2022| 0.05 | 0.06 | 0.08 | 0.22 | 0.40 | 0.78 | 1.04 | 1.37 | 1.55 | 1.63 | 2.05 | 2.01 |
        | 01/04/2022| 0.06 | 0.05 | 0.08 | 0.22 | 0.38 | 0.77 | 1.02 | 1.37 | 1.57 | 1.66 | 2.10 | 2.07 |
        ...
        | 08/28/2023| 5.56 | 5.53 | 5.58 | 5.56 | 5.44 | 4.98 | 4.69 | 4.38 | 4.32 | 4.20 | 4.48 | 4.29 |
        | 08/29/2023| 5.54 | 5.53 | 5.56 | 5.52 | 5.37 | 4.87 | 4.56 | 4.26 | 4.21 | 4.12 | 4.42 | 4.23 |

        ```
        - vol: Type of volatility to use ("simple", "garch", or "ewma"). Default is "simple".
        - interval: The time interval in days for which to calculate VaR (Value at Risk) and ES (Expected Shortfall). Default is 1.
          - The 'interval' is intended to represent a period over which the risk (VaR/ES) of the bond is assessed.
          - Limited to a maximum of 150 days to ensure meaningful and reliable results for the following reasons:
             1. Numerical Stability: Limiting the interval to 150 days helps maintain the numerical stability of the VaR and ES calculations.
             2. Market Conditions: An interval longer than 150 days may not accurately capture the fast-changing market conditions affecting the bond's risk profile.
             3. Model Assumptions: The underlying statistical and financial models are more accurate over shorter time frames.
        - alpha: Significance level for VaR/ES. Default is 0.05.
        - p: The 'p' parameter for the GARCH model. Default is 1.
        - q: The 'q' parameter for the GARCH model. Default is 1.
        - lambda_ewma: The lambda parameter for Exponentially Weighted Moving Average (EWMA) model. Default is 0.94.

        Methodology:
        ------------
        - Input Checks: Validates the types and values of input parameters.
        - Interval Constraints: Throws errors if the interval is inappropriate relative to the bond's maturity.
        - Interpolation:
            - The 'target_month' is calculated based on the Modified Duration of the bond, scaled to a monthly time frame (Modified Duration * 12).
            - Cubic Spline Interpolation is employed to get a smoother yield curve and extract yields for target months, IF NOT directly available in the original yield curve.
            - A window of months around the 'target_month' is selected to perform the cubic spline interpolation.
            Edge cases are handled by ensuring the interpolation window fits within the available yield curve data.
            - Once the interpolated yield curve is available, the yield for the 'target_month' is extracted. 
        - Volatility Estimation: Depending on the selected model ("simple", "garch", "ewma"), calculates the volatility of yield changes.
        - VaR and ES Calculation: Applies the Modified Duration mapping approach to calculate VaR and ES.

        Returns:
        --------
        DURATION_NORMAL : dict
            A dictionary containing calculated VaR, ES and other relevant metrics.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from scipy.stats import norm ; import numpy as np ; from scipy.interpolate import CubicSpline ; import pandas as pd
        
        from inputsControlFunctions import (check_yield_curve_df, check_vol, check_interval, check_alpha)

        # check procedure 
        yield_curve = check_yield_curve_df(yield_curve) ; vol = check_vol(vol) ; interval = check_interval(interval) ; alpha = check_alpha(alpha) 

        if interval >= (self.maturity * 12 * 25):
            raise ValueError(f"""
            The maturity of the bond is {self.maturity} years.
            Which is equivalent to {self.maturity * 12} months or {self.maturity * 12 * 25} days.
            The 'interval' for the VaR/ES (Value at Risk/Expected Shortfall) calculation cannot exceed, or be equal to, {self.maturity * 12 * 25} days.
            Right now, it is set to {interval}!
            The 'interval' is meant to represent a time period over which you're assessing the risk (VaR/ES) of the bond. 
            If the 'interval' were to match or exceed the bond's maturity, the risk metrics would no longer be meaningful. 
            """)
        
        limit = 25*6
        
        if interval > (limit):
            raise Exception(f"""
            The 'interval' parameter is intended to specify a time frame for assessing the risk metrics VaR/ES 
            (Value at Risk/Expected Shortfall) of the bond. 
            Setting this interval to a period longer than {limit} days (equivalent to 6 months) is not allowed due to the following reasons:

            1. Numerical Instability: Long time periods can exacerbate the small errors or approximations inherent in numerical methods,
            making the resulting risk metrics unreliable.
            
            2. Huge Approximations: Over such a long horizon, many factors like interest rate changes, market volatility, etc.,
            are likely to significantly affect the bond's price. These factors are often not captured accurately in the models,
            leading to overly simplistic and potentially misleading results.
            
            Therefore, the maximum allowed "interval" for risk metric calculation is set to be {limit} days, which is equivalent to 25 trading days * 6 months.
            """)
                
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # Note:
        # Modified Duration (MD) serves as a scaling factor or "mapping multiplier," analogous to beta and delta in other financial instruments.
        # By analyzing the bond in the FixedIncomePrmSingle class through the lens of its Modified Duration and Price, we can effectively treat it as if 
        # it were a Zero-Coupon Bond (ZCB) with a maturity equal to the bond's Modified Duration (in years).

        # To obtain the yield for this "equivalent ZCB," we use cubic spline interpolation to find the yield for a ZCB with the same maturity as 
        # the Modified Duration from the existing (monthly) yield curve.
        # ----------------------------------------------------------------------------------------------------------------------------------------

        target_month = self.modified_duration * 12

        # initializing an empty DataFrame to store the interpolated yields
        interpolated_df = pd.DataFrame()

        # checking if target_month is already in the columns of the yield_curve
        if target_month in yield_curve.columns: 
            y = yield_curve.loc[:, target_month].values
            y = y / 100

        else:                                   
            # handling edge cases
            if np.floor(target_month) < 3: 
                left = 1
                right = left + 6
            elif np.ceil(target_month) > max(yield_curve.columns) - 3:
                right = max(yield_curve.columns)
                left = right - 6
            else:
                left = np.floor(target_month) - 3
                right = np.ceil(target_month) + 3

            # defining new columns for the interpolated DataFrame
            new_columns = np.arange(left, right, 0.1) 
            interpolated_df = pd.DataFrame(index=yield_curve.index, columns=new_columns)

            # looping through all rows of the existing yield curve DataFrame
            for i, row in yield_curve.iterrows():
                # converting index and values to lists
                existing_terms = row.index.astype(int).tolist() ; existing_yields = row.values.tolist()
                cs = CubicSpline(existing_terms, existing_yields)
                interpolated_yields = cs(new_columns)
                # adding the interpolated yields to the DataFrame
                interpolated_df.loc[i] = interpolated_yields

            # converting the DataFrame to float dtype
            interpolated_df = interpolated_df.astype(float)

            # ////////////////////////////////////////////////////////////////////////////////////////////////

            closest_column = min(interpolated_df.columns, key = lambda x: abs(x - target_month))
            y = (interpolated_df[closest_column].values) / 100

        delta_y = np.diff(y)

        # Sigma of the delta_y **********************************************************************************************************************************

        # Simple historical Volatility -----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------
        
        if vol == "simple":
            if interval == 1:
                sigma_delta_y = np.std(delta_y)
            elif interval > 1:
                sigma_delta_y = np.std(delta_y) * np.sqrt(interval)

        # GARCH (p,q) Volatility -----------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        elif vol == "garch":

            from inputsControlFunctions import (check_p, check_q)

            # check procedure
            p = check_p(p) ; q = check_q(q) 

            if p > 1 or q > 1:
                raise Exception("p and q are limited to 1 here in order to ensure numerical stability")

            # to ignore warnings
            import os ; import sys ; import numpy as np ; from scipy.stats import norm ; from arch import arch_model

            model = arch_model(delta_y, vol="GARCH", p=p, q=q, power= 2.0, dist = "normal") 

            # redirecting stderr to devnull
            stderr = sys.stderr ; sys.stderr = open(os.devnull, 'w')

            try:
                model_fit = model.fit(disp='off')
            except Exception as e:
                print("Garch fitting did not converge:", str(e))
                return
            finally:
                # Reset stderr
                sys.stderr = stderr

            # horizon is 1 --- better always stay at the highest frequency and then aggregate
            horizon = 1

            # forecasting
            forecasts = model_fit.forecast(start=0, horizon=horizon)

            # getting the variance forecasts
            variance_forecasts = forecasts.variance.dropna().iloc[-1][0]

            # ------------------------------------------------------------------------------------------------------------------------

            if interval == 1:
                sigma_delta_y = np.sqrt(variance_forecasts)
            elif interval > 1:
                cumulative_variance = variance_forecasts * interval # assuming normality i.i.d for the future
                sigma_delta_y = np.sqrt(cumulative_variance)

         # EWMA (lamdba = lambda_ewma) Volatility ------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        elif vol == "ewma":

            from inputsControlFunctions import check_lambda_ewma

            # check procedure
            lambda_ewma = check_lambda_ewma(lambda_ewma) 

            # creatting the fitted time series of the ewma(lambda_ewma) variances ------------

            n = len(delta_y)

            variance_time_series = np.repeat(delta_y[0]**2, n)

            for t in range(1,n):
                variance_time_series[t] = lambda_ewma * variance_time_series[t-1] + (1-lambda_ewma) * delta_y[t-1]**2

            variance_t_plus1 = lambda_ewma * variance_time_series[-1] + (1-lambda_ewma) * delta_y[-1]**2

            # ------------------------------------------------------------------------------------------------------------------------

            if interval == 1: #########

                sigma_delta_y = np.sqrt(variance_t_plus1)    

            if interval >1 : #########

                sigma_delta_y = np.sqrt(variance_t_plus1) * np.sqrt(interval) # assuming normality i.i.d for the future

        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # ----------------------------------------------------------------------------------------------------------------------------------

        absOverAllPos = abs(self.bond_price * self.num_assets * self.modified_duration)

        quantile = norm.ppf(1-alpha, loc=0, scale=1) 
        q_es = ( ( np.exp( - (quantile**2)/2)) / ( (np.sqrt(2*np.pi)) * (1-(1-alpha))))

        # Value at Risk and Expected Shortfall 
        VaR = quantile * absOverAllPos * sigma_delta_y ; ES = q_es * absOverAllPos * sigma_delta_y

        # consolidating results into a dictionary to return
        DURATION_NORMAL = {"var" : round(VaR, 4),
                           "es:" : round(ES, 4),
                           "T" : interval,
                           "BondPod" : round(self.bond_price * self.num_assets,4),
                           "Maturity" : self.maturity, 
                           "Duration" : self.duration,
                           "ModifiedDuration" : self.modified_duration,
                           "MappedBondPos" : round(absOverAllPos,4),
                           "BondPrice" : round(self.bond_price, 4),
                           "SigmaDeltaY" : round(sigma_delta_y,5)}

        return DURATION_NORMAL

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Historical Simulation <<<<<<<<<<<<<<<

    def HistoricalSimulation(self , yield_curve: pd.DataFrame, interval: int = 1, alpha: float = 0.05):

        """
        Main
        -----
        In this approach, historical data is used to generate a distribution of potential outcomes, which can be more realistic compared to model-based approaches like 
        the Duration-Normal method.

        Parameters
        -----------
        - yield_curve : The yield curve DataFrame with different maturities, with terms in months as columns and the yield rates as row values.

        ```
        | Date      |   1  |   2  |   3  |  4   |  5   | ...  | 355  |  356 |  357 |  358 |  359 |  360 |
        |-----------|------|------|------|------|------|------|------|------|------|------|------|------|
        | 01/03/2022| 0.05 | 0.06 | 0.08 | 0.22 | 0.40 | 0.78 | 1.04 | 1.37 | 1.55 | 1.63 | 2.05 | 2.01 |
        | 01/04/2022| 0.06 | 0.05 | 0.08 | 0.22 | 0.38 | 0.77 | 1.02 | 1.37 | 1.57 | 1.66 | 2.10 | 2.07 |
        ...
        | 08/28/2023| 5.56 | 5.53 | 5.58 | 5.56 | 5.44 | 4.98 | 4.69 | 4.38 | 4.32 | 4.20 | 4.48 | 4.29 |
        | 08/29/2023| 5.54 | 5.53 | 5.56 | 5.52 | 5.37 | 4.87 | 4.56 | 4.26 | 4.21 | 4.12 | 4.42 | 4.23 |

        ```
        - interval: The time interval in days for which to calculate VaR (Value at Risk) and ES (Expected Shortfall).
          - The 'interval' is intended to represent a period over which the risk (VaR/ES) of the bond is assessed. Default is 1.
          - Limited to a maximum of 150 days to ensure meaningful and reliable results for the following reasons:
             1. Numerical Stability: Limiting the interval to 150 days helps maintain the numerical stability of the VaR and ES calculations.
             2. Market Conditions: An interval longer than 150 days may not accurately capture the fast-changing market conditions affecting the bond's risk profile.
             3. Model Assumptions: The underlying statistical and financial models are more accurate over shorter time frames.
        - alpha : Significance level for VaR/ES. Default is 0.05.

        Methodology:
        ------------
        1. Initialization and Input Checks:
        - Validate yield_curve, interval, and alpha.
        - Check maturity and interval constraints.

        2. Yield Curve Interpolation:
        - Convert yield rates to decimals.
        - Apply cubic spline interpolation.

        3. Data Preprocessing:
        - Round column names for precision.
        - Calculate yield curve differences at specified interval.
        - Generate simulated yield curves.

        4. Calculating Profit/Loss:
        - For Zero-Coupon Bonds:
            - Calculate new maturity date.
            - Calculate time-effect only price.
            - Calculate yield-shift effect and Profit/Loss.
        - For Non-Zero Coupon Bonds:
            - Prepare payment schedule.
            - Calculate time-effect only price.
            - Calculate yield-shift effect and Profit/Loss.

        5. Risk Metrics Calculation:
        - Generate a losses array.
        - Calculate Value at Risk (VaR).
        - Calculate Expected Shortfall (ES).

        Returns:
        --------
        DURATION_NORMAL : dict
            A dictionary containing calculated VaR, ES and other relevant metrics.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np ; from scipy.interpolate import CubicSpline ; import pandas as pd
        
        from inputsControlFunctions import (check_yield_curve_df, check_interval, check_alpha)

        # check procedure 
        yield_curve = check_yield_curve_df(yield_curve) ; interval = check_interval(interval) ; alpha = check_alpha(alpha) 

        if interval >= (self.maturity * 12 * 25):
            raise ValueError(f"""
            The maturity of the bond is {self.maturity} years.
            Which is equivalent to {self.maturity * 12} months or {self.maturity * 12 * 25} days.
            The 'interval' for the VaR/ES calculation cannot exceed, or be equal to, {self.maturity * 12 * 25} days.
            Right now, it is set to {interval}!
            The 'interval' is meant to represent a time period over which you're assessing the risk (VaR/ES) of the bond. 
            If the 'interval' were to match or exceed the bond's maturity, the risk metrics would no longer be meaningful. 
            """)
        
        limit = 25*6
        
        if interval > (limit):
            raise Exception(f"""
            The 'interval' parameter is intended to specify a time frame for assessing the risk metrics VaR/ES 
            (Value at Risk/Expected Shortfall) of the bond. 
            Setting this interval to a period longer than {limit} days (equivalent to 6 months) is not allowed due to the following reasons:

            1. Numerical Instability: Long time periods can exacerbate the small errors or approximations inherent in numerical methods,
            making the resulting risk metrics unreliable.
            
            2. Huge Approximations: Over such a long horizon, many factors like interest rate changes, market volatility, etc.,
            are likely to significantly affect the bond's price. These factors are often not captured accurately in the models,
            leading to overly simplistic and potentially misleading results.
            
            Therefore, the maximum allowed "interval" for risk metric calculation is set to be {limit} days, which is equivalent to 25 trading days * 6 months.
            """)

        # dividing each entry by 100 to convert percentages to decimals
        yield_curve = yield_curve / 100

        # creating an array for new columns; each step is 0.04 to represent a day (1/25 month)
        new_columns = np.arange(0.04, yield_curve.shape[1] + 0.04 , 0.04)

        # initializing an empty DataFrame with the same row index as 'yield_curve' and columns as 'new_columns'
        interpolated_df = pd.DataFrame(index = yield_curve.index, columns=new_columns)

        # iterating through each row in the 'yield_curve' DataFrame
        for i, row in yield_curve.iterrows():
            # converting the index and row values to lists for cubic spline interpolation
            existing_terms = row.index.astype(int).tolist() ; existing_yields = row.values.tolist()
            # applying cubic spline interpolation
            cs = CubicSpline(existing_terms, existing_yields)
            # obtaining new interpolated yields
            interpolated_yields = cs(new_columns)
            # storing the interpolated yields into the new DataFrame
            interpolated_df.loc[i] = interpolated_yields

        # converting the data type of all elements in the DataFrame to float
        interpolated_df = interpolated_df.astype(float)

        # initializing an empty array to store rounded column names
        columns_rounded = np.zeros(len(interpolated_df.columns))

        # iterating through each column and rounding the name to 2 decimal places
        for i, col in enumerate(interpolated_df.columns):
            columns_rounded[i] = round(col, 2)

        # updating the column names with the rounded values
        interpolated_df.columns = columns_rounded

        # calculating the difference between rows at a given interval, then removing NA rows
        DiffIntervalYieldCurve = interpolated_df.diff(periods = interval, axis=0).dropna()

        # calculating the new maturity by deducting the interval and scaling it down to months
        NewMaturity = round(((self.maturity * 12 * 25) - interval) / 25, 2)

        # determining the maximum number of possible simulations based on the number of rows
        NumSims = DiffIntervalYieldCurve.shape[0]

        # creating a DataFrame of yield curves for simulation, based on the differences and the last yield curve
        SimuYieldCurve = DiffIntervalYieldCurve + interpolated_df.iloc[-1]

        # Time Decay and Yield Curve Shifts
        # ---------------------------------
        # This method will consider both the natural time decay of the bond (i.e., approaching maturity) and potential shifts in the yield curve
        # ------------------------------------------------------------------------------------------------------------------------------------------

        PL = np.zeros(NumSims)

        # if self.is_zcb == True ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if self.is_zcb == True:

            # looping through all rows in the simulated yield curve. Each row represents a different scenario.
            for i in range(SimuYieldCurve.shape[0]):

                # calculating the new maturity date, considering the interval step. 
                # the original maturity of the bond is given by 'self.maturity' (in years), 
                # which is converted to days and then adjusted by the 'interval'.
                # it's then rounded to 2 decimal places and converted back to years.
                NewMaturity = round(((self.maturity * 12 * 25) - interval) / 25, 2)

                # Part 1: Price of the bond without any change (time effect only)
                # ----------------------------------------------------------------
                # this is done to isolate the time effect on the bond price, holding everything else constant.
                # using today's yield curve (the last row of the interpolated_df), find the yield for the NewMaturity.
                exact_yield = interpolated_df.iloc[-1].loc[NewMaturity]
                
                # using this yield, calculate what the bond's fair price would be at the NewMaturity date, 
                # without considering any shift in the yield curve.
                fair_price_at_new_maturity = round(self.face_value / ((1 + (exact_yield)) ** (NewMaturity / 12)), 4)
                
                # Part 2: Price of the bond with a change in the yield curve
                # -----------------------------------------------------------
                # this accounts for any shifts in the yield curve.
                # retrieving the yield for the bond at its NewMaturity under the current simulation scenario.
                exact_yield = SimuYieldCurve.iloc[i].loc[NewMaturity]
                
                # calculate the new price of the bond under the current simulation scenario.
                NewPrice = round(self.face_value / ((1 + (exact_yield)) ** (NewMaturity / 12)), 4)

                # computing the Profit/Loss only due to the simulated shifts in the interpolated yield curve
                # -------------------------------------------------------------------------------------------
                # PL[i] is the difference between the new price (NewPrice) and the 'time effect only' price (fair_price_at_new_maturity),
                # both scaled by the number of assets. This captures the effect of the shift in the yield curve.
                PL[i] = (NewPrice * self.num_assets) - (fair_price_at_new_maturity * self.num_assets)

        # if self.is_zcb == False ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if self.is_zcb == False:

            # Part 1: Calculating the bond's price accounting for time effects only
            # -----------------------------------------------------------------------
            
            # determining if payments are yearly
            if not self.semi_annual_payment:
                # calculating the payment schedule in terms of months, adjusting for the interval
                paymentSchedules = (np.arange(1 ,(self.maturity + 1), 1) * 12) - round(interval / 25,2)

            # determining if payments are semi-annual
            if self.semi_annual_payment:
                # calculating the semi-annual payment schedule in terms of months, adjusting for the interval
                paymentSchedules = (np.arange(0.5, (self.maturity + 0.5), 0.5) * 12) - round(interval / 25,2)

            # rounding each value in the paymentSchedules array to two decimal places for precision
            for i in range(len(paymentSchedules)):
                paymentSchedules[i] = round(paymentSchedules[i], 2)

            # filtering the paymentSchedules array to keep only values that are strictly greater than zero
            # if a value is less or equal to 0 in the paymentSchedules, it indicates that the bond has already paid the coupons for those periods.
            paymentSchedules = [x for x in paymentSchedules if x > 0]

            # initializing a zero array to hold the bond's cash flows
            cashFlows = np.zeros(len(paymentSchedules))

            # iterating through each payment schedule to set the cash flows
            for z in range(len(cashFlows)):
                # assigning coupon payments for all but the last payment
                if z != (len(cashFlows) - 1):
                    cashFlows[z] = self.face_value * (self.coupon_rate / 100)
                # assigning coupon plus face value for the last payment
                else:
                    cashFlows[z] = (self.face_value * (self.coupon_rate / 100)) + self.face_value

            # initializing a zero array to hold the present value of cash flows
            pvCashFlow = np.zeros(len(paymentSchedules))

            # iterating through each payment schedule to calculate the present value of cash flows
            for idx, vle in enumerate(paymentSchedules):
                # extracting the yield corresponding to the new maturity
                exact_yield = interpolated_df.iloc[-1].loc[vle]
                # calculating the present value of each cash flow
                pvCashFlow[idx] = cashFlows[idx] / ((1 + (exact_yield)) ** (vle / 12))

            # calculating the bond's fair price at the new maturity, rounding to 4 decimal places
            fair_price_at_new_maturity = round(np.sum(pvCashFlow), 4)

            # Part 2: Calculating the bond's price accounting for changes in the yield curve
            # ------------------------------------------------------------------------------
            
            # iterating through each simulated yield curve
            for i in range(SimuYieldCurve.shape[0]):
                # re-initializing the zero array to hold the present value of cash flows
                pvCashFlow = np.zeros(len(paymentSchedules))

                # iterating through each payment schedule to calculate the present value of cash flows
                for idx, vle in enumerate(paymentSchedules):
                    # extracting the simulated yield corresponding to the new maturity
                    exact_yield = SimuYieldCurve.iloc[i].loc[vle]
                    # calculating the present value of each cash flow using the simulated yield
                    pvCashFlow[idx] = cashFlows[idx] / ((1 + (exact_yield)) ** (vle / 12))

                # calculating the bond's new price based on the simulated yield, rounding to 4 decimal places
                NewPrice = round(np.sum(pvCashFlow), 4)

                # computing the profit/loss due to the change in yield, scaled by the number of assets
                PL[i] = (NewPrice * self.num_assets) - (fair_price_at_new_maturity * self.num_assets)

        # -------------------------------------------------------------------------------------------------------------------------

        # losses
        losses = PL[PL<0]

        if len(losses) == 0:
            raise Exception("""
            No losses were generated in the simulation based on the current data and 'interval' settings.            
            Consider doing one or more of the following:

            1. Review the quality of your input data to ensure it's suitable for simulation.
            2. Slightly adjust the 'interval' parameter to potentially produce a different simulation outcome.
            """)

        NewMaturity = round(((self.maturity * 12 * 25) - interval) / 25, 2)

        # Value at Risk and Expected Shortfall 
        VaR = np.quantile(losses, alpha) * -1 ; ES = np.mean(losses[losses < - VaR]) * -1

        # consolidating results into a dictionary to return
        HS = {"var" : round(VaR, 4),
              "es" : round(ES, 4),
              "T" : interval,
              "BondPod" : round(self.bond_price * self.num_assets,4),
              "Maturity" : self.maturity,
              "NewMaturity" : round(NewMaturity/12,4), 
              "BondPrice" : round(self.bond_price, 4),
              "NumSims" : len(PL)}

        return HS

##########################################################################################################################################################################
##########################################################################################################################################################################
## FixedIncomePrmPort  ################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class FixedIncomePrmPort:
    
    """
    Main
    ----
    This class offers the functionalities for pricing and risk assessment of a portfolio of bond securities. 
    It not only facilitates the calculation of portfolio-level risk metrics such as duration, yield-to-maturity, Value-at-Risk (VaR), and Expected Shortfall (ES),
    but also provides the same metrics for each individual bond within the portfolio.

    Important Features and Constraints
    ----------------------------------
    1. `maturity_s` Array Parameter:
    - Accepts either integer values to represent full years or decimals like 0.5 for semi-annual periods.
    - Optimized for risk and return evaluation at specific time-points: either at the time of issuance or 
      just after a coupon payment for coupon-bearing bonds.
    
    2. Alignment Within Maturities:
    - The class is designed to account for alignment within the maturities of the bonds in the portfolio.
    - The system in place requires every bond to either mature or have its subsequent payment due in intervals that are multiples of six months.
    - While this is a limitation since it doesn't account for bonds with non-standard payment schedules or already issued ones, it provides a base that future developers can build upon to create more advanced evaluation mechanisms.
    - Therefore, optimal usage is at the time of issuance of all bonds bond or every six months thereafter.
    
    3. Bond Compatibility:
    - Compatible with portfolios that include both zero-coupon bonds and coupon-bearing bonds.
    - Provides options for annual or semi-annual coupon payments for coupon-bearing bonds.

    Purpose
    -------
    This class aims to provide a comprehensive toolkit for understanding, analyzing, and managing portfolios of fixed-income securities. 
    It is tailored to serve two main audiences:

    1. Portfolio Managers and Quantitative Analysts: 
        - For professionals who aim to develop or integrate more sophisticated calculations and risk management strategies, 
        this class serves as a robust foundation. It offers a reliable basis upon which additional features, functionalities, and asset classes can be incorporated.

    2. Individual Investors and Educators: 
        - For those who intend to use the class for straightforward portfolio pricing and risk assessment, the class offers an array of ready-to-use methods. 
        These methods enable the calculation of key portfolio-level metrics such as aggregate bond price, portfolio duration, portfolio yield-to-maturity,
        Value-at-Risk (VaR), and Expected Shortfall (ES). 

    In addition, this class also allows users to drill down into the specifics of each bond within the portfolio, offering metrics like individual bond price, duration,
    and yield-to-maturity.
    By catering to both of these needs, the class offers a flexible and robust starting point for a range of fixed-income analysis and risk management tasks.

    Initial Parameters
    ------------------
    - yield_curve_today : The DataFrame containing the current yield curve, with terms in months as columns and the yield rates as row values.
    The yield rates should be represented as a percentage, e.g., 3% should be given as 3.

    ```
    | Date      |   1  |   2  |   3  |  4   |  5   | ...  | 355  |  356 |  357 |  358 |  359 |  360 |
    |-----------|------|------|------|------|------|------|------|------|------|------|------|------|
    | 08/29/2023| 5.54 | 5.53 | 5.52 | 5.51 | 5.51 | 4.87 | 4.56 | 4.26 | 4.21 | 4.19 | 4.18 | 4.19 |
    ```

    - maturity_s: Array of times to maturity for the bonds in the portfolio, in years. Each value must be either an integer for full years or a decimal like 0.5
    to indicate semi-annual periods.
    - num_assets_s: Array representing the number of identical assets/bonds for each bond in the portfolio.
    - face_value_s: Array containing the face values of the bonds in the portfolio.
    - is_zcb_s: Array of boolean indicators to specify whether each bond in the portfolio is a zero-coupon bond.
    - coupon_rate_s: Array of the annual coupon rates for the bonds in the portfolio, represented as percentages 
    (e.g., 3% should be entered as 3). This array is required if `is_zcb` contains any `False` values. Default is None.
    - semi_annual_payment_s: Array of boolean indicators to specify whether each bond in the portfolio pays coupons semi-annually.
    This array is required if `is_zcb` contains any `False` values. Default is None.

    Example of Initial Parameters
    -----------------------------
    - maturity_s = array([4, 1, 2.5, 2, 4.5])
    - num_assets_s = array([ 100,  40, -35,  25, -75])
    - face_value_s = array([1000, 1000, 1000, 1000, 1000])
    - is_zcb_s = array([ True,  True, False, False, False])
    - coupon_rate_s = array([0, 0, 2, 2, 3])
    - semi_annual_payment_s = array([False, False,  True, False,  True])

    Methods
    -------
    - DurationNormal(): Calculates the Value-at-Risk (VaR) and Expected Shortfall (ES) of a portfolio of bond using a normal distribution model.
    This method assumes a normal distribution of yield changes and uses the modified duration of the portfolio
    to approximate the change in bond price for a given change in yield.
    - HistoricalSimulation(): In this approach, historical data is used to generate a distribution of potential outcomes,
    which can be more realistic compared to model-based approaches like the Duration-Normal method.

    Attributes
    ----------
    - bond_prices: The price of each bond within the portfolio, calculated based on whether it is a zero-coupon or coupon-bearing bond.
    - durations: The Macaulay duration (in years) for each bond within the portfolio.
    - modified_durations: The modified duration (in years) for each bond within the portfolio.
    - yield_to_maturities: The yield to maturity (annualized) for each bond within the portfolio.
    - initial_position: Initial financial position of the portfolio, considering the characteristics of each bond.
    - tot_future_payments: Total future payments to be received from all bonds within the portfolio.
    - summary: A comprehensive summary of the portfolio's attributes.

    Exceptions
    ----------
    - Raises an exception if the coupon_rate and semi_annual_payment are not provided when is_zcb is set to False.
    - Raises an exception if the maturity provided is beyond the highest maturity available in the yield_curve_today DataFrame.

    Note
    ----
    The yield curve provided should be a monthly yield curve, with maturity ranging from 1 to n months.

    """

    import warnings ; warnings.filterwarnings("ignore") 

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def __init__(self, yield_curve_today: pd.DataFrame,               # not in decimal point. For instance: for 3% it should be 3.
                       maturity_s: np.ndarray,                        # array - in years
                       num_assets_s: np.ndarray, 
                       face_value_s: np.ndarray, 
                       is_zcb_s: np.ndarray, 
                       coupon_rate_s: np.ndarray = None,              # array - not in decimal point. For instance: for 3% it should be 3.
                       semi_annual_payment_s: np.ndarray = None):
        

        from inputsControlFunctions import (check_yield_curve_today, check_maturity_s, check_num_assets_s,check_face_value_s, check_is_zcb_s,
                                                       check_coupon_rate_s, check_semi_annual_payment_s)

        # check procedure
        self.yield_curve_today = check_yield_curve_today(yield_curve_today) ; self.maturity_s = check_maturity_s(maturity_s) ; self.num_assets_s = check_num_assets_s(num_assets_s)
        self.face_value_s = check_face_value_s(face_value_s) ; self.is_zcb_s = check_is_zcb_s(is_zcb_s) # here it returns the value

        # -----------------------------------------------------------------------------------------------------
        
        # first checking
        # checking the same length for all the imputs *args

        def check_same_length(*args):
            """
            Checks if all given numpy arrays have the same length.
            """
            it = iter(args)
            the_len = len(next(it))

            if not all(len(l) == the_len for l in it):
                raise ValueError("Not all the input arrays in the FixedIncomePrmPort class have the same length!")
            
        # Here we are ensuring that all arrays have the same length
        check_same_length(maturity_s, num_assets_s, face_value_s, is_zcb_s)

        # ------------------------------------------------------------------------------------------------

        # checking if at least one value in is_zcb_s the is False
        if not all(is_zcb_s):

            if coupon_rate_s is None and semi_annual_payment_s is None: 
                raise Exception("Both coupon_rate_s and semi_annual_payment_s must be provided when at least one element in the is_zcb_s is False.")
            if coupon_rate_s is None or semi_annual_payment_s is None: 
                raise Exception("Either coupon_rate_s or semi_annual_payment_s is missing. Both are required when at least one element in the is_zcb_s is False.")
            
            if coupon_rate_s is not None and semi_annual_payment_s is not None: 

                # check procedure
                self.coupon_rate_s = check_coupon_rate_s(coupon_rate_s) ; self.semi_annual_payment_s = check_semi_annual_payment_s(semi_annual_payment_s)

               # second checking
                # Here we are ensuring that all arrays have the same length
                check_same_length(maturity_s, num_assets_s, face_value_s, is_zcb_s,coupon_rate_s,semi_annual_payment_s)

            for index, (mat, semi) in enumerate(zip(self.maturity_s, self.semi_annual_payment_s)):
                if (mat * 2) % 2 != 0 and not semi:
                    raise Exception(f"""
                    If the maturity of a bond is in fractions of .5 years, then element-wise semi_annual_payment must be set to True. 
                                    
                    This class checks every bond and for those that have a "half-year" maturity makes sure that they have semi-annual payments.
                    This ensures that the payment schedule aligns with the bond's maturity period, preventing logical inconsistencies.
                                    
                    Issue at index {index}.
                      
                    More precisely, the following bond:
                    - maturity: {self.maturity_s[index]}, 
                    - semi_annual_payment value set to: {self.semi_annual_payment_s[index]},
                    - num_asset: {self.num_assets_s[index]},
                    - face_value: {self.face_value_s[index]}
                    """)
                
    # *******************************************************************************************************************************************
        
        for index in range(len(self.num_assets_s)):

            if (self.maturity_s[index] * 12) > self.yield_curve_today.columns[-1]:
                raise Exception(f"""
                Issue at index {index}.
                More precisely, the following bond:
                - maturity: {self.maturity_s[index]}, or {self.maturity_s[index] * 12} months,
                - semi_annual_payment value set to: {self.semi_annual_payment_s[index]},
                - num_asset: {self.num_assets_s[index]},
                - face_value: {self.face_value_s[index]}
                                
                The highest maturity available in the 'yield_curve_today' DataFrame is {self.yield_curve_today.columns[-1]} months.
                Therefore, the method does not have the required yield for the maturity of the bond at index {index}.
                """)

        self.bond_prices = None  ; self.durations = None ; self.modified_durations = None ; self.yield_to_maturities = None
        self.initial_position = None ; self.tot_future_payments = None ; self.summary = None

        self.fit()  # Automatically call the fit method

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    def fit(self):

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np ; from scipy.optimize import newton ; import pandas as pd
 
        # *****************************************************************************************************************************************************

        def AttributesCalculator(yield_curve_today, maturity_s, num_assets_s, face_value_s, is_zcb_s, coupon_rate_s, semi_annual_payment_s):
 
            # initializing arrays to store bond attributes
            bond_prices = np.zeros(len(num_assets_s)) ; durations = np.zeros(len(num_assets_s)) ; modified_durations = np.zeros(len(num_assets_s))
            yield_to_maturities = np.zeros(len(num_assets_s)) ; tot_future_payments = np.zeros(len(num_assets_s))

            # ------------------------------------------------------------------------------------------------------------------

            # iterating over each bond
            for bond_num in range(len(num_assets_s)):

                # if is_zcb_s[bond_num] == True /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                if is_zcb_s[bond_num] == True:
  
                    # calculating the exact yield and bond price for zero-coupon bonds
                    maturity = maturity_s[bond_num] * 12 ; exact_yield = yield_curve_today.iloc[-1].loc[maturity]

                    # calculating the bond price using the zero-coupon bond pricing formula: P = F / (1 + r)^n
                    bond_prices[bond_num] = round(face_value_s[bond_num] / ( (1 + (exact_yield/100))  ** maturity_s[bond_num]),4)
 
                    # assigning duration and yield to maturity
                    durations[bond_num] = maturity_s[bond_num] ; yield_to_maturities[bond_num] = round(exact_yield/100,6)

                    # calculating modified duration and setting total future payments as 1 for zero-coupon bonds
                    modified_durations[bond_num] = round(durations[bond_num] / (1 + (yield_to_maturities[bond_num])), 4) ; tot_future_payments[bond_num] = 1

                # if is_zcb_s[bond_num] == False ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

                if is_zcb_s[bond_num] == False:

                    # determining the payment schedules based on the type of payments (yearly or semi-annual)
                    # if yearly payments -------------------------------------------------------------------------------------
                    if not semi_annual_payment_s[bond_num]: 
                        paymentSchedules = np.arange(1,(maturity_s[bond_num] + 1),1) * 12 
                    # if semi annual payment payments ------------------------------------------------------------------------
                    if semi_annual_payment_s[bond_num]:
                        paymentSchedules = np.arange(0.5,(maturity_s[bond_num] + 0.5),0.5) * 12 
                        coupon_rate_s[bond_num] = coupon_rate_s[bond_num] / 2 # PIVOTAL
                    # --------------------------------------------------------------------------------------------------------

                    # calculating cash flows and their present values
                    cashFlows = np.zeros(len(paymentSchedules))

                    for i in range(len(cashFlows)):
                        if i != (len(cashFlows)-1):
                            cashFlows[i] = face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)
                        else:
                            cashFlows[i] = (face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)) + face_value_s[bond_num]

                    pvCashFlow = np.zeros(len(paymentSchedules))

                    for index, value in enumerate(paymentSchedules):
                        exact_yield = yield_curve_today.iloc[-1].loc[value]
                        pvCashFlow[index] = cashFlows[index] /  ((1+((exact_yield/100)))**(value/12)) 
                    
                    # calculating the bond price by summing up the present values and setting total future payments for the cb
                    bond_prices[bond_num] = round(np.sum(pvCashFlow),4) ; tot_future_payments[bond_num] = len(paymentSchedules)

                    # finding yield to maturity by minimizing the function npv_minus_price ----------------------------------------

                    def npv_minus_price(irr): #irr here
                        cashFlows = np.zeros(len(paymentSchedules))
                        for i in range(len(cashFlows)):
                            if i != (len(cashFlows)-1):
                                cashFlows[i] = face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)
                            else:
                                cashFlows[i] = (face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)) + face_value_s[bond_num]
                        pvCashFlow = np.zeros(len(paymentSchedules))
                        for index, value in enumerate(paymentSchedules):
                            pvCashFlow[index] = cashFlows[index] /  ((1+((irr/100)))**(value/12)) # #irr here
                        sumPvCashFlow = np.sum(pvCashFlow)
                        return sumPvCashFlow - bond_prices[bond_num]
                    
                    # --------------------------------------------------------------------------------------------------------------------
                    
                    # assuming the initial guess for IRR is 5%
                    initial_guess = 5.0 ; irr_result = newton(npv_minus_price, initial_guess)

                    # storing the result
                    yield_to_maturities[bond_num] = round(irr_result/100, 6)

                    # calculating duration and modified duration 
                    durationWeights = pvCashFlow / bond_prices[bond_num] ; Weights_times_t = np.zeros(len(durationWeights))

                    for index, value in enumerate(durationWeights):
                        Weights_times_t[index] = durationWeights[index] * (paymentSchedules[index] / 12)

                    durations[bond_num] = round(np.sum(Weights_times_t), 4)

                    if semi_annual_payment_s[bond_num]:
                        frequency = 2  # payments are semi-annual
                    else:
                        frequency = 1  # payments are annual

                    modified_durations[bond_num] = durations[bond_num] / (1 + (yield_to_maturities[bond_num] / frequency))
                    modified_durations[bond_num] = round(modified_durations[bond_num], 4)

            # ----------------------------------------------------------------------------------------------------------------
            
            # calculating the initial portfolio position
            initial_position = round(np.dot(num_assets_s, bond_prices),4)

            return bond_prices, durations, modified_durations, yield_to_maturities, initial_position, tot_future_payments
        
        # *****************************************************************************************************************************************************

        # calling the AttributesCalculator function and storing the results
        self.bond_prices, self.durations, self.modified_durations, self.yield_to_maturities, self.initial_position, self.tot_future_payments = AttributesCalculator(
            self.yield_curve_today, self.maturity_s, self.num_assets_s, self.face_value_s, self.is_zcb_s, self.coupon_rate_s, self.semi_annual_payment_s)

        # creating a summary DataFrame
        self.summary = pd.DataFrame({"ZCB" : self.is_zcb_s, "Semi" : self.semi_annual_payment_s, "BondPrice" : self.bond_prices, "FaceValue" : self.face_value_s,
            "Maturity" : self.maturity_s, "TotFutPay" : self.tot_future_payments, "NumAsset" : self.num_assets_s, "Pos" : self.bond_prices * self.num_assets_s, 
            "Duration" : self.durations, "MD" : self.modified_durations, "YTM" : self.yield_to_maturities, "CouponRate" : self.coupon_rate_s / 100})
        
    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # DURATION METHOD <<<<<<<<<<<<<<<

    def DurationNormal(self, yield_curve: pd.DataFrame, vol: str = "simple", interval: int = 1, alpha: float = 0.05, p: int = 1, q: int = 1, lambda_ewma: float = 0.94):

        """
        Main
        -----
        This method calculates the Value at Risk (VaR) and Expected Shortfall (ES) for a given bond portfolio using the Modified Duration mapping approach.

        Parameters
        -----------
        - yield_curve : The yield curve DataFrame with different maturities, with terms in months as columns and the yield rates as row values.

        ```
        | Date      |   1  |   2  |   3  |  4   |  5   | ...  | 355  |  356 |  357 |  358 |  359 |  360 |
        |-----------|------|------|------|------|------|------|------|------|------|------|------|------|
        | 01/03/2022| 0.05 | 0.06 | 0.08 | 0.22 | 0.40 | 0.78 | 1.04 | 1.37 | 1.55 | 1.63 | 2.05 | 2.01 |
        | 01/04/2022| 0.06 | 0.05 | 0.08 | 0.22 | 0.38 | 0.77 | 1.02 | 1.37 | 1.57 | 1.66 | 2.10 | 2.07 |
        ...
        | 08/28/2023| 5.56 | 5.53 | 5.58 | 5.56 | 5.44 | 4.98 | 4.69 | 4.38 | 4.32 | 4.20 | 4.48 | 4.29 |
        | 08/29/2023| 5.54 | 5.53 | 5.56 | 5.52 | 5.37 | 4.87 | 4.56 | 4.26 | 4.21 | 4.12 | 4.42 | 4.23 |

        ```
        - vol : Type of volatility to use ("simple", "garch", or "ewma").
        - interval: The time interval in days for which to calculate VaR (Value at Risk) and ES (Expected Shortfall). Default is 1.
          - The 'interval' is intended to represent a period over which the risk (VaR/ES) of the bond is assessed.
          - Limited to a maximum of 150 days to ensure meaningful and reliable results for the following reasons:
             1. Numerical Stability: Limiting the interval to 150 days helps maintain the numerical stability of the VaR and ES calculations.
             2. Market Conditions: An interval longer than 150 days may not accurately capture the fast-changing market conditions affecting the bond's risk profile.
             3. Model Assumptions: The underlying statistical and financial models are more accurate over shorter time frames.
        - alpha : Significance level for VaR/ES. Default is 0.05.
        - p: The 'p' parameter for the GARCH model. Default is 1.
        - q: The 'q' parameter for the GARCH model. Default is 1.
        - lambda_ewma: The lambda parameter for Exponentially Weighted Moving Average (EWMA) model. Default is 0.94.

        Methodology:
        ------------
        - Input Checks: Validates the types and values of input parameters.
        - Validates the 'interval' parameter against two key constraints:
            - a) Max Possible Interval: Calculated based on the bond with the shortest maturity.
            - b) Predefined Limit: Set to a fixed number of days (150 days, equivalent to 6 months).
        - Raises exceptions if the provided 'interval' exceeds either constraint, offering reasons and suggestions for adjustment.
        - Interpolation:
            - The 'target_month' is calculated based on the Modified Duration of the bond, scaled to a monthly time frame (Modified Duration * 12).
            - Cubic Spline Interpolation is employed to get a smoother yield curve and extract yields for target months, IF NOT directly available in the original yield curve.
            - A window of months around the 'target_month' is selected to perform the cubic spline interpolation.
            Edge cases are handled by ensuring the interpolation window fits within the available yield curve data.
            - Once the interpolated yield curve is available, the yield for the 'target_month' is extracted. 
        - Volatility Estimation: Depending on the selected model ("simple", "garch", "ewma"), calculates the volatility of yield changes.
        - VaR and ES Calculation: Applies the Modified Duration mapping approach to calculate VaR and ES.

        Returns:
        --------
        DURATION_NORMAL : dict
            A dictionary containing calculated VaR, ES and other relevant metrics.
        """

        import warnings ; warnings.filterwarnings("ignore")

        from scipy.stats import norm ; import numpy as np ; from scipy.interpolate import CubicSpline ; import pandas as pd
        
        from inputsControlFunctions import (check_yield_curve_df, check_vol, check_interval, check_alpha)

        # check procedure 
        yield_curve = check_yield_curve_df(yield_curve) ; vol = check_vol(vol) ; interval = check_interval(interval) ; alpha = check_alpha(alpha) 

        # calculating the maximum possible interval based on the bond with the shortest maturity
        max_possible_interval = min(self.maturity_s) * 12 * 25  # assuming 25 trading days in a month
        limit = 25 * 6  # a predefined limit set to 150 days (equivalent to 6 months)

        # checking if the provided 'interval' exceeds either of the two constraints: max_possible_interval or limit
        if interval >= max_possible_interval or interval >= limit:
            if interval >= max_possible_interval:
                reason1 = f"""
                The bond with the shortest maturity in the portfolio has a remaining time of {min(self.maturity_s)} years,
                equivalent to {min(self.maturity_s) * 12} months or {max_possible_interval} days."""
                reason2 = f"""
                The provided 'interval' of {interval} days exceeds or matches this limit, making the VaR/ES metrics unreliable."""
                suggestion1 = f"""
                Please choose an 'interval' less than {max_possible_interval} days.
                """
            else:
                reason1 = f"""
                Setting this interval to a period longer than {limit} days (equivalent to 6 months) is not allowed for the following reasons:"""
                reason2 = """
                1. Numerical Instability: Long time periods can exacerbate the small errors or approximations inherent in numerical methods.
                2. Huge Approximations: Over such a long horizon, many factors like interest rate changes, market volatility, etc., are likely to 
                significantly affect the bond's price."""
                suggestion1 = f"""
                Therefore, the maximum allowed 'interval' for risk metric calculation is set to be {limit} days.
                """

            raise Exception(f"""
                Invalid 'interval' for VaR/ES (Value at Risk/Expected Shortfall) calculation.
            {reason1}
            {reason2}
            {suggestion1}
            """)
        
        

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------
        # Note:
        # When dealing with a portfolio of bonds, the concept of Modified Duration extends to the portfolio level as the weighted average of the Modified Durations
        # of the individual assets in the portfolio. This portfolio-level Modified Duration serves as a composite "mapping multiplier," 
        # similar to how beta and delta function in other financial contexts.

        # By analyzing the portfolio's Modified Duration and Price, one can treat the entire portfolio as if it were a single Zero-Coupon Bond (ZCB) 
        # with a maturity equal to the portfolio's weighted average Modified Duration (in years).

        # To estimate the yield for this "equivalent ZCB," cubic spline interpolation can be applied to the existing (monthly) yield curve. 
        # The aim is to find the yield associated with a ZCB that has the same maturity as the portfolio's weighted average Modified Duration.
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------

        # calculating the absolute values of the initial positions for each asset (here represented by the product of bond prices and the number of such assets)
        self.abs_initial_position = [abs(x) for x in (self.bond_prices * self.num_assets_s)]

        # summing up all the absolute values to get the total absolute position of the portfolio
        AbsPosPort = np.sum(self.abs_initial_position)

        # calculating the weights of each asset in the portfolio by dividing its absolute value by the total absolute position
        weights = self.abs_initial_position / AbsPosPort

        # calculating the weighted average duration of the portfolio using dot product
        duration_port = np.dot(weights, self.durations)

        # calculating the weighted average modified duration of the portfolio using dot product
        modified_duration_port = np.dot(weights, self.modified_durations)

        # setting the target_month value to be the weighted average modified duration of the portfolio
        target_month = modified_duration_port

        # initializing an empty DataFrame to store the interpolated yields
        interpolated_df = pd.DataFrame()

        # if target_month is already in the yield_curve.columns
        if target_month in yield_curve.columns: 
            y = yield_curve.loc[:, target_month].values
            y = y / 100

        else:                                   
            # handling edge cases
            if np.floor(target_month) < 3: 
                left = 1
                right = left + 6
            elif np.ceil(target_month) > max(yield_curve.columns) - 3:
                right = max(yield_curve.columns)
                left = right - 6
            else:
                left = np.floor(target_month) - 3
                right = np.ceil(target_month) + 3

            # defining new columns for the interpolated DataFrame
            new_columns = np.arange(left, right, 0.1) 

            interpolated_df = pd.DataFrame(index=yield_curve.index, columns=new_columns)

            # looping through all rows of the existing yield curve DataFrame
            for i, row in yield_curve.iterrows():
                # converting index and values to lists
                existing_terms = row.index.astype(int).tolist() ; existing_yields = row.values.tolist()
                cs = CubicSpline(existing_terms, existing_yields)
                interpolated_yields = cs(new_columns)
                # adding the interpolated yields to the DataFrame
                interpolated_df.loc[i] = interpolated_yields

            # converting the DataFrame to float dtype
            interpolated_df = interpolated_df.astype(float)

            # ////////////////////////////////////////////////////////////////////////////////////////////////

            closest_column = min(interpolated_df.columns, key = lambda x: abs(x - target_month))
            y = (interpolated_df[closest_column].values) / 100

        delta_y = np.diff(y)

        # Sigma of the delta_y **********************************************************************************************************************************

        # Simple historical Volatility -----------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------
        
        if vol == "simple":
            if interval == 1:
                sigma_delta_y = np.std(delta_y)
            elif interval > 1:
                sigma_delta_y = np.std(delta_y) * np.sqrt(interval)

        # GARCH (p,q) Volatility -----------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        elif vol == "garch":

            from inputsControlFunctions import (check_p, check_q)

            # check procedure
            p = check_p(p) ; q = check_q(q) 

            if p > 1 or q > 1:
                raise Exception("p and q are limited to 1 here in order to ensure numerical stability")

            # to ignore warnings
            import os ; import sys ; import numpy as np ; from scipy.stats import norm ; from arch import arch_model

            model = arch_model(delta_y, vol="GARCH", p=p, q=q, power= 2.0, dist = "normal") 

            # redirecting stderr to devnull
            stderr = sys.stderr ; sys.stderr = open(os.devnull, 'w')

            try:
                model_fit = model.fit(disp='off')
            except Exception as e:
                print("Garch fitting did not converge:", str(e))
                return
            finally:
                # Reset stderr
                sys.stderr = stderr

            # horizon is 1 --- better always stay at the highest frequency and then aggregate
            horizon = 1

            # forecasting
            forecasts = model_fit.forecast(start=0, horizon=horizon)

            # getting the variance forecasts
            variance_forecasts = forecasts.variance.dropna().iloc[-1][0]

            # ------------------------------------------------------------------------------------------------------------------------

            if interval == 1:
                sigma_delta_y = np.sqrt(variance_forecasts)
            elif interval > 1:
                cumulative_variance = variance_forecasts * interval # assuming normality i.i.d for the future
                sigma_delta_y = np.sqrt(cumulative_variance)

         # EWMA (lamdba = lambda_ewma) Volatility ------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        elif vol == "ewma":

            from inputsControlFunctions import check_lambda_ewma

            # check procedure
            lambda_ewma = check_lambda_ewma(lambda_ewma) 

            # creatting the fitted time series of the ewma(lambda_ewma) variances ------------

            n = len(delta_y)

            variance_time_series = np.repeat(delta_y[0]**2, n)

            for t in range(1,n):
                variance_time_series[t] = lambda_ewma * variance_time_series[t-1] + (1-lambda_ewma) * delta_y[t-1]**2

            variance_t_plus1 = lambda_ewma * variance_time_series[-1] + (1-lambda_ewma) * delta_y[-1]**2

            # ------------------------------------------------------------------------------------------------------------------------

            if interval == 1: #########

                sigma_delta_y = np.sqrt(variance_t_plus1)    

            if interval >1 : #########

                sigma_delta_y = np.sqrt(variance_t_plus1) * np.sqrt(interval) # assuming normality i.i.d for the future

        # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # ----------------------------------------------------------------------------------------------------------------------------------

        absOverAllPos = np.sum(self.abs_initial_position) * modified_duration_port

        quantile = norm.ppf(1-alpha, loc=0, scale=1) 
        q_es = ((np.exp(-(quantile**2)/2)) / ((np.sqrt(2*np.pi)) * (1-(1-alpha))))

        # Value at Risk and Expected Shortfall 
        VaR = quantile * absOverAllPos * sigma_delta_y ; ES = q_es * absOverAllPos * sigma_delta_y

        # consolidating results into a dictionary to return
        DURATION_NORMAL = {"var" : round(VaR, 4),
                           "es" : round(ES, 4),
                           "T" : interval,
                           "DurationPort" : round(duration_port,4),
                           "ModifiedDurationPort" : round(modified_duration_port,4),
                           "MappedBondPos" : round(absOverAllPos,4), 
                           "OrigPos" : self.initial_position,
                           "SigmaDeltaY" : round(sigma_delta_y,5)}

        return DURATION_NORMAL

    # ******************************************************************************************************************************************************************
    # ******************************************************************************************************************************************************************

    # Historical Simulation <<<<<<<<<<<<<<<

    def HistoricalSimulation(self, yield_curve: pd.DataFrame, interval: int = 1, alpha: float = 0.05):

        """
        Main
        -----
        In this approach, historical data is used to generate a distribution of potential outcomes, which can be more realistic compared to model-based approaches like 
        the Duration-Normal method.

        Parameters
        -----------
        - yield_curve : The yield curve DataFrame with different maturities, with terms in months as columns and the yield rates as row values.

        ```
        | Date      |   1  |   2  |   3  |  4   |  5   | ...  | 355  |  356 |  357 |  358 |  359 |  360 |
        |-----------|------|------|------|------|------|------|------|------|------|------|------|------|
        | 01/03/2022| 0.05 | 0.06 | 0.08 | 0.22 | 0.40 | 0.78 | 1.04 | 1.37 | 1.55 | 1.63 | 2.05 | 2.01 |
        | 01/04/2022| 0.06 | 0.05 | 0.08 | 0.22 | 0.38 | 0.77 | 1.02 | 1.37 | 1.57 | 1.66 | 2.10 | 2.07 |
        ...
        | 08/28/2023| 5.56 | 5.53 | 5.58 | 5.56 | 5.44 | 4.98 | 4.69 | 4.38 | 4.32 | 4.20 | 4.48 | 4.29 |
        | 08/29/2023| 5.54 | 5.53 | 5.56 | 5.52 | 5.37 | 4.87 | 4.56 | 4.26 | 4.21 | 4.12 | 4.42 | 4.23 |

        ```
        - interval: The time interval in days for which to calculate VaR (Value at Risk) and ES (Expected Shortfall). Default is 1.
          - The 'interval' is intended to represent a period over which the risk (VaR/ES) of the bond is assessed.
          - Limited to a maximum of 150 days to ensure meaningful and reliable results for the following reasons:
             1. Numerical Stability: Limiting the interval to 150 days helps maintain the numerical stability of the VaR and ES calculations.
             2. Market Conditions: An interval longer than 150 days may not accurately capture the fast-changing market conditions affecting the bond's risk profile.
             3. Model Assumptions: The underlying statistical and financial models are more accurate over shorter time frames.
        - alpha : Significance level for VaR/ES. Default is 0.05.

        Methodology:
        ------------
        1. Initialization and Input Checks:
        - Validate yield_curve, interval, and alpha.
        - Check maturity and interval constraints.

        2. Yield Curve Interpolation:
        - Convert yield rates to decimals.
        - Apply cubic spline interpolation.

        3. Data Preprocessing:
        - Round column names for precision.
        - Calculate yield curve differences at specified interval.
        - Generate simulated yield curves.
        
        4. Calculating Profit/Loss:
            - For each bond and position in the portfolio:
                - For Zero-Coupon Bonds:
                    - Calculate new maturity date.
                    - Calculate time-effect only price.
                    - Calculate yield-shift effect and Profit/Loss.
                - For Non-Zero Coupon Bonds:
                    - Prepare payment schedule.
                    - Calculate time-effect only price.
                    - Calculate yield-shift effect and Profit/Loss.
            - Finally, the aggregate value of the portfolio for each simulation is contrasted with the 
            value that arises solely from the passage of time, in order to compute the Profit/Loss.

        5. Risk Metrics Calculation:
        - Generate a losses array.
        - Calculate Value at Risk (VaR).
        - Calculate Expected Shortfall (ES).

        Returns:
        --------
        DURATION_NORMAL : dict
            A dictionary containing calculated VaR, ES and other relevant metrics.
        """

        import warnings ; warnings.filterwarnings("ignore")

        import numpy as np ; from scipy.interpolate import CubicSpline ; import pandas as pd 
        
        from inputsControlFunctions import (check_yield_curve_df, check_interval, check_alpha)

        # check procedure 
        yield_curve = check_yield_curve_df(yield_curve) ; interval = check_interval(interval) ; alpha = check_alpha(alpha) 

        # calculating the maximum possible interval based on the bond with the shortest maturity
        max_possible_interval = min(self.maturity_s) * 12 * 25  # assuming 25 trading days in a month
        limit = 25 * 6  # a predefined limit set to 150 days (equivalent to 6 months)

        # checking if the provided 'interval' exceeds either of the two constraints: max_possible_interval or limit
        if interval >= max_possible_interval or interval >= limit:
            if interval >= max_possible_interval:
                reason1 = f"""
                The bond with the shortest maturity in the portfolio has a remaining time of {min(self.maturity_s)} years,
                equivalent to {min(self.maturity_s) * 12} months or {max_possible_interval} days."""
                reason2 = f"""
                The provided 'interval' of {interval} days exceeds or matches this limit, making the VaR/ES metrics unreliable."""
                suggestion1 = f"""
                Please choose an 'interval' less than {max_possible_interval} days.
                """
            else:
                reason1 = f"""
                Setting this interval to a period longer than {limit} days (equivalent to 6 months) is not allowed for the following reasons:"""
                reason2 = """
                1. Numerical Instability: Long time periods can exacerbate the small errors or approximations inherent in numerical methods.
                2. Huge Approximations: Over such a long horizon, many factors like interest rate changes, market volatility, etc., are likely to 
                significantly affect the bond's price."""
                suggestion1 = f"""
                Therefore, the maximum allowed 'interval' for risk metric calculation is set to be {limit} days.
                """

            raise Exception(f"""
                Invalid 'interval' for VaR/ES (Value at Risk/Expected Shortfall) calculation.
            {reason1}
            {reason2}
            {suggestion1}
            """)
        
        # -------------------------------------------------------------------------------------------------------------------------------------
        # >>>>>> although the handler for interval >= max_possible_interval appears redundant, it's retained for future development <<<<<<<<<
        # -------------------------------------------------------------------------------------------------------------------------------------

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # dividing each entry by 100 to convert percentages to decimals
        yield_curve = yield_curve / 100

        # creating an array for new columns; each step is 0.04 to represent a day (1/25 month)
        new_columns = np.arange(0.04, yield_curve.shape[1] + 0.04 , 0.04)

        # initializing an empty DataFrame with the same row index as 'yield_curve' and columns as 'new_columns'
        interpolated_df = pd.DataFrame(index = yield_curve.index, columns=new_columns)

        # iterating through each row in the 'yield_curve' DataFrame
        for i, row in yield_curve.iterrows():
            # converting the index and row values to lists for cubic spline interpolation
            existing_terms = row.index.astype(int).tolist() ; existing_yields = row.values.tolist()
            # applying cubic spline interpolation
            cs = CubicSpline(existing_terms, existing_yields)
            # obtaining new interpolated yields
            interpolated_yields = cs(new_columns)
            # storing the interpolated yields into the new DataFrame
            interpolated_df.loc[i] = interpolated_yields

        # converting the data type of all elements in the DataFrame to float
        interpolated_df = interpolated_df.astype(float)

        # initializing an empty array to store rounded column names
        columns_rounded = np.zeros(len(interpolated_df.columns))

        # iterating through each column and rounding the name to 2 decimal places
        for i, col in enumerate(interpolated_df.columns):
            columns_rounded[i] = round(col, 2)

        # updating the column names with the rounded values
        interpolated_df.columns = columns_rounded

        # calculating the difference between rows at a given interval, then removing NA rows
        DiffIntervalYieldCurve = interpolated_df.diff(periods = interval, axis=0).dropna()

        # creating a DataFrame of yield curves for simulation, based on the differences and the last yield curve
        SimuYieldCurve = DiffIntervalYieldCurve + interpolated_df.iloc[-1]

        # Time Decay and Yield Curve Shifts
        # ---------------------------------
        # This method will consider both the natural time decay of each bond (i.e., approaching maturity) and potential shifts in the yield curve
        # ---------------------------------------------------------------------------------------------------------------------------------------

        # PL calculator ***************************************************************************************************************************************
        
        def PlCalculator(SimuYieldCurve,interpolated_df, interval, num_assets_s, is_zcb_s,face_value_s,maturity_s, semi_annual_payment_s, coupon_rate_s):
            
            # 1.FAIR PRICES

            # initializing arrays to store bond attributes
            bond_prices_fair = np.zeros(len(num_assets_s))

            # ------------------------------------------------------------------------------------------------------------------

            # iterating over each bond
            for bond_num in range(len(num_assets_s)):

                # if is_zcb_s[bond_num] == True /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                if is_zcb_s[bond_num] == True:

                    # calculating the exact yield and bond price for zero-coupon bonds
                    NewMaturity =  round(((maturity_s[bond_num] * 12 * 25) - interval) / 25,2)
                    exact_yield = interpolated_df.iloc[-1].loc[NewMaturity]

                    # calculating the bond price using the zero-coupon bond pricing formula: P = F / (1 + r)^n
                    bond_prices_fair[bond_num] = face_value_s[bond_num] / ( (1 + (exact_yield))  ** (NewMaturity/12))

                # if is_zcb_s[bond_num] == False ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

                if is_zcb_s[bond_num] == False:

                    # determining the payment schedules based on the type of payments (yearly or semi-annual)

                    # if yearly payments -------------------------------------------------------------------------------------
                    if not semi_annual_payment_s[bond_num]: 
                        paymentSchedules = (np.arange(1 ,(maturity_s[bond_num] + 1), 1) * 12) - (interval / 25)
                    # if semi annual payment payments ------------------------------------------------------------------------
                    if semi_annual_payment_s[bond_num]:
                        paymentSchedules = (np.arange(0.5 ,(maturity_s[bond_num] + 0.5), 0.5) * 12) - (interval / 25)
                        coupon_rate_s[bond_num] = coupon_rate_s[bond_num] / 2 # PIVOTAL
                    # --------------------------------------------------------------------------------------------------------

                    # rounding each value in the paymentSchedules array to two decimal places for precision
                    for u in range(len(paymentSchedules)):
                        paymentSchedules[u] = round(paymentSchedules[u], 2)

                    # filtering the paymentSchedules array to keep only values that are strictly greater than zero
                    # if a value is less or equal to 0 in the paymentSchedules, it indicates that the bond has already paid the coupons for those periods.
                    # FOR FUTURE DEVELOPEMENT */*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
                    paymentSchedules = [x for x in paymentSchedules if x > 0]

                    # calculating cash flows and their present values
                    cashFlows = np.zeros(len(paymentSchedules))

                    for i in range(len(cashFlows)):
                        if i != (len(cashFlows)-1):
                            cashFlows[i] = face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)
                        else:
                            cashFlows[i] = (face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)) + face_value_s[bond_num]

                    pvCashFlow = np.zeros(len(paymentSchedules))
                    for index, value in enumerate(paymentSchedules):
                        exact_yield = interpolated_df.iloc[-1].loc[value]
                        pvCashFlow[index] = cashFlows[index] /  ((1+((exact_yield)))**(value/12)) 
                    
                    # calculating the bond price by summing up the present values and setting total future payments for the cb
                    bond_prices_fair[bond_num] = np.sum(pvCashFlow)
            
            # *************************************************************************

            fair_positions_at_new_interval = np.dot(num_assets_s,bond_prices_fair)

            # --------------------------------------------------------------------------------------------------------------------------------

            # 1.SIMULATED PRICES

            PL = np.zeros(SimuYieldCurve.shape[0])
            for j in range(SimuYieldCurve.shape[0]):

                # initializing arrays to store bond attributes
                bond_prices = np.zeros(len(num_assets_s))

                # ------------------------------------------------------------------------------------------------------------------

                # iterating over each bond
                for bond_num in range(len(num_assets_s)):

                    # if is_zcb_s[bond_num] == True /////////////////////////////////////////////////////////////////////////////////////////////////////////

                    if is_zcb_s[bond_num] == True:

                        # calculating the exact yield and bond price for zero-coupon bonds
                        NewMaturity =  round(((maturity_s[bond_num] * 12 * 25) - interval) / 25,2)
                        exact_yield = SimuYieldCurve.iloc[j].loc[NewMaturity]
                        # calculating the bond price using the zero-coupon bond pricing formula: P = F / (1 + r)^n
                        bond_prices[bond_num] = face_value_s[bond_num] / ( (1 + (exact_yield))  ** (NewMaturity/12))

                    # if is_zcb_s[bond_num] == False ////////////////////////////////////////////////////////////////////////////////////////////////////////

                    if is_zcb_s[bond_num] == False:

                        # determining the payment schedules based on the type of payments (yearly or semi-annual)
                        # if yearly payments -------------------------------------------------------------------------------------
                        if not semi_annual_payment_s[bond_num]: 
                            paymentSchedules = (np.arange(1 ,(maturity_s[bond_num] + 1), 1) * 12) - (interval / 25)
                        # if semi annual payment payments ------------------------------------------------------------------------
                        if semi_annual_payment_s[bond_num]:
                            paymentSchedules = (np.arange(0.5 ,(maturity_s[bond_num] + 0.5), 0.5) * 12) - (interval / 25)
                            coupon_rate_s[bond_num] = coupon_rate_s[bond_num] / 2 # PIVOTAL
                        # --------------------------------------------------------------------------------------------------------

                        # rounding each value in the paymentSchedules array to two decimal places for precision
                        for u in range(len(paymentSchedules)):
                            paymentSchedules[u] = round(paymentSchedules[u], 2)

                        # filtering the paymentSchedules array to keep only values that are strictly greater than zero
                        # if a value is less or equal to 0 in the paymentSchedules, it indicates that the bond has already paid the coupons for those periods.
                        paymentSchedules = [x for x in paymentSchedules if x > 0]

                        # calculating cash flows and their present values
                        cashFlows = np.zeros(len(paymentSchedules))

                        for i in range(len(cashFlows)):
                            if i != (len(cashFlows)-1):
                                cashFlows[i] = face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)
                            else:
                                cashFlows[i] = (face_value_s[bond_num] * (coupon_rate_s[bond_num]/100)) + face_value_s[bond_num]

                        pvCashFlow = np.zeros(len(paymentSchedules))

                        for index, value in enumerate(paymentSchedules):
                            exact_yield = SimuYieldCurve.iloc[j].loc[value]
                            pvCashFlow[index] = cashFlows[index] /  ((1+((exact_yield)))**(value/12)) 
                        
                        # calculating the bond price by summing up the present values and setting total future payments for the cb
                        bond_prices[bond_num] = np.sum(pvCashFlow)

            # *************************************************************************

                PL[j] = np.dot(num_assets_s,bond_prices) - fair_positions_at_new_interval

            return PL
            
        # -------------------------------------------------------------------------------------------------------------------------------------------------

        print(f"\rHistorical Simulation for {len(self.maturity_s)} Bonds ----> In progress ...", end=" " * 90)

        PL = PlCalculator(SimuYieldCurve, interpolated_df, interval, self.num_assets_s, self.is_zcb_s,self.face_value_s,self.maturity_s, self.semi_annual_payment_s, self.coupon_rate_s)
        
        print("\rHistorical Simulation --->  Done", end=" " * 100)
        print("")

        # losses
        losses = PL[PL<0]

        if len(losses) == 0:
            raise Exception("""
            No losses were generated in the simulation based on the current data and 'interval' settings.            
            Consider doing one or more of the following:

            1. Review the quality of your input data to ensure it's suitable for simulation.
            2. Slightly adjust the 'interval' parameter to potentially produce a different simulation outcome.
            """)

        # Value at Risk and Expected Shortfall 
        VaR = np.quantile(losses, alpha) * -1 ; ES = np.mean(losses[losses < - VaR]) * -1

        # consolidating results into a dictionary to return
        HS = {"var" : round(VaR, 4),
              "es:" : round(ES, 4),
              "T" : interval,
              "OrigPos" : self.initial_position}

        return HS
    









        