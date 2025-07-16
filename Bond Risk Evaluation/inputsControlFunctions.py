from typing import Union, List ; import numpy as np ; import pandas as pd

## check_yield_curve_today #################################################################################################################################################
############################################################################################################################################################################

def check_yield_curve_today(yield_curve_today):
    """
    Checks the validity of the yield_curve_today DataFrame.
    
    Parameters
    ----------
    yield_curve_today : A DataFrame containing yield curve data for today.
        
    Returns
    -------
    yield_curve_today :  A checked DataFrame containing yield curve data for today.
    
    Raises
    ------
    ValueError: If any of the validation checks fail.
    """

    import pandas as pd ; import numpy as np
    
    # Check if yield_curve_today is a DataFrame
    if not isinstance(yield_curve_today, pd.DataFrame):
        raise ValueError("yield_curve_today must be a DataFrame.")
    
    # Check if yield_curve_today has only one row
    if yield_curve_today.shape[0] != 1:
        raise ValueError("yield_curve_today must have only one row: today's value of the yield curve")
    
    # Check if columns start from 1 and increment by 1
    expected_columns = list(range(1, yield_curve_today.shape[1] + 1))
    if list(yield_curve_today.columns) != expected_columns:
        raise ValueError("Columns must start from the 1 month value and increments by 1.")
    
    # Check if all column values are integers
    if not all(isinstance(col, (int, np.int32, np.int64, float, np.float32, np.float64)) for col in yield_curve_today.columns):
        raise ValueError("All column names must be either integers or floats.")
    
    # Check if all row values are either int, np.int32, np.int64, float, np.float32, or np.float64
    if not all(yield_curve_today.iloc[0].apply(lambda x: isinstance(x, (int, np.int32, np.int64, float, np.float32, np.float64)))):
        raise ValueError("All values in the row must be either int, np.int32, np.int64, float, np.float32, or np.float64.")
    
    return yield_curve_today

## check_maturity #############################################################################################################################################################
############################################################################################################################################################################

def check_maturity(maturity):
    """
    This function checks if the provided maturity parameter is valid for statistical analysis.
    Args:
    maturity: (np.ndarray, pd.core.series.Series, list, number)
        The maturity parameter to be checked for validity.
    Returns:
    maturity: (int)
        The validated maturity to be used for statistical analysis.
    Raises:
        TypeError:
            - If the maturity parameter is a pandas DataFrame object.
            - If the maturity parameter is not a number or an array with valid values.
        Exception:
            - If the Series or array/list with the maturity parameter is empty.
            - If the maturity array/list is more than one-dimensional.
            - If there is more than one maturity parameter provided.
            - If the maturity parameter is not positive and greater than 0.
            - If the maturity is not an integer or a value with a 0.5 decimal part.
    """
    
    import pandas as pd ; import numpy as np

    # Raise TypeError if maturity is a DataFrame
    if isinstance(maturity, pd.core.frame.DataFrame):
        raise TypeError("The maturity parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if maturity is a Series
    if isinstance(maturity, pd.core.series.Series):
        if len(maturity) == 0:
            raise Exception("The Series with the maturity parameter is empty")
        maturity = list(maturity)

    # Handle if maturity is a list or ndarray
    if isinstance(maturity, (np.ndarray, list)):
        if len(maturity) == 0:
            raise Exception("The array/list with the maturity parameter is empty")
        dim = np.ndim(maturity)
        if dim > 1:
            raise Exception("The maturity array/list should be one-dimensional")
        if len(maturity) > 1:
            raise Exception("More than one maturity provided")
        maturity = maturity[0]
        
    # Check if maturity is a number and if it's positive and greater than 0
    if not isinstance(maturity, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)):
        raise TypeError("The maturity parameter must be a number")
    if maturity <= 0:
        raise Exception("The maturity parameter must be positive and greater than 0")

    # Check if maturity is either an integer or a number with a 0.5 decimal part
    if maturity != int(maturity) and (maturity - int(maturity)) != 0.5:
        raise Exception("The maturity (in years) must be either an integer or a number with a 0.5 decimal part")

    return maturity

## check_num_assets #############################################################################################################################################################
############################################################################################################################################################################

def check_num_assets(num_assets):
    """
    This function checks if the provided num_assets parameter is valid for statistical analysis.
    Args:
    num_assets: (np.ndarray, pd.core.series.Series, list, number)
        The num_assets parameter to be checked for validity.
    Returns:
    num_assets: (int)
        The validated num_assets to be used for statistical analysis.
    Raises:
    TypeError: 
        - If the num_assets parameter is a pandas DataFrame object.
        - If the num_assets parameter is not an integer or an array with all integer values.
    Exception:
        - If the Series or array/list with the num_assets parameter is empty.
        - If the num_assets array/list is more than one-dimensional.
        - If there is more than one num_assets parameter provided.
        - If the num_assets parameter is not an integer greater than 1.
    """

    import pandas as pd ; import numpy as np

    # Raise TypeError if num_assets is a DataFrame
    if isinstance(num_assets, pd.core.frame.DataFrame):
        raise TypeError("The num_assets parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if num_assets is a Series
    if isinstance(num_assets, pd.core.series.Series):
        if len(num_assets) == 0:
            raise Exception("The Series with the num_assets parameter is empty")
        num_assets = list(num_assets)

    # Handle if num_assets is a list or ndarray
    if isinstance(num_assets, (np.ndarray, list)):
        if len(num_assets) == 0:
            raise Exception("The array/list with the num_assets parameter is empty")
        dim = np.ndim(num_assets)
        if dim > 1:
            raise Exception("The num_assets array/list should be one-dimensional")
        if len(num_assets) > 1:
            raise Exception("More than one num_assets provided")
        num_assets = num_assets[0]
        
    # Handle if num_assets is a number
    if not isinstance(num_assets, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)):
        raise TypeError("The num_assets parameter must be an number")

    return num_assets

## check_face_value ################################################################################################################################################################
############################################################################################################################################################################

def check_face_value(face_value):
    """
    Checks the validity of the input face_value.

    Parameters:
    face_value: int, float, or array-like
        The number of simulations samples to use for calculations.
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If face_value is not of the correct type.
    ValueError: If no value is provided for face_value or if more than one value is provided.

    Returns:
    face_value: int or float
        The validated number of face_value.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(face_value, (int, np.int32, np.int64, float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The face_value should be a number!")

    # Convert to list
    if isinstance(face_value, (list, pd.Series, np.ndarray)):
        face_value = list(face_value)
        if len(face_value) == 0: 
            raise ValueError("No face_value parameter provided")
        # Check for single value
        if len(face_value) > 1:
            raise ValueError(f"More than one face_value has been provided. This function works only with one face_value parameter at the time")
    
    if isinstance(face_value, (list)):
        face_value = face_value[0]

    if not isinstance(face_value, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError(f"face_value should be a number!")
    
    if face_value <= 0:
        raise TypeError(f"face_value should be a number! higher than 0 --- face_values cannot be negative...")

    return face_value

## check_is_zcb ###########################################################################################################################################################
############################################################################################################################################################################

def check_is_zcb(is_zcb, allowed_values=[True, False]):
    """
    Check if a is_zcb value or a list of is_zcb values is valid.
    Args:
    is_zcb: A boolean, string, list, numpy array, or pandas series representing one or more is_zcb values.
    allowed_values: A list of valid is_zcb values.
    Returns:
    None
    Raises:
    TypeError: If the is_zcb value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one is_zcb value is provided, or if the is_zcb value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: (bool, str, list, np.ndarray, pd.Series)")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(is_zcb, (bool, str, list, np.ndarray, pd.Series), "is_zcb")
    is_zcb = convert_to_list(is_zcb)
    check_single_value(is_zcb, "is_zcb")
    check_in_values(is_zcb, allowed_values, "is_zcb")

## check_coupon_rate ################################################################################################################################################################
############################################################################################################################################################################

def check_coupon_rate(coupon_rate):
    """
    Checks the validity of the input coupon_rate.

    Parameters:
    coupon_rate: int, float, or array-like
        The number of simulations samples to use for calculations.
        Can be a single number or an array-like object containing a single number.

    Raises:
    TypeError: If coupon_rate is not of the correct type.
    ValueError: If no value is provided for coupon_rate or if more than one value is provided.

    Returns:
    coupon_rate: int or float
        The validated number of coupon_rate.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(coupon_rate, (int, np.int32, np.int64, float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The coupon_rate should be a number!")

    # Convert to list
    if isinstance(coupon_rate, (list, pd.Series, np.ndarray)):
        coupon_rate = list(coupon_rate)
        if len(coupon_rate) == 0: 
            raise ValueError("No coupon_rate parameter provided")
        # Check for single value
        if len(coupon_rate) > 1:
            raise ValueError(f"More than one coupon_rate has been provided. This function works only with one coupon_rate parameter at the time")
    
    if isinstance(coupon_rate, (list)):
        coupon_rate = coupon_rate[0]

    if not isinstance(coupon_rate, (int, np.int32, np.int64, float, np.float32, np.float64)):
        raise TypeError(f"coupon_rate should be a number!")
    
    if coupon_rate <= 0:
        raise TypeError(f"coupon_rate should be a number! higher than 0 --- coupon_rate cannot be negative...")

    return coupon_rate

## check_semi_annual_payment ###########################################################################################################################################################
############################################################################################################################################################################

def check_semi_annual_payment(semi_annual_payment, allowed_values=[True, False]):
    """
    Check if a semi_annual_payment value or a list of semi_annual_payment values is valid.
    Args:
    semi_annual_payment: A boolean, string, list, numpy array, or pandas series representing one or more semi_annual_payment values.
    allowed_values: A list of valid semi_annual_payment values.
    Returns:
    None
    Raises:
    TypeError: If the semi_annual_payment value is not a boolean, string, list, numpy array, or pandas series.
    ValueError: If more than one semi_annual_payment value is provided, or if the semi_annual_payment value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: (bool, str, list, np.ndarray, pd.Series)")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            return list(input)
        if isinstance(input, list):
            return input
        return [input]

    def check_in_values(input, values, name):
        if input[0] not in values:
            raise ValueError(f"Please, be sure to use a correct {name}! {values}")

    check_input_type(semi_annual_payment, (bool, str, list, np.ndarray, pd.Series), "semi_annual_payment")
    semi_annual_payment = convert_to_list(semi_annual_payment)
    check_single_value(semi_annual_payment, "semi_annual_payment")
    check_in_values(semi_annual_payment, allowed_values, "semi_annual_payment")

## check_maturity_s ####################################################################################################################################################
############################################################################################################################################################################

def check_maturity_s(maturity_s):
    """
    This function validates the input 'maturity_s', ensuring that it is a one-dimensional array-like object (numpy array, pandas series, or list)
    that contains only real numbers, and has at least one non-zero element.

    Parameters
    ----------
    maturity_s : np.ndarray, pd.core.series.Series, list
        The maturities of a portfolio of bonds. This should be a one-dimensional array-like object containing only real numbers.

    Raises
    ------
    TypeError
        If 'maturity_s' is not a numpy array, pandas Series or list.
        If 'maturity_s' contains elements that are not numbers.
    Exception
        If 'maturity_s' is empty.
        If 'maturity_s' is multi-dimensional.
        If all elements of 'maturity_s' are zero.

    Returns
    -------
    maturity_s : np.ndarray
        The validated maturity_s array.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(maturity_s, (np.ndarray, pd.core.series.Series, list)):
        raise TypeError("maturity_s should be provided in the following object: [ np.ndarray, pd.core.series.Series, list ]")
    
    if isinstance(maturity_s, pd.Series):
        maturity_s = maturity_s.to_numpy()

        if len(maturity_s) == 0: 
            raise Exception("The Series with the maturity_s is empty")
        
    if isinstance(maturity_s, np.ndarray):
        if len(maturity_s) == 0: 
            raise Exception("The array with the maturity_s is empty")
        
        if maturity_s.ndim > 1:
            raise Exception("The maturity_s array should be a one-dimensional array")
        
        if not np.all(np.isreal(maturity_s)):
            raise TypeError("The array of maturity_s should contain only numbers")
        
    if isinstance(maturity_s, list):
        if len(maturity_s) == 0: 
            raise Exception("The list with the maturity_s is empty")
        
        if np.ndim(maturity_s) > 1:
            raise Exception("The maturity_s list should be a one-dimensional list")
        
        if not all(isinstance(item, (int, np.int32, np.int64, float, np.float32, np.float64)) for item in maturity_s):
            raise TypeError("The list of maturity_s should contain only numbers")
        
        maturity_s = np.array(maturity_s)

    if np.all(maturity_s <= 0):
        raise Exception("All elements in maturity_s must be greater than 0.")
        
    # Check if the numbers are integers or float with .5 increment
    if not all((x > 0 and (x == int(x) or (x - int(x)) == 0.5)) for x in maturity_s):
        raise Exception("All elements in maturity_s should be positive integers or floats ending in 0.5.")

    return maturity_s

## check_num_assets_s #####################################################################################################################################################
############################################################################################################################################################################

def check_num_assets_s(num_assets_s):
    """
    Check if a list, pandas series, numpy array or dataframe of num_assets_s is valid and return a numpy array of num_assets_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of num_assets_s for a multiple options.
    Returns:
    A numpy array of num_assets_s.
    Raises:
    TypeError: If the num_assets_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of S0_s is empty.
    Exception: If the num_assets_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the num_assets_s contain non-numeric types.
    Exception: If the num_assets_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(num_assets_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("num_assets_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(num_assets_s,list):
        if len(num_assets_s) == 0:
            raise Exception("The list of num_assets_s is empty")
        dim = np.ndim(num_assets_s)
        if dim > 1:
            raise Exception("The num_assets_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_assets_s):
            raise TypeError("The list of num_assets_s should contain only numbers")
        num_assets_s = np.asarray(num_assets_s)

    if isinstance(num_assets_s,pd.core.series.Series):
        num_assets_s = num_assets_s.values
        if len(num_assets_s) == 0:
            raise Exception("The Series of num_assets_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_assets_s):
            raise TypeError("The Series of num_assets_s should contain only numbers")
        num_assets_s = np.asarray(num_assets_s)

    if isinstance(num_assets_s,np.ndarray):
        if len(num_assets_s) == 0:
            raise Exception("The array of num_assets_s is empty")
        dim = np.ndim(num_assets_s)
        if dim > 1:
            raise Exception("The num_assets_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_assets_s):
            raise TypeError("The array of num_assets_s should contain only numbers")

    if isinstance(num_assets_s,pd.DataFrame):
        if num_assets_s.empty:
            raise Exception("The DataFrame with the num_assets_s is empty")
        if num_assets_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in num_assets_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in num_assets_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if num_assets_s.shape[1] == 1:
            num_assets_s = num_assets_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(num_assets_s).any():
        raise Exception("The num_assets_s contain NaN values")

    return num_assets_s

## check_face_value_s ####################################################################################################################################################
############################################################################################################################################################################

def check_face_value_s(face_value_s):
    """
    This function validates the input 'face_value_s', ensuring that it is a one-dimensional array-like object (numpy array, pandas series, or list)
    that contains only real numbers, and has at least one non-zero element.

    Parameters
    ----------
    face_value_s : np.ndarray, pd.core.series.Series, list
        The maturities of a portfolio of bonds. This should be a one-dimensional array-like object containing only real numbers.

    Raises
    ------
    TypeError
        If 'face_value_s' is not a numpy array, pandas Series or list.
        If 'face_value_s' contains elements that are not numbers.
    Exception
        If 'face_value_s' is empty.
        If 'face_value_s' is multi-dimensional.
        If all elements of 'face_value_s' are zero.

    Returns
    -------
    face_value_s : np.ndarray
        The validated face_value_s array.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(face_value_s, (np.ndarray, pd.core.series.Series, list)):
        raise TypeError("face_value_s should be provided in the following object: [ np.ndarray, pd.core.series.Series, list ]")
    
    if isinstance(face_value_s, pd.Series):
        face_value_s = face_value_s.to_numpy()

        if len(face_value_s) == 0: 
            raise Exception("The Series with the face_value_s is empty")
        
    if isinstance(face_value_s, np.ndarray):
        if len(face_value_s) == 0: 
            raise Exception("The array with the face_value_s is empty")
        
        if face_value_s.ndim > 1:
            raise Exception("The face_value_s array should be a one-dimensional array")
        
        if not np.all(np.isreal(face_value_s)):
            raise TypeError("The array of face_value_s should contain only numbers")
        
    if isinstance(face_value_s, list):
        if len(face_value_s) == 0: 
            raise Exception("The list with the face_value_s is empty")
        
        if np.ndim(face_value_s) > 1:
            raise Exception("The face_value_s list should be a one-dimensional list")
        
        if not all(isinstance(item, (int, np.int32, np.int64, float, np.float32, np.float64)) for item in face_value_s):
            raise TypeError("The list of face_value_s should contain only numbers")
        
        face_value_s = np.array(face_value_s)

    if np.all(face_value_s <= 0):
        raise Exception("All elements in face_value_s must be greater than 0.")
        
    if not all((x > 0) for x in face_value_s):
        raise Exception("All elements in face_value_s should be positive integers or floats - face value cannot be negative")

    return face_value_s

## check_is_zcb_s #####################################################################################################################################################
############################################################################################################################################################################

def check_is_zcb_s(is_zcb_s):
    """
    Check if a list, pandas series, numpy array, or dataframe of is_zcb_s is valid and return a numpy array of is_zcb_s.
    
    Parameters
    ----------
    is_zcb_s : list, pd.Series, np.ndarray, pd.DataFrame
        A list, pandas series, numpy array, or dataframe of is_zcb_s for multiple options.

    Raises
    ------
    TypeError
        If the is_zcb_s are not provided in a list, pandas series, numpy array, or dataframe.
        If the is_zcb_s contain non-boolean types.
    Exception
        If the list, series, array, or dataframe of is_zcb_s is empty.

    Returns
    -------
    np.ndarray
        A numpy array of is_zcb_s with dtype np.bool_.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(is_zcb_s, (list, pd.Series, np.ndarray, pd.DataFrame)):
        raise TypeError("is_zcb_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    # Convert different types to numpy array for uniform handling
    is_zcb_s = np.asarray(is_zcb_s)

    # Check for empty array
    if is_zcb_s.size == 0:
        raise Exception("The array of is_zcb_s is empty")
    
    # Check for bool dtype
    if is_zcb_s.dtype != np.bool_:
        raise TypeError("The array of is_zcb_s should contain only bools [True,False]")

    return is_zcb_s

## check_coupon_rate_s ###############################################################################################################################################################
############################################################################################################################################################################

def check_coupon_rate_s(coupon_rate_s):
    """
    Check if a list, pandas series, numpy array or dataframe of coupon_rate_s is valid and return a numpy array of coupon_rate_s.
    Args:
    returns: A list, pandas series, numpy array, or dataframe of coupon_rate_s for a multiple options.
    Returns:
    A numpy array of coupon_rate_s.
    Raises:
    TypeError: If the coupon_rate_s are not provided in a list, pandas series, numpy array or dataframe.
    Exception: If the list, series, array or dataframe of coupon_rate_s is empty.
    Exception: If the coupon_rate_s list, series, array or dataframe has more than one dimension or more than one column.
    TypeError: If the coupon_rate_s contain non-numeric types.
    Exception: If the coupon_rate_s contain NaN values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(coupon_rate_s,(list,pd.core.series.Series,np.ndarray, pd.DataFrame)):
        raise TypeError("coupon_rate_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    if isinstance(coupon_rate_s,list):
        if len(coupon_rate_s) == 0:
            raise Exception("The list of coupon_rate_s is empty")
        dim = np.ndim(coupon_rate_s)
        if dim > 1:
            raise Exception("The coupon_rate_s list should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in coupon_rate_s):
            raise TypeError("The list of coupon_rate_s should contain only numbers")
        coupon_rate_s = np.asarray(coupon_rate_s)

    if isinstance(coupon_rate_s,pd.core.series.Series):
        coupon_rate_s = coupon_rate_s.values
        if len(coupon_rate_s) == 0:
            raise Exception("The Series of coupon_rate_s is empty")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in coupon_rate_s):
            raise TypeError("The Series of coupon_rate_s should contain only numbers")
        coupon_rate_s = np.asarray(coupon_rate_s)

    if isinstance(coupon_rate_s,np.ndarray):
        if len(coupon_rate_s) == 0:
            raise Exception("The array of coupon_rate_s is empty")
        dim = np.ndim(coupon_rate_s)
        if dim > 1:
            raise Exception("The coupon_rate_s array should be one-dimensional")
        if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in coupon_rate_s):
            raise TypeError("The array of coupon_rate_s should contain only numbers - (%)")

    if isinstance(coupon_rate_s,pd.DataFrame):
        if coupon_rate_s.empty:
            raise Exception("The DataFrame with the coupon_rate_s is empty")
        if coupon_rate_s.shape[1] > 1:
            raise Exception("A DataFrame with more than one column has been provided")
        for col in coupon_rate_s.columns:
            if not all(isinstance(item, (int, np.int16, np.int32, np.int64, float, np.float16, np.float32, np.float64)) for item in coupon_rate_s[col]):
                raise TypeError(f"The DataFrame column {col} should contain only numbers")
        if coupon_rate_s.shape[1] == 1:
            coupon_rate_s = coupon_rate_s.values
    
    # After converting to numpy array, check for NaN values
    if np.isnan(coupon_rate_s).any():
        raise Exception("The coupon_rate_s contain NaN values")

    # Check if any value is less than zero
    if (coupon_rate_s < 0).any():
        raise Exception("All values in coupon_rate_s must be greater or equal to 0")
        
    return coupon_rate_s

## check_semi_annual_payment_s #####################################################################################################################################################
############################################################################################################################################################################

def check_semi_annual_payment_s(semi_annual_payment_s):
    """
    Check if a list, pandas series, numpy array, or dataframe of semi_annual_payment_s is valid and return a numpy array of semi_annual_payment_s.
    
    Parameters
    ----------
    semi_annual_payment_s : list, pd.Series, np.ndarray, pd.DataFrame
        A list, pandas series, numpy array, or dataframe of semi_annual_payment_s for multiple options.

    Raises
    ------
    TypeError
        If the semi_annual_payment_s are not provided in a list, pandas series, numpy array, or dataframe.
        If the semi_annual_payment_s contain non-boolean types.
    Exception
        If the list, series, array, or dataframe of semi_annual_payment_s is empty.

    Returns
    -------
    np.ndarray
        A numpy array of semi_annual_payment_s with dtype np.bool_.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(semi_annual_payment_s, (list, pd.Series, np.ndarray, pd.DataFrame)):
        raise TypeError("semi_annual_payment_s must be provided in the form of: [list, pd.Series, np.array, pd.DataFrame]")

    # Convert different types to numpy array for uniform handling
    semi_annual_payment_s = np.asarray(semi_annual_payment_s)

    # Check for empty array
    if semi_annual_payment_s.size == 0:
        raise Exception("The array of semi_annual_payment_s is empty")
    
    # Check for bool dtype
    if semi_annual_payment_s.dtype != np.bool_:
        raise TypeError("The array of semi_annual_payment_s should contain only bools [True,False]")

    return semi_annual_payment_s

## check_yield_curve_df #################################################################################################################################################
############################################################################################################################################################################

def check_yield_curve_df(yield_curve):
    """
    Checks the validity of the yield_curve DataFrame.
    
    Parameters
    ----------
    yield_curve : A DataFrame containing yield curve data for today.
        
    Returns
    -------
    yield_curve :  A checked DataFrame containing yield curve data for today.
    
    Raises
    ------
    ValueError: If any of the validation checks fail.
    """

    import pandas as pd ; import numpy as np
    
    # Check if yield_curve is a DataFrame
    if not isinstance(yield_curve, pd.DataFrame):
        raise ValueError("yield_curve must be a DataFrame.")
    
    # Check if columns start from 1 and increment by 1
    expected_columns = list(range(1, yield_curve.shape[1] + 1))
    if list(yield_curve.columns) != expected_columns:
        raise ValueError("Columns must start from the 1 month value and increments by 1.")
    
    # Check if all column values are integers
    if not all(isinstance(col, (int, np.int32, np.int64, float, np.float32, np.float64)) for col in yield_curve.columns):
        raise ValueError("All column names must be either integers or floats.")
    
    # Check if all row and column values are of the required types
    if not yield_curve.applymap(lambda x: isinstance(x, (int, np.int32, np.int64, float, np.float32, np.float64))).all().all():
        raise ValueError("All values in the DataFrame must be either int, np.int32, np.int64, float, np.float32, or np.float64.")
    
    return yield_curve

## check_vol ###############################################################################################################################################################
############################################################################################################################################################################

def check_vol(vol, allowed_values=["garch", "ewma","simple"]):
    """
    Check if a vol value or a list of vol values is valid.
    Args:
    vol: A string, list, numpy array, or pandas series representing one or more vol values.
    allowed_values: A list of valid vol values.
    Returns:
    None
    Raises:
    TypeError: If the vol value is not a string, list, numpy array, or pandas series.
    ValueError: If more than one vol value is provided, or if the vol value is not one of the allowed values.
    """

    import pandas as pd ; import numpy as np

    if not isinstance(vol, (str, list, np.ndarray, pd.Series)):
        raise TypeError("The vol should be one of the following types: (str, list, np.ndarray, pd.Series)")

    if isinstance(vol, (pd.Series, np.ndarray)):
        vol = list(vol)

    if isinstance(vol, list):
        if len(vol) > 1:
            raise ValueError("More than one vol has been provided. This function works only with one vol at a time")
        if len(vol) == 1:
            vol = vol[0]

    if vol not in allowed_values:
        raise ValueError(f"Please, be sure to use a correct vol! {allowed_values}")
    
    return vol

## check_interval #########################################################################################################################################################
############################################################################################################################################################################

def check_interval(interval):
    """
    This function checks if the provided interval parameter is valid for statistical analysis.
    Args:
    interval: (np.ndarray, pd.core.series.Series, list, number)
        The interval parameter to be checked for validity.
    Returns:
    interval: (int)
        The validated interval to be used for statistical analysis.
    Raises:
    TypeError: 
        - If the interval parameter is a pandas DataFrame object.
        - If the interval parameter is not an integer or an array with all integer values.
    Exception:
        - If the Series or array/list with the interval parameter is empty.
        - If the interval array/list is more than one-dimensional.
        - If there is more than one interval parameter provided.
        - If the interval parameter is not an integer greater than 1.
    """

    import pandas as pd ; import numpy as np

    # Raise TypeError if interval is a DataFrame
    if isinstance(interval, pd.core.frame.DataFrame):
        raise TypeError("The interval parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if interval is a Series
    if isinstance(interval, pd.core.series.Series):
        if len(interval) == 0:
            raise Exception("The Series with the interval parameter is empty")
        interval = list(interval)

    # Handle if interval is a list or ndarray
    if isinstance(interval, (np.ndarray, list)):
        if len(interval) == 0:
            raise Exception("The array/list with the interval parameter is empty")
        dim = np.ndim(interval)
        if dim > 1:
            raise Exception("The interval array/list should be one-dimensional")
        if len(interval) > 1:
            raise Exception("More than one interval provided")
        interval = interval[0]
        
    # Handle if interval is a number
    if not isinstance(interval, (int, np.int32, np.int64)):
        raise TypeError("The interval parameter must be an integer ")

    # Ensure the value of interval higher than one
    if interval < 1:
        raise Exception("Please, insert a correct value for interval parameter! > 1)")

    return interval

## check_alpha #############################################################################################################################################################
############################################################################################################################################################################

def check_alpha(alpha):
    """
    Check if an alpha value or a list of alpha values is valid.
    Args:
    alpha: A float, list, numpy array, or pandas series representing one or more alpha values.
    Returns:
    alpha: A float representing a valid alpha value.
    Raises:
    TypeError: If the alpha value is not a float, list, numpy array, or pandas series.
    ValueError: If more than one alpha value is provided, or if the alpha value is not within the range of 0 to 1.
    """

    import pandas as pd ; import numpy as np

    # Check input type
    if not isinstance(alpha, (float, np.float32, np.float64, list, np.ndarray, pd.Series)):
        raise TypeError(f"The alpha should be one of the following types: float, list, np.ndarray, pd.Series")

    # Convert to list if input is pd.Series or np.ndarray
    if isinstance(alpha, (pd.Series, np.ndarray)):
        if len(alpha) == 0: 
            raise ValueError("No alpha provided")
        if len(alpha) > 1:
            raise ValueError("More than one alpha provided")
        alpha =  list(alpha)

    # Check if input is list and convert to single float
    if isinstance(alpha, list):
        if len(alpha) == 0: 
            raise ValueError("No alpha provided")
        if len(alpha) > 1:
            raise ValueError("More than one alpha provided")
        alpha = alpha[0]

    # Check alpha range
    if not isinstance(alpha, (float, np.float32, np.float64)):
        raise TypeError("The alpha value should be a number (float)")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("The alpha value should be between 0 and 1")

    return alpha

## check_p #################################################################################################################################################################
############################################################################################################################################################################

def check_p(p):
    """
    This function checks if the provided p parameter is valid for statistical analysis.
    Args:
    p: (np.ndarray, pd.core.series.Series, list, number)
        The p parameter to be checked for validity.
    Returns:
    p: (int)
        The validated p to be used for statistical analysis.
    Raises:
    TypeError: 
        - If the p parameter is a pandas DataFrame object.
        - If the p parameter is not an integer or an array with all integer values.
    Exception:
        - If the Series or array/list with the p parameter is empty.
        - If the p array/list is more than one-dimensional.
        - If there is more than one p parameter provided.
        - If the p parameter is not an integer greater than 1.
    """

    import pandas as pd ; import numpy as np

    # Raise TypeError if p is a DataFrame
    if isinstance(p, pd.core.frame.DataFrame):
        raise TypeError("The p parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if p is a Series
    if isinstance(p, pd.core.series.Series):
        if len(p) == 0:
            raise Exception("The Series with the p parameter is empty")
        p = list(p)

    # Handle if p is a list or ndarray
    if isinstance(p, (np.ndarray, list)):
        if len(p) == 0:
            raise Exception("The array/list with the p parameter is empty")
        dim = np.ndim(p)
        if dim > 1:
            raise Exception("The p array/list should be one-dimensional")
        if len(p) > 1:
            raise Exception("More than one p provided")
        p = p[0]
        
    # Handle if p is a number
    if not isinstance(p, (int, np.int32, np.int64)):
        raise TypeError("The p parameter must be an integer ")

    # Ensure the value of p higher than one
    if p < 1:
        raise Exception("Please, insert a correct value for p parameter! > 1)")

    return p

## check_q #################################################################################################################################################################
############################################################################################################################################################################

def check_q(q):
    """
    This function checks if the provided q parameter is valid for statistical analysis.
    Args:
    q: (np.ndarray, pd.core.series.Series, list, number)
        The q parameter to be checked for validity.
    Returns:
    q: (int)
        The validated q to be used for statistical analysis.
    Raises:
    TypeError: 
        - If the q parameter is a pandas DataFrame object.
        - If the q parameter is not an integer or an array with all integer values.
    Exception:
        - If the Series or array/list with the q parameter is empty.
        - If the q array/list is more than one-dimensional.
        - If there is more than one q parameter provided.
        - If the q parameter is not an integer greater than 1.
    """

    import pandas as pd ; import numpy as np

    # Raise TypeError if q is a DataFrame
    if isinstance(q, pd.core.frame.DataFrame):
        raise TypeError("The q parameter should be provided in the following object: [np.ndarray, pd.core.series.Series, list, number]")

    # Handle if q is a Series
    if isinstance(q, pd.core.series.Series):
        if len(q) == 0:
            raise Exception("The Series with the q parameter is empty")
        q = list(q)

    # Handle if q is a list or ndarray
    if isinstance(q, (np.ndarray, list)):
        if len(q) == 0:
            raise Exception("The array/list with the q parameter is empty")
        dim = np.ndim(q)
        if dim > 1:
            raise Exception("The q array/list should be one-dimensional")
        if len(q) > 1:
            raise Exception("More than one q provided")
        q = q[0]
        
    # Handle if q is a number
    if not isinstance(q, (int, np.int32, np.int64 )):
        raise TypeError("The q parameter must be an integer ")

    # Ensure the value of q higher than one
    if q < 1:
        raise Exception("Please, insert a correct value for q parameter! > 1)")

    return q

## check_lambda_ewma ######################################################################################################################################################
############################################################################################################################################################################

def check_lambda_ewma(lambda_ewma):
    """
    Check if an lambda_ewma value or a list of lambda_ewma values is valid.
    Args:
    lambda_ewma: A float, list, numpy array, or pandas series representing one or more lambda_ewma values.
    Returns:
    lambda_ewma
    Raises:
    TypeError: If the lambda_ewma value is not a float, list, numpy array, or pandas series.
    ValueError: If more than one lambda_ewma value is provided, or if the lambda_ewma value is not within the range of 0 to 1.
    """

    import pandas as pd ; import numpy as np

    def check_input_type(input, types, name):
        if not isinstance(input, types):
            raise TypeError(f"The {name} should be one of the following types: [float, list, np.ndarray, pd.Series]")

    def check_single_value(input, name):
        if isinstance(input, (list, np.ndarray, pd.Series)) and len(input) > 1:
            raise ValueError(f"More than one {name} has been provided. This function works only with one {name} at a time")

    def convert_to_list(input):
        if isinstance(input, (pd.Series, np.ndarray)):
            if len(input) == 0: 
                raise ValueError("No lambda_ewma provided")
            if len(input) > 1:
                raise Exception("More than one lambda_ewma provided")
            input=  list(input)
        if isinstance(input, list):
            if len(input) == 0: 
                raise ValueError("No lambda_ewma provided")
            if len(input) > 1:
                raise Exception("More than one lambda_ewma provided")
            input =  input[0]
        return input

    def check_lambda_ewma(lambda_ewma):
        if not isinstance(lambda_ewma, (float, np.float16, np.float32, np.float64)):
            raise TypeError("The lambda_ewma value should be a number (float)")
        if lambda_ewma <= 0 or lambda_ewma >= 1:
            raise ValueError("The lambda_ewma value should be between 0 and 1")

    check_input_type(lambda_ewma, (float, np.float16, np.float32, np.float64, list, np.ndarray, pd.Series), "lambda_ewma")
    lambda_ewma = convert_to_list(lambda_ewma)
    check_single_value(lambda_ewma, "lambda_ewma")
    check_lambda_ewma(lambda_ewma)

    return lambda_ewma
