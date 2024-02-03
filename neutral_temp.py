import numpy as np


def op_a(bldg):
    """
    TSV = a * top + b
    Regressing Thermal Sensation Vote (TSV) against indoor operative temperature (top).    
    Use the regression model to derive the neutral temperature by letting TSV equals 0.
    The upper and lower limit of 80% comfort zone are derived by letting TSV equals +-0.85.

    Parameters
    --------------------
    bldg : pandas dataframe
    each row of the dataframe is a building. Columns are mean indoor operative temperature,
    mean thermal sensation vote, and other characteristics of each building.  

    Returns
    --------------------
    sig_model : list
    a list of model parameters, including the slope, intercept, neutral temperature, upper
    and lower limit of 80% comfort zone. If the slope is not significant, return np.nan.
    """
    try:
        lm_result = smf.ols(formula='thermal_sensation ~ top', data=bldg).fit()
        slope = lm_result.params['top']
        intercept = lm_result.params['Intercept']
        # check whether the slope is significant
        if lm_result.pvalues['top'] < 0.05:
            a = slope
            b = intercept
            temp_n = -b / a
            temp_up = (0.85-b)/a
            temp_low = (-0.85-b)/a
            sig_model = [slope, intercept, temp_n, temp_up, temp_low]
            return sig_model
    except (ValueError, TypeError):
        pass
    return [np.nan] * 5

def op_b(bldg):
    """
    top = a * TSV + b
    Regressing indoor operative temperature (top) against Thermal Sensation Vote (TSV).
    Use the regression model to derive the neutral temperature by letting TSV equals 0.
    The upper and lower limit of 80% comfort zone are derived by letting TSV equals +-0.85.

    Parameters
    --------------------
    bldg : pandas dataframe
    each row of the dataframe is a building. Columns are mean indoor operative temperature,
    mean thermal sensation vote, and other characteristics of each building. 

    Returns
    --------------------
    sig_model : list
    a list of model parameters, including the slope, intercept, neutral temperature, upper
    and lower limit of 80% comfort zone. If the slope is not significant, return np.nan.
    """
    try:
        lm_result = smf.ols(formula='top ~ thermal_sensation', data=bldg).fit()
        slope = lm_result.params['thermal_sensation']
        intercept = lm_result.params['Intercept']
        # check whether the slope is significant
        if lm_result.pvalues['thermal_sensation'] < 0.05:
            a = slope
            b = intercept
            temp_n = b
            temp_up = 0.85*a + b
            temp_low = (-0.85)*a + b
            sig_model = [slope, intercept, temp_n, temp_up, temp_low]
            return sig_model
    except (ValueError, TypeError):
        pass
    return [np.nan] * 5

def SET_a(bldg):
    """
    TSV = a * set + b
    Regressing Thermal Sensation Vote (TSV) against Standard Effective Temperature (set).
    Use the regression model to derive the neutral temperature by letting TSV equals 0.
    The upper and lower limit of 80% comfort zone are derived by letting TSV equals +-0.85.

    Parameters
    --------------------
    bldg : pandas dataframe
    each row of the dataframe is a building. Columns are mean indoor operative temperature,
    mean thermal sensation vote, and other characteristics of each building. 

    Returns
    --------------------
    sig_model : list
    a list of model parameters, including the slope, intercept, neutral temperature, upper
    and lower limit of 80% comfort zone. If the slope is not significant, return np.nan.
    """
    try:
        lm_result = smf.ols(formula='thermal_sensation ~ set', data=bldg).fit()
        slope = lm_result.params['set']
        intercept = lm_result.params['Intercept']
        # check whether the slope is significant
        if lm_result.pvalues['set'] < 0.05:
            a = slope
            b = intercept
            temp_n = -b / a
            temp_up = (0.85-b)/a
            temp_low = (-0.85-b)/a
            sig_model = [slope, intercept, temp_n, temp_up, temp_low]
            return sig_model
    except (ValueError, TypeError):
        pass
    return [np.nan] * 5

def SET_b(bldg):
    """
    set = a * TSV + b
    Regressing Standard Effective Temperature (set) against Thermal Sensation Vote (TSV).
    Use the regression model to derive the neutral temperature by letting TSV equals 0.
    The upper and lower limit of 80% comfort zone are derived by letting TSV equals +-0.85.

    Parameters
    --------------------
    bldg : pandas dataframe
    each row of the dataframe is a building. Columns are mean indoor operative temperature,
    mean thermal sensation vote, and other characteristics of each building. 

    Returns
    --------------------
    sig_model : list
    a list of model parameters, including the slope, intercept, neutral temperature, upper
    and lower limit of 80% comfort zone. If the slope is not significant, return np.nan.
    """
    try:
        lm_result = smf.ols(formula='set ~ thermal_sensation', data=bldg).fit()
        slope = lm_result.params['thermal_sensation']
        intercept = lm_result.params['Intercept']
        # check whether the slope is significant
        if lm_result.pvalues['thermal_sensation'] < 0.05:
            a = slope
            b = intercept
            temp_n = b
            temp_up = 0.85*a + b
            temp_low = (-0.85)*a + b
            sig_model = [slope, intercept, temp_n, temp_up, temp_low]
            return sig_model
    except (ValueError, TypeError):
        pass
    return [np.nan] * 5