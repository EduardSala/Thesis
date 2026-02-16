from sklearn.metrics import root_mean_squared_error
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def bias(df_mooring, df_sat, variable) -> float:
    """
    Compute the bias between the mooring and satellite data for a given variable.
    Args:
        df_mooring: DataFrame containing the mooring data.
        df_sat: DataFrame containing the satellite data.
        variable: Variable for which to compute the bias.

    Returns:
        (float): Bias
    """
    y = df_sat[variable].values
    x = df_mooring[variable].values
    bias_coef = np.mean(y-x)
    return np.round(bias_coef, 5)


def rmse(df_mooring, df_sat, variable) -> float:
    """
        Compute the root mean squared error between the mooring and satellite data for a given variable.
    Args:
        df_mooring: DataFrame containing the mooring data.
        df_sat: DataFrame containing the satellite data.
        variable: DataFrame containing the satellite data.

    Returns:
        (float): Root mean squared error
    """
    y = df_sat[variable].values
    x = df_mooring[variable].values
    rmse_coef = root_mean_squared_error(y, x)
    return np.round(rmse_coef, 5)


def si(df_mooring, df_sat, variable) -> float:
    """
        Compute the scatter index between the mooring and satellite data for a given variable.
    Args:
        df_mooring:
        df_sat:
        variable:

    Returns:
        (float): Scatter index
    """
    y = df_sat[variable].values
    si_coef = rmse(df_mooring, df_sat, variable)/np.mean(y)
    return np.round(si_coef, 5)


def cc(df_mooring, df_sat, variable) -> float:
    """
        Compute the correlation coefficient between the mooring and satellite data for a given variable.
    Args:
        df_mooring:
        df_sat:
        variable:

    Returns:
        (float): Correlation coefficient
    """
    y = df_sat[variable].values
    x = df_mooring[variable].values
    cc_coef, _ = pearsonr(y, x)
    return np.round(cc_coef, 5)


def metrics_array(df_mooring: pd.DataFrame, df_sat: pd.DataFrame, variable: str)\
        -> np.ndarray:
    """
        Compute the bias, root mean squared error, correlation coefficient
        and scatter index between the mooring and satellite data for a given variable and return them as an array.
    Args:
        df_mooring: DataFrame containing the mooring data.
        df_sat: DataFrame containing the satellite data.
        variable: Variable for which to compute the metrics.

    Returns:
        (np.ndarray): Array containing the bias, root mean squared error, correlation coefficient and scatter index.
    """
    return np.array(
        [bias(df_mooring, df_sat, variable), rmse(df_mooring, df_sat, variable), cc(df_mooring, df_sat, variable),
         si(df_mooring, df_sat, variable)], dtype=float)
