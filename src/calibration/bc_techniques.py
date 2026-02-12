import numpy as np
import pandas as pd
from calibration import calibration_methods as cal_mthd


def ecdf(x: np.ndarray):
    bins = np.sort(x)
    quantiles = np.arange(1, len(bins) + 1 / len(bins))
    return bins, quantiles


def linear_cal(df_sat: pd.DataFrame, df_mooring: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Calibrates a satellite variable using the first 10 days of reference mooring data, then applies the linear
    transformation to the calibration dataset. The coefficients b and a are applied to the satellite validation dataset.

    Parameters:
        df_sat (pd.DataFrame): DataFrame containing satellite data.
        df_mooring (pd.DataFrame): DataFrame containing mooring data.
        variable (str): The name of the variable to calibrate.

    Returns:
        pd.DataFrame: A new DataFrame containing the calibrated satellite validation data for
        the specified variable.
    """

    df_mooring_cal, df_sat_cal, df_mooring_val, df_sat_val = cal_mthd.calib_df_first_ten_days(df_mooring, df_sat)

    x = df_mooring_cal[variable]
    y = df_sat_cal[variable]

    b, a = np.polyfit(y, x, deg=1)
    new_y_validation = a + b * df_sat_val[variable].values
    new_df_sat_validation = df_sat_val.copy()
    new_df_sat_validation[variable] = new_y_validation

    return new_df_sat_validation


def delta_cal(df_sat: pd.DataFrame, df_mooring: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Bias correction method that calculates the mean difference (delta) between the satellite and mooring data for the
    first 10 days of calibration, then applies this delta to adjust the satellite validation dataset.
    Args:
        df_sat: DataFrame containing satellite data.
        df_mooring: DataFrame containing mooring data.
        variable: Variable name to calibrate.

    Returns:
        A new DataFrame containing the calibrated satellite validation data for the specified variable.
    """

    df_mooring_cal, df_sat_cal, df_mooring_val, df_sat_val = cal_mthd.calib_df_first_ten_days(df_mooring, df_sat)

    x = df_mooring_cal[variable]
    y = df_sat_cal[variable]

    delta_factor = x.values.mean() - y.values.mean()
    new_y_validation = df_sat_val[variable] + delta_factor
    new_df_sat_validation = df_sat_val.copy()
    new_df_sat_validation[variable] = new_y_validation

    return new_df_sat_validation


def fdm_correction(df_sat: pd.DataFrame, df_mooring: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Full Distribution Mapping (FDM) bias correction method that adjusts the satellite validation dataset based on the
    cumulative distribution functions (CDFs) of the calibration datasets. The method involves interpolating the
    satellite calibration CDF to the mooring calibration CDF, calculating the quantile differences, fitting a
    polynomial to these differences, and applying the correction to the satellite validation dataset.
    Args:
        df_sat: DataFrame containing satellite data.
        df_mooring: DataFrame containing mooring data.
        variable: Variable name to calibrate.

    Returns:
        A new DataFrame containing the calibrated satellite validation data for the specified variable.
    """

    df_mooring_cal, df_sat_cal, df_mooring_val, df_sat_val = cal_mthd.calib_df_first_ten_days(df_mooring, df_sat)

    x_cal = df_mooring_cal[variable]
    y_cal = df_sat_cal[variable]

    x_val = df_mooring_val[variable]
    y_val = df_sat_val[variable]

    x_cal_sorted, cdf_cal_mooring = ecdf(x_cal)
    y_cal_sorted, cdf_cal_sat = ecdf(y_cal)
    y_val_sorted, cdf_val_sat = ecdf(y_val)

    bias_corrected_fm = np.interp(x=cdf_cal_mooring, xp=cdf_cal_sat, fp=y_cal_sorted)
    x_q_fm = x_cal_sorted - bias_corrected_fm
    coef_p_fm = np.polyfit(bias_corrected_fm, x_q_fm, deg=2)
    y_val_corrected = np.polyval(coef_p_fm, y_val) + y_val

    df_sat_val[variable] = y_val_corrected
    mask = df_sat_val[variable] >= 0
    df_sat_val = df_sat_val[mask]
    df_sat_val_final = df_sat_val.copy()

    return df_sat_val_final


def qm_correction(df_sat: pd.DataFrame, df_mooring: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Quantile Mapping (QM) bias correction method that adjusts the satellite validation dataset based on the quantiles of the calibration datasets. The
    method involves dividing the calibration datasets into quantiles, calculating the CDFs for each quantile, interpolating the satellite calibration CDF to the mooring calibration CDF for each quantile, calculating the quantile differences, fitting a polynomial to these differences, and applying the correction to the satellite validation dataset for each quantile.

    This method allows for a more detailed correction that accounts for differences in the distribution of the data across different quantiles, potentially improving the accuracy of the bias correction, especially when the relationship between the satellite and mooring data is not linear or when there are significant differences in the distribution of the data across different quantiles.

    Args:
        df_sat: DataFrame containing satellite data.
        df_mooring: DataFrame containing mooring data.
        variable: Variable name to calibrate.
    Returns:
        A new DataFrame containing the calibrated satellite validation data for the specified variable.
    """

    df_mooring_cal, df_sat_cal, df_mooring_val, df_sat_val = cal_mthd.calib_df_first_ten_days(df_mooring, df_sat)

    q = np.linspace(0, 1, 10)
    rank_q = np.linspace(1, 10, 10)

    quantiles_sat = np.quantile(df_sat_cal[variable], q)
    quantiles_moor = np.quantile(df_mooring_cal[variable], q)
    quantiles_sat_val = np.quantile(df_sat_val[variable], q)
    df_sat_cal['quantile'] = np.digitize(df_sat_cal[variable], quantiles_sat)
    df_mooring_cal['quantile'] = np.digitize(df_mooring_cal[variable], quantiles_moor)
    df_sat_val['quantile'] = np.digitize(df_sat_val[variable], quantiles_sat_val)

    for i in rank_q:
        mask_moor_cal = df_mooring_cal['quantile'] == i
        mask_sat_cal = df_sat_cal['quantile'] == i
        mask_sat_val = df_sat_val['quantile'] == i
        x_cal_sorted, cdf_cal_mooring = ecdf(df_mooring_cal[variable].loc[mask_moor_cal])
        y_cal_sorted, cdf_cal_sat = ecdf(df_sat_cal[variable].loc[mask_sat_cal])

        bias_corrected = np.interp(x=cdf_cal_mooring, xp=cdf_cal_sat, fp=y_cal_sorted)
        x_q = x_cal_sorted - np.sort(bias_corrected)
        coef_p = np.polyfit(bias_corrected, x_q, deg=3)

        values_corrected = np.polyval(coef_p, df_sat_val[variable].loc[mask_sat_val]) + df_sat_val[variable].loc[
            mask_sat_val]
        df_sat_val.loc[mask_sat_val, variable] = values_corrected

    return df_sat_val.copy()

