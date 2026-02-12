import numpy as np
import pandas as pd
from calibration import calibration_methods as cal_mthd


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes the empirical cumulative distribution function (ECDF) for a given array of data. The ECDF is a step function that represents the proportion of data points that are less than or equal to a given value. It is calculated by sorting the data and assigning quantiles based on the rank of each data point.

    Parameters:
        x: Vector of data points for which to compute the ECDF.

    Returns:
        A tuple containing two arrays: the first array contains the sorted data points (bins), and the second array contains the corresponding quantiles (proportions) for each data point.
    """
    bins = np.sort(x)
    quantiles = np.arange(1, len(bins) + 1) / len(bins)
    return bins, quantiles


def linear_cal(df_sat_cal: pd.DataFrame, df_mooring_cal: pd.DataFrame, df_sat_val: pd.DataFrame, df_mooring_val: pd.DataFrame,variable: str) -> pd.DataFrame:

    """Calibrates the satellite validation dataset using a linear regression method. The method fits a linear model to the calibration datasets and applies the resulting coefficients to adjust the satellite validation dataset.

    Parameters:
        df_mooring_cal: DataFrame containing the mooring calibration data.
        df_sat_cal: DataFrame containing the satellite calibration data.
        df_mooring_val: DataFrame containing the mooring validation data.
        df_sat_val: DataFrame containing the satellite validation data.
        variable: Variable name to calibrate.
    Returns:
        pd.DataFrame: A new DataFrame containing the calibrated satellite validation data for
        the specified variable.
    """

    x = df_mooring_cal[variable]
    y = df_sat_cal[variable]

    b, a = np.polyfit(y, x, deg=1)
    new_y_validation = a + b * df_sat_val[variable].values
    new_df_sat_validation = df_sat_val.copy()
    new_df_sat_validation[variable] = new_y_validation

    return new_df_sat_validation


def delta_cal(df_sat_cal: pd.DataFrame, df_mooring_cal: pd.DataFrame, df_sat_val: pd.DataFrame, df_mooring_val: pd.DataFrame,variable: str) -> pd.DataFrame:
    """Bias correction method that calculates the mean difference (delta) between the satellite and mooring data, then applies this delta to adjust the satellite validation dataset.

    Parameters:
        df_mooring_cal: DataFrame containing the mooring calibration data.
        df_sat_cal: DataFrame containing the satellite calibration data.
        df_mooring_val: DataFrame containing the mooring validation data.
        df_sat_val: DataFrame containing the satellite validation data.
        variable: Variable name to calibrate.

    Returns:
        A new DataFrame containing the calibrated satellite validation data for the specified variable.
    """

    x = df_mooring_cal[variable]
    y = df_sat_cal[variable]

    delta_factor = x.values.mean() - y.values.mean()
    new_y_validation = df_sat_val[variable] + delta_factor
    new_df_sat_validation = df_sat_val.copy()
    new_df_sat_validation[variable] = new_y_validation

    return new_df_sat_validation


def fdm_correction(df_sat_cal: pd.DataFrame, df_mooring_cal: pd.DataFrame, df_sat_val: pd.DataFrame, df_mooring_val: pd.DataFrame,variable: str) -> pd.DataFrame:
    """Full Distribution Mapping (FDM) is bias correction method that adjusts the satellite validation dataset based
    on the cumulative distribution functions (CDFs) of the calibration datasets. The method involves interpolating the
    satellite calibration CDF to the mooring calibration CDF and then calculating the quantile differences. After
    that a polynomial interpolation is applied to these differences, which returns correction coefficients. These
    factors are then applied to the satellite validation dataset.

    Parameters:
        df_mooring_cal: DataFrame containing the mooring calibration data.
        df_sat_cal: DataFrame containing the satellite calibration data.
        df_mooring_val: DataFrame containing the mooring validation data.
        df_sat_val: DataFrame containing the satellite validation data.
        variable: Variable name to calibrate.
    Returns:
        A new DataFrame containing the calibrated satellite validation data for the specified variable.
    """

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
    df_sat_val_final = df_sat_val.copy()

    return df_sat_val_final


def qm_correction(df_sat_cal: pd.DataFrame, df_mooring_cal: pd.DataFrame, df_sat_val: pd.DataFrame, df_mooring_val: pd.DataFrame,variable: str) -> pd.DataFrame:
    """Quantile Mapping (QM) is bias correction method that adjusts the satellite validation dataset based on the quantiles of the calibration datasets. The
    method involves dividing the calibration datasets into quantiles, calculating the CDFs for each quantile and then interpolating the satellite calibration CDF to the mooring calibration CDF for each quantile. After the interpolation, quantile differences are calculated, and fitted with a polynomial of 2nd degree, which return the correction factors and applying themto the satellite validation dataset for each quantile.

    This method allows for a more detailed correction that accounts for differences in the distribution of the data across different quantiles, potentially improving the accuracy of the bias correction, especially when the relationship between the satellite and mooring data is not linear or when there are significant differences in the distribution of the data across different quantiles.

    Parameters:
        df_mooring_cal: DataFrame containing the mooring calibration data.
        df_sat_cal: DataFrame containing the satellite calibration data.
        df_mooring_val: DataFrame containing the mooring validation data.
        df_sat_val: DataFrame containing the satellite validation data.
        variable: Variable name to calibrate.
    Returns:
        A new DataFrame containing the calibrated satellite validation data for the specified variable.
    """

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

