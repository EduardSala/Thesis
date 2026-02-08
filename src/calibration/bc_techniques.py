import numpy as np
import pandas as pd
from calibration import calibration_methods as cal_mthd


def linear_cal(df_sat: pd.DataFrame, df_mooring: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Calibrates a satellite variable using the first 10 days of reference mooring data, then applies the linear
    transformation to the calibration dataset. The coefficients b and a are applied to the satellite validation dataset.
    Args:
        df_sat (pd.DataFrame): DataFrame containing satellite data.
        df_mooring (pd.DataFrame): DataFrame containing mooring data.
        variable (str): The name of the variable to calibrate.
    Returns:
        new_df_sat_validation (pd.DataFrame): A new DataFrame containing the calibrated satellite validation data for
         the specified variable.
    """

    df_mooring_cal, df_sat_cal, df_mooring_val, df_sat_val = cal_mthd.calib_df_first_ten_days(df_mooring, df_sat)

    x = df_mooring_cal[variable]
    y = df_sat_cal[variable]

    b, a = np.polyfit(y, x, deg=1)
    new_y_validation = a + b*df_sat_val[variable].values
    new_df_sat_validation = df_sat_val.copy()
    new_df_sat_validation[variable] = new_y_validation

    return new_df_sat_validation
