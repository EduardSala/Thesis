import numpy as np
import pandas as pd
from calibration import calibration_methods as cal_mthd


def linear_cal(df_sat: pd.DataFrame, df_mooring: pd.DataFrame, variable: str):

    df_mooring_cal, df_sat_cal, df_mooring_val, df_sat_val = cal_mthd.calib_df_first_ten_days(df_mooring, df_sat)

    x = df_mooring_cal[variable]
    y = df_sat_cal[variable]

    b, a = np.polyfit(y, x, deg=1)
    new_y_validation = a + b*df_sat_val[variable].values
    new_df_sat_validation = df_sat_val.copy()
    new_df_sat_validation[variable] = new_y_validation

    return new_df_sat_validation
