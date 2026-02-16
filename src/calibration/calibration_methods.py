import pandas as pd


def calib_df_first_ten_days(df_mooring: pd.DataFrame, df_sat: pd.DataFrame) \
        -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calibrates the satellite data using the first ten days of mooring data as the calibration set and the remaining
    days as the validation set. The function creates boolean masks based on the day of the month extracted from the
    `time` column in `df_mooring`, and applies these masks to both `df_mooring` and `df_sat` to create the respective
    calibration and validation DataFrames.

    Parameters:
        df_mooring (pd.DataFrame): DataFrame containing mooring data.
        df_sat (pd.DataFrame): DataFrame containing satellite data.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the following DataFrames:
            - df_mooring_cal:
            - df_sat_cal:
            - df_mooring_val:
            - df_sat_val:
    """

    # Work on a copy to avoid mutating the original df_mooring in-place
    df_mooring_copy = df_mooring.copy()

    # Compute day from mooring time
    day = pd.to_datetime(df_mooring_copy['time']).dt.day
    df_mooring_copy['day'] = day

    # Create boolean masks for the first ten days
    mask_cal = day <= 10
    mask_val = day > 10

    # Use the masks directly for df_mooring_copy (same index)
    df_mooring_cal = df_mooring_copy.loc[mask_cal]
    df_mooring_val = df_mooring_copy.loc[mask_val]

    # Use positional boolean indexing for df_sat to avoid index alignment issues
    mask_cal_array = mask_cal.to_numpy()
    mask_val_array = mask_val.to_numpy()
    df_sat_cal = df_sat.iloc[mask_cal_array]
    df_sat_val = df_sat.iloc[mask_val_array]

    return df_mooring_cal, df_sat_cal, df_mooring_val, df_sat_val


def calib_df_last_ten_days(df_mooring: pd.DataFrame, df_sat: pd.DataFrame) \
        -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
     Calibrates the satellite data using the last ten days of mooring data as the calibration set and the remaining
     days as the validation set. The function creates boolean masks based on the day of the month extracted from the
     `time` column in `df_mooring`, and applies these masks to both `df_mooring` and `df_sat` to create the respective
     calibration and validation DataFrames.

     Parameters:
         df_mooring (pd.DataFrame): DataFrame containing mooring data.
         df_sat (pd.DataFrame): DataFrame containing satellite data.
     Returns:
         tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the following DataFrames:
             - df_mooring_cal:
             - df_sat_cal:
             - df_mooring_val:
             - df_sat_val:
     """

    # Work on a copy to avoid mutating the original df_mooring in-place
    df_mooring_copy = df_mooring.copy()

    # Compute day from mooring time
    day = pd.to_datetime(df_mooring_copy['time']).dt.day
    df_mooring_copy['day'] = day

    # Create boolean masks for the last ten days (days >= 20)
    mask_cal = day >= 20
    mask_val = day < 20

    # Use the masks directly for df_mooring_copy (same index)
    df_mooring_cal = df_mooring_copy.loc[mask_cal]
    df_mooring_val = df_mooring_copy.loc[mask_val]

    # Use positional boolean indexing for df_sat to avoid index alignment issues
    mask_cal_array = mask_cal.to_numpy()
    mask_val_array = mask_val.to_numpy()
    df_sat_cal = df_sat.iloc[mask_cal_array]
    df_sat_val = df_sat.iloc[mask_val_array]
    return df_mooring_cal, df_sat_cal, df_mooring_val, df_sat_val
