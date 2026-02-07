import pandas as pd


def calib_df_first_ten_days(df_mooring: pd.DataFrame, df_sat: pd.DataFrame):

    df_mooring['day'] = df_mooring['time'].astype('datetime64[ns]').dt.day
    df_mooring_cal = df_mooring.loc[df_mooring['day'] <= 10]
    df_sat_cal = df_sat.loc[df_mooring['day'] <= 10]
    df_mooring_val = df_mooring.loc[df_mooring['day'] > 10]
    df_sat_val = df_sat.loc[df_mooring['day'] > 10]

    return df_mooring_cal, df_sat_cal, df_mooring_val, df_sat_val


def calib_df_last_ten_days(df_mooring: pd.DataFrame, df_sat: pd.DataFrame):

    df_mooring['day'] = df_mooring['time'].astype('datetime64[ns]').dt.day
    df_mooring_cal = df_mooring.loc[df_mooring['day'] >= 20]
    df_sat_cal = df_sat.loc[df_mooring['day'] >= 20]
    df_mooring_val = df_mooring.loc[df_mooring['day'] < 20]
    df_sat_val = df_sat.loc[df_mooring['day'] < 20]

    return df_mooring_cal, df_sat_cal, df_mooring_val, df_sat_val
