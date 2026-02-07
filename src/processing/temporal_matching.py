import pandas as pd
import numpy as np

def temporal_cross(df_sat_after_coLoc: pd.DataFrame,df_mooring,cross_time_val: int|float,cross_time_unit: str) -> pd.DataFrame:

    t_mooring = df_mooring['time'].to_numpy(dtype='datetime64[ns]')
    t_sat = df_sat_after_coLoc['time'].to_numpy(dtype='datetime64[ns]')

    diff_time = np.abs(t_sat[:, None] - t_mooring[None, :])

    tol_time = np.timedelta64(cross_time_val,cross_time_unit)
    rows, col = np.where(diff_time <= tol_time)
    diff_time_filtered = diff_time[rows,col].astype('timedelta64[m]')

    df_mooring = df_mooring.iloc[col].reset_index(drop=True)
    df_mooring['N_cross'] = rows + 1
    df_mooring['N_cross'] = df_mooring['N_cross'].astype(int)
    df_mooring['deltaTime [min]'] = diff_time_filtered

    df_mooring_final = df_mooring.copy()

    return df_mooring_final

def temporal_coLoc_closestMeasure(df_mooring_after_temp_cross: pd.DataFrame) -> pd.DataFrame:

    ncross_unique = np.unique(df_mooring_after_temp_cross['N_cross']).astype(int)
    df_mooring_after_temp_cross['N_cross'] = (df_mooring_after_temp_cross['N_cross']).astype(int)
    min_deltaT_idx = df_mooring_after_temp_cross.groupby('N_cross')['deltaTime [min]'].idxmin()
    df_mooring = df_mooring_after_temp_cross.loc[min_deltaT_idx].reset_index(drop=True)

    return df_mooring

def temporal_matching(df_sat: pd.DataFrame,df_mooring: pd.DataFrame,cross_time_val: int|float,cross_time_unit: str) -> pd.DataFrame:

    df_mooring_stp1 = temporal_cross(df_sat,df_mooring,cross_time_val,cross_time_unit)
    if df_mooring_stp1.empty:
        return pd.DataFrame()
    return temporal_coLoc_closestMeasure(df_mooring_stp1)