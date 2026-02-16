import pandas as pd
import numpy as np


def temporal_cross(df_sat_after_co_loc: pd.DataFrame, df_mooring, cross_time_val: int | float, cross_time_unit: str)\
        -> pd.DataFrame:
    """
    For each cross-over point identified in the spatial matching step, this function performs a temporal cross-matching
    between the satellite and mooring data.

    It calculates the absolute time difference between each satellite data point and each mooring data point, and
    identifies pairs of points where the time difference is within a specified tolerance (defined by `cross_time_val`
    and `cross_time_unit`). The function then creates a new DataFrame that includes only the mooring data points that
    have at least one corresponding satellite point within the temporal matching window, along with the number of
    cross-over points (`N_cross`) and the time difference in minutes for each matched pair.

    This step is crucial for ensuring that the satellite and data are not only spatially close but also temporally
    aligned for accurate comparison and analysis.

    Args:
        df_sat_after_co_loc: DataFrame containing satellite data points that have been identified as cross-over points
        based on spatial proximity to a mooring point.
        df_mooring: DataFrame containing mooring data points that are candidates for temporal matching with the
        satellite data.
        cross_time_val: Cross-matching time tolerance value, which defines the maximum allowed time difference between
        satellite and mooring data points for them to be considered a match.
        cross_time_unit: Cross-matching time tolerance unit, which specifies the unit of time for the `cross_time_val`
        (e.g., 's' for seconds, 'm' for minutes, 'h' for hours). This unit will be used to convert the time difference
        into a consistent format for comparison.

    Returns:
        A DataFrame containing the mooring data points that have been temporally matched with the satellite data points,
        along with the number of cross-over points (`N_cross`) and the time difference in minutes for each matched pair.
    """
    t_mooring = df_mooring['time'].to_numpy(dtype='datetime64[ns]')
    t_sat = df_sat_after_co_loc['time'].to_numpy(dtype='datetime64[ns]')

    diff_time = np.abs(t_sat[:, None] - t_mooring[None, :])

    tol_time = np.timedelta64(cross_time_val, cross_time_unit)
    rows, col = np.where(diff_time <= tol_time)
    diff_time_filtered = diff_time[rows, col].astype('timedelta64[m]')

    df_mooring = df_mooring.iloc[col].reset_index(drop=True)
    df_mooring['N_cross'] = rows + 1
    df_mooring['N_cross'] = df_mooring['N_cross'].astype(int)
    df_mooring['deltaTime [min]'] = diff_time_filtered

    df_mooring_final = df_mooring.copy()

    return df_mooring_final


def temporal_co_loc_closest_measure(df_mooring_after_temp_cross: pd.DataFrame) -> pd.DataFrame:
    """
    For each cross-over point identified in the temporal matching step, this function co-locates the mooring data point
    that is closest in time to the satellite data point.

    It groups the mooring data by cross-over point number (`N_cross`) and selects the row with the minimum time
    difference for each group, resulting in a DataFrame that contains only the closest mooring point for each
    cross-over point.

    This ensures that for each temporally matched cross-over point, only the most relevant mooring data point (the one
    closest in time to the satellite point) is retained for further analysis.

    Args:
        df_mooring_after_temp_cross: dataframe containing mooring data points that have been identified as cross-over po
        ints based on temporal proximity to satellite data points.

    Returns:
        A DataFrame containing only the closest mooring data point for each cross-over point, based on the minimum time
        difference to the satellite point.
    """

    n_cross_unique = np.unique(df_mooring_after_temp_cross['N_cross']).astype(int)
    df_mooring_after_temp_cross['N_cross'] = (df_mooring_after_temp_cross['N_cross']).astype(int)
    min_delta_t_idx = df_mooring_after_temp_cross.groupby('N_cross')['deltaTime [min]'].idxmin()
    df_mooring = df_mooring_after_temp_cross.loc[min_delta_t_idx].reset_index(drop=True)

    return df_mooring


def temporal_matching(df_sat: pd.DataFrame, df_mooring: pd.DataFrame, cross_time_val: int | float,
                      cross_time_unit: str):

    df_mooring_stp1 = temporal_cross(df_sat, df_mooring, cross_time_val, cross_time_unit)
    if df_mooring_stp1.empty:
        return pd.DataFrame()
    return temporal_co_loc_closest_measure(df_mooring_stp1)
