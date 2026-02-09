import pandas as pd
import numpy as np


def calc_haversine(lat_mooring: float, lon_mooring: float, lat_sat: np.ndarray, lon_sat: np.ndarray) -> np.ndarray:
    """
        Calculates the Haversine distance between a mooring point and multiple satellite points.
    Args:
        lat_mooring: Latitude of the mooring point in degrees.
        lon_mooring: Longitude of the mooring point in degrees.
        lat_sat: Latitudes of the satellite points as a NumPy array in degrees.
        lon_sat: Longitudes of the satellite points as a NumPy array in degrees.

    Returns:
        A NumPy array containing the Haversine distances in kilometers between the mooring point and each
        satellite point.
    """

    r = 6371.0   # Earth radius

    lat1, lon1 = np.radians(lat_mooring), np.radians(lon_mooring)
    lat2, lon2 = np.radians(lat_sat), np.radians(lon_sat)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return r * c  # Distance in km


def spatial_cross(df_s: pd.DataFrame, df_m: pd.DataFrame, cross_radius: float) -> pd.DataFrame:
    """
    Matching between satellite and mooring data based on spatial proximity. This function identifies satellite
    data points that are within a specified radius of a mooring point and assigns cross-over point numbers to the
    satellite data based on their proximity to the mooring point. It also calculates the distance projected on Earth,
    using Haversine function, from each satellite point to the mooring point and adds this information to the
    resulting DataFrame.

    Args:
        df_s: DataFrame containing satellite data.
        df_m: DataFrame containing mooring data.
        cross_radius: Cross-matching radius in kilometers. Satellite points within this radius from the
        mooring point will be considered as cross-over points.
    Returns:
        Dataframe with satellite points that are within the specified `cross_radius` of the mooring point, along with
        additional columns. If no points are within `cross_radius`, an empty DataFrame is returned.
        Key columns:

            - `latitude` (float) — satellite point latitude in degrees.
            - `longitude` (float) — satellite point longitude in degrees.
            - `N_cross` (int) — sequential cross\-over number
            - `mooring_ref` (same type as `df_mooring['platfID']`) — identifier of the associated platform/mooring.
            - `distance` (float, km) — Haversine distance between satellite point and mooring.
    """
    df_mooring = df_m
    df_sat = df_s

    lat_sat = df_sat['latitude'].values
    lon_sat = df_sat['longitude'].values

    lat_mooring = df_mooring['latitude'].iloc[0]
    lon_mooring = df_mooring['longitude'].iloc[0]

    distance = calc_haversine(lat_mooring, lon_mooring, lat_sat, lon_sat)
    dist_filtered_idx = np.where(distance <= cross_radius)

    if len(dist_filtered_idx[0]) == 0:
        return pd.DataFrame()

    else:
        array_diff_idx = np.diff(dist_filtered_idx[0])
        array_n_cross = np.zeros(len(dist_filtered_idx[0]))
        internal_cross = (array_diff_idx > 1).astype(int)
        array_n_cross[0] = 1  # initialize starting cross point
        array_n_cross[1:] = internal_cross
        i_th_cross = np.cumsum(array_n_cross)

        df_sat = df_sat.iloc[dist_filtered_idx[0]].copy()
        df_sat['N_cross'] = i_th_cross
        df_sat['mooring_ref'] = np.full(len(dist_filtered_idx[0]), df_mooring['platfID'].iloc[0])
        df_sat['distance'] = distance[dist_filtered_idx[0]]

        # print(f"Satellite has {(df_sat['N_cross'].iloc[-1]).astype(int)} cross-over points with
        # {df_mooring['platfID'].iloc[0]}!")

        return df_sat.reset_index(drop=True)


def spatial_co_loc_min_dist(df_sat_after_cross: pd.DataFrame) -> pd.DataFrame:
    """
    For each cross-over point withing the spatial matching radius, this function co-locates the satellite data point
    that is closest to the mooring point based on the calculated Haversine distance. It groups the satellite data by
    cross-over point number (`N_cross`) and selects the row with the minimum distance for each group, resulting in a
    DataFrame that contains only the closest satellite point for each cross-over point. This ensures that for each
    spatially matched cross-over point, only the most relevant satellite data point (the one closest to the mooring) is
    retained for further analysis.

    Args:
        df_sat_after_cross: DataFrame containing satellite data points that have been identified as cross-over points
        based on spatial proximity to a mooring point.

    Returns:
        A DataFrame containing only the closest satellite data point for each cross-over point, based on the minimum
        Haversine distance to the mooring point
    """

    ncross_unique = np.unique(df_sat_after_cross['N_cross']).astype(int)
    df_sat_after_cross['N_cross'] = (df_sat_after_cross['N_cross']).astype(int)
    min_dist_idx = df_sat_after_cross.groupby('N_cross')['distance'].idxmin()
    df_sat = df_sat_after_cross.loc[min_dist_idx]

    return df_sat.reset_index(drop=True)


def spatial_matching(df_sat: pd.DataFrame, df_mooring: pd.DataFrame, cross_radius: float) -> pd.DataFrame:

    df_sat_stp1 = spatial_cross(df_sat, df_mooring, cross_radius)
    if df_sat_stp1.empty:
        return pd.DataFrame()
    return spatial_co_loc_min_dist(df_sat_stp1)
