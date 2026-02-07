import pandas as pd
import numpy as np



def calc_haversine(lat_mooring: float,lon_mooring: float,lat_sat: np.ndarray,lon_sat: np.ndarray)-> np.ndarray:

    R = 6371.0   # Earth radius

    lat1, lon1 = np.radians(lat_mooring), np.radians(lon_mooring)
    lat2, lon2 = np.radians(lat_sat), np.radians(lon_sat)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c  # Distance in km
def spatial_cross(df_s: pd.DataFrame, df_m: pd.DataFrame, cross_radius: float) -> pd.DataFrame:



    df_mooring = df_m
    df_sat = df_s

    lat_sat = df_sat['latitude'].values
    lon_sat = df_sat['longitude'].values

    lat_mooring = df_mooring['latitude'].iloc[0]
    lon_mooring = df_mooring['longitude'].iloc[0]

    distance = calc_haversine(lat_mooring,lon_mooring, lat_sat, lon_sat)
    dist_filtered_idx = np.where(distance <= cross_radius)

    if (len(dist_filtered_idx[0]) == 0):
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
        df_sat['mooring_ref'] = np.full(len(dist_filtered_idx[0]),df_mooring['platfID'].iloc[0])
        df_sat['distance'] = distance[dist_filtered_idx[0]]

        #print(f"Satellite has {(df_sat['N_cross'].iloc[-1]).astype(int)} cross-over points with {df_mooring['platfID'].iloc[0]}!")

        return df_sat.reset_index(drop=True)

def spatial_coLoc_min_dist(df_sat_after_cross:pd.DataFrame) -> pd.DataFrame:

    ncross_unique = np.unique(df_sat_after_cross['N_cross']).astype(int)
    df_sat_after_cross['N_cross'] = (df_sat_after_cross['N_cross']).astype(int)
    min_dist_idx = df_sat_after_cross.groupby('N_cross')['distance'].idxmin()
    df_sat = df_sat_after_cross.loc[min_dist_idx]

    return df_sat.reset_index(drop=True)


def spatial_matching(df_sat: pd.DataFrame, df_mooring: pd.DataFrame, cross_radius: float) -> pd.DataFrame:

    df_sat_stp1 = spatial_cross(df_sat,df_mooring,cross_radius)
    if df_sat_stp1.empty:
        return pd.DataFrame()
    return spatial_coLoc_min_dist(df_sat_stp1)







