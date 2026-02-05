import pandas as pd
import numpy as np
import export_data_to_csv as data_ex


def calc_haversine(lat_mooring,lon_mooring,lat_sat,lon_sat):

    R = 6371.0   # Earth radius

    lat1, lon1 = np.radians(lat_m), np.radians(lon_m)
    lat2, lon2 = np.radians(lat_sat), np.radians(lon_sat)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c  # Distance in km
def spatial_cross(filepath_sat_csv,filepath_mooring_csv,cross_radius,var_name):



    df_mooring = pd.read_csv(filepath_mooring_csv)
    df_sat = data_ex.extr_sat_data_from_csv(filepath_sat_csv,var_name)

    lat_sat = df_sat['latitude'].values
    lon_sat = df_sat['longitude'].values

    lat_mooring = df_mooring['latitude'][0]
    lon_mooring = df_mooring['longitude'][0]

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
        df_sat['ref_platfID'] = np.full(len(dist_filtered_idx[0]),df_mooring['platfID'][0])

        return df_sat




