import pandas as pd
import numpy as np
import data_extraction_md as data_ex


def calc_haversine(lat_m,lon_m,lat_sat,lon_sat):

    R = 6371.0   # Earth radius

    lat1, lon1 = np.radians(lat_m), np.radians(lon_m)
    lat2, lon2 = np.radians(lat_sat), np.radians(lon_sat)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c  # Distance in km
def spatial_cross(filepath_sat_csv,filepath_mooring_csv,cross_radius,var_name,dist_tol):

    dist_lim = dist_tol + 0.1

    df_m = pd.read_csv(filepath_mooring_csv)
    df_sat = data_ex.extr_sat_data_from_csv(filepath_sat_csv,var_name)

    lat_sat = df_sat['latitude'].values
    lon_sat = df_sat['longitude'].values

    lat_m = df_m['latitude'][0]
    lon_m = df_m['longitude'][0]

    distance = calc_haversine(lat_m,lon_m, lat_sat, lon_sat)
    idx_dist_filtered = np.where(distance <= cross_radius)

    if (len(idx_dist_filtered[0]) == 0):
        return pd.DataFrame()

    else:
        diff_idx_filtered = np.diff(idx_dist_filtered[0])
        array_n_cross = np.zeros(len(idx_dist_filtered[0]))
        internal_cross = (diff_idx_filtered > 1).astype(int)
        array_n_cross[0] = 1  # initialize starting cross point
        array_n_cross[1:] = internal_cross
        i_th_cross = np.cumsum(array_n_cross)

        df_sat = df_sat.iloc[idx_dist_filtered[0]].copy()
        df_sat['N_cross'] = i_th_cross
        df_sat['ref_platfID'] = np.full(len(idx_dist_filtered[0],df_m['platfID'][0]))

        return df_sat




