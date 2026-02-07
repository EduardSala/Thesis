import pandas as pd
import numpy as np
from io_data import load_dataframe as load_data
from processing import spatial_matching as spatial_match
from processing import temporal_matching as temp_match
from pathlib import Path
from tqdm import tqdm


def align_dataframes(df_sat, df_mooring):

    common_cross = np.intersect1d(list(df_mooring['N_cross']), list(df_sat['N_cross']))

    df_mooring_final = df_mooring[df_mooring['N_cross'].isin(common_cross)].reset_index(drop=True)
    df_sat_final = df_sat[df_sat['N_cross'].isin(common_cross)].reset_index(drop=True)

    return df_sat_final, df_mooring_final


def spatio_temp_matching(params):

    dir_path_sat = Path(params['dir_paths']['dir_input_sat_csv'])
    dir_path_mooring = Path(params['dir_paths']['dir_input_mooring_nc'])
    fp_single_file_sat = Path(params['dir_paths']['path_sat_singleFile'])

    mooring_list = []
    sat_list = []

    list_path_mooring = list(dir_path_mooring.glob("*.nc"))

    for fp in tqdm(list_path_mooring, desc="Spatio-temporal matching progress"):

        df_sat = load_data.load_sat_data_csv(fp_single_file_sat, params['extraction']['var_name'])
        df_mooring = load_data.load_moor_data_nc(fp, params['extraction']['var_name'],
                                                 params['extraction']['deph_val'])
        df_sat_spatial_cross = spatial_match.spatial_matching(
            df_sat, df_mooring, params['crossMatching']['cross_radius'])

        if df_sat_spatial_cross.empty:
            # print(f"Satellite has no cross-over points with {df_mooring['platfID'].iloc[0]}!")
            continue
        # <------------------------------------------------------------------------------------------------------------>
        df_mooring_temporal_cross = temp_match.temporal_matching(
            df_sat_spatial_cross, df_mooring, params['crossMatching']['cross_time_val'],
            params['crossMatching']['cross_time_unit'])

        if df_mooring_temporal_cross.empty:
            # print(f"{df_mooring['platfID'].iloc[0]} has no data matching time-wise with satellite!")
            continue
        # <------------------------------------------------------------------------------------------------------------>
        df_sat_final, df_mooring_final = align_dataframes(df_sat_spatial_cross, df_mooring_temporal_cross)
        mask_nan = np.isnan(df_mooring_final[params['crossMatching']['var_name']])
        idx_nan = (np.where(mask_nan)[0])
        df_mooring_final = df_mooring_final.drop(idx_nan)
        df_sat_final = df_sat_final.drop(idx_nan)

        mooring_list.append(df_mooring_final)
        sat_list.append(df_sat_final)

    return pd.concat(sat_list, ignore_index=True), pd.concat(mooring_list, ignore_index=True)
