import pandas as pd
import numpy as np
import yaml

from io_data import load_dataframe as load_data
from processing import spatial_matching as spatial_match
from processing import temporal_matching as temp_match
from pathlib import Path
from tqdm import tqdm


def align_dataframes(df_sat: pd.DataFrame, df_mooring: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns the satellite and mooring dataframes based on their common cross-over points.
    Args:
        df_sat: DataFrame containing satellite data with a column 'N_cross' representing cross-over points.
        df_mooring: DataFrame containing mooring data with a column 'N_cross' representing cross-over points.

    Returns:
        A tuple containing the aligned satellite and mooring dataframes, filtered to include only rows with common
        cross-over points.
    """
    common_cross = np.intersect1d(list(df_mooring['N_cross']), list(df_sat['N_cross']))

    df_mooring_final = df_mooring[df_mooring['N_cross'].isin(common_cross)].reset_index(drop=True)
    df_sat_final = df_sat[df_sat['N_cross'].isin(common_cross)].reset_index(drop=True)

    return df_sat_final, df_mooring_final


def spatio_temp_matching(params: yaml.YAMLObject) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
        Performs spatio-temporal matching between satellite and mooring data based on the provided
        configuration parameters.
    Args:
        params: A dictionary containing configuration parameters for the spatio-temporal matching process, including
                directory paths for input satellite CSV files and mooring NetCDF files, variable names,
                cross-matching criteria (spatial radius and temporal window), and other relevant settings.

    Returns:
        A tuple containing two DataFrames: the first with the matched satellite data and the second with the
        corresponding mooring data. If no matches are found, both DataFrames will be empty.
    """
    cfg_params0 = params['data_extraction']
    cfg_params1 = params['spatio_temp_matching']

    cfg_dir_path_sat = cfg_params0['dir_paths']['dir_input_sat_csv']
    cfg_dir_path_mooring = cfg_params0['dir_paths']['dir_input_mooring_nc']

    cfg_var_name = cfg_params1['variable']['var_name']
    cfg_depth_val = cfg_params0['variable']['depth_val']
    cfg_cross_radius = cfg_params1['variable']['cross_radius']
    cfg_cross_time_val = cfg_params1['variable']['cross_time_val']
    cfg_cross_time_unit = cfg_params1['variable']['cross_time_unit']

    dir_path_sat = Path(cfg_dir_path_sat)
    dir_path_mooring = Path(cfg_dir_path_mooring)

    mooring_list = []
    sat_list = []

    list_path_mooring = list(dir_path_mooring.glob("*.nc"))
    print(list_path_mooring)
    for fp in tqdm(list_path_mooring, desc="Spatio-temporal matching progress"):

        df_sat = load_data.load_sat_data_csv(dir_path_sat, cfg_var_name)
        df_mooring = load_data.load_moor_data_nc(fp, cfg_var_name, cfg_depth_val)
        df_sat_spatial_cross = spatial_match.spatial_matching(
            df_sat, df_mooring, cfg_cross_radius)

        if df_sat_spatial_cross.empty:
            # print(f"Satellite has no cross-over points with {df_mooring['platfID'].iloc[0]}!")
            continue
        # <------------------------------------------------------------------------------------------------------------>
        df_mooring_temporal_cross = temp_match.temporal_matching(
            df_sat_spatial_cross, df_mooring, cfg_cross_time_val, cfg_cross_time_unit)

        if df_mooring_temporal_cross.empty:
            # print(f"{df_mooring['platfID'].iloc[0]} has no data matching time-wise with satellite!")
            continue
        # <------------------------------------------------------------------------------------------------------------>
        df_sat_final, df_mooring_final = align_dataframes(df_sat_spatial_cross, df_mooring_temporal_cross)
        mask_nan = np.isnan(df_mooring_final[cfg_var_name])
        idx_nan = (np.where(mask_nan)[0])
        df_mooring_final = df_mooring_final.drop(idx_nan)
        df_sat_final = df_sat_final.drop(idx_nan)

        mooring_list.append(df_mooring_final)
        sat_list.append(df_sat_final)

    if not sat_list or not mooring_list:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(sat_list, ignore_index=True), pd.concat(mooring_list, ignore_index=True)
