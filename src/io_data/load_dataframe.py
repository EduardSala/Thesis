import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path


def load_sat_data_csv(filepath_sat_csv: str | Path, var_name: str) -> pd.DataFrame:
    """
    Load satellite data from a CSV file, filter it by the specified variable name,
    and return a DataFrame with relevant columns.
    Args:
        filepath_sat_csv: Path to the satellite CSV file as a string or Path object.
        var_name: Name of the variable to filter the satellite data by (e.g., 'VAVH' | 'WSPD').

    Returns:
        (pd.DataFrame): A DataFrame containing the filtered satellite data with columns:

            - `platfID`: Platform ID (same for all rows, taken from the first entry).
            - `time`: Time of the observation, converted to datetime format.
            - `latitude`: Latitude of the observation.
            - `longitude`: Longitude of the observation.
            - var_name: The values of the specified variable.
            - `valueQC`: Quality control values for the specified variable.
    """

    filepath_sat_csv = Path(filepath_sat_csv)
    dataframe_sat_in = pd.read_csv(filepath_sat_csv, skiprows=5)
    df_sat = dataframe_sat_in[dataframe_sat_in['parameter'] == var_name]
    df_sat_out = pd.DataFrame(
        {
            'platfID': np.full(len(df_sat), str(df_sat['platformId'].iloc[0])),

            'time': np.array(df_sat['time'].values, dtype='datetime64[ns]'),

            'latitude': df_sat['latitude'].values,

            'longitude': df_sat['longitude'].values,

            var_name: df_sat['value'].values,

            'valueQC': df_sat['valueQc'].values
        }
    )
    # print("Satellite data has been loaded!")
    return df_sat_out


def load_moor_data_nc(filepath_file_nc: str | Path, var_name: str, deph_val: int) -> pd.DataFrame | None:
    """
    Load mooring data from a NetCDF file, filter it by the specified variable name and depth value,
    and return a DataFrame with relevant columns. If the variable name is not found
    or no matching depth is found, return None.
    Args:
        filepath_file_nc: String or Path object representing the path to the NetCDF file containing mooring data.
        var_name: Variable name to filter the mooring data by (e.g., 'VAVH' | 'WSPD').
        deph_val: Depth value to filter the mooring data by. Only rows with a depth value equal to deph_val will be included in the output DataFrame. `0` for VAVH and `-10` for WSPD.
    Returns:
        (pd.DataFrame | None): A DataFrame containing the filtered mooring data with columns:

            - `platfID`: Platform ID (same for all rows, taken from the 'STATION' variable in the NetCDF file).
            - `time`: Time of the observation, converted to datetime format.
            - `latitude`: Latitude of the observation (same for all rows, taken from the 'LATITUDE' variable in the NetCDF file).
            - `longitude`: Longitude of the observation (same for all rows, taken from the 'LONGITUDE' variable in the NetCDF file).
            - var_name: The values of the specified variable at the specified depth.

        If the variable name is not found in the NetCDF file or no matching depth is found, returns None.
    """
    with xr.open_dataset(filepath_file_nc) as dataset_nc:
        if var_name in dataset_nc.data_vars:
            cond = (dataset_nc['DEPH'].values >= deph_val) & (dataset_nc['DEPH'].values <= deph_val)
            idx = np.where(cond)[0]
            # ----------------------------------------------
            if idx.size > 0:
                mooring_name = str(dataset_nc['STATION'].values).split(sep="'")[1]
                variable = np.array(dataset_nc[var_name][:, idx].values).flatten()
                time = dataset_nc['TIME'].values
                latitude = dataset_nc['LATITUDE'].values
                longitude = dataset_nc['LONGITUDE'].values
                # --------------------------------------
                dataframe_mooring = pd.DataFrame(
                    {
                        'platfID': np.full(len(time), mooring_name),
                        'time': np.array(time, dtype='datetime64[ns]'),
                        'latitude': np.full(len(time), latitude),
                        'longitude': np.full(len(time), longitude),
                        var_name: variable
                    }
                )
                # print("In-situ data has been extracted!")
                return dataframe_mooring
            else:
                # No matching depth found for the requested deph_val
                return None
        else:
            # print("No dataframe has been extracted!")
            return None
