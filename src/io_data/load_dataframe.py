import pandas as pd
import xarray as xr
import numpy as np


def load_sat_data_csv(filepath_sat_csv, var_name):

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


def load_moor_data_nc(filepath_file_nc, var_name, deph_val):

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
            # print("No dataframe has been extracted!")
            return None
