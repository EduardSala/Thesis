import xarray as xr
import pandas as pd
import numpy as np
import os

def extr_insitu_data_from_nc(filepath_file_nc,var_name,field,deph_val,tol=0.1):
    dataset_nc = xr.open_dataset(filepath_file_nc)

    if var_name in dataset_nc.data_vars:
        cond = (dataset_nc['DEPH'].values >= deph_val - tol) & (dataset_nc['DEPH'].values <= deph_val + tol)
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
                    'time': pd.to_datetime(time),
                 'latitude': np.full(time.size,latitude),
                 'longitude': np.full(time.size,longitude),
                 var_name: variable, 'platfID': np.full(time.size,mooring_name)
                 }
            ).fillna(-1)

            print("In-situ data has been extracted!")
            return dataframe_mooring
        else:
            print("No dataframe has been extracted!")
            return None

def exp_dataframe_to_file(dataframe_insitu,field,dir_output):
    if (len(dataframe_insitu) > 0):
        mooring_name = dataframe_insitu['platfID'][0]
        dataframe_insitu.to_csv(dir_output + f"/" + mooring_name + f"_{field}" + ".csv")
    else:
        print("No file has been generated!")

# -----------------------------------------------------------

def extr_sat_data_from_csv(filepath_sat_csv,var_name):

    dataframe_sat_in = pd.read_csv(filepath_sat_csv,skiprows=5)
    df_sat = dataframe_sat_in[dataframe_sat_in['parameter']==var_name]
    df_sat_out = pd.DataFrame(
        {
            'time': pd.to_datetime(df_sat['time'].values),

            'latitude': df_sat['latitude'].values,

            'longitude': df_sat['longitude'].values,

            var_name: df_sat['value'].values,

            'valueQC': df_sat['valueQc'].values
        }
    )
    print("Satellite data has been extracted!")
    return df_sat_out










