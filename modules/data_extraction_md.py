import xarray as xr
import pandas as pd
import numpy as np
import os

def extr_insitu_data(filepath_file_nc,var_name,field,deph_val,tol=0.1):
    dataset_nc = xr.open_dataset(filepath_file_nc)

    if var_name in dataset_nc.data_vars:
        cond = (dataset_nc['DEPH'].values >= deph_val - tol) & \
               (dataset_nc['DEPH'].values <= deph_val + tol)
        idx = np.where(cond)[0]

        if idx.size > 0:
            mooring_name = str(dataset_nc['STATION'].values).split(sep="'")[1]
            variable = np.array(dataset_nc[var_name][:, idx].values).flatten()
            time = np.array(dataset_nc['TIME'][idx].values).flatten()
            latitude = np.array(dataset_nc['LATITUDE'].values).flatten()
            longitude = np.array(dataset_nc['LONGITUDE'].values).flatten()

            dataframe_mooring = pd.DataFrame(
                {'time': time, 'latitude': latitude, 'longitude': longitude,
                 variable_name: variable}).fillna(-1)
            print("Extraction done!")
            return dataframe_mooring,mooring_name,field
        else:
            print("Extraction done!")
            return None



def export_dataframe_to_file(dataframe_insitu,mooring_name,field,dir_output):
    dataframe_insitu.to_csv(dir_output + f"/" + mooring_name + f"_{field}" + ".csv")










