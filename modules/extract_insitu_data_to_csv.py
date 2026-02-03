import xarray as xr
import pandas as pd
import numpy as np
from modules import Module_all_functions as md
import os
filepath_files = md.pass_filepath("../datasets/moorings-nc")

# -----------------------------------------------------------------------------
#variable_name = 'VAVH'
#application = 'wave'

variable_name = 'WSPD'
application = 'wind'
# -----------------------------------------------------------------------------

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
            return [dataframe_mooring,mooring_name,field]
            # dataframe_mooring.to_csv(r"../datasets/mooring-csv/" + mooring_name + "_wave" + ".csv")
            dataframe_mooring.to_csv(folderDir_csv + f"/" + mooring_name + f"_{field}" + ".csv")
        else:
            return None



def export_dataframe_to_file(dataframe_insitu,mooring_name,field,dir_output):
    dataframe_insitu.to_csv(dir_output + f"/" + mooring_name + f"_{field}" + ".csv")










