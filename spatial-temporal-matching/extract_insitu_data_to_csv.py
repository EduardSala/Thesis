import xarray as xr
import pandas as pd
import numpy as np
from modules import Module_all_functions as md

filepath_files = md.pass_filepath("../datasets/moorings-nc")

# -----------------------------------------------------------------------------
#variable_name = 'VAVH'
#application = 'wave'

variable_name = 'WSPD'
application = 'wind'
# -----------------------------------------------------------------------------

# After downloading all the moorings data in .nc format, I "trasform" them into .csv files with only information needed such as longitude - latitude - time - variable

for path in filepath_files:
    dataset_nc = xr.load_dataset(path)

    if variable_name in dataset_nc.data_vars:
        mooring_name = str(dataset_nc['STATION'].values).split(sep="'")[1]

        # variable = np.array(dataset_nc[variable_name][:, np.where(dataset_nc['DEPH'].values == 0)[0]].values).flatten()
        variable = np.array(
            dataset_nc[variable_name][:, np.where(dataset_nc['DEPH'].values == -10)[0]].values).flatten()

        time = np.array(
            dataset_nc['TIME'][np.where(dataset_nc['DEPH'].values == -10)[0]].values).flatten()

        latitude = np.array(dataset_nc['LATITUDE'].values).flatten()

        longitude = np.array(dataset_nc['LONGITUDE'].values).flatten()

        dataframe_mooring = pd.DataFrame(
            {'time': np.array(time), 'latitude': np.array(latitude), 'longitude': np.array(longitude),
             variable_name: np.array(variable)}).fillna(-1)
        # dataframe_mooring.to_csv(r"../datasets/mooring-csv/" + mooring_name + "_wave" + ".csv")
        dataframe_mooring.to_csv(r"../datasets/mooring-csv/" + mooring_name + "_wind" + ".csv")

    else:
        continue







