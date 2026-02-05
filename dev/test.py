import xarray as xr
import pandas as pd
import numpy as np
import os
import yaml
from modules import Module_all_functions as md
from modules import load_dataframe as loadData
from modules import load_configuration as loadCFG
from pathlib import Path


cfg = loadCFG.load_config("../config/config.yaml")
cfg_var = cfg['extraction']['variable']['var_name']
cfg_field = cfg['extraction']['variable']['field']
cfg_deph = cfg['extraction']['variable']['deph_val']

dir_path_sat = Path(cfg['extraction']['dir_paths']['dir_input_sat_csv'])
dir_path_mooring = Path(cfg['extraction']['dir_paths']['dir_input_mooring_nc'])


for fp in dir_path_sat.glob("*.csv"):
    print(loadData.load_satData_csv(fp, cfg_var))

for fp in dir_path_mooring.glob("*.nc"):
    print(loadData.load_moorData_nc(fp, cfg_var, cfg_field, cfg_deph))



