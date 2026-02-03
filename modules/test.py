import xarray as xr
import pandas as pd
import numpy as np
from modules import Module_all_functions as md
import os
dir_input_sat_csv = "../datasets/satellite-csv"

filePath = [os.path.join(dir_input_sat_csv, name) for name in os.listdir(dir_input_sat_csv)]
print(filePath)

df = pd.read_csv(dir_input_sat_csv,skiprows=5)
print(df)





