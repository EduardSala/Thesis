import xarray as xr
import pandas as pd
import numpy as np
from modules import Module_all_functions as md
import os
from data_extraction_md import extr_sat_data_from_csv as extr_sat
import data_extraction_md as data_extr

dir_input_sat_csv = "../datasets/satellite-csv"
path = "../datasets/satellite-csv/ds_S6A_wave.csv"
filePath = [os.path.join(dir_input_sat_csv, name) for name in os.listdir(dir_input_sat_csv)]

#print(filePath)
#df = pd.read_csv(filePath[0],skiprows=5)
df_sat = extr_sat(path,"VAVH")
var_name = "VAVH"
#print(df)
path_m = "../datasets/moorings-nc/AR_TS_MO_Oseberg-A.nc"
df_m,field= data_extr.extr_insitu_data_from_nc(path_m,var_name,"wave",0)


lat_sat = df_sat['latitude'].values
lon_sat = df_sat['longitude'].values

lat_m = df_m['latitude'].values[0]
lon_m = df_m['longitude'].values[0]

from spatial_crossover_md import calc_haversine as  haversine

distance = haversine(lat_m,lon_m,lat_sat,lon_sat)

#print(distance)
indici = np.where(distance<=50)
differenza = np.diff(indici[0])

cambi_cross = np.zeros(len(indici[0]),dtype=int)
#print(cambi_cross)
mark_cross = (differenza>1).astype(int)
cambi_cross[0] = 1
cambi_cross[1:] = mark_cross

df_sat = df_sat.loc[indici[0]]
df_sat['N_cross'] = np.cumsum(cambi_cross)
print(df_sat)


# when mark_cross==1, it means that a new crossover begins or an older one ends




