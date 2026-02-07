import pathlib
import typing
from optparse import Values
from string import printable
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import os
import geopandas as gp
import geopy.distance
import cartopy.mpl.ticker as cticker
from datetime import datetime
from datetime import date
import math
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr
from scipy import interpolate

# ---------------------
path_coastlines = "https://www.ngdc.noaa.gov/mgg/shorelines/gshhs.html"
path_lakes = "https://www.ngdc.noaa.gov/mgg/shorelines/gshhs.html"

def pass_filepath(folderPath:pathlib.Path) ->list[str]:
    """
    
    Questa funzione prende il percorso di una cartella
    con dei file e mi ritorna ogni percorso di ogni file.
    :param folderPath:
    :return: filePath
    """
    filePath = [os.path.join(folderPath, name) for name in os.listdir(folderPath)]
    return filePath


def haversine(lon1:float, lat1:float, lon2:float, lat2:float) -> float:
    """
    This function calculates the distance between the mooring location and the projected position by the satellite on Earth, using Haversine function.
    :param lon1: Longitudine del corpo 1
    :param lat1: Latitudine del corpo 1
    :param lon2: Longitudine del corpo 2
    :param lat2: Latitudine del corpo 2
    :return: distance
    """
    # Converting from degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # -----------------
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # Haversine Formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    # Earth radius in Km
    r = 6371
    # ----------------
    distance = c * r
    # ----------------
    return distance


def save_Sat_data(filePath_sat_data:pathlib.Path,variable) -> pd.DataFrame:


    """
    Questa funzione salva i dati del satellite in un DataFrame di Pandas.

    Si basa sulla struttura dati dei file .csv presi da CMEMS.
    :param filePath_sat_data:
    :param variable: nome della variabile da salvare
    :return df_sat: dataframe del satellite
    """

    #filepath_sat_data = [os.path.join(folderPath_sat,name) for name in os.listdir(folderPath_sat)]
    df =  pd.read_csv(filePath_sat_data,skiprows=5)
    time = pd.to_datetime(df[df['parameter']==variable]['time'])
    var = df[df['parameter']==variable]['value'].values
    lon = df[df['parameter']==variable]['longitude'].values
    lat = df[df['parameter'] == variable]['latitude'].values


    dict_sat = {'latitude':lat,'longitude':lon,variable:var,'time':pd.to_datetime(time)}
    df_sat = pd.DataFrame(dict_sat).sort_values(by='time')
    return df_sat

def dist_platf_coast(filepath_platf:pathlib.Path) -> None:
    """
    Funzione che calcola la minima distanza di una piattaforma dalla costa.
    :param filepath_platf:
    :return nome,dist: nome della piattaforma, distanza minima dalla costa
    """

    path_coastlines = r"D:\Materiale_Uni\Tesi_Magsitrale\Dataset\GSHHS_shp\i\GSHHS_i_L1.shp"
    coast = gp.read_file(path_coastlines)
    #lakes = gp.read_file(path_lakes)
    #lakes.crs = "EPSG:4326"
    coast.crs = "EPSG:4326"
    count = 0
    platf = xr.load_dataset(filepath_platf)
    flag = 0
    min = 1000000
    punto = [platf['LATITUDE'].values, platf['LONGITUDE'].values]
    for c in coast.geometry:
        for coord in c.exterior.coords:
            coord_from = [coord[1], coord[0]]
            distanza = geopy.distance.GeodesicDistance(punto, coord_from).km
            if distanza < min:
                min = distanza
                coord_min = coord_from
    print("La distanza minima dalla costa della piattaforma ", platf.attrs['platform_code'], "è: ", min)
    nome = platf.attrs['platform_code']
    dist = min

    return nome,dist


"""
def dist_platf_coast2(coord_array,filepath_platf:pathlib.Path) -> None:
    platf = xr.load_dataset(filepath_platf)
    min = 1000000
    punto = [platf['LATITUDE'].values, platf['LONGITUDE'].values]
    for coord in coord_array:
        coord_from = [coord[0], coord[1]]
        distanza = geopy.distance.GeodesicDistance(punto, coord_from).km
        if distanza < min:
            min = distanza
            coord_min = coord_from
    print("La distanza minima dalla costa della piattaforma ", platf.attrs['platform_code'], "è: ", min)
    nome = platf.attrs['platform_code']
    dist = min
    return nome, dist
"""

def calcolo_count_CrossPoints(dataFrame_sat:pd.DataFrame,platform:xr.Dataset,distance_tol:float) -> int:

    """
    Calcola i cross-points: quante volte il satellite  passa sopra la piattaforma.

    :param dataFrame_sat: dataFrame del satellite
    :param platform: dataSet della piattaforma (file .xr)
    :param distance_tol: massima distanza del cross-radius
    :return:
    """

    dist_tol = distance_tol
    dist_lim = 1
    lon_sat = dataframe_sat['longitude'].values
    lat_sat = dataframe_sat['latitude'].values
    time_sat = pd.to_datetime(dataframe_sat['time'])
    lon_sat_filter = lon_sat
    lat_sat_filter = lat_sat
    time_sat_filter = time_sat
    #------------------------------------
    lon_platf = platform['LONGITUDE'].values
    lat_platf = platform['LATITUDE'].values
    platf_code = platform.attrs['platform_code']

    lon_sat_filter = np.array(lon_sat_filter)
    lat_sat_filter = np.array(lat_sat_filter)
    time_sat_filter = np.array(time_sat_filter)

    index_i = 1
    n_cross = 0
    while index_i < len(time_sat_filter) - 2:
        index_i_minus_1 = index_i - 1
        index_i_plus_1 = index_i + 1

        #dist_i = np.sqrt((lon_platf - lon_sat_filter[index_i]) ** 2 + (lat_platf - lat_sat_filter[index_i]) ** 2)
        #dist_i_minus_1 = np.sqrt((lon_platf - lon_sat_filter[index_i_minus_1]) ** 2 + (lat_platf - lat_sat_filter[index_i_minus_1]) ** 2)
        #dist_i_plus_1 = np.sqrt((lon_platf - lon_sat_filter[index_i_plus_1]) ** 2 + (lat_platf - lat_sat_filter[index_i_plus_1]) ** 2)

        dist_i = haversine(lon_platf, lat_platf, lon_sat_filter[index_i], lat_sat_filter[index_i])
        dist_i_minus_1 = haversine(lon_platf, lat_platf, lon_sat_filter[index_i_minus_1],
                                   lat_sat_filter[index_i_minus_1])
        dist_i_plus_1 = haversine(lon_platf, lat_platf, lon_sat_filter[index_i_plus_1], lat_sat_filter[index_i_plus_1])
        if (dist_i_minus_1) > dist_tol and (dist_i) <= dist_tol and (dist_i_plus_1) <= dist_tol:
            j_loop = index_i
            inizio = j_loop
            condizione = True
            while condizione == True:
                j_plus_1 = j_loop + 1
                if j_plus_1 == (len(time_sat_filter) - 1):
                    fine = j_plus_1
                    index_i = fine
                    dist_j = haversine(lon_platf, lat_platf, lon_sat_filter[j_loop], lat_sat_filter[j_loop])
                    dist_j_plus_1 = haversine(lon_platf, lat_platf, lon_sat_filter[j_plus_1], lat_sat_filter[j_plus_1])
                    condizione = False
                else:

                    #dist_j = np.sqrt((lon_platf - lon_sat_filter[j_loop]) ** 2 + (lat_platf - lat_sat_filter[j_loop]) ** 2)
                    #dist_j_plus_1 = np.sqrt((lon_platf - lon_sat_filter[j_plus_1]) ** 2 + (lat_platf - lat_sat_filter[j_plus_1]) ** 2)

                    dist_j = haversine(lon_platf,lat_platf,lon_sat_filter[j_loop],lat_sat_filter[j_loop])
                    dist_j_plus_1 = haversine(lon_platf, lat_platf, lon_sat_filter[j_plus_1], lat_sat_filter[j_plus_1])
                    if (dist_j) <= dist_tol and (dist_j_plus_1) <= dist_tol:
                        j_loop = j_loop + 1
                    elif (dist_j) <= dist_tol and (dist_j_plus_1) > dist_tol:
                        fine = j_loop
                        index_i = fine
                        condizione = False
                    index_i = index_i + 2
            n_cross = n_cross + 1
        else:
            index_i = index_i + 1
    print(platf_code," ha:",n_cross," cross-points ")
    return n_cross

def filtering_crossover_spatial(dataFrame_sat:pd.DataFrame,platf:pd.DataFrame,distance_tol:float,platf_code:str,variable:str) -> pd.DataFrame:
    """
    Il matching spazio-temporale viene fatto insieme ma per comodità di scrittura dell'algoritmo, ho preferito
    spezzettare in due fasi.

    :param dataFrame_sat: dataFrame del satellite .csv
    :param platf: dataFrame della piattaforma .csv
    :param distance_tol: massima distanza tollerabile = cross-radius
    :return: dataFrame_sat_cross,dataFrame_platf
    """
    dist_tol = distance_tol
    dist_lim = distance_tol + 0.1
    dataFrame_sat = dataFrame_sat.fillna(-1)
    lon_sat = dataFrame_sat['longitude'].values
    lat_sat = dataFrame_sat['latitude'].values
    hs_sat = dataFrame_sat[variable].values
    time_sat = pd.to_datetime(dataFrame_sat['time']).dt.tz_localize(None)

    platform = platf
    lon_platf = platform['longitude'].values[0]
    lat_platf = platform['latitude'].values[0]

    lon_sat_filter = []
    lat_sat_filter = []
    hs_sat_filter = []
    time_sat_filter = []

    for lon, lat, hs,t in zip(lon_sat, lat_sat, hs_sat ,time_sat):
        if (lon) <= (lon_platf + dist_lim) and (lon) >= (lon_platf - dist_lim) and (lat) <= (
                lat_platf + dist_lim) and (lat) >= (lat_platf - dist_lim):
            lon_sat_filter.append(lon)
            lat_sat_filter.append(lat)
            hs_sat_filter.append(hs)
            time_sat_filter.append(t)
    lon_sat_filter = np.array(lon_sat_filter)
    lat_sat_filter = np.array(lat_sat_filter)
    hs_sat_filter = np.array(hs_sat_filter)
    time_sat_filter = np.array(time_sat_filter)

    time_sat_cross = []
    hs_sat_cross = []
    lon_sat_cross = []
    lat_sat_cross = []
    index_i = 1
    n_cross = 0
    number_crossover = []
    cross = 0

    # questo ciclo iterativo serve per salvare i dati del cross-over, ma siccome a priori non so effettivamente quando
    # inizia e finisce il cross, ho implementato questo algoritmo che permette di identificare i dati satellitari
    # in base al i-esimo passaggio del satellite sopra la piattaforma.

    while index_i < len(time_sat_filter) - 2:
        index_i_minus_1 = index_i - 1
        index_i_plus_1 = index_i + 1
        dist_i = np.sqrt((lon_platf - lon_sat_filter[index_i]) ** 2 + (lat_platf - lat_sat_filter[index_i]) ** 2)
        dist_i_minus_1 = np.sqrt(
            (lon_platf - lon_sat_filter[index_i_minus_1]) ** 2 + (lat_platf - lat_sat_filter[index_i_minus_1]) ** 2)
        dist_i_plus_1 = np.sqrt(
            (lon_platf - lon_sat_filter[index_i_plus_1]) ** 2 + (lat_platf - lat_sat_filter[index_i_plus_1]) ** 2)
        if (dist_i_minus_1) > dist_tol and (dist_i) <= dist_tol and (dist_i_plus_1) <= dist_tol:
            n_cross = n_cross + 1
            condizione = True
            j_loop = index_i
            inizio = j_loop
            while condizione == True:
                j_plus_1 = j_loop + 1
                if j_plus_1 == (len(time_sat_filter) - 1):
                    fine = j_plus_1
                    index_i = fine
                    condizione = False
                    break
                else:
                    dist_j = np.sqrt(
                        (lon_platf - lon_sat_filter[j_loop]) ** 2 + (lat_platf - lat_sat_filter[j_loop]) ** 2)
                    dist_j_plus_1 = np.sqrt(
                        (lon_platf - lon_sat_filter[j_plus_1]) ** 2 + (lat_platf - lat_sat_filter[j_plus_1]) ** 2)
                    if (dist_j) <= dist_tol and (dist_j_plus_1) <= dist_tol:
                        j_loop = j_loop + 1
                    elif (dist_j) <= dist_tol and (dist_j_plus_1) > dist_tol:
                        fine = j_loop
                        index_i = fine
                        condizione = False
                        break
            if fine != (len(time_sat_filter) - 1):
                for h, t, lon, lat in zip(hs_sat_filter[slice(inizio, fine)],time_sat_filter[slice(inizio, fine)],
                                          lon_sat_filter[slice(inizio, fine)], lat_sat_filter[slice(inizio, fine)]):
                    time_sat_cross.append(pd.to_datetime(t))
                    hs_sat_cross.append(h)
                    lon_sat_cross.append(lon)
                    lat_sat_cross.append(lat)
                    number_crossover.append(n_cross)
                index_i = index_i + 1
            elif (fine == (len(time_sat_filter) - 1)):
                for h,t, lon, lat in zip(hs_sat_filter[slice(inizio, fine)],time_sat_filter[slice(inizio, fine)],
                                          lon_sat_filter[slice(inizio, fine)], lat_sat_filter[slice(inizio, fine)]):
                    time_sat_cross.append(pd.to_datetime(t))
                    hs_sat_cross.append(h)
                    lon_sat_cross.append(lon)
                    lat_sat_cross.append(lat)
                    number_crossover.append(n_cross)
        else:
            index_i = index_i + 1

    if len(hs_sat_cross) > 0:
        dict_sat = {'N_cross': np.array(number_crossover), 'time': np.array(time_sat_cross),
                    variable: np.array(hs_sat_cross),'longitude': np.array(lon_sat_cross),
                    'latitude': np.array(lat_sat_cross)}
        dataFrame_sat_cross = pd.DataFrame(dict_sat)

        # gli ultimi due metodi dei dataFrame permettono di  salvare i dati in file .csv, per comodità
        #dataFrame_sat_cross.to_csv(seq_file_name + str(count) + ".csv")
        #dataFrame_plat.to_csv(seq_file_name + str(count) + "__" + platf_code + ".csv")

        return dataFrame_sat_cross
    else:
        #print("No cross-over.")
        return []

def filtering_crossover_temporal(dataFrame_sat_cross_spatial:pd.DataFrame,dataFrame_platf:pd.DataFrame,time_tol:int,variable) -> list[pd.DataFrame,pd.DataFrame]:

    """
    In questa parte viene fatto il filtraggio dei dati a livello temporale. Dato in input la lunghezza della finestra
    temporale, vengono quindi salvati i dati che rientrano in quel range. Vengono salvate tutte le misurazioni della
    piattaforma in quella finestra temporale e poi succesivamente tramite le co-location techniques si prendono
    le misurazioni finali.

    :param dataFrame_sat_cross_spatial: dataFrame del satellite .csv
    :param dataFrame_platf: dataFrame della piattaforma .csv
    :param time_tol: tolleranza sulla finestra temporale
    :param variable: nome della variabile
    :return:
    """

    time_sat_cross = pd.to_datetime(dataFrame_sat_cross_spatial['time'],format='mixed').dt.tz_localize(None)
    hs_sat_cross = dataFrame_sat_cross_spatial[variable]
    lon_sat_cross = dataFrame_sat_cross_spatial['longitude']
    lat_sat_cross = dataFrame_sat_cross_spatial['latitude']
    number_crossover_sat = dataFrame_sat_cross_spatial['N_cross']

    if len(dataFrame_sat_cross_spatial) > 0 and len(dataFrame_platf) > 0:

        delta_time = np.timedelta64(time_tol, 's')

        j = 0
        n_cross = 1
        time_platf_cross = []
        hs_platf_cross = []
        times_sat_cross_final = []
        hs_sat_cross_final = []
        lon_sat_cross_final = []
        lat_sat_cross_final = []
        number_crossover_final = []
        cross_points = 0
        number_crossover_platf = []

        platform = dataFrame_platf.fillna(-1)
        lon_platf = platform['longitude'][0]
        lat_platf = platform['latitude'][0]
        qc_platf = platform['QUALITY_FLAG'].values
        time_platf = pd.to_datetime(platform['time'],format='mixed').dt.tz_localize(None)
        hs_platf = platform[variable]
        lon_sat_filter = []
        lat_sat_filter = []
        hs_sat_filter = []
        time_sat_filter = []
        numb_end_cross = number_crossover_sat[len(number_crossover_sat) - 1]

        while j < len(time_sat_cross) - 1:
            start_loop = j
            start_cross = number_crossover_sat[start_loop]
            if number_crossover_sat[start_loop] != numb_end_cross:
                flag_end = 0
                j_loop = j
                while flag_end == 0:
                    if number_crossover_sat[j_loop + 1] == start_cross:
                        j_loop = j_loop + 1
                    elif number_crossover_sat[j_loop + 1] != start_cross:
                        end_loop = j_loop
                        j = end_loop
                        flag_end = 1
                        break
                t_sx = time_sat_cross[start_loop] - delta_time
                t_dx = time_sat_cross[end_loop] + delta_time
                no_nan = False
                for t, hs,qc in zip(time_platf, hs_platf,qc_platf):
                    if ((hs) != (-1)):
                        if (t >= t_sx) and (t <= t_dx):
                            if (qc==0) or (qc==1) or (qc==2):
                                no_nan = True
                                time_platf_cross.append(pd.to_datetime(t))
                                hs_platf_cross.append(hs)
                                number_crossover_platf.append(start_cross)
                if no_nan == True:
                    for lon, lat, hs, t, n in zip(lon_sat_cross[slice(start_loop, end_loop)],
                                                  lat_sat_cross[slice(start_loop, end_loop)],
                                                  hs_sat_cross[slice(start_loop, end_loop)],
                                                  time_sat_cross[slice(start_loop, end_loop)],
                                                  number_crossover_sat[slice(start_loop, end_loop)]):
                        lon_sat_cross_final.append(lon)
                        lat_sat_cross_final.append(lat)
                        hs_sat_cross_final.append(hs)
                        times_sat_cross_final.append(pd.to_datetime(t))
                        number_crossover_final.append(n)
                j = j + 1
            elif number_crossover_sat[start_loop] == numb_end_cross:
                end_loop = len(number_crossover_sat) - 1
                t_sx = pd.to_datetime(time_sat_cross[start_loop]) - delta_time
                t_dx = pd.to_datetime(time_sat_cross[end_loop]) + delta_time
                no_nan = False

                for t, hs,qc in zip(time_platf, hs_platf,qc_platf):
                    if ((hs) != (-1)):
                        if (t >= t_sx) and (t <= t_dx):
                            if (qc == 0) or (qc == 1) or (qc == 2):
                                no_nan = True
                                time_platf_cross.append(pd.to_datetime(t))
                                hs_platf_cross.append(hs)
                                number_crossover_platf.append(start_cross)

                if no_nan == True:
                    for lon, lat, hs, t, n in zip(lon_sat_cross[slice(start_loop, end_loop)],
                                                  lat_sat_cross[slice(start_loop, end_loop)],
                                                  hs_sat_cross[slice(start_loop, end_loop)],
                                                  time_sat_cross[slice(start_loop, end_loop)],
                                                  number_crossover_sat[slice(start_loop, end_loop)]):
                        lon_sat_cross_final.append(lon)
                        lat_sat_cross_final.append(lat)
                        hs_sat_cross_final.append(hs)
                        times_sat_cross_final.append(pd.to_datetime(t))
                        number_crossover_final.append(n)
                break



        print("Array is not empty!\t", "Platform", "has been analyzed! \n")
        datadict_sat = {"N_cross": np.array(number_crossover_final), "time": np.array(times_sat_cross_final),
                        variable: np.array(hs_sat_cross_final), "longitude": np.array(lon_sat_cross_final),
                        "latitude": np.array(lat_sat_cross_final)}
        datadict_platf = {"N_cross": np.array(number_crossover_platf), "time": np.array(time_platf_cross),
                          variable: np.array(hs_platf_cross),'longitude':lon_platf*np.ones(len(hs_platf_cross)),'latitude':lat_platf*np.ones(len(hs_platf_cross))}
        df_sat = pd.DataFrame(datadict_sat).dropna()
        df_platf = pd.DataFrame(datadict_platf).dropna()
        #df_sat.to_csv(seq_file_sat + '_' + str(n_platf) + ".csv", mode='w')
        #df_platf.to_csv(seq_file_platf + '_' + str(n_platf) + ".csv", mode='w')
        return df_sat,df_platf

    else:
        print("Array is empty!\t", "Platform  ", "has been analyzed! \n")
        return [],[]




def LIDW_function(df_sat:pd.DataFrame,df_platf:pd.DataFrame,beta_interp:float,variable:str) ->list[pd.DataFrame,pd.DataFrame]:
    """
    Spatial Co-location technique

    Linear Interpolation Distance Inverse Weighting

    :param df_sat: dataFrame del satellite
    :param df_platf: dataFrame della piattaforma
    :param beta_interp: fattore beta che può essere -2 oppure -1
    :param variable: nome della variabile
    :return: dataFrame_sat, dataFrame_paltform
    """
    beta = beta_interp
    sat = df_sat
    platf = df_platf


    time_sat = pd.to_datetime(sat['time'])
    number_cross = sat['N_cross'].values
    lat_sat = sat['latitude'].values
    lon_sat = sat['longitude'].values
    hs_sat = sat[variable].values
    lat_platf = platf['latitude'][0]
    lon_platf = platf['longitude'][0]
    j = 0
    hs_final = []
    time_final = []
    lon_final = []
    lat_final = []
    n_cross_final = []

    cross_n = []
    cross_j = 0
    numb_end_cross = number_cross[len(number_cross) - 1]
    while j < len(time_sat) - 1:
        array_wi = []
        array_zi = []
        start_loop = j
        if number_cross[start_loop] != numb_end_cross:
            j_loop = start_loop
            flag_end = 0
            start_cross = number_cross[start_loop]
            while flag_end == 0:
                if number_cross[j_loop + 1] == start_cross:
                    j_loop = j_loop + 1
                else:  # number_cross[j_loop + 1] != start_cross:
                    end_loop = j_loop
                    j = end_loop
                    flag_end = 1
                    break
            if flag_end == 1:
                for hs, lon, lat in zip(hs_sat[slice(start_loop, end_loop)], lon_sat[slice(start_loop, end_loop)],lat_sat[slice(start_loop, end_loop)]):
                    dist_i = np.sqrt((lon_platf - lon) ** 2 + (lat_platf - lat) ** 2)
                    w_i = dist_i ** (-beta)
                    array_wi.append(dist_i)
                    array_zi.append(hs)
            arr_wi = np.array(array_wi)
            arr_zi = np.array(array_zi)
            num = np.multiply(arr_wi, arr_zi).sum()
            den = arr_wi.sum()
            #print(num, '--', den)
            if den==0 and num ==0:
                hs_final.append(-1)
                time_final.append(time_sat[start_loop])
                lon_final.append(lon_sat[start_loop])
                lat_final.append(lat_sat[start_loop])
                n_cross_final.append(number_cross[start_loop])
            elif den!=0 and  num!=0:
                zi = num / den
                hs_final.append(zi)
                time_final.append(time_sat[start_loop])
                lon_final.append(lon_sat[start_loop])
                lat_final.append(lat_sat[start_loop])
                n_cross_final.append(number_cross[start_loop])
            j = end_loop
            j = j + 1
        elif number_cross[start_loop] == numb_end_cross:
            end_loop = len(number_cross) - 1
            for hs, lon, lat in zip(hs_sat[slice(start_loop, end_loop)], lon_sat[slice(start_loop, end_loop)],lat_sat[slice(start_loop, end_loop)]):
                dist_i = np.sqrt((lon_platf - lon) ** 2 + (lat_platf - lat) ** 2)
                w_i = dist_i ** (-beta)
                array_wi.append(dist_i)
                array_zi.append(hs)
            arr_wi = np.array(array_wi)
            arr_zi = np.array(array_zi)
            num = np.multiply(arr_wi, arr_zi).sum()
            den = arr_wi.sum()
            if den == 0 and num == 0:
                hs_final.append(-1)
                time_final.append(time_sat[start_loop])
                lon_final.append(lon_sat[start_loop])
                lat_final.append(lat_sat[start_loop])
                n_cross_final.append(number_cross[start_loop])
            elif den!=0 and  num!=0:
                zi = num / den
                hs_final.append(zi)
                time_final.append(time_sat[start_loop])
                lon_final.append(lon_sat[start_loop])
                lat_final.append(lat_sat[start_loop])
                n_cross_final.append(number_cross[start_loop])
            j =  end_loop

    dataframe_sat = pd.DataFrame({'N_cross':n_cross_final,"time":pd.to_datetime(time_final),variable:hs_final,"longitude":lon_final,"latitude":lat_final})
    #dataframe_sat.to_csv(filepath_out + "_sat" + ".csv")
    #platf.to_csv(filepath_out + "_p" + ".csv")
    return dataframe_sat,platf

def minimum_distance(df_sat:pd.DataFrame,df_platf:pd.DataFrame,variable:str) -> list[pd.DataFrame,pd.DataFrame]:
    """

    Spatial Co-location technique

    Minimum distance

    :param df_sat: dataFrame del satellite
    :param df_platf: dataFrame della piattaforma
    :param variable: nome della variabile
    :return: dataframe_sat,dataframe_platf
    """
    sat = df_sat
    platf = df_platf

    time_sat = pd.to_datetime(sat['time'])
    number_cross = sat['N_cross']
    lat_sat = sat['latitude'].values
    lon_sat = sat['longitude'].values
    hs_sat = sat[variable].values
    lat_platf = platf['latitude'].values[0]
    lon_platf = platf['longitude'].values[0]
    j = 0
    hs_final = []
    time_final = []
    lon_final = []
    lat_final = []
    cross_n = []
    cross_j = 0
    numb_end_cross = number_cross[len(number_cross) - 1]
    while j < len(time_sat) - 1:
        min = 10000
        start_loop = j
        if number_cross[start_loop] != numb_end_cross:
            j_loop = start_loop
            flag_end = 0
            start_cross = number_cross[start_loop]
            hs_min = hs_sat[start_loop]
            while flag_end == 0:
                if number_cross[j_loop + 1] == start_cross:
                    j_loop = j_loop + 1
                else: #number_cross[j_loop + 1] != start_cross:
                    end_loop = j_loop
                    j = end_loop
                    flag_end = 1
                    break
            if flag_end == 1:
                for hs, lon, lat in zip(hs_sat[slice(start_loop, end_loop)], lon_sat[slice(start_loop, end_loop)],lat_sat[slice(start_loop, end_loop)]):
                    dist = np.sqrt((lon_platf - lon) ** 2 + (lat_platf - lat) ** 2)
                    if dist <= min:
                        min = dist
                        hs_min = hs
                hs_final.append(hs_min)
                time_final.append(time_sat[start_loop])
                lon_final.append(lon_sat[start_loop])
                lat_final.append(lat_sat[start_loop])
                cross_n.append(start_cross)
            j = j + 1
        elif number_cross[start_loop] == numb_end_cross:
            end_loop = len(number_cross) - 1
            hs_min  = hs_sat[start_loop]
            for hs, lon, lat in zip(hs_sat[slice(start_loop, end_loop)], lon_sat[slice(start_loop, end_loop)],
                                    lat_sat[slice(start_loop, end_loop)]):
                dist = np.sqrt((lon_platf - lon) ** 2 + (lat_platf - lat) ** 2)
                if dist <= min:
                    min = dist
                    hs_min = hs
            hs_final.append(hs_min)
            time_final.append(time_sat[start_loop])
            lon_final.append(lon_sat[start_loop])
            lat_final.append(lat_sat[start_loop])
            cross_n.append(number_cross[start_loop])
            j = end_loop






    dataframe_sat = pd.DataFrame({"N_cross":cross_n,"time": np.array(pd.to_datetime(time_final)), variable: hs_final,"longitude":lon_final,"latitude":lat_final})
    #dataframe_sat.to_csv(filepath_out + "_sat" + ".csv")
    #platf.to_csv(filepath_out + "_p" + ".csv")
    return dataframe_sat,platf

def coLocation_temporal(df_sat:pd.DataFrame,df_platf:pd.DataFrame,variable) -> list[pd.DataFrame,pd.DataFrame]:
    """
    :param df_sat
    :param df_platf
    :param filepath_out

    :return:TS_sat,TS_platf
    """


    sat = df_sat
    platf =  df_platf
    var_sat = sat[variable].values
    time_sat = sat['time'].values.astype('datetime64[s]')
    ncross_sat = sat['N_cross'].values
    var_p = platf[variable].values
    time_p = platf['time'].values.astype('datetime64[s]')
    ncross_p = platf['N_cross'].values
    lon_p = platf['longitude'].values
    lat_p = platf['latitude'].values
    TS_sat = pd.DataFrame({'N_cross':ncross_sat,variable: var_sat,'time':time_sat}, index=ncross_sat)
    TS_sat.columns = ['N_cross',variable,'time']
    TS_platf = pd.DataFrame({'N_cross':ncross_p,variable: var_p,'time':time_p}, index=ncross_p)
    TS_platf.columns = ['N_cross',variable,'time']

    length_array = len(TS_platf)
    TS_platf['longitude'] = lon_p[0] * np.ones([length_array,1])
    TS_platf['latitude'] = lat_p[0] * np.ones([length_array, 1])
    #TS_platf.columns = ['N_cross', 'Hs', 'time','longitude','latitude']
    #TS_platf = TS_platf.drop_duplicates(subset=['N_cross'])


    # in questa ciclo vengono prese tutte le misurazioni del cross-over e viene fatta una media temporale
    # produce risultati molto scarsi ma era giusto citarla, siccome è stato verificato
    """
    for p in TS_platf.index:
        TS_platf.loc[p, variable] = TS_platf[variable].loc[p].mean()
        TS_platf.loc[p, 'time'] = TS_platf['time'][p]
    TS_platf = TS_platf.drop_duplicates(subset=[variable])
    """

    # in questo ciclo viene presa la misurazione più vicina a livello temporale rispetto alla misurazione del satellite
    for i in TS_sat.index.values:
        condizione = np.isin(i,TS_platf.index.values)
        size = np.array(TS_platf.loc[i]).size
        if size > 5:
            time_s = pd.to_datetime(TS_sat.loc[i, 'time'])
            hs_p = []
            time_platf = []
            min = 100000000
            for t, hs in zip(TS_platf.loc[i, 'time'].values, TS_platf.loc[i, variable].values):
                delta = pd.Timedelta(np.abs(time_s - t), 's').seconds
                if delta < min:
                    min = delta
                    time_p = t
                    hs_p = hs
            TS_platf.loc[i, variable] = hs_p
            TS_platf.loc[i, 'time'] = time_p
        else:
            continue



    TS_platf = TS_platf.drop_duplicates('N_cross')
    if len(TS_sat) < len(TS_platf):
        for i_p in TS_platf.index.values:
            cond = np.isin(i_p,TS_sat.index.values)
            if cond == False:
                TS_platf = TS_platf.drop(i_p)
    elif len(TS_sat) > len(TS_platf):
        for i_s in TS_sat.index.values:
            cond = np.isin(i_s, TS_platf.index.values)
            if cond == False:
                TS_sat = TS_sat.drop(i_s)

    ind = TS_sat[TS_sat[variable]== (-1)].index
    TS_sat = TS_sat[TS_sat[variable]!= (-1)]
    TS_platf = TS_platf.drop(ind)

    return TS_sat,TS_platf

def read_ALL_data(filepath:pathlib.Path) -> list[pd.DataFrame,pd.DataFrame]:
    """
    Funzione che legge il file .csv che è stato prodotto dopo aver fatto il matching spazio-temporale.

    :param filepath
    :return: [dataframe_sat,dataframe_p]
    """
    foldMd = filepath
    hs_p = []
    time_p = []
    hs_s = []
    time_s = []
    for fp in pass_filepath(foldMd):
        strumento = fp.split(sep='\\')[9].split(sep='_')[1].split(sep='.')[0]
        df = pd.read_csv(fp)
        if strumento == 'p':
            for hs,time in zip(df['hs'].values,df['time'].values):
                hs_p.append(hs)
                time_p.append(pd.to_datetime(time))
        else:
            for hs, time in zip(df['hs'].values, df['time'].values):
                hs_s.append(hs)
                time_s.append(pd.to_datetime(time))

    dic_p = {'time': time_p, 'hs': hs_p}
    dic_s = {'time': time_s, 'hs': hs_s}
    df_s = pd.DataFrame(dic_s)
    df_p = pd.DataFrame(dic_p)

    return df_s,df_p

def get_season(date_str:str) -> str:
    """
        Convert a date to the corresponding season.
        :param date_str: A string date in 'YYYY-MM-DD' format.
        :return: The season as a string.
        """
    try:
        date_pd = pd.to_datetime(date_str)
    except ValueError:
        return "Formato data non valido. Usa 'YYYY-MM-DD'."

    # Estrai il mese e il giorno dalla data
    month_day = (date_pd.month, date_pd.day)

    # Definisci i periodi di stagione
    seasons = {
        "Spring": ((3, 1), (5, 31)),
        "Summer": ((6, 1), (8, 31)),
        "Autumn": ((9, 1), (11, 30)),
        "Winter": ((12, 1), (2, 28))
    }

    # Determina la stagione corrispondente
    for season, (start, end) in seasons.items():
        if (start <= month_day <= end):
            return season

    return 'Winter'


def Bias(df_real,df_pred,variable):

    hs_pred =  df_pred[variable].values
    hs_real =  df_real[variable].values
    bias = np.mean(hs_pred - hs_real)

    return bias

def RMSE(df_real,df_pred,variable):

    hs_pred = df_pred[variable].values
    hs_real = df_real[variable].values
    rmse =  root_mean_squared_error(hs_real,hs_pred)

    return rmse

def SI(df_real,df_pred,variable):

    hs_pred = df_pred[variable].values
    hs_real = df_real[variable].values
    si = RMSE(df_real,df_pred,variable)/np.mean(hs_real)

    return si

def CC(df_real,df_pred,variable):

    hs_pred = df_pred[variable]
    hs_real = df_real[variable]
    cc,_ = pearsonr(hs_real,hs_pred)

    return cc

def save_data_p(platform_nc_file:xr.Dataset,folderP:pathlib.Path,start_time:str,end_time:str) -> None:
    """
    Funzione che prende i  file .xr delle piattaforme e li salva in formato .csv, con i relativi dati.

    :param platform_nc_file: file .nc della piattaforma
    :param folderP: path della cartella dove voglio salvare il file
    :param start_time: periodo d'inizio in cui  voglio i dati
    :param end_time: periodo finale in cui  voglio i dati
    :return:
    """

    platform = platform_nc_file.sel(TIME=slice(start_time,end_time))
    lon_platf = platform['LONGITUDE'].values
    lat_platf = platform['LATITUDE'].values
    platf_code = platform.attrs['platform_code']
    time_platf = []
    hs_platf = []
    qc_hs = []
    dir_platf = []

    # questo ciclo che inizia serve per salvare i dati dell' Hs, ma siccome si usa nei file una  diversa denominazione,
    # con questo ciclo vado a filtrare e salvare i dati quando mi trova il nome della variabile giusta.
    # Inoltre spesso accade che i dati vengono forniti per diverse profondità, in quanto oltre al dato dell'Hs viene
    # anche fornito il dato della Wind_speed_10_m
    trovato = 0
    while trovato == 0:
        size = platform['DEPTH'].size
        match size:
            case x if x > 2:
                for var in platform.variables.keys():
                    match var:
                        case 'VAVH':
                            trovato = 1
                            for t in platform['TIME'].values:
                                for deph in platform['DEPTH'].values:
                                    var_deph = platform['DEPH'].sel(TIME=t).values[deph]
                                    if var_deph == (0.0):
                                        hs_platf.append(platform['VAVH'].sel(TIME=t).values[deph].item())
                                        time_platf.append(pd.to_datetime(t))
                                        qc_hs.append(platform['VAVH_QC'].sel(TIME=t).values[deph].item())
                            break
                        case 'VHM0':
                            trovato = 1
                            for t in platform['TIME'].values:
                                for deph in platform['DEPTH'].values:
                                    var_deph = platform['DEPH'].sel(TIME=t).values[deph]
                                    if var_deph == (0.0):
                                        hs_platf.append(platform['VHM0'].sel(TIME=t).values[deph].item())
                                        time_platf.append(pd.to_datetime(t))
                                        qc_hs.append(platform['VAVH_QC'].sel(TIME=t).values[deph].item())
                            break
                if trovato == 1:
                    break
            case 2:
                for var in platform.variables.keys():
                    match var:
                        case 'VAVH':
                            trovato = 1
                            for t in platform['TIME'].values:
                                for deph in platform['DEPTH'].values:
                                    var_deph = platform['DEPH'].sel(TIME=t).values[deph]
                                    if var_deph == (0.0):
                                        hs_platf.append(platform['VAVH'].sel(TIME=t).values[deph].item())
                                        time_platf.append(pd.to_datetime(t))
                                        qc_hs.append(platform['VAVH_QC'].sel(TIME=t).values[deph].item())
                            break
                        case 'VHM0':
                            trovato = 1
                            for t in platform['TIME'].values:
                                for deph in platform['DEPTH'].values:
                                    var_deph = platform['DEPH'].sel(TIME=t).values[deph]
                                    if var_deph == (0.0):
                                        hs_platf.append(platform['VHM0'].sel(TIME=t).values[deph].item())
                                        time_platf.append(pd.to_datetime(t))
                                        qc_hs.append(platform['VAVH_QC'].sel(TIME=t).values[deph].item())
                            break
                if trovato == 1:
                    break
            case 1:
                for var in platform.variables.keys():
                    match var:
                        case 'VAVH':
                            trovato = 1
                            for t in platform['TIME'].values:
                                hs_platf.append(platform['VAVH'].sel(TIME=t).values.item())
                                time_platf.append(pd.to_datetime(t))
                                qc_hs.append(platform['VAVH_QC'].sel(TIME=t).values.item())
                            break
                        case 'VHM0':
                            trovato = 1
                            for t in platform['TIME'].values:
                                hs_platf.append(platform['VHM0'].sel(TIME=t).values.item())
                                time_platf.append(pd.to_datetime(t))
                                qc_hs.append(platform['VAVH_QC'].sel(TIME=t).values.item())
                            break
                if trovato == 1:
                    break
    df_p = pd.DataFrame(
        {'VAVH': hs_platf,'QUALITY_FLAG':qc_hs ,'time': pd.to_datetime(time_platf), 'longitude': lon_platf * np.ones(len(time_platf)),
         'latitude': lat_platf * np.ones(len(time_platf))})
    fd_out = folderP + '\\' + platf_code + '_.csv'
    df_p.to_csv(fd_out)


    print(fd_out)
    return 0

def save_wspd_data(platform_nc_file:xr.Dataset,folderP:pathlib.Path,start_time:str,end_time:str) -> None:

    """
    Funzione che prende i  file .xr delle piattaforme e li salva in formato .csv, con i relativi dati.

    :param platform_nc_file: file .nc della piattaforma
    :param folderP: path della cartella dove voglio salvare il file
    :param start_time: periodo d'inizio in cui  voglio i dati
    :param end_time: periodo finale in cui  voglio i dati
    :return:
    """
    platform = platform_nc_file.sel(TIME=slice(start_time, end_time))
    lon_platf = platform['LONGITUDE'].values
    lat_platf = platform['LATITUDE'].values
    platf_code = platform.attrs['platform_code']
    time_platf = []
    qualityFlag = []
    wspd_platf = []
    dir_platf = []

    for t in platform['TIME'].values:
        for deph in platform['DEPTH'].values:
            var_deph = platform['DEPH'].sel(TIME=t).values[deph]
            if var_deph == (-10.0):
                time_platf.append(pd.to_datetime(t))
                wspd_platf.append(platform['WSPD'].sel(TIME=t).values[deph])
                qualityFlag.append(platform['WSPD_QC'].sel(TIME=t).values[deph])
                break

    df_p = pd.DataFrame(
        {'WIND_SPEED': wspd_platf,'QUALITY_FLAG':qualityFlag,'time': pd.to_datetime(time_platf),
         'longitude': lon_platf * np.ones(len(time_platf)),
         'latitude': lat_platf * np.ones(len(time_platf))})
    fd_out = folderP + '\\' + platf_code + '_.csv'
    df_p.to_csv(fd_out)
    return 0


def hs_resampled_time(time_arr_resample,dataframe,timeDelta):
    """
    Funzione che fa un resampling dei dati rispetto ad un timeDelta che diamo in input.

    :param time_arr_resample:
    :param dataframe:
    :param timeDelta:
    :return: hs_array_resampled
    """
    df = dataframe
    time_arr = time_arr_resample

    df['time'] = pd.to_datetime(df['time'])
    time_df = pd.DataFrame({'time': time_arr})

    df = df.sort_values('time')
    time_df = time_df.sort_values('time')

    merged_df = pd.merge_asof(time_df, df, on='time', tolerance=timeDelta, direction='nearest')
    hs_p = merged_df['hs'].tolist()

    return hs_p

def hs_mean_min_max(folderPath_cluster,time_arr):
    """
    Funzione che viene utlizzata per calcolare Min - Max - Mean tra i vari elementi del 'cluster'

    :param folderPath_cluster:
    :param time_arr: array temporale su cui è stato fatto il resampling
    :return: hs_mean,hs_min,hs_max
    """
    arr_p = []
    hs_df = []
    for fp in pass_filepath(folderPath_cluster):
        df = pd.read_csv(fp, index_col=0)
        hs_p = hs_resampled_time(time_arr, df, pd.Timedelta(minutes=5))
        hs_df.append(hs_p)

    indici = np.arange(0, len(hs_df[0]))
    hs_min = []
    hs_max = []
    hs_mean = []
    for i in indici:
        hs_fake = []
        for hs_loop in hs_df:
            hs_fake.append(hs_loop[i])
        hs_fake = np.array(hs_fake)
        if np.isnan(hs_fake).all():
            hs_min.append(np.nan)
            hs_max.append(np.nan)
            hs_mean.append(np.nan)
        else:
            hs_min.append(np.nanmin(hs_fake))
            hs_max.append(np.nanmax(hs_fake))
            hs_mean.append(np.nanmean(hs_fake))
    hs_min = np.array(hs_min)
    hs_max = np.array(hs_max)
    hs_mean = np.array(hs_mean)
    hs_max[hs_max == 0] = np.nan
    hs_mean[hs_mean == 0] = np.nan
    mean_series = pd.DataFrame({'hs': hs_mean})
    hs_min = pd.Series(hs_min, index=time_arr)
    hs_mean = pd.Series(hs_mean, index=time_arr)
    hs_max = pd.Series(hs_max, index=time_arr)

    return hs_mean,hs_min,hs_max


def Split_dataframe_season(df:pd.DataFrame,variable:str):

    stagioni = {'Autumn','Winter','Spring','Summer'}
    array_stagioni = []
    min = []
    max = []
    mean = []

    df['stagione'] = df['time'].apply(get_season)


    for name in stagioni:
        array_stagioni.append(name)
        min.append(df[df['stagione']==name][variable].min())
        max.append(df[df['stagione']==name][variable].max())
        mean.append(df[df['stagione']==name][variable].mean())

    df_finale = pd.DataFrame({'min':min,'max':max,'mean':mean},index=array_stagioni)
    return df_finale


def Split_dataframe_calibration(df_x:pd.DataFrame,df_y:pd.DataFrame,variable:str):

    """
    Funzione che divide i dataFrame in due parti, un dataframe "calibration" ed un altro "validation" in cui si verifica/valida
    il metodo di calibrazione utilizzato precedentemente.

    :param df_x:
    :param df_y:
    :param variable:
    :return:
    """

    """
    # Divido il dataframe in base all'anno (i vari df non sono bilanciati in termini di proprozioni dei dati
    df_x['year'] = pd.to_datetime(df_x['time'].values).year
    df_x_cal = df_x[df_x['year']==2023]
    df_y_cal = df_y[df_x['year']==2023]
    df_x_val = df_x[df_x['year']!=2023]
    df_y_val = df_y[df_x['year']!=2023]
    """

    """
    # Divido il dataframe in base alla stagione
    df_x['stagione'] = df_x['time'].apply(get_season)
    df_y['stagione'] = df_y['time'].apply(get_season)
    indici = df_x[df_x[variable] == 0].index
    df_y = df_y.drop(indici)
    df_x = df_x.drop(indici)

    indici_cal = df_y[df_y['stagione'] == 'Winter'].index
    indici_val = df_y[df_y['stagione'] != 'Winter'].index
    df_y_cal = df_y.loc[indici_cal]
    df_x_cal = df_x.loc[indici_cal]
    df_y_val = df_y.loc[indici_val]
    df_x_val = df_x.loc[indici_val]
    """


    df_x['day'] = pd.to_datetime(df_x['time'].values).day
    df_x_cal = df_x[df_x['day'] <= 10]
    df_y_cal = df_y[df_x['day'] <= 10]
    df_x_val = df_x[df_x['day'] > 10]
    df_y_val = df_y[df_x['day'] > 10]

    """
    df_x['year'] = pd.to_datetime(df_x['time'].values).year
    df_x_cal = df_x[df_x['year'] == 2022]
    df_y_cal = df_y[df_x['year'] == 2022]
    df_x_val = df_x[df_x['year'] != 2022]
    df_y_val = df_y[df_x['year'] != 2022]
    """
    return df_x_cal,df_y_cal,df_x_val,df_y_val

def Describe_dataframe(df_x:pd.DataFrame,df_y:pd.DataFrame,variable:str,name_technique:str):

    bias = Bias(df_x, df_y, variable)
    #bias = "{:.4g}".format(bias)
    rmse = RMSE(df_x, df_y, variable)
    #rmse = "{:.4g}".format(rmse)
    cc = CC(df_x, df_y, variable)
    #cc = "{:.4g}".format(cc)
    si = SI(df_x, df_y, variable)
    #si = "{:.4g}".format(si)

    #print("Bias correction technique = ",name_technique)
    #print("Bias \t RMSE \t CC \t SI")
    #print(bias,'\t',rmse,'\t',cc,'\t',si)

    return [bias,rmse,cc,si]

    #df = pd.DataFrame({'Bias before cal.':bias_cal,'Bias after cal.':bias_cal_new,'RMSE before cal.':rmse_cal,'RMSE after cal.':rmse_cal_new,'CC before cal.':cc_cal,'CC after cal.':cc_cal_new,'SI before cal.':si_cal,'SI after cal.':si_cal_new})

    #return [bias_cal,bias_cal_new,rmse_cal,rmse_cal_new,cc_cal,cc_cal_new,si_cal,si_cal_new]





def LinearCalibration(df_cal_x:pd.DataFrame,df_cal_y:pd.DataFrame,df_y_validation,variable:str) -> pd.DataFrame:
    """
    :param df_cal_x:
    :param df_cal_y:
    :param variable: name of the variable
    :return  [b,a] --> b=slope and a=intercept
    """
    variable_x = df_cal_x[variable].values
    variable_y = df_cal_y[variable].values

    b, a = np.polyfit(variable_y, variable_x, 1)
    variable_y_validation_new = a + b*df_y_validation[variable].values
    df_y_val_new = pd.DataFrame({variable:variable_y_validation_new,'time':pd.to_datetime(df_y_validation['time'].values)})
    #print("Coefficient a = ",a)
    #print("Coefficient b = ",b)
    return df_y_val_new,b,a

def ECDF(x):
    """
    # Empirical CDF computation
    :param x: data
    :return: bins, quantiles
    """
    bins = np.sort(x)
    quantiles = np.arange(1, len(bins)+1)/len(bins)
    return bins, quantiles


def DeltaCalibration(df_calibration_obs:pd.DataFrame, df_calibration_model:pd.DataFrame,df_validation_model:pd.DataFrame,variable):

    DeltaFactor = df_calibration_obs[variable].values.mean() - df_calibration_model[variable].values.mean()
    df_y_val_new = df_validation_model[variable].values + DeltaFactor
    #print("Delta factor = ",DeltaFactor)
    #print("Delta Factor = ",DeltaFactor)
    return pd.DataFrame({variable:df_y_val_new})

def QM_Calibration(df_insitu:pd.DataFrame, df_sat:pd.DataFrame,variable,n):
    """
    Tecnica del Quantile Mapping: vengono utilizzati 10 quantili ma si può cambiare il numero di quantili 'q'.
    All'interno di questa function viene fatta in automatico lo splitting  del dataframe in "calibration" e "validation".

    :param df_insitu: dataframe in-situ
    :param df_sat: dataframe satellite
    :param variable: nome della variabile
    :param n: grado del polinomio interpolante utillizzato nell'interpolazione della funzione X
    :return:
    """
    df_insitu['orig_index'] = df_insitu.index.values
    df_sat['orig_index'] = df_sat.index.values
    q = np.linspace(0, 1, 10)
    rank_q = np.linspace(1, 10, 10)

    df_cal_insitu, df_cal_sat, df_val_insitu, df_val_sat = Split_dataframe_calibration(df_insitu, df_sat, variable)

    original_index = []
    values_array = []
    all_coeff_p = []

    values_insitu = []
    index_insitu = []
    # creazione degli array
    array_cal_insitu = np.array(df_cal_insitu[variable].values)
    array_cal_sat = np.array(df_cal_sat[variable].values)
    array_val_sat = np.array(df_val_sat[variable].values)
    array_val_insitu = np.array(df_val_insitu[variable].values)
    # creazione dei quantili
    quantiles_insitu = np.quantile(array_cal_insitu, q=q)
    quantiles_sat = np.quantile(array_cal_sat, q=q)
    quantiles_val_sat = np.quantile(array_val_sat, q=q)
    quantiles_val_insitu = np.quantile(array_val_insitu, q=q)
    # creazione dell'etichette dei quantili
    quantile_labels_insitu = np.digitize(array_cal_insitu, quantiles_insitu)
    quantile_labels_sat = np.digitize(array_cal_sat, quantiles_sat)
    quantile_labels_sat_val = np.digitize(array_val_sat, quantiles_val_sat)
    quantile_labels_insitu_val = np.digitize(array_val_insitu, quantiles_val_insitu)
    # creazione delle copie dei dataframe originali
    df_cal_insitu_copy = df_cal_insitu.copy()
    df_cal_sat_copy = df_cal_sat.copy()
    df_val_sat_copy = df_val_sat.copy()
    df_val_insitu_copy = df_val_insitu.copy()
    # creo una nuova colonna nel dataframe a cui associo le label che ho creato prima
    df_cal_insitu_copy['quantile'] = quantile_labels_insitu
    df_cal_sat_copy['quantile'] = quantile_labels_sat
    df_val_sat_copy['quantile'] = quantile_labels_sat_val
    df_val_insitu_copy['quantile'] = quantile_labels_insitu_val

    original_index = []
    values_array = []
    all_coeff_p = []
    values_insitu = []
    index_insitu = []
    # con questo ciclo vado ad operare la tecnica QM ad ogni quantile 'q'
    for i in rank_q:
        values_q_to_correct = df_val_sat_copy.loc[df_val_sat_copy['quantile'] == i]
        Data_obs = df_cal_insitu_copy.loc[df_cal_insitu_copy['quantile'] == i]
        Data_mod = df_cal_sat_copy.loc[df_cal_sat_copy['quantile'] == i]
        Data_obs_val = df_val_insitu_copy.loc[df_val_insitu_copy['quantile'] == i]
        # determino la ECDF per i dataframe "calibration" e "validation"
        sorted_cal_insitu, cdf_cal_insitu = ECDF(Data_obs.loc[:, variable])
        sorted_cal_sat, cdf_cal_sat = ECDF(Data_mod.loc[:, variable])
        sorted_val_sat, cdf_val_sat = ECDF(values_q_to_correct.loc[:, variable])
        # funzione interpolante tra la CDF del dataframe_insitu e CDF del dataframe_satellite
        bias_corrected = np.interp(x=cdf_cal_insitu, xp=cdf_cal_sat, fp=sorted_cal_sat)
        # la funzione X sarebbe l'inversa della CDF
        X_q = sorted_cal_insitu - np.sort(bias_corrected)
        # trovo i coefficienti del polinomio interpolante che andrò ad utilizzare tramite un 'polyval' con i valori da
        # correggere
        coeff_p = np.polyfit(bias_corrected, X_q, deg=n)
        values_corrected = np.polyval(coeff_p, values_q_to_correct.loc[:, variable]) + values_q_to_correct.loc[:,variable]
        for value, index in zip(values_corrected, values_q_to_correct.loc[:, 'orig_index']):
            values_array.append(value)
            original_index.append(index)
        for value, index in zip(Data_obs_val.loc[:, variable], Data_obs_val.loc[:, 'orig_index']):
            values_insitu.append(value)
            index_insitu.append(index)
    # alla fine mi salvo i dati che sono stati corretti tramite la QM e creo un dataframe finale con gli indici originali
    df_corrected = pd.DataFrame({variable: values_array, 'orig_index': original_index}).sort_values(by='orig_index')
    df_insitu_corrected = pd.DataFrame({variable: values_insitu, 'orig_index': index_insitu}).sort_values(by='orig_index')

    index_negative = df_corrected[df_corrected[variable]<0].index.values
    df_corrected = df_corrected.drop(index_negative)
    df_insitu_corrected = df_insitu_corrected.drop(index_negative)

    return df_corrected,df_insitu_corrected


def FDM_correction(df_insitu:pd.DataFrame, df_sat:pd.DataFrame,variable,n):
    """
    Tecnica della FDM: in maniera similare a quanto fatto nella QM, solo che la parte di interpolazione dei dati viene
    fatta a livello della CDF globale e non per ogni CDF relativa al quantile 'q'

    :param df_insitu:
    :param df_sat:
    :param variable:
    :param n: grado del polinomio interpolante utillizzato nell'interpolazione della funzione X
    :return:
    """

    df_insitu['orig_index'] = df_insitu.index.values
    df_sat['orig_index'] = df_sat.index.values

    df_cal_insitu, df_cal_sat, df_val_insitu, df_val_sat = Split_dataframe_calibration(df_insitu, df_sat, variable)

    sorted_cal_insitu_FM, cdf_cal_insitu_FM = ECDF(df_cal_insitu.loc[:, variable])
    sorted_cal_sat_FM, cdf_cal_sat_FM = ECDF(df_cal_sat.loc[:, variable])
    sorted_val_sat_FM, cdf_val_sat_FM = ECDF(df_val_sat.loc[:, variable])

    bias_corrected_FM = np.interp(x=cdf_cal_insitu_FM, xp=cdf_cal_sat_FM, fp=sorted_cal_sat_FM)
    X_q_FM = sorted_cal_insitu_FM - bias_corrected_FM
    coeff_p_FM = np.polyfit(bias_corrected_FM, X_q_FM, deg=n)
    values_corrected_FM = np.polyval(coeff_p_FM, df_val_sat.loc[:, variable]) + df_val_sat.loc[:, variable]

    df_corrected_FM = pd.DataFrame({variable: values_corrected_FM, 'orig_index': df_val_sat.loc[:, 'orig_index']}).sort_values(by='orig_index')

    index_negative = df_corrected_FM[df_corrected_FM[variable]<0].index.values
    df_corrected_FM = df_corrected_FM.drop(index_negative)
    df_val_insitu = df_val_insitu.drop(index_negative)

    return df_corrected_FM,df_val_insitu
