from string import printable
import geopy.distance
import netCDF4
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import os 
import shapely as shp
import filtering
import geopandas as gp
import cartopy.mpl.ticker as cticker
import matplotlib.dates as mdates
from cartopy.util import add_cyclic_point
from sklearn.linear_model import  LinearRegression
import statistics
from scipy.stats import norm
import Module_all_functions as flt



# abilito il font Tex per i grafici + grandezza del font = 36
plt.rcParams['text.usetex'] = True
plt.rcParams['xtick.labelsize'] = 36
plt.rcParams['ytick.labelsize'] = 36


path_coastlines = "https://www.ngdc.noaa.gov/mgg/shorelines/gshhs.html"
path_lakes = "https://www.ngdc.noaa.gov/mgg/shorelines/gshhs.html"

fp_sat = r""
fp_p = r""
foldPath_clusterA = r""
foldPath_clusterB = r""
foldPath_clusterC = r""
fp_platf_csv = flt.pass_filepath(r"")
fp_platf_csv_cluster = flt.pass_filepath(r"")
fp_platf = flt.pass_filepath(r"")
fp_platf2 = flt.pass_filepath(r"")


start='2021-01-01'
end='2022-01-01'

time_arr = np.array(pd.date_range(start=start, end=end, freq='60min').to_pydatetime())
timeD = pd.Timedelta(minutes=5)

array_marker =  ["o","v","^","p","*","+","x","1","2","s","P","s","4","H","D","<","h","X"]


# -1
# Plotting del cluster
"""
i = 0
filtering.Mappa()
for fp in fp_platf_csv:
    nome_p = fp.split(sep='\\')[-1].split(sep='.')[0].split(sep='_')[0]
    df = pd.read_csv(fp,index_col=0)
    plt.scatter(df['longitude'],df['latitude'],label=nome_p,marker=array_marker[i])
    i = i + 1
plt.legend()
plt.show()
#---
"""


# -2
# Per ogni cluster, ho preso una boa di riferimento e ho confrontato i valori Min-Max-Mean (nel tempo)
"""
fp_A = r""
p_A = pd.read_csv(fp_A)
nome_pA = fp_A.split(sep='\\')[-1].split(sep='.')[0].split(sep='_')[0]
hs_pA = flt.hs_resampled_time(time_arr,p_A,timeD)
hs_pA = np.array(hs_pA)
hs_pA[hs_pA == 0] = np.nan

fp_B = r""
p_B = pd.read_csv(fp_B)
nome_pB = fp_B.split(sep='\\')[-1].split(sep='.')[0].split(sep='_')[0]
hs_pB = flt.hs_resampled_time(time_arr,p_B,timeD)
hs_pB = np.array(hs_pB)
hs_pB[hs_pB == 0] = np.nan

fp_C = r""
p_C = pd.read_csv(fp_C)
nome_pC = fp_C.split(sep='\\')[-1].split(sep='.')[0].split(sep='_')[0]
hs_pC = flt.hs_resampled_time(time_arr,p_C,timeD)
hs_pC = np.array(hs_pC)
hs_pC[hs_pC == 0] = np.nan


hs_mean_A,hs_min_A,hs_max_A = flt.hs_mean_min_max(foldPath_clusterA,time_arr)
hs_mean_B,hs_min_B,hs_max_B = flt.hs_mean_min_max(foldPath_clusterB,time_arr)
hs_mean_C,hs_min_C,hs_max_C = flt.hs_mean_min_max(foldPath_clusterC,time_arr)


fig,ax = plt.subplots(3,figsize=(14,12),dpi=400)
#ax.plot(time_arr,hs_mean_A ,color='orange',linewidth=2 ,marker='',alpha=0.4)
#ax.plot(time_arr,hs_mean_B ,color='grey',linewidth=2 ,marker='',alpha=0.4)
#ax.plot(time_arr,hs_mean_C ,color='navy',linewidth=2 ,marker='',alpha=0.4)

ax[0].plot(time_arr,hs_mean_A,color='orange',linewidth=1 ,marker='',alpha=0.7,label='Cluster A')
ax[0].plot(time_arr,hs_mean_B,color='navy',linewidth=0.7 ,marker='',alpha=0.4,label='Other clusters')
ax[0].plot(time_arr,hs_mean_C,color='navy',linewidth=0.7 ,marker='',alpha=0.4)

ax[1].plot(time_arr,hs_mean_B,color='orange',linewidth=1 ,marker='',alpha=0.7,label='Cluster B')
ax[1].plot(time_arr,hs_mean_A,color='navy',linewidth=0.7 ,marker='',alpha=0.4,label='Other clusters')
ax[1].plot(time_arr,hs_mean_C,color='navy',linewidth=0.7 ,marker='',alpha=0.4)

ax[2].plot(time_arr,hs_mean_C ,color='orange',linewidth=1 ,marker='',alpha=0.7,label='Cluster C')
ax[2].plot(time_arr,hs_mean_A ,color='navy',linewidth=0.7 ,marker='',alpha=0.4,label='Other clusters')
ax[2].plot(time_arr,hs_mean_B,color='navy',linewidth=0.7 ,marker='',alpha=0.4)

#ax[0].set_title("Cluster A", loc='center', fontsize=20, fontweight='bold')
#ax[1].set_title("Cluster B", loc='center', fontsize=20, fontweight='bold')
#ax[2].set_title("Cluster C", loc='center', fontsize=20, fontweight='bold')

ax[0].legend(fontsize=22,markerscale=2)
ax[1].legend(fontsize=22,markerscale=2)
ax[2].legend(fontsize=22,markerscale=2)
#ax[1].plot(time_arr,hs_mean_B ,color='grey',linewidth=1 ,marker='',alpha=0.4)
#ax[2].plot(time_arr,hs_mean_C,color='grey',linewidth=1 ,marker='',alpha=0.4)

fig.autofmt_xdate()

ax[0].grid(which='minor',linestyle=':',linewidth=0.2,color='#EEEEEE')
ax[0].grid(which='major', linewidth=0.8, color='#DDDDDD')
ax[0].minorticks_on()

ax[1].grid(which='minor',linestyle=':',linewidth=0.2,color='#EEEEEE')
ax[1].grid(which='major', linewidth=0.8, color='#DDDDDD')
ax[1].minorticks_on()

ax[2].grid(which='minor',linestyle=':',linewidth=0.2,color='#EEEEEE')
ax[2].grid(which='major', linewidth=0.8, color='#DDDDDD')
ax[2].minorticks_on()

ax[0].set_ylabel(r"$H_{s} \ [m]$", fontsize=22)
ax[1].set_ylabel(r"$H_{s} \ [m]$", fontsize=22)
ax[2].set_ylabel(r"$H_{s} \ [m]$", fontsize=22)

fig.suptitle(r" Time series of mean $H_{s}$ of every cluster compared to one reference buoy 2021-2022: ", fontweight='bold', fontsize=24)

#plt.show()


"""

# -3
# ho fatto un resampling dei valori in un arco temporale prestabilito (ad esempio volevo plottare  i valori ogni 20-min)
"""
fp1 = r""
fp2 = r""
p1 = pd.read_csv(fp1)
p2 = pd.read_csv(fp2)
nome_p1 = fp1.split(sep='\\')[-1].split(sep='.')[0].split(sep='_')[0]
nome_p2 = fp2.split(sep='\\')[-1].split(sep='.')[0].split(sep='_')[0]

hs_p1 = flt.hs_resampled_time(time_arr,p1,timeD)
hs_p1 = np.array(hs_p1)
hs_p1[hs_p1 == 0] = np.nan
hs_p2 = flt.hs_resampled_time(time_arr,p2,timeD)
hs_p2 = np.array(hs_p2)
hs_p2[hs_p2 == 0] = np.nan

fig,ax = plt.subplots(figsize=(9, 6))
# Plotting del Delta = Hs_1 - Hs_2
ax.plot(time_arr,hs_p1-hs_p2,color='navy',marker='d',markersize=3.5)
#ax.plot(time_arr,hs_p2,linestyle='--',color='black',marker='d',markersize=2.5)
plt.show()
"""


# -4
# Time series plots
"""

fp_platf_csv = flt.pass_filepath(r"")

for fp in fp_platf_csv:
    nome_p = fp.split(sep='\\')[-1].split(sep='.')[0].split(sep='_')[0]
    df = pd.read_csv(fp,index_col=0)
    df['time'] = pd.to_datetime(df['time'],format='mixed')
    time = df['time']
    timeD = pd.Timedelta(minutes=5)
    hs_p = flt.hs_resampled_time(time_arr,df,timeD)
    hs_p = np.array(hs_p)
    hs_p[hs_p == 0] = np.nan
    df = df.drop(0)
    j = j + 1
    
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    ax.plot(time_arr,hs_p , color='orange',linewidth=0.5,marker='o',markersize=0.6)
    #ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(which='minor',linestyle=':',linewidth=0.5,color='#EEEEEE')
    ax.grid(which='major', linewidth=0.8, color='#DDDDDD')
    ax.minorticks_on()
    ax.set_ylabel(r"$H_{s} \ [m]$")
    ax.set_title(r"Time series: " + nome_p, fontweight='bold',fontsize=10)
    #ax.legend(fontsize=10)
    fig.autofmt_xdate()

    #fig.savefig(r"" + "\\" + nome_p, dpi=300)

plt.show()
"""


# -5
# confronto dell'Hs di ogni boa del cluster rispetto alla boa 'centrale' del cluster
"""
coord_from = [p1['latitude'][0],p1['longitude'][0]]
j = 0

array_nome = []
array_dist = []
array_rmse = []
array_lat = []

for fp in fp_platf_csv:
    nome_p = fp.split(sep='\\')[-1].split(sep='.')[0].split(sep='_')[0]
    array_nome.append(nome_p)
    df = pd.read_csv(fp, index_col=0)
    array_lat.append(df['latitude'][0])
    timeD = pd.Timedelta(minutes=30)
    
    hs_p = flt.hs_resampled_time(time_arr, df, timeD)
    hs_p = np.array(hs_p)
    hs_p[hs_p == 0] = np.nan
    punto  = [df['latitude'][0],df['longitude'][0]]
    j = j + 1

    N = len(hs_p)
    N1 = len(hs_p1)
    
    # calcolo dei coefficienti statistici per confrontare i valori tra boa 'centrale' e le altre del gruppo
    bias = np.nansum(np.diff(hs_p - hs_p1)) / N
    rmse = np.sqrt(np.nansum(np.square(np.diff(hs_p - hs_p1))) / N)
    si = np.nansum(np.square((hs_p - np.nanmean(hs_p)) - (hs_p - np.nanmean(hs_p1)))) / np.nansum(np.square(hs_p1))
    cc = np.nansum(np.multiply((hs_p - np.nanmean(hs_p)), (hs_p1 - np.nanmean(hs_p1)))) / np.sqrt(
        np.nansum(np.square(hs_p - np.nanmean(hs_p))) * np.nansum(np.square(hs_p1 - np.nanmean(hs_p1))))
    textstr = '\n'.join((
        r'$RMSE=%.4f \ [m]$' % (rmse,),
        r'$BIAS=%.4f \ [m]$' % (bias,),
        r'$SI=%.4f$' % (si,),
        r'$CC=%.4f$' % (cc,)))
    distanza = geopy.distance.GeodesicDistance(punto, coord_from).km
    print(distanza)
    array_rmse.append(rmse)
    array_dist.append(distanza)
    
    
    fig, ax = plt.subplots(figsize=(12, 12),dpi=400)
    array_t = pd.to_datetime(df['time'],format='mixed')
    time_days = np.diff(array_t) / pd.Timedelta(days=1)
    # Plot per vedere i 'buchi temporali' tra una misurazione e l'altra
    ax.plot(array_t[1::],time_days,color='navy',linewidth=2.5)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, color='#EEEEEE', alpha=0.8)
    ax.grid(which='major', linewidth=0.8, color='#DDDDDD', alpha=0.6)

    # confronto dell'Hs di ogni boa del cluster rispetto alla boa 'centrale' del cluster
    #ax.plot(time_arr, hs_p - hs_mean, color='navy',linewidth=2,marker='o', markersize=3,linestyle='-')
    #ax.plot(time_arr, hs_p - hs_p1, color='red', linewidth=2, marker='o', markersize=3,linestyle='-')
    #ax.text(0.4, 0.95, textstr, transform=ax.transAxes, fontsize=12,verticalalignment='top',bbox=dict(facecolor='yellow',alpha=0.4))
    ax.minorticks_on()
    ax.set_xlabel(r"Time", fontsize=36)
    ax.set_ylabel(r"$\Delta T$ [days]", fontsize=36)
    ax.set_title(nome_p, fontweight='bold', fontsize=36)
    #ax.axhline(y=1, linestyle='-')
    #ax.axhline(y=-1, linestyle='-')
    #ax.spines["right"].set_visible(False)
    #ax.spines["left"].set_visible(False)
    #ax.spines["top"].set_visible(False)
    #ax.grid(which='minor', linestyle=':', linewidth=0.5, color='#EEEEEE')
    #ax.grid(which='major', linewidth=0.8, color='#DDDDDD')
    #ax.minorticks_on()
    #ax.set_ylabel(r"$\Delta H_{s} \ [m]$", fontsize=16)
    #ax.set_title(r" $\Delta H_{s} \ $ 2021-2023: " + nome_p, fontweight='bold', fontsize=15)
    #ax.legend(fontsize=10)
    fig.autofmt_xdate()
    fig.savefig(r"" + nome_p + ".png")
plt.show()
"""

# -6
# Plotting dei precedenti coefficienti statistici calcolati
"""
fig, ax = plt.subplots(figsize=(12, 12),dpi=400)
datafr = pd.DataFrame({'name':array_nome,'distance':array_dist,'latitude':array_lat,'rmse':array_rmse})
datafr = datafr.sort_values(by='latitude')
datafr = datafr.loc[datafr['distance']!=0]
print(datafr)

ax.scatter(datafr['distance'],datafr['rmse'],s=400,color='orange')
texts = []
for name, x1, y1 in zip(datafr['name'], datafr['distance'], datafr['rmse']):
    texts.append(ax.annotate(name, xy=(x1, y1), fontsize=32))
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray'))


for name,x1,y1 in zip(datafr['name'],datafr['distance'],datafr['rmse']):
    ax.annotate(name,xy=(x1-5,y1),fontsize=30)
    
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='#EEEEEE',alpha=0.8)
ax.grid(which='major', linewidth=0.8, color='#DDDDDD',alpha=0.6)
#ax.spines["right"].set_visible(False)
#ax.spines["top"].set_visible(False)
ax.minorticks_on()
ax.set_xlabel(r"Distance $[km]$",fontsize=30)
ax.set_ylabel(r"RMSE $[m]$",fontsize=30)
ax.set_title(r"Reference buoy: - ", fontweight='bold', fontsize=30)
"""























"""
for d, hs, t in zip(ds_filt['DEPH'].values, ds_filt['VAVH'].values, ds_filt['TIME'].values):
    for d_2, hs_2 in zip(d, hs):
        if d_2 == 0:
            #print(d_2,hs_2,t)
            hs_platf.append(hs_2.item())
            time_platf.append(pd.to_datetime(t))
            break
"""


# VDIR
"""
fold = r"D:\Materiale_Uni\Tesi_Magsitrale\Dataset\Dataset_output_codice\MareNord\Al\1h\Md"
df_s,df_p = flt.read_ALL_data(fold)
N = len(df_s)
hs_p = df_p['hs']
hs_s = df_s['hs']
bias = flt.Bias(df_s,df_p)
rmse = flt.RMSE(df_s,df_p)
si = flt.SI(df_s,df_p)
cc = flt.CC(df_s,df_p)
print('Bias: ',bias)
print('RMSE: ',rmse)
print('SI: ',si)
print('CC: ',cc)
"""


"""
N_2021 = []
N_2022 = []
N_2023 = []
N_2024 = []

for fp in flt.pass_filepath(r"D:\Materiale_Uni\Tesi_Magsitrale\Dataset\Dataset_insitu\groupA"):
    ds = xr.load_dataset(fp)
    bo = ds.sel(TIME=slice('2021-01-01', '2024-05-31'))
    gruppi = bo.groupby('TIME.year')
    nome_arr.append(ds.attrs['platform_code'])
    #N_2021.append(len(gruppi[2021]['TIME']))
    #N_2022.append(len(gruppi[2022]['TIME']))
    #N_2023.append(len(gruppi[2023]['TIME']))
    #N_2024.append(len(gruppi[2024]['TIME']))
    N_records.append(len(bo['TIME']))
diz = {'platform_code':nome_arr,'2021':N_2021,'2022':N_2022,'2023':N_2023,'2024':N_2024}
#diz = {'platform_code':nome_arr,}
df = pd.DataFrame(diz)
"""



"""
# calcolo dei collocation points per ogni satellite
nome_arr = []
tot_arr = []
true_arr = []
nan_arr =  []
for fp in flt.pass_filepath(r"D:\Materiale_Uni\Tesi_Magsitrale\Dataset\Dataset_output_codice\MareNord\Al\1h\Md"):
    stringa = fp
    stringa_split = stringa.split(sep="\\", maxsplit=9)[9].split(sep='_')
    stringa_nome_p = stringa_split[0]
    stringa_platf = stringa_split[1].split(sep='.')[0]
    #print(stringa_platf)
    if stringa_platf == 'p':
        p = pd.read_csv(stringa,index_col=0)
        tot_val = p['N_cross'].values[-1]
        true_val = len(p['N_cross'].values)
        nan_val = tot_val - true_val
        nome_arr.append(stringa_nome_p)
        tot_arr.append(tot_val)
        true_arr.append(true_val)
        nan_arr.append(nan_val)

diz = {'platform_code':nome_arr,'tot_val':tot_arr,'true_val':true_arr,'nan_val':nan_arr}
df = pd.DataFrame(diz)
nome_sat = 'Al'
path_ex = r"D:\Materiale_Uni\Tesi_Magsitrale\Dataset\Dataset_output_codice" + '\\' + nome_sat + '_recap_1h.xlsx'
print(path_ex)
df.to_excel(path_ex)
"""


"""
# calcolo distanza dalla costa + distanza tra le piattaforme
d = []
nomi = []
nomi2 = []
for fp in flt.pass_filepath(r"D:\Materiale_Uni\Tesi_Magsitrale\Dataset\Dataset_insitu\platf_A"):
    p = xr.load_dataset(fp)
    #flt.dist_platf_coast(fp)

    min = 10000000
    coord = [p['LATITUDE'].values,p['LONGITUDE'].values]
    name = None
    for fp_2 in flt.pass_filepath(r"D:\Materiale_Uni\Tesi_Magsitrale\Dataset\Dataset_insitu\platf_A"):
        p2 = xr.load_dataset(fp_2)
        coord2 = [p2['LATITUDE'].values,p2['LONGITUDE'].values]
        if p2.attrs['platform_code'] == p.attrs['platform_code']:
            continue
        else:
            dist = geopy.distance.GeodesicDistance(coord2, coord).km
            if dist < min:
                min = dist
                name = p2.attrs['platform_code']
    print(p.attrs['platform_code'], ": la piattaforma più vicina è:", name, " e dista", min, " km")
    nomi2.append(name)
    d.append(min)
    nomi.append(p.attrs['platform_code'])


dict = {'name':nomi,'closest':nomi2,'distance':d}
dataf = pd.DataFrame(dict)

#plt.show()
"""


"""
# creazione dell'excel con le varie piattaforme e le relative informazioni
for fp in flt.pass_filepath(r"D:\Materiale_Uni\Tesi_Magsitrale\Dataset\Dataset_insitu\platf_A"):
    platf = xr.load_dataset(fp).fillna(-1)
    nome_p.append(platf.attrs['platform_code'])
    lon_p.append(platf['LONGITUDE'].values)
    lat_p.append(platf['LATITUDE'].values)
    t_start.append(platf.attrs['time_coverage_start'])
    t_end.append(platf.attrs['time_coverage_end'])

dict = {'platform_code':nome_p,'time_coverage_start':t_start,'time_coverage_end':t_end,'longitude':lon_p,'latitude':lat_p}
df = pd.DataFrame(dict)

"""







