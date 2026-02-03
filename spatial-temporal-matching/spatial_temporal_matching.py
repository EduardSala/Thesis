import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import linear_model
import sklearn.metrics as metrics
import statistics
import scipy
import seaborn as  sns
from modules import Module_all_functions as flt
from scipy.stats import gaussian_kde

plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26

#fp_platf = flt.pass_filepath(r"platfCSV_hs")
fp_sat = "../datasets/ds_S6A_wave.csv"
dataframe_test = flt.save_Sat_data(fp_sat,"VAVH")
start='2021-01-01'
end='2024-01-01'

time_arr = np.array(pd.date_range(start=start, end=end, freq='30min').to_pydatetime())

#variable = 'VAVH'
variable = 'WIND_SPEED'
counter = 0
START = 0
cross_radius = [0.5]
sns.set_theme()

for cross_r in cross_radius:
    nome_sat_array = []
    rmse_array = []
    bias_array = []
    cc_array = []
    si_array = []
    rad_cross = []
    time_cross = []
    number_data = []

    for fp_sat in flt.pass_filepath(r""):
        sat = flt.save_Sat_data(fp_sat, variable)
        nome = fp_sat.split(sep='\\')[5].split(sep='.')[0]
        time_sat = []
        var_sat = []
        time_p = []
        var_p = []
        for fp in fp_platf:
            platf = pd.read_csv(fp, index_col=0)
            name_platf = fp.split(sep='\\')[6].split(sep='_')[0]
            df_sat = flt.filtering_crossover_spatial(sat, platf, cross_r, name_platf, variable)
            if len(df_sat) > 0:
                df_sat2, df_p2 = flt.filtering_crossover_temporal(df_sat, platf, 3600, variable)
                if len(df_sat2) > 0 and len(df_p2) > 0:
                    # df_sat3, df_p3 = flt.minimum_distance(df_sat2, df_p2, variable)
                    df_sat3, df_p3 = flt.LIDW_function(df_sat2, df_p2, 2, variable)
                    s, p = flt.coLocation_temporal(df_sat3, df_p3, variable)

                    for t_sat, t_p, v_sat, v_p in zip(s['time'].values, p['time'].values, s[variable].values,
                                                      p[variable].values):
                        time_sat.append(t_sat)
                        time_p.append(t_p)
                        var_sat.append(v_sat)
                        var_p.append(v_p)
                print(len(s), len(p))
        dic_p = {'time': time_p, variable: var_p}
        dic_s = {'time': time_sat, variable: var_sat}
        df_s = pd.DataFrame(dic_s)
        df_p = pd.DataFrame(dic_p)
        indici_insitu = df_p[df_p[variable] <= 0].index.values
        df_p = df_p.drop(indici_insitu)
        df_s = df_s.drop(indici_insitu)

        outPut_sat = r"" + "\\" + nome + ".csv"
        outPut_insitu = r"" + "\\" + nome + "_insitu" + ".csv"
        df_s.to_csv(outPut_sat)
        df_p.to_csv(outPut_insitu)
        # questi file .csv verranno poi utilizzati nel modulo 'calibration.py' per calibrare i dati satellitari rispetto
        # ai dati delle boe
        """
        
        #  parte dei grafici
        
        x = df_p[variable]
        y = df_s[variable]
        max = np.max([x,y])
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        
        dpi = 400
        fig, ax = plt.subplots(figsize=(13, 13), dpi=dpi)
        
        sns.scatterplot(x=x, y=y, s=125, c=z, cmap='rainbow')
        sns.set_style({'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})
        ax.text(0.05, 0.95, '60-min time-frame', transform=plt.gca().transAxes, verticalalignment='top',
                fontsize=30, fontweight='bold')
        #graf1 = ax.scatter(x, y,alpha=1, s=100, c=z, cmap='rainbow', marker='.')
        #cbar = fig.colorbar(graf1)
        #cbar.set_label(label='Density', fontsize=22)

        ax.plot(np.linspace(0, max + 0.5), np.linspace(0, max + 0.5), linestyle='--', linewidth=4, color='black',
                alpha=0.5)
        ax.set_title('Dataset: ' + nome, fontweight='heavy', loc='center',fontsize=30)
        #ax.grid(True, color='gray', linestyle='-', linewidth=0.4, alpha=0.3)
        #ax.set_xlabel("$H_s$ from in-situ observations [m]", fontsize=30)
        #ax.set_ylabel("$H_s$ from satellite observations [m]", fontsize=30)
        ax.tick_params(axis='x', labelsize=26)  # Cambia la dimensione dei tick sull'asse X
        ax.tick_params(axis='y', labelsize=26)  # Cambia la dimensione dei tick sull'asse Y
        #ax.set_ylabel("Bias [m/s]", fontsize=30)
        ax.set_xlabel("$U_{10}$ from in-situ observations [m/s]", fontsize=30)
        ax.set_ylabel("$U_{10}$ from satellite observations [m/s]", fontsize=30)
        #ax.set_xlim([0, max + 0.5])
        #ax.set_ylim([0, max + 0.5])

        #textstr = "15-min time-frame"
        #font = {'family': 'Times New Roman',  # Cambia il font family
               # 'color': 'darkred',  # Cambia il colore del testo
                #'weight': 'bold',  # Imposta uno stile bold
                #'size': 26}  # Imposta la dimensione del font
        #ax.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
         #       verticalalignment='top', bbox=dict(boxstyle='square', facecolor='wheat', alpha=0.2), fontdict=font)
        fig.savefig(r"" + "\\" + nome + "_wspd_60min.png", dpi=dpi)
        # -------------------------------------------------------------------------------------------------------------
        """

        if len(df_s) > 0 and len(df_p) > 0:
            rmse = flt.RMSE(df_p, df_s, variable)
            bias = flt.Bias(df_p, df_s, variable)
            cc = flt.CC(df_p, df_s, variable)
            si = flt.SI(df_p, df_s, variable)
        else:
            rmse = np.nan
            bias = np.nan
            cc = np.nan
            si = np.nan
        rmse_array.append(rmse)
        bias_array.append(bias)
        cc_array.append(cc)
        si_array.append(si)
        number_data.append(len(df_s))
        nome_sat_array.append(nome)
        rad_cross.append(cross_r)
        time_cross.append(1800)

    dataframe_finale = pd.DataFrame(
        {'name': nome_sat_array, 'rad_cross': rad_cross, 'time_cross': time_cross, 'bias': bias_array,
         'rmse': rmse_array, 'cc': cc_array, 'si': si_array, 'number_data': number_data})
    dataframe_finale.to_csv(r"" + str(counter) + ".csv")
    counter = counter + 1

