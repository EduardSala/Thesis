import numpy as np
import pandas as pd
from scipy import interpolate
import filtering2 as flt
import matplotlib.pyplot as plt
from pypalettes import load_cmap
from cmethods import adjust
from cmethods import utils
from scipy.interpolate import UnivariateSpline
import scipy.stats as stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import floating_axes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
from scipy.stats import pearsonr
import warnings
import seaborn as sns
from openpyxl import load_workbook
import os
from scipy.stats import gaussian_kde

# Ignora il RankWarning per tutto il programma
warnings.simplefilter('ignore', np.RankWarning)
cmap = load_cmap("Bodianus_rufus")

# Global plot settings
#plt.rcParams['text.usetex'] = True
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
#plt.rcParams['figure.dpi'] = 400
plt.rcParams["font.family"] = "Times New Roman"

fp_insitu_hs = flt.pass_filepath(r"")
fp_sat_hs = flt.pass_filepath(r"")
fp_insitu_wspd = flt.pass_filepath(r"")
fp_sat_wspd = flt.pass_filepath(r"")

variable = 'VAVH'
#variable = 'WIND_SPEED'

sns.set_theme()
sns.set_style({'font.family':'sans-serif','font.sans-serif':'Arial'})

stats_array = []
sat_array = []
a_array = []
b_array = []

df_sat_ALL_var = []
df_sat_ALL_time = []
df_insitu_ALL_var = []
df_insitu_ALL_time = []

iterazione = 0
# ciclo in  cui vado ad applicare le varie BC ad ogni dataset satellitare
for filepath_sat,filepath_insitu,iteration in zip(fp_sat_hs,fp_insitu_hs,range(1, 10)):

    dpi = 400
    fig, ax = plt.subplots(figsize=(13, 13), dpi=dpi)
    nome_sat = filepath_sat.split('\\')[6].split('_')[0]
    df_sat = pd.read_csv(filepath_sat,index_col=0)
    df_insitu = pd.read_csv(filepath_insitu,index_col=0)
    print("Il satellite è: ", nome_sat)

    # questo ciclo for viene fatto siccome vado a raccogliere i dati di tutti i dataset satellitari in unico dataset finale
    # i.e 'Ensemble'
    file_path = r"C:\Users\eduar\Desktop\file_finale_HS.xlsx"
    for var_sat,t_sat,var_insitu,t_insitu in zip(df_sat[variable].values,df_sat['time'].values,df_insitu[variable].values,df_insitu['time'].values):
        df_sat_ALL_var.append(var_sat)
        df_sat_ALL_time.append(t_sat)
        df_insitu_ALL_var.append(var_insitu)
        df_insitu_ALL_time.append(t_insitu)

    # vado ad togliere i dati con valore nullo
    indici_insitu = df_insitu[df_insitu[variable]<=0].index.values
    df_insitu = df_insitu.drop(indici_insitu)
    df_sat = df_sat.drop(indici_insitu)

    #max = np.max([df_sat[variable],df_insitu[variable]])
    # inserisco una colonna per identificare l'indice originale di partenza, che poi utilizzerò per fare il sorting originale (dopo le BC)
    df_insitu['orig_index'] = df_insitu.index.values
    df_sat['orig_index'] = df_sat.index.values
    q = np.linspace(0,1,10)
    rank_q = np.linspace(1,10,10)

    x = df_insitu[variable]
    y = df_sat[variable]
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    z = kde(xy)

    # ------------------------------------------------------------------------------------------------------------------

    # grafici Seaborn style
    """
    sns.set_style({'font.family':'sans-serif','font.sans-serif':'Arial'})
    sns.regplot(x=x, y=y, scatter=False, color='navy', line_kws={"linewidth": 3})
    #ax.text(0.05, 0.95, '30-km cross-radius', transform=plt.gca().transAxes,verticalalignment='top',fontsize=30,fontweight='bold')
    #sns.histplot(x=x, y=y, bins=20, pthresh=.1, cmap="hot",color=z)
    #sns.kdeplot(x=x, y=y, levels=10, color="w", linewidths=1)
    """

    # ------------------------------------------------------------------------------------------------------------------

    # applicazione delle 4 tecniche di Bias Correction
    df_cal_insitu, df_cal_sat, df_val_insitu, df_val_sat = flt.Split_dataframe_calibration(df_insitu, df_sat, variable)
    df_QM_validated_sat,df_insitu_corrected = flt.QM_Calibration(df_insitu,df_sat,variable,n=3)
    df_DeltaFactor_sat = flt.DeltaCalibration(df_cal_insitu,df_cal_sat,df_val_sat,variable)
    df_LR_sat,b,a = flt.LinearCalibration(df_cal_insitu,df_cal_sat,df_val_sat,variable)
    df_FDM_corrected,df_FDM_insitu = flt.FDM_correction(df_insitu,df_sat,variable,3)

    # Differente metologia di plotting dei dati
    #df_DeltaFactor_sat[variable].plot(kind='kde', label='Linear Calibration',color='black')
    #df_FDM_corrected[variable].plot(kind='kde', label='FDM', color='green')
    #df_QM_validated_sat[variable].plot(kind='kde', label='QM', color='grey')
    #df_val_sat[variable].plot(kind='kde', label='Satellite',color='navy')
    #df_QM_validated_sat[variable].plot(kind='kde',label='QM',color='red',linestyle='--')

    # ------------------------------------------------------------------------------------------------------------------

    # calcolo dei coefficienti statistici per ogni tecnica
    """
    bias = []
    rmse = []
    cc = []
    si = []
    bias.append(flt.Describe_dataframe(df_val_insitu,df_val_sat,variable,"No calibration")[0])
    rmse.append(flt.Describe_dataframe(df_val_insitu,df_val_sat,variable,"No calibration")[1])
    cc.append(flt.Describe_dataframe(df_val_insitu,df_val_sat,variable,"No calibration")[2])
    si.append(flt.Describe_dataframe(df_val_insitu,df_val_sat,variable,"No calibration")[3])

    bias.append(flt.Describe_dataframe(df_val_insitu,df_DeltaFactor_sat,variable,"Delta")[0])
    rmse.append(flt.Describe_dataframe(df_val_insitu,df_DeltaFactor_sat,variable,"Delta")[1])
    cc.append(flt.Describe_dataframe(df_val_insitu,df_DeltaFactor_sat,variable,"Delta")[2])
    si.append(flt.Describe_dataframe(df_val_insitu,df_DeltaFactor_sat,variable,"Delta")[3])

    bias.append(flt.Describe_dataframe(df_val_insitu, df_LR_sat, variable, "Linear")[0])
    rmse.append(flt.Describe_dataframe(df_val_insitu, df_LR_sat, variable, "Linear")[1])
    cc.append(flt.Describe_dataframe(df_val_insitu, df_LR_sat, variable, "Linear")[2])
    si.append(flt.Describe_dataframe(df_val_insitu, df_LR_sat, variable, "Linear")[3])

    bias.append(flt.Describe_dataframe(df_FDM_insitu, df_FDM_corrected, variable, "FDM")[0])
    rmse.append(flt.Describe_dataframe(df_FDM_insitu, df_FDM_corrected, variable, "FDM")[1])
    cc.append(flt.Describe_dataframe(df_FDM_insitu, df_FDM_corrected, variable, "FDM")[2])
    si.append(flt.Describe_dataframe(df_FDM_insitu, df_FDM_corrected, variable, "FDM")[3])

    bias.append(flt.Describe_dataframe(df_insitu_corrected, df_QM_validated_sat, variable, "QM")[0])
    rmse.append(flt.Describe_dataframe(df_insitu_corrected, df_QM_validated_sat, variable, "QM")[1])
    cc.append(flt.Describe_dataframe(df_insitu_corrected, df_QM_validated_sat, variable, "QM")[2])
    si.append(flt.Describe_dataframe(df_insitu_corrected, df_QM_validated_sat, variable, "QM")[3])

    #flt.Describe_dataframe(df_val_insitu,df_val_sat,variable,"No calibration")
    #flt.Describe_dataframe(df_val_insitu,df_DeltaFactor_sat,variable,"Delta")
    #flt.Describe_dataframe(df_val_insitu, df_LR_sat, variable, "Linear")
    #flt.Describe_dataframe(df_FDM_insitu, df_FDM_corrected, variable, "FDM")
    #flt.Describe_dataframe(df_insitu_corrected, df_QM_validated_sat, variable, "QM")
    """

    # ------------------------------------------------------------------------------------------------------------------

    sheet_name = f"Iteration_{nome_sat}"
    final_df = pd.DataFrame({'bias':bias,'rmse':rmse,'cc':cc,'si':si},index=["No calibration","Delta","Linear","FDM","QM "])
    if iteration == 1 and not os.path.exists(file_path):
        final_df.to_excel(file_path, sheet_name=sheet_name, index=True)
    else:
        # Altrimenti, aggiungi il nuovo DataFrame come un nuovo foglio
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a") as writer:
            final_df.to_excel(writer, sheet_name=sheet_name, index=True)
    # mi salva in un foglio Excel i coefficienti statistici calcolati per ogni tecnica di correzzione

    # ------------------------------------------------------------------------------------------------------------------

    #df_val_insitu = df_val_insitu.sort_values(by=variable)
    #df_val_sat = df_val_sat.sort_values(by=variable)
    #df_LR_sat = df_LR_sat.sort_values(by=variable)
    #df_DeltaFactor_sat = df_DeltaFactor_sat.sort_values(by=variable)
    #df_FDM_corrected = df_FDM_corrected.sort_values(by=variable)
    #df_FDM_insitu = df_FDM_insitu.sort_values(by=variable)
    #df_QM_validated_sat = df_QM_validated_sat.sort_values(by=variable)
    #df_insitu_corrected = df_insitu_corrected.sort_values(by=variable)

    # ------------------------------------------------------------------------------------------------------------------

    # vari grafici che ho realizzato
    """
    #ax.plot(pd.to_datetime(df_val_insitu['time']),df_val_insitu[variable],color='blue',marker='d',linewidth=1.2,markersize=3)
    #ax.plot(pd.to_datetime(df_val_sat['time']), df_val_sat[variable], color='red', marker='x', linewidth=1.2, markersize=3)
    #ax.plot(pd.to_datetime(df_LR_sat['time']), df_LR_sat[variable], color='lightgrey', marker='s', linewidth=1.2,
            #markersize=3)
    #ax.legend(fontsize=26)
    #sns.scatterplot(x=df_val_insitu[variable].values, y=df_val_sat[variable].values, s=150,color='navy')
    #sns.scatterplot(x=df_insitu_corrected[variable].values, y=df_QM_validated_sat[variable].values, s=150, color='green')
    #sns.scatterplot(x=df_val_insitu[variable].values, y=df_LR_sat[variable].values, s=150, color='grey')
    #ax.scatter(df_val_insitu[variable],df_LR_sat[variable])
    #sns.scatterplot(x=np.quantile(df_val_insitu[variable].values, q=q), y=q, color='purple', s=1200,marker='d')

    #sns.scatterplot(x=df_val_insitu[variable].values[::50], y=df_val_sat[variable].values[::50], s=250, color='navy',linewidth=3)
    #sns.scatterplot(x=df_val_insitu[variable].values[::10], y=df_DeltaFactor_sat[variable].values[::10], s=250, color='green')
    #sns.scatterplot(x=df_val_insitu[variable].values[::10], y=df_LR_sat[variable].values[::10], s=250, color='grey')
    #sns.scatterplot(x=df_insitu_corrected[variable].values[::50], y=df_QM_validated_sat[variable].values[::50], s=250, color='red')
    #sns.scatterplot(x=df_FDM_insitu[variable].values[::50], y=df_FDM_corrected[variable].values[::50], s=250, color='yellow')

    #sns.lineplot(x=df_val_insitu[variable].values[::50],y=df_val_sat[variable].values[::50],color='navy', linewidth=3)
    #sns.lineplot(x=df_insitu_corrected[variable].values[::50],y=df_QM_validated_sat[variable].values[::50],color='red', linewidth=3)
    #sns.lineplot(x=df_FDM_insitu[variable].values[::50],y=df_FDM_corrected[variable].values[::50],color='yellow', linewidth=3)

    sns.ecdfplot(x=df_val_insitu[variable].values,color='navy',linewidth=5)
    sns.ecdfplot(x=df_val_sat[variable].values, color='red', linewidth=5)
    sns.ecdfplot(x=df_QM_validated_sat[variable].values, color='green', linewidth=5)
    #sns.ecdfplot(x=df_FDM_corrected[variable].values, color='black', linewidth=5)
    sns.scatterplot(x=np.quantile(df_val_insitu[variable].values, q=q),y=q, color='purple',s=1000,marker='d')

    #sns.regplot(x=df_val_insitu[variable].values, y=df_val_sat[variable].values, scatter=False,color='navy', line_kws={"linewidth": 4.5},scatter_kws={"edgecolor": "white",'s':100},)
    #sns.regplot(x=df_insitu_corrected[variable].values, y=df_QM_validated_sat[variable].values, scatter=False, color='green',line_kws={"linewidth": 4.5},scatter_kws={"edgecolor": "white",'s':100})
    #sns.regplot(x=df_val_insitu[variable].values, y=df_LR_sat[variable].values, scatter=False, color='grey',line_kws={"linewidth": 4.5},scatter_kws={"edgecolor": "white",'s':100})

    #ax.scatter([], [], color='navy', label='No calibration', s=100)
    #ax.scatter([], [], color='green', label='QM technique', s=100)
    #ax.scatter([], [], color='grey', label='Linear regression calibration', s=100)

    ax.plot([], [], color='navy', label='Insitu data',linewidth=7)
    ax.plot([], [], color='red', label='Satellite data', linewidth=7)
    ax.plot([], [], color='green', label='Satellite data corrected via QM', linewidth=7)
    #ax.scatter([], [], color='green', label='Delta technique',s=150)
    #ax.scatter([], [], color='grey', label='Linear calibration', s=150)
    #ax.scatter([], [], color='red', label='Satellite data corrected via QM', s=150)
    #ax.scatter([], [], color='yellow', label='Satellite data corrected via FDM', s=150)
    ax.scatter([],[],color='purple',label='Quantiles $q_j$',s=150,marker='d')
    ax.legend(fontsize=26,markerscale=2)
    """

    # ------------------------------------------------------------------------------------------------------------------

    """
    df_val_insitu = df_val_insitu.sort_values(by=variable)
    df_val_sat = df_val_sat.sort_values(by=variable)
    df_DeltaFactor_sat = df_DeltaFactor_sat.sort_values(by=variable)
    df_LR_sat = df_LR_sat.sort_values(by=variable)
    df_QM_validated_sat = df_QM_validated_sat.sort_values(by=variable)
    df_FDM_corrected = df_FDM_corrected.sort_values(by=variable)
    df_FDM_insitu = df_FDM_insitu.sort_values(by=variable)
    df_insitu_corrected = df_insitu_corrected.sort_values(by=variable)
    """

    # ------------------------------------------------------------------------------------------------------------------
    """
    max = np.max([df_val_insitu[variable],df_val_sat[variable]])
    #max = np.max([df_val_insitu[variable], df_val_sat[variable],df_LR_sat[variable]])
    fattore_plotting = 1

    axins = inset_axes(ax, width="50%", height="50%", loc="center right") 
    # mi crea uno 'zoom' di una specifica area nel plot

    sns.ecdfplot(x=df_val_insitu[variable].values, color='navy', linewidth=5,ax=axins)
    sns.ecdfplot(x=df_val_sat[variable].values, color='red', linewidth=5,ax=axins)
    sns.ecdfplot(x=df_QM_validated_sat[variable].values, color='green', linewidth=5,ax=axins)
    #sns.ecdfplot(x=df_FDM_corrected[variable].values, color='black', linewidth=5,ax=axins)
    sns.scatterplot(x=np.quantile(df_val_insitu[variable].values, q=q), y=q, color='purple', s=1000, marker='d',ax=axins)
    #np.quantile(df_val_sat[variable].values, q=q)
    #sns.scatterplot(x=df_val_insitu[variable].values,y=df_val_sat[variable].values,s=100, color='navy', ax=axins)
    #sns.scatterplot(x=df_val_insitu[variable].values,y=df_DeltaFactor_sat[variable].values,s=100, color='green', ax=axins)
    #sns.scatterplot(x=df_val_insitu[variable].values,y=df_LR_sat[variable].values,s=100, color='grey', ax=axins)
    #sns.scatterplot(x=df_insitu_corrected[variable].values, y=df_QM_validated_sat[variable].values, s=100, color='red',ax=axins)
    #sns.scatterplot(x=df_FDM_insitu[variable].values, y=df_FDM_corrected[variable].values, s=100, color='purple', ax=axins)
    #sns.scatterplot(x=df_val_insitu[variable].values[::10], y=df_val_sat[variable].values[::10], s=250, color='navy', ax=axins)
    #sns.scatterplot(x=df_val_insitu[variable].values[::10], y=df_DeltaFactor_sat[variable].values[::10], s=250,color='green', ax=axins)
    #sns.scatterplot(x=df_val_insitu[variable].values[::10], y=df_LR_sat[variable].values[::10], s=250, color='grey', ax=axins)
    #sns.scatterplot(x=df_insitu_corrected[variable].values[::10], y=df_QM_validated_sat[variable].values[::10], s=250,color='red', ax=axins)
    #sns.scatterplot(x=df_FDM_insitu[variable].values[::10], y=df_FDM_corrected[variable].values[::10], s=250,color='yellow', ax=axins)
    #axins.plot(np.linspace(0, max), np.linspace(0, max), linestyle='--', linewidth=4.5, color='black', alpha=0.8)
    #sns.regplot(x=df_val_insitu[variable].values,y=df_val_sat[variable].values,scatter=False, color='navy',line_kws={"linewidth": 4.5}, ax=axins)
    #sns.regplot(x=df_val_insitu[variable].values,y=df_DeltaFactor_sat[variable].values,scatter=False, color='green',line_kws={"linewidth": 4.5}, ax=axins)
    #sns.regplot(x=df_val_insitu[variable].values,y=df_LR_sat[variable].values,scatter=False, color='grey',line_kws={"linewidth": 4.5}, ax=axins)

    # ------------------------------------------------------------------------------------------------------------------
    
    axins.set_xlim(2.,3)
    axins.set_ylim(0.6, 0.8)
    #axin.plot(bins_sat, cdf_sat, color='navy', linewidth=3, label='Satellite data')
    #axin.plot(bins_insitu, cdf_insitu, color='red', linewidth=3, label='Insitu data')
    #axin.plot(bins_corrected, cdf_corrected, color='lightgrey', linewidth=3, label='Bias corrected data via QM')
    #axin.plot(bins_corrected_FM, cdf_corrected_FM, color='green', linewidth=3, label='Bias corrected data via FDM')
    #axins.scatter(quantiles_val_sat, q, marker='d', color='navy', edgecolor='yellow', s=200, label='$q_j$ quantiles')
    axins.tick_params(axis='y', labelright=True, right=True)
    axins.tick_params(axis='y', labelleft=False, left=False)
    # axin.set_ylim(0.42, 0.5)
    #axins.grid(True, alpha=0.4)
    #axins.indicate_inset_zoom(axins, edgecolor="black", alpha=0.7)
    axins.tick_params(axis='y', labelright=True, right=True)
    axins.tick_params(axis='y', labelleft=False, left=False)
    axins.tick_params(axis='x', labeltop=True, top=True)
    axins.tick_params(axis='x', labelbottom=False, bottom=False)
    #axins.set_xlabel("$U_{10}$ from in-situ observations [m/s]", fontsize=30)
    axins.set_ylabel('', fontsize=30)
    axins.tick_params(axis='x', labelsize=26)  # Cambia la dimensione dei tick sull'asse X
    axins.tick_params(axis='y', labelsize=26)  # Cambia la dimensione dei tick sull'asse Y
    axins.indicate_inset_zoom(axins, edgecolor="black", alpha=0.7)
    
    #ax.plot(bins_sat,cdf_sat,color='navy',linewidth=1,label='Satellite data')
    #ax.plot(bins_insitu,cdf_insitu,color='red',linewidth=1,label='Insitu data')
    #ax.plot(bins_corrected,cdf_corrected,color='lightgrey',linewidth=1,label='Bias corrected data via QM')
    #ax.plot(bins_corrected_FM, cdf_corrected_FM, color='green', linewidth=1, label='Bias corrected data via FDM')
    #ax.scatter(quantiles_val_sat, q, marker='d', color='navy', edgecolor='yellow', s=120, label='$q_j$ quantiles')
    """

    # ------------------------------------------------------------------------------------------------------------------

    #y2 = df_DeltaFactor_sat[variable].values
    #print(y2)
    #xy2 = np.vstack([x,y2-x])
    #z2 = gaussian_kde(xy2)(xy2)

    #ax.scatter(, , marker='d', color='navy', edgecolor='lightgrey',label='$q_j$ quantiles', s=50)
    #graf1 = ax.scatter(x, y-x, alpha=1, s=150,c=z,cmap='viridis', marker='.')
    #ax.scatter(x, y - x, alpha=1, s=15, color='navy', marker='o')
    #ax.scatter(x, df_DeltaFactor_sat[variable] - x, alpha=1, s=15, color='red', marker='x')
    #ax.scatter(x, df_LR_sat[variable] - x, alpha=1, s=15, color='grey', marker='s')
    #graf2 = ax.scatter(x, y2-x, alpha=1, s=100, c=z2, cmap='viridis', marker='.')

    #cbar = fig.colorbar(graf1)
    #cbar.set_label(label='Density',fontsize=22)
    #ax.scatter(df_val_insitu[variable][::fattore_plotting], df_val_sat[variable][::fattore_plotting], alpha=1, s=40, label='No correction',color=cmap.colors[0], marker='o',facecolors='none')
    #ax.scatter(df_FDM_insitu[variable][::fattore_plotting], df_FDM_corrected[variable][::fattore_plotting], alpha=1, s=40,label='FDM correction', color='red', marker='o',facecolors='none')
    # (1-df_val_sat[variable][::fattore_plotting]/np.max(df_val_sat[variable][::fattore_plotting]))/1.1
    #ax.scatter(df_val_insitu[variable][::fattore_plotting], df_DeltaFactor_sat[variable][::fattore_plotting],alpha=1,s=40 ,label='Delta Technique', color=cmap.colors[2], marker='o',facecolors='none')
    # (1-df_DeltaFactor_sat[variable][::fattore_plotting]/np.max(df_DeltaFactor_sat[variable][::fattore_plotting]))/1.1
    #ax.scatter(df_insitu_corrected[variable][::fattore_plotting], df_QM_validated_sat[variable][::fattore_plotting],alpha=1,s=40 ,label='QM correction', color=cmap.colors[3], marker='o',facecolors='none')
    #ax.scatter(df_val_insitu[variable][::fattore_plotting], df_corrected_FM[variable][::fattore_plotting], alpha=(1 -df_corrected_FM[variable][::fattore_plotting] / np.max(df_corrected_FM[variable][::fattore_plotting])) / 2,s=25, label='FDM correction', color=cmap.colors[4], marker='o')
    #ax.scatter(df_val_insitu[variable][::fattore_plotting], df_LR_sat[variable][::fattore_plotting],alpha=1,s=40 ,label='Linear Regression', color=cmap.colors[4], marker='o',facecolors='none')
    #ax.plot(df_val_insitu[variable],a+b*df_val_insitu[variable],linewidth=2,color='navy')
    #ax.plot(df_val_insitu[variable], a + b * df_val_sat.sort_values(by=variable)[variable], linewidth=2, color='red')
    # (1 - df_LR_sat[variable][::fattore_plotting]/np.max(df_LR_sat[variable][::fattore_plotting]))/1.1
    #ax.scatter(quantiles_val_sat,quantiles_val_sat,marker='d',color='navy',edgecolor='lightgrey',label='$q_j$ quantiles',s=125)
    #ax.plot(np.linspace(0, max ), np.linspace(0, max ),linestyle='--', linewidth=4.5,color='black', alpha=0.8)

    # ------------------------------------------------------------------------------------------------------------------

    ax.set_title('Dataset: ' + nome_sat, fontweight='heavy', loc='center',fontsize=30)
    #ax.grid(True, color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
    #ax.set_xlabel("$H_s$ from in-situ observations [m]", fontsize=30)
    ax.tick_params(axis='x', labelsize=26)  # Cambia la dimensione dei tick sull'asse X
    ax.tick_params(axis='y', labelsize=26)  # Cambia la dimensione dei tick sull'asse Y
    #ax.set_ylabel("Bias [m]", fontsize=30)
    #ax.set_xlabel("$U_{10}$ from in-situ observations [m/s]", fontsize=30)
    ax.set_xlabel("$H_s$ [m]", fontsize=30)
    #ax.set_xlabel("$U_{10}$ [m/s]", fontsize=30)
    ax.set_ylabel("CDF [-]", fontsize=30)
    #ax.set_ylabel("$U_{10}$ from satellite observations [m/s]", fontsize=30)
    #ax.set_ylabel("$H_s$ from satellite observations [m]", fontsize=30)

    #ax.set_ylabel("Bias [m/s]", fontsize=22)
    #ax.set_xlim([0,max + 0.5])
    #ax.set_ylim([0,max + 0.5])
    #ax.set_xlim([4,6])
    #ax.set_ylim([4,6])
    #ax.legend(fontsize=22,markerscale=3)

    # ------------------------------------------------------------------------------------------------------------------

    """
    textstr = "30 km cross-radius"
    font = {'family': 'Times New Roman',  # Cambia il font family
            'color': 'darkred',  # Cambia il colore del testo
            'weight': 'bold',  # Imposta uno stile bold
            'size': 20}  # Imposta la dimensione del font
    ax.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             verticalalignment='top',bbox=dict(boxstyle='square', facecolor='wheat', alpha=0.2),fontdict=font)
    
    """

    # ------------------------------------------------------------------------------------------------------------------

    """
    axin = inset_axes(ax, width="40%", height="40%", loc="lower right")
    fattore_plotting_2 = 2
    axin.scatter(df_val_insitu[variable][::fattore_plotting_2], df_val_sat[variable][::fattore_plotting_2], alpha=1,s=40,label='No correction', color=cmap.colors[0], marker='o')
    axin.scatter(df_val_insitu[variable][::fattore_plotting], df_corrected_FM[variable][::fattore_plotting], alpha=1, s=40,label='FDM correction', color=cmap.colors[1], marker='x')
    axin.scatter(df_val_insitu[variable][::fattore_plotting_2], df_DeltaFactor_sat[variable][::fattore_plotting_2], alpha=1,s=40,label='Delta Technique', color=cmap.colors[2], marker='d')
    axin.scatter(df_val_insitu[variable][::fattore_plotting], df_QM_validated_sat[variable][::fattore_plotting], alpha=1, s=40,label='QM correction', color=cmap.colors[3], marker='s')
    axin.scatter(df_val_insitu[variable][::fattore_plotting_2], df_LR_sat[variable][::fattore_plotting_2], alpha=1,s=40,label='Linear Regression', color=cmap.colors[4], marker='p')
    axin.plot(np.linspace(0, max + 0.5), np.linspace(0, max + 0.5), linestyle='--', linewidth=2, color='black',alpha=0.7)

    #axin.plot(df_val_insitu[variable][::fattore_plotting], df_val_sat[variable][::fattore_plotting], alpha=1,label='No correction', color=cmap.colors[0], marker='o')
    #axin.plot(df_val_insitu[variable][::fattore_plotting], df_corrected_FM[variable][::fattore_plotting], alpha=1, label='FDM correction', color=cmap.colors[1], marker='x')
    #axin.plot(df_val_insitu[variable][::fattore_plotting], df_DeltaFactor_sat[variable][::fattore_plotting], alpha=1,label='Delta Technique', color=cmap.colors[2], marker='d')
    #axin.plot(df_val_insitu[variable][::fattore_plotting], df_QM_validated_sat[variable][::fattore_plotting], alpha=1, label='QM correction', color=cmap.colors[3], marker='s')
    #axin.plot(df_val_insitu[variable][::fattore_plotting], df_LR_sat[variable][::fattore_plotting], alpha=1,label='Linear Regression', color=cmap.colors[4], marker='p')
    #axin.plot(np.linspace(0, max + 0.5), np.linspace(0, max + 0.5), linestyle='--', linewidth=2, color='black', alpha=0.7)
    
    #axin.plot(bins_sat, cdf_sat, color='navy', linewidth=2, label='Satellite data')
    #axin.plot(bins_insitu, cdf_insitu, color='red', linewidth=2, label='Insitu data')
    #axin.plot(bins_corrected, cdf_corrected, color='lightgrey', linewidth=2, label='Bias corrected data')
    #axin.plot(bins_corrected_FM, cdf_corrected_FM, color='green', linewidth=2, label='Bias corrected data via FDM')
    #axin.scatter(quantiles_val_sat, q, marker='d', color='navy', edgecolor='yellow', s=120, label='$q_j$ quantiles')
    
    axin.set_xlim(5, 10)
    axin.set_ylim(5, 10)
    axin.tick_params(axis='y', labelright=True, right=True)
    axin.tick_params(axis='y', labelleft=False, left=False)
    axin.tick_params(axis='x', labeltop=True, top=True)
    axin.tick_params(axis='x', labelbottom=False, bottom=False)
    #axin.set_ylim(0.42, 0.5)
    axin.grid(True, alpha=0.4)
    axin.indicate_inset_zoom(axin, edgecolor="black", alpha=0.7)
    
plt.show()
"""

# ------------------------------------------------------------------------------------------------------------------

# in questa sezione unisco tutti i dataset in unico dataset e applico le varie BC
"""
df_sat_ALL = pd.DataFrame({variable:df_sat_ALL_var,'time':df_sat_ALL_time})
df_insitu_ALL = pd.DataFrame({variable:df_insitu_ALL_var,'time':df_insitu_ALL_time})

indici_insitu = df_insitu_ALL[df_insitu_ALL[variable]<=0].index.values
df_insitu_ALL = df_insitu_ALL.drop(indici_insitu)
df_sat_ALL = df_sat_ALL.drop(indici_insitu)

df_cal_insitu_all,df_cal_sat_all,df_val_insitu_all,df_val_sat_all = flt.Split_dataframe_calibration(df_insitu_ALL,df_sat_ALL,variable)
df_LR_sat,b,a = flt.LinearCalibration(df_cal_insitu_all,df_cal_sat_all,df_val_sat_all,variable)
df_DeltaFactor_sat = flt.DeltaCalibration(df_cal_insitu_all,df_cal_sat_all,df_val_sat_all,variable)
df_QM_validated_sat,df_QM_insitu = flt.QM_Calibration(df_insitu_ALL,df_sat_ALL,variable,3)
df_FDM_sat,df_FDM_insitu = flt.FDM_correction(df_insitu_ALL,df_sat_ALL,variable,3)

flt.Describe_dataframe(df_val_insitu_all,df_val_sat_all,variable,"No calibration")
flt.Describe_dataframe(df_val_insitu_all,df_DeltaFactor_sat,variable,"Delta")
flt.Describe_dataframe(df_val_insitu_all, df_LR_sat, variable, "Linear")
flt.Describe_dataframe(df_FDM_insitu, df_FDM_sat, variable, "FDM")
flt.Describe_dataframe(df_QM_insitu, df_QM_validated_sat, variable, "QM")

dpi = 400
df_val_insitu_all = df_val_insitu_all.sort_values(by=variable)
df_val_sat_all = df_val_sat_all.sort_values(by=variable)
df_DeltaFactor_sat = df_DeltaFactor_sat.sort_values(by=variable)
df_QM_validated_sat = df_QM_validated_sat.sort_values(by=variable)
df_QM_insitu = df_QM_insitu.sort_values(by=variable)
df_FDM_sat = df_FDM_sat.sort_values(by=variable)
df_FDM_insitu = df_FDM_insitu.sort_values(by=variable)
df_LR_sat = df_LR_sat.sort_values(by=variable)

# ------------------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 13), dpi=dpi)
max = np.max([df_val_insitu_all[variable],df_val_sat_all[variable]])

ax.scatter(df_val_insitu_all[variable], df_DeltaFactor_sat[variable],alpha=1,s=40 ,label='Delta Technique', color=cmap.colors[0], marker='o',facecolors='none')
ax.scatter(df_QM_insitu[variable], df_QM_validated_sat[variable],alpha=1,s=40 ,label='QM correction', color=cmap.colors[1], marker='o',facecolors='none')
ax.scatter(df_val_insitu_all[variable], df_LR_sat[variable],alpha=1,s=40 ,label='Linear Regression', color=cmap.colors[2], marker='o',facecolors='none')
ax.scatter(df_val_insitu_all[variable], df_val_sat_all[variable],alpha=1,s=40 ,label='No correction', color=cmap.colors[3], marker='o',facecolors='none')
#ax.scatter(df_val_insitu_all[variable], df_val_sat_all[variable],alpha=1,s=10 ,label='No correction', color=cmap.colors[3], marker='o',facecolors='none')
ax.scatter(df_FDM_insitu[variable], df_FDM_sat[variable],alpha=1,s=40 ,label='FDM correction', color=cmap.colors[4], marker='o',facecolors='none')

fattore_plotting = 150
sns.scatterplot(x=df_val_insitu_all[variable].values[::fattore_plotting], y=df_val_sat_all[variable].values[::fattore_plotting], s=250, color='navy',linewidth=3)
sns.scatterplot(x=df_val_insitu_all[variable].values[::fattore_plotting], y=df_DeltaFactor_sat[variable].values[::fattore_plotting], s=250, color='green',linewidth=3)
sns.scatterplot(x=df_val_insitu_all[variable].values[::fattore_plotting], y=df_LR_sat[variable].values[::fattore_plotting], s=250, color='grey',linewidth=3)
sns.scatterplot(x=df_QM_insitu[variable].values[::fattore_plotting], y=df_QM_validated_sat[variable].values[::fattore_plotting], s=250, color='red')
sns.scatterplot(x=df_FDM_insitu[variable].values[::fattore_plotting], y=df_FDM_sat[variable].values[::fattore_plotting], s=250, color='yellow')

sns.lineplot(x=df_val_insitu_all[variable].values[::fattore_plotting],y=df_val_sat_all[variable].values[::fattore_plotting],color='navy', linewidth=3)
sns.lineplot(x=df_val_insitu_all[variable].values[::fattore_plotting],y=df_DeltaFactor_sat[variable].values[::fattore_plotting],color='green', linewidth=3)
sns.lineplot(x=df_val_insitu_all[variable].values[::fattore_plotting],y=df_LR_sat[variable].values[::fattore_plotting],color='grey', linewidth=3)
sns.lineplot(x=df_QM_insitu[variable].values[::fattore_plotting],y=df_QM_validated_sat[variable].values[::fattore_plotting],color='red', linewidth=3)
sns.lineplot(x=df_FDM_insitu[variable].values[::fattore_plotting],y=df_FDM_sat[variable].values[::fattore_plotting],color='yellow', linewidth=3)

ax.scatter([], [], color='navy', label='No correction',s=150)
ax.scatter([], [], color='green', label='Delta technique',s=150)
ax.scatter([], [], color='grey', label='Linear calibration', s=150)
ax.scatter([], [], color='red', label='Satellite data corrected via QM', s=150)
ax.scatter([], [], color='yellow', label='Satellite data corrected via FDM', s=150)
    #ax.scatter([],[],color='purple',label='Quantiles $q_j$',s=150,marker='d')

ax.plot(np.linspace(0, max+1), np.linspace(0, max+1),
            linestyle='--', linewidth=3,
            color='black', alpha=0.8)
ax.set_title("Ensemble Dataset ", fontweight='heavy', loc='center',fontsize=28)
#ax.grid(True, color='gray', linestyle='-', linewidth=0.4, alpha=0.3)
#ax.set_xlabel("Hs from in-situ observations [m]", fontsize=30)
#ax.set_ylabel("Hs from satellite observations [m]", fontsize=30)
ax.set_xlabel("$U_{10}$ from in-situ observations [m/s]", fontsize=30)
#ax.set_xlabel("$H_s$ [m/s]", fontsize=22)
#ax.set_ylabel("CDF [-]", fontsize=22)
ax.tick_params(axis='x', labelsize=26)  # Cambia la dimensione dei tick sull'asse X
ax.tick_params(axis='y', labelsize=26)  # Cambia la dimensione dei tick sull'asse Y
ax.set_ylabel("$U_{10}$ from satellite observations [m/s]", fontsize=30)
#ax.set_ylabel("Bias [m/s]", fontsize=22)
#ax.set_xlim([0,max + 0.5])
#ax.set_ylim([0,max + 0.5])
ax.legend(fontsize=26,markerscale=2)

# ------------------------------------------------------------------------------------------------------------------

axin = inset_axes(ax, width="40%", height="40%", loc="lower right")
axin.set_xlim(10,15)
axin.set_ylim(10, 15)
axin.tick_params(axis='y', labelright=True, right=True)
axin.tick_params(axis='y', labelleft=False, left=False)
axin.tick_params(axis='x', labeltop=True, top=True)
axin.tick_params(axis='x', labelbottom=False, bottom=False)
#axin.set_ylim(0.42, 0.5)
axin.grid(True, alpha=0.4)
axin.indicate_inset_zoom(axin, edgecolor="black", alpha=0.7)

sns.scatterplot(x=df_val_insitu_all[variable].values[::fattore_plotting], y=df_val_sat_all[variable].values[::fattore_plotting], s=250, color='navy',linewidth=3,ax=axin)
sns.scatterplot(x=df_val_insitu_all[variable].values[::fattore_plotting], y=df_DeltaFactor_sat[variable].values[::fattore_plotting], s=250, color='green',linewidth=3,ax=axin)
sns.scatterplot(x=df_val_insitu_all[variable].values[::fattore_plotting], y=df_LR_sat[variable].values[::fattore_plotting], s=250, color='grey',linewidth=3,ax=axin)
sns.scatterplot(x=df_QM_insitu[variable].values[::fattore_plotting], y=df_QM_validated_sat[variable].values[::fattore_plotting], s=250, color='red',ax=axin)
sns.scatterplot(x=df_FDM_insitu[variable].values[::fattore_plotting], y=df_FDM_sat[variable].values[::fattore_plotting], s=250, color='yellow',ax=axin)

sns.lineplot(x=df_val_insitu_all[variable].values[::fattore_plotting],y=df_val_sat_all[variable].values[::fattore_plotting],color='navy', linewidth=3,ax=axin)
sns.lineplot(x=df_val_insitu_all[variable].values[::fattore_plotting],y=df_DeltaFactor_sat[variable].values[::fattore_plotting],color='green', linewidth=3,ax=axin)
sns.lineplot(x=df_val_insitu_all[variable].values[::fattore_plotting],y=df_LR_sat[variable].values[::fattore_plotting],color='grey', linewidth=3,ax=axin)
sns.lineplot(x=df_QM_insitu[variable].values[::fattore_plotting],y=df_QM_validated_sat[variable].values[::fattore_plotting],color='red', linewidth=3,ax=axin)
sns.lineplot(x=df_FDM_insitu[variable].values[::fattore_plotting],y=df_FDM_sat[variable].values[::fattore_plotting],color='yellow', linewidth=3,ax=axin)
axin.tick_params(axis='x', labelsize=26)  # Cambia la dimensione dei tick sull'asse X
axin.tick_params(axis='y', labelsize=26)
axin.plot(np.linspace(0, max+1), np.linspace(0, max+1),
            linestyle='--', linewidth=3,
            color='black', alpha=0.8)

#fig.savefig(r"C:Users\eduar\Desktop\screen Tesi\prova"+ "\\" + "ensemble_wspd"+ ".png",dpi=dpi)

#plt.show()
"""