from sklearn.metrics import root_mean_squared_error
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def BIAS(df_mooring,df_sat,variable):

    y = df_sat[variable].values
    x = df_mooring[variable].values
    bias = np.mean(y-x)
    return np.round(bias,5)

def RMSE(df_mooring,df_sat,variable):

    y = df_sat[variable].values
    x = df_mooring[variable].values
    rmse = root_mean_squared_error(y,x)
    return np.round(rmse,5)

def SI(df_mooring,df_sat,variable):

    y = df_sat[variable].values
    x = df_mooring[variable].values
    si = RMSE(df_mooring,df_sat,variable)/np.mean(y)
    return np.round(si,5)

def CC(df_mooring,df_sat,variable):

    y = df_sat[variable].values
    x = df_mooring[variable].values
    cc,_ = pearsonr(y,x)
    return np.round(cc,5)

def metrics_array(df_mooring: pd.DataFrame,df_sat:pd.DataFrame,variable:str):

    return np.array(
        [
        BIAS(df_mooring,df_sat,variable),
        RMSE(df_mooring,df_sat,variable),
        CC(df_mooring,df_sat,variable),
        SI(df_mooring,df_sat,variable)
        ]
    )