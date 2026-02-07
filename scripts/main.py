from modules import load_configuration as lc
from modules import processing_df as pr_df
from modules import metrics
from tqdm import tqdm
from modules import calibration_methods as cal_meth
from modules import bc_techniques as bc_techn

cfg = lc.load_config("../config/config_local.yaml")
cfg_var = cfg['crossMatching']['var_name']
df_sat,df_mooring = pr_df.spatio_temp_matching(cfg)

df_mooring_cal,df_sat_cal,df_mooring_val,df_sat_val = cal_meth.calib_df_first_ten_days(df_mooring,df_sat)
df_sat_val_linearCal = bc_techn.linearCal(df_sat,df_mooring,cfg_var)

print(metrics.metrics_array(df_mooring_val,df_sat_val,cfg_var))


"""
bias = metrics.BIAS(df_mooring,df_sat,cfg_var)
rmse = metrics.RMSE(df_mooring,df_sat,cfg_var)
cc = metrics.CC(df_mooring,df_sat,cfg_var)
si = metrics.SI(df_mooring,df_sat,cfg_var)
"""




