import load_configuration as lc
from processing import processing_df as pr_df
from calibration import calibration_methods as cal_meth
from calibration import bc_techniques as bc_techn
from io_data import export_data_to_csv as exp_data


def main():

    # 1. Load configuration
    cfg = lc.load_config("config/config.yaml")
    # 2. Print configuration to verify it was loaded correctly
    # 3 Start spatio-temporal matching to align satellite and mooring data based on the configuration parameters
    df_sat, df_mooring = pr_df.spatio_temp_matching(cfg)
    # 4. Print the resulting dataframes to verify the spatio-temporal matching process
    # 5. Prepare calibration and validation datasets by splitting the aligned dataframes into calibration and
    # validation subsets typically using the first ten days of data for calibration and the remaining data for validation.
    df_mooring_cal, df_sat_cal, df_mooring_val, df_sat_val = cal_meth.calib_df_first_ten_days(df_mooring, df_sat)
    list_dataframes = [df_sat_cal, df_mooring_cal, df_sat_val, df_mooring_val]
    # 6. Perform bias correction on the satellite validation dataset using the specified techniques and evaluate the
    # results using various metrics.
    results = bc_techn.bias_correction(cfg, list_dataframes)
    exp_data.export_dict_to_file(results, cfg)


if __name__ == "__main__":
    main()


