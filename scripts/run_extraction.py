from modules import export_data_to_csv as data_extr
from pathlib import Path
from modules import load_configuration
import os


# loading config file
cfg = load_configuration.load_config("../config/config.yaml")
# saving directory path from config file
input_dir = Path(cfg['extraction']['dir_paths']['dir_input_mooring_nc'])
output_dir = Path(cfg['extraction']['dir_paths']['dir_output_mooring_csv'])
file_paths = sorted(input_dir.glob("*.nc"))
print(f"Found {len(file_paths)} NetCDF files")

# saving parameters from config  file
cfg_var = cfg['extraction']['variable']['var_name']
cfg_deph = cfg['extraction']['variable']['deph_val']
cfg_tol = cfg['extraction']['variable']['tol']
cfg_field = cfg['extraction']['variable']['field']

# extraction of each insitu data and exporting it as .csv into output directory (config file)
for fp in file_paths:
    print(f"Processing {fp.name}")

    try:
        df = data_extr.extr_insitu_data_from_nc(fp, cfg_var, cfg_field, cfg_deph, cfg_tol)
        data_extr.exp_dataframe_to_file(df, cfg_field, output_dir)
    # managing errors when the file is not processed
    except Exception as e:
        print(f"Error processing {fp.name}: {e}")











