from io_data import export_data_to_csv as data_extr
from io_data import load_dataframe as load_data
from pathlib import Path
from config import load_configuration


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
        df = load_data.load_moor_data_nc(fp, cfg_var, cfg_field, cfg_deph)
        data_extr.export_dataframe_to_file(df, cfg_field, output_dir)
    # managing errors when the file is not processed
    except Exception as e:
        print(f"Error processing {fp.name}: {e}")
