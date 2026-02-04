from modules import data_extraction_md as data_extr
import yaml
import os


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


cfg = load_config("../config/config.yaml")

filePath = [os.path.join(cfg['extraction']
                         ['dir_paths']['dir_input_mooring_nc'],name)
            for name in os.listdir(cfg['extraction']['dir_paths']['dir_input_mooring_nc'])]
for fp in filePath:
    data_extr.exp_dataframe_to_file(
        data_extr.extr_insitu_data_from_nc(
            fp,cfg['extraction']['variable']['var_name'],cfg['extraction']['variable']['field'],
            cfg['extraction']['variable']['deph_val'],cfg['extraction']['variable']['tol']
        ),
        cfg['extraction']['variable']['field'],cfg['extraction']['dir_paths']['dir_output_mooring_csv']
    )










