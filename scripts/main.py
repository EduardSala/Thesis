import yaml


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

cfg = load_config("../config/config.yaml")
print(cfg['setup_extraction'])