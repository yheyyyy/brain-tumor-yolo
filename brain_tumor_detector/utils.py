import os
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_dataset():
    """Download the brain tumor dataset."""
    config = load_config('config/default.yaml')
    os.system(f"wget -q {config['data']['yaml_url']} -O {config['data']['local_path']}")
