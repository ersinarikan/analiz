import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi

# Kaggle credentials
kaggle_json = {
    "username": "ersnarikan",
    "key": "45500aee9bf7abc8ac97d6007af1b52c"
}

# Save Kaggle credentials
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
    json.dump(kaggle_json, f)

# Set permissions
os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Download dataset
dataset_name = "jangedoo/utkface-new"
download_path = "storage/datasets"
os.makedirs(download_path, exist_ok=True)

print("Downloading UTKFace dataset...")
api.dataset_download_files(dataset_name, path=download_path, unzip=False)
print("Download completed!") 