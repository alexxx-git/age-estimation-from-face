import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)


os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

from kaggle import api  # импорт после установки env

data_dir = Path(os.getenv("DATA_DIR"))
data_dir.mkdir(parents=True, exist_ok=True)

kaggle_dataset = "yapwh1208/face-data"

print("Downloading dataset from Kaggle...")
api.dataset_download_files(kaggle_dataset, path=str(data_dir), unzip=True)
print(f"Dataset downloaded to {data_dir}")
