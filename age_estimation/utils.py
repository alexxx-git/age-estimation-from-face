import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv(override=True)
data_dir = Path(os.getenv("FACE_DATA_DIR"))
bucket = os.getenv("DVC_BUCKET")
endpoint = f"{os.getenv('MINIO_URL')}:{os.getenv('MINIO_PORT')}"
access_key = os.getenv("MINIO_ROOT_USER")
secret_key = os.getenv("MINIO_ROOT_PASSWORD")

# def ensure_data() -> Path:

#     if data_dir.exists() and any(data_dir.iterdir()):
#         return data_dir

#     try:
#         subprocess.run(["dvc", "pull"], check=True)
#     except subprocess.CalledProcessError as exc:
#         raise RuntimeError("Failed to pull data via DVC") from exc

#     if not data_dir.exists() or not any(data_dir.iterdir()):
#         raise RuntimeError("Dataset not found after DVC pull")

#     return data_dir


def check_minio(endpoint_url, access_key, secret_key):
    try:
        print(f"{endpoint_url},{access_key}, {secret_key}")
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        # Пробуем получить список бакетов
        s3.list_buckets()
        return True
    except:
        return False


def dvc_init():
    subprocess.run(["dvc", "remote", "add", "-d", "myremote", f"s3://{bucket}"], check=True)

    subprocess.run(["dvc", "remote", "modify", "myremote", "access_key_id", access_key, "--local"], check=True)
    subprocess.run(["dvc", "remote", "modify", "myremote", "secret_access_key", secret_key, "--local"], check=True)
    subprocess.run(["dvc", "remote", "modify", "myremote", "endpointurl", endpoint, "--local"], check=True)


def download_data() -> Path:
    if check_minio(endpoint_url=endpoint, access_key=access_key, secret_key=secret_key):
        load_dotenv(override=True)
        os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")
        from kaggle import api  # импорт после установки env

        print(os.getenv("FACE_DATA_DIR"))
        data_dir = Path(os.getenv("FACE_DATA_DIR"))
        if data_dir.exists() and any(data_dir.iterdir()):
            print(f"{data_dir} уже существует и содержит файлы. Скачивание не требуется.")
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_path = Path(tmp_dir)
                kaggle_dataset = os.getenv("KAGGLE_DATASET")
                api.dataset_download_files(kaggle_dataset, path=str(temp_path), unzip=True)
                downloaded_subdir = next(temp_path.iterdir())  #
                for item in downloaded_subdir.iterdir():
                    shutil.move(str(item), data_dir)
            dvc_init()
            subprocess.run(["dvc", "pull"], check=True)
    else:
        print("minio no start")
    return data_dir
