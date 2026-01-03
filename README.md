# Age Estimation from Face Images

## Overview
This project solves the age estimation problem from facial images using deep learning.
We use s3 storage with dvc
Dataset load from kaggle. U must get your secret in .env file in root of project

## Setup

insert your data in this text and save like .env in root folder project like this
```bash
# MinIO
MINIO_URL="http://localhost"
MINIO_ROOT_USER=your data
MINIO_ROOT_PASSWORD=your data
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001
DATA_BUCKET=face_data
DVC_BUCKET=dvc-face-data

# Kaggle (только для первичной загрузки)
KAGGLE_USERNAME=your data
KAGGLE_KEY=your data


#NO_CHANGE
KAGGLE_DATASET = yapwh1208/face-data
FACE_DATA_DIR=age_estimation/data/face_data
```
Start project
```bash
uv venv
source .venv/bin/activate
uv sync
pre-commit install


curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py
python get-pip.py
python -m pip --version
python -m pip install pre-commit
```

start s3
```bash
docker compose up -d
```

## Train



## Data

Данные хранятся локально в папке `data/face_data`. Хеши данных dvc расположены в s3 хранилище.

При первом запуске `train.py` или `infer.py` датасет автоматически скачивается
из открытого источника (Kaggle) и сохраняется локально.
