# Age Estimation from Face Images

## Overview
This project solves the age estimation problem from facial images using deep learning.

## Setup
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

## Train



## Data

Данные хранятся локально в папке `data/face_dataset`.

При первом запуске `train.py` или `infer.py` датасет автоматически скачивается
из открытого источника (Kaggle) и сохраняется локально.
