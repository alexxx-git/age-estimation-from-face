# train.py

import subprocess
from pathlib import Path

DATA_DIR = Path("data/face_dataset")

# Проверяем, есть ли данные, если нет — подтягиваем через DVC
if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
    print("Downloading dataset via DVC...")
    subprocess.run(["dvc", "pull"], check=True)

# Дальше уже обычный импорт вашего DataModule и LightningModule
from age_estimation.datamodule import FaceAgeDataModule
from age_estimation.model import FaceAgeClassifier  # пример


def main():
    data_module = FaceAgeDataModule(data_dir=DATA_DIR)
    model = FaceAgeClassifier()

    # Здесь обычная тренировка PyTorch Lightning
    import pytorch_lightning as pl

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
