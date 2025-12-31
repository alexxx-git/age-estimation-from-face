# age_estimation/train.py

import pytorch_lightning as pl

from age_estimation.datamodule import FaceAgeDataModule
from age_estimation.model import FaceAgeClassifier
from age_estimation.utils import ensure_data


def main() -> None:
    data_dir = ensure_data()

    data_module = FaceAgeDataModule(data_dir=data_dir)
    model = FaceAgeClassifier()

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
