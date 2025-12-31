from pathlib import Path
from typing import Callable, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .dataset import FaceAgeDataset


class FaceAgeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 224,
        val_split: float = 0.2,
        transform_train: Optional[Callable] = None,
        transform_val: Optional[Callable] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.val_split = val_split
        self.seed = seed

        # трансформы, если не переданы
        if transform_train is None:
            self.transform_train = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform_train = transform_train

        if transform_val is None:
            self.transform_val = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform_val = transform_val

        self.train_dataset: Optional[FaceAgeDataset] = None
        self.val_dataset: Optional[FaceAgeDataset] = None

    def setup(self, stage: Optional[str] = None):
        """
        Создает датасеты train/val. Вызывается автоматически Lightning.
        """
        full_dataset = FaceAgeDataset(root_dir=self.data_dir, transform=self.transform_train)
        val_len = int(len(full_dataset) * self.val_split)
        train_len = len(full_dataset) - val_len

        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            lengths=[train_len, val_len],
            generator=None,  # reproducibility можно добавить через torch.Generator().manual_seed(self.seed)
        )

        # Для val_dataset используем transform_val
        self.val_dataset.dataset.transform = self.transform_val

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
