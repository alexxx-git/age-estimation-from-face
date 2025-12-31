# Standard Library Imports
from pathlib import Path
from typing import Callable, Optional

import torch

# Third-Party Imports
from PIL import Image
from torch.utils.data import Dataset

AGE_GROUPS = {
    "young": (1, 25),
    "adult": (26, 60),
    "elderly": (61, 101),
}


AGE_GROUP_TO_LABEL = {
    "young": 0,
    "adult": 1,
    "elderly": 2,
}


def age_to_group(age: int) -> int:
    """
    Convert numeric age to age group label.
    """
    for group_name, (min_age, max_age) in AGE_GROUPS.items():
        if min_age <= age <= max_age:
            return AGE_GROUP_TO_LABEL[group_name]
    raise ValueError(f"Age {age} is out of supported range")


class FaceAgeDataset(Dataset):
    """
    Example:
    23_0_0_20161219140623097.jpg.chip.jpg
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = self._collect_images()

    def _collect_images(self) -> list[Path]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        image_paths = sorted(self.root_dir.glob("**/*.jpg"))
        if not image_paths:
            raise RuntimeError(f"No images found in {self.root_dir}")

        return image_paths

    @staticmethod
    def _parse_age_from_filename(path: Path) -> int:
        try:
            age_str = path.name.split("_")[0]
            return int(age_str)
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid filename format: {path.name}") from exc

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path = self.image_paths[index]

        age = self._parse_age_from_filename(image_path)
        label = age_to_group(age)

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }
