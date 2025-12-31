import subprocess
from pathlib import Path

DATA_DIR = Path("data/face_dataset")


def ensure_data() -> Path:
    """
    Ensure dataset is available locally.

    Priority:
    1. Existing local data
    2. DVC pull
    """
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        return DATA_DIR

    try:
        subprocess.run(["dvc", "pull"], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to pull data via DVC") from exc

    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        raise RuntimeError("Dataset not found after DVC pull")

    return DATA_DIR
