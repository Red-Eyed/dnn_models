from pathlib import Path


class Config:
    ROOT_DIR = Path(__file__).absolute().parent
    MODELS_DIR = ROOT_DIR / "trained_models"
    DATASETS_DIR = ROOT_DIR / "datasets"
