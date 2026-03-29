from dataclasses import dataclass, fields
from pathlib import Path
import yaml
from torchvision import models


@dataclass
class Config:
    ROOT_DIR: Path | str = "."
    DATA_DIR: Path | str = "../data/200_patients/200_middle7"
    CSV_PATH: Path | str = "data/acl.csv"

    CKPT_DIR: Path | None = None
    LOG_DIR: Path | None = None
    CKPT_PATH: Path | None = None
    ROOT_OUTPUT_DIR: Path | None = None
    BASE_OUTPUT_DIR: Path | None = None
    PROJECT_ROOT: Path | None = None

    IMG_SIZE: int = 224
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 8
    LR: float = 1e-5
    WEIGHT_DECAY: float = 1e-3
    DROPOUT: int = 0.3
    NUM_EPOCH: int = 50
    SEED: int = 42
    THRESHOLD: float = 0.5
    PATIENCE: int = 15
    MIN_DELTA: float = 1e-4
    TOP_K: int = 3
    USE_AMP: bool = True
    MODE: str = "max"
    AGG: str = "mean"
    POS_WEIGHT: float = 1.06
    MODEL_WEIGHT: object = models.ResNet50_Weights.DEFAULT
    MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def _resolve_path(project_root: Path, value):
    p = Path(value)
    return p if p.is_absolute() else (project_root / p)


def load_config(config_path: str | Path, project_root: str | Path) -> Config:
    config_path = Path(config_path)
    project_root = Path(project_root).resolve()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()
    valid_fields = {f.name for f in fields(Config)}

    for k, v in raw.items():
        if k in valid_fields:
            setattr(cfg, k, v)

    cfg.PROJECT_ROOT = project_root
    cfg.THRESHOLD = float(cfg.THRESHOLD)
    cfg.ROOT_DIR = _resolve_path(project_root, cfg.ROOT_DIR)
    cfg.DATA_DIR = _resolve_path(project_root, cfg.DATA_DIR)
    cfg.CSV_PATH = _resolve_path(project_root, cfg.CSV_PATH)

    if isinstance(cfg.MODEL_WEIGHT, str):
        key = cfg.MODEL_WEIGHT.upper()
        if key == "DEFAULT":
            cfg.MODEL_WEIGHT = models.ResNet50_Weights.DEFAULT
        elif key == "NONE":
            cfg.MODEL_WEIGHT = None
        else:
            raise ValueError(f"Unsupported MODEL_WEIGHT: {cfg.MODEL_WEIGHT}")

    return cfg