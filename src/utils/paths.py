from pathlib import Path
import datetime


def prepare_train_paths(cfg):
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.ROOT_OUTPUT_DIR = cfg.PROJECT_ROOT / "outputs"
    cfg.BASE_OUTPUT_DIR = cfg.ROOT_OUTPUT_DIR / f"Output_{run_id}"
    cfg.CKPT_DIR = cfg.BASE_OUTPUT_DIR / "checkpoints"
    cfg.LOG_DIR = cfg.BASE_OUTPUT_DIR / "logs"
    cfg.CKPT_PATH = cfg.CKPT_DIR / "best.ckpt"

    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    return cfg


def prepare_test_paths(cfg, ckpt_path):
    ckpt_path = Path(ckpt_path).resolve()

    cfg.ROOT_OUTPUT_DIR = cfg.PROJECT_ROOT / "outputs"
    cfg.BASE_OUTPUT_DIR = ckpt_path.parent.parent
    cfg.CKPT_DIR = cfg.BASE_OUTPUT_DIR / "checkpoints"
    cfg.LOG_DIR = cfg.BASE_OUTPUT_DIR / "logs"
    cfg.CKPT_PATH = ckpt_path

    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    return cfg