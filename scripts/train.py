import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.paths import prepare_train_paths
from src.engine.trainer import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config, PROJECT_ROOT)
    cfg = prepare_train_paths(cfg)

    print(f"Project root: {cfg.PROJECT_ROOT}")
    print(f"Data dir:      {cfg.DATA_DIR}")
    print(f"Output dir:    {cfg.BASE_OUTPUT_DIR}")

    train(cfg)


if __name__ == "__main__":
    main()