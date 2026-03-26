from pathlib import Path
import torch
import torch.nn as nn


def save_checkpoint(path: Path, *, model, optimizer=None, scaler=None, scheduler=None,
                    epoch: int, best_val: float, cfg=None, extra: dict | None = None):
    ckpt = {
        "epoch": epoch,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": (optimizer.state_dict() if optimizer is not None else None),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "scheduler": (scheduler.state_dict() if scheduler is not None else None),
        "cfg": (cfg.__dict__ if cfg is not None else None),
        "extra": (extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


class EarlyStopping:
    def __init__(self, cfg):
        """
        mode="min": expects metric to go DOWN (e.g., val_loss)
        mode="max": expects metric to go UP   (e.g., val_f1)
        """
        assert cfg.MODE in ["min", "max"]
        self.patience = int(cfg.PATIENCE)
        self.min_delta = float(cfg.MIN_DELTA)
        self.mode = cfg.MODE
        self.ckpt_path = cfg.CKPT_PATH

        self.best = None
        self.num_bad = 0
        self.best_state = None

    def _is_improvement(self, current):
        if self.best is None:
            return True
        if self.mode == "min":
            return current < (self.best - self.min_delta)
        else:
            return current > (self.best + self.min_delta)

    def step(self, current_metric: float, model: nn.Module) -> tuple[bool, bool]:
        if self._is_improvement(current_metric):
            self.best = current_metric
            self.num_bad = 0
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            return False, True

        self.num_bad += 1
        return self.num_bad >= self.patience, False

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state, strict=True)