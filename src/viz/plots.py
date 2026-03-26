from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_train_val_curves(history, save_path):
    """
    history:
      - "train_loss", "val_loss", "train_acc", "val_acc": list[float]
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, ax_loss = plt.subplots(figsize=(10, 5))
    ax_acc = ax_loss.twinx()

    l1, = ax_loss.plot(epochs, history["train_loss"], label="train loss", color="C0")
    l2, = ax_loss.plot(epochs, history["val_loss"],   label="val loss",   color="C1")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")

    l3, = ax_acc.plot(epochs, history["train_acc"], label="train acc", color="C2")
    l4, = ax_acc.plot(epochs, history["val_acc"],   label="val acc",   color="C3")
    ax_acc.set_ylabel("Accuracy")

    lines = [l1, l2, l3, l4]
    labels = [ln.get_label() for ln in lines]
    ax_loss.legend(lines, labels, loc="best")

    ax_loss.set_title("Training / Validation Curves")
    ax_loss.grid(True)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.show()


def plot_prf_acc_curves(history: dict, save_path=None, split: str = "val") -> None:
    k_acc = f"{split}_acc"
    k_p = f"{split}_precision"
    k_r = f"{split}_recall"
    k_f1 = f"{split}_f1"

    for k in (k_acc, k_p, k_r, k_f1):
        if k not in history:
            raise KeyError(f"Missing history['{k}']. Available keys: {list(history.keys())}")

    epochs = list(range(1, len(history[k_acc]) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history[k_acc], label=f"{split} acc",       color="C0")
    ax.plot(epochs, history[k_p],   label=f"{split} precision", color="C1")
    ax.plot(epochs, history[k_r],   label=f"{split} recall",    color="C2")
    ax.plot(epochs, history[k_f1],  label=f"{split} f1",        color="C3")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"{split.upper()} Precision / Recall / F1 / Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.show()

from pathlib import Path
import matplotlib.pyplot as plt


def plot_patient_val_loss(history, save_path=None):
    epochs = list(range(1, len(history["val_pat_loss"]) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["val_pat_loss"], label="val patient loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Patient-wise Val Loss")
    ax.set_title("Patient-wise Validation Loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_patient_val_f1(history, save_path=None):
    epochs = list(range(1, len(history["val_pat_f1"]) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["val_pat_f1"], label="val patient F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Patient-wise F1")
    ax.set_title("Patient-wise Validation F1")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_patient_roc_curve(y_true, y_score, save_path=None):
    if len(set(y_true)) < 2:
        print("[WARN] ROC curve cannot be plotted because only one class is present.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Patient-wise ROC Curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()