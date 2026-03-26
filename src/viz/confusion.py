from pathlib import Path
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.metrics.patientwise import evaluate_patientwise


@torch.no_grad()
def print_patient_confusion_matrix(model, loader, device, cfg, threshold, save_path=None):
    model.eval()

    class_names = ("0", "1")

    metrics = evaluate_patientwise(model, loader, device, cfg, threshold)

    y_true = []
    y_pred = []

    for pid, info in sorted(metrics["detail"].items()):
        y_true.append(int(info["y"]))
        y_pred.append(int(info["pred"]))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\nPatient-level confusion matrix (rows=true, cols=pred):")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(class_names)
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Patient-level Confusion Matrix @ threshold={threshold:.2f}")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return cm, metrics