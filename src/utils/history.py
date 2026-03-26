from pathlib import Path
import json


def _to_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def save_json(data, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_to_serializable(data), f, indent=2)


def patient_case(y_true, y_pred):
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    return "FN"


def build_patient_rows(metrics):
    rows = []
    for pid, info in sorted(metrics["detail"].items()):
        y_true = int(info["y"])
        y_pred = int(info["pred"])
        rows.append({
            "patient_id": pid,
            "true_label": y_true,
            "pred_label": y_pred,
            "case": patient_case(y_true, y_pred),
            "score": float(info["score"]),
            "n_slices": int(info["n_slices"]),
        })
    return rows


def build_case_groups(patient_rows):
    groups = {"TP": [], "FP": [], "TN": [], "FN": []}
    for row in patient_rows:
        groups[row["case"]].append({
            "patient_id": row["patient_id"],
            "score": row["score"],
            "n_slices": row["n_slices"],
        })
    return groups


def build_split_summary(split_name, metrics):
    return {
        "split": split_name,
        "threshold": float(metrics["threshold"]),
        "metrics": {
            "patients": int(metrics["patients"]),
            "loss": float(metrics["loss"]),
            "roc_auc": None if str(metrics["roc_auc"]) == "nan" else float(metrics["roc_auc"]),
            "acc": float(metrics["acc"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "tp": int(metrics["tp"]),
            "fp": int(metrics["fp"]),
            "tn": int(metrics["tn"]),
            "fn": int(metrics["fn"]),
        }
    }


def build_split_patient_report(split_name, metrics):
    patient_rows = build_patient_rows(metrics)
    return {
        "split": split_name,
        "threshold": float(metrics["threshold"]),
        "metrics": {
            "patients": int(metrics["patients"]),
            "loss": float(metrics["loss"]),
            "roc_auc": None if str(metrics["roc_auc"]) == "nan" else float(metrics["roc_auc"]),
            "acc": float(metrics["acc"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "tp": int(metrics["tp"]),
            "fp": int(metrics["fp"]),
            "tn": int(metrics["tn"]),
            "fn": int(metrics["fn"]),
        },
        "case_groups": build_case_groups(patient_rows),
        "patients": patient_rows,
    }

