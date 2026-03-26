from collections import defaultdict
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def prf_from_counts(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def bce_from_probs(y_true, y_prob, eps=1e-8):
    """
    Binary cross-entropy computed from probabilities.
    y_true: list/array of 0/1
    y_prob: list/array of probabilities in [0,1]
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_prob = np.clip(y_prob, eps, 1.0 - eps)

    loss = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(np.mean(loss))


def aggregate_patient_score(probs, cfg):
    mode = cfg.AGG
    if len(probs) == 0:
        return 0.0

    if mode == "mean":
        return float(np.mean(probs))
    if mode == "max":
        return float(np.max(probs))
    if mode == "topk_mean":
        k = min(cfg.TOP_K, len(probs))
        return float(np.mean(np.sort(probs)[-k:]))

    raise ValueError(f"Unknown aggregation mode: {mode}")

def get_patient_case_rows(metrics, case):
    """
    case: 'fp', 'fn', 'tp', 'tn'
    """
    case = case.lower()
    if case not in {"fp", "fn", "tp", "tn"}:
        raise ValueError("case must be one of: 'fp', 'fn', 'tp', 'tn'")

    threshold = metrics.get("threshold", None)
    rows = []

    for pid, info in sorted(metrics["detail"].items()):
        score = float(info["score"])
        pred = int(info["pred"])
        ytrue = int(info["y"])
        n_slices = int(info["n_slices"])

        # consistency check
        if threshold is not None:
            pred_from_score = int(score >= threshold)
            if pred != pred_from_score:
                raise ValueError(
                    f"Inconsistent metrics for pid={pid}: "
                    f"score={score:.4f}, pred={pred}, threshold={threshold:.4f}, "
                    f"but int(score >= threshold)={pred_from_score}"
                )

        if case == "fp" and pred == 1 and ytrue == 0:
            rows.append((pid, score, n_slices))
        elif case == "fn" and pred == 0 and ytrue == 1:
            rows.append((pid, score, n_slices))
        elif case == "tp" and pred == 1 and ytrue == 1:
            rows.append((pid, score, n_slices))
        elif case == "tn" and pred == 0 and ytrue == 0:
            rows.append((pid, score, n_slices))

    # sort for readability
    if case in {"fp", "tp"}:
        rows.sort(key=lambda x: x[1], reverse=True)
    else:
        rows.sort(key=lambda x: x[1])

    return rows


def print_patient_case_rows(metrics, case):
    rows = get_patient_case_rows(metrics, case)
    threshold = metrics.get("threshold", None)

    titles = {
        "fp": "false-positive",
        "fn": "false-negative",
        "tp": "true-positive",
        "tn": "true-negative",
    }
    title = titles[case.lower()]

    if threshold is not None:
        print(f"\nFinal {title} patients: {len(rows)} (threshold={threshold:.2f})")
    else:
        print(f"\nFinal {title} patients: {len(rows)}")

    for pid, score, n_slices in rows:
        print(f"pid={pid} | score={score:.4f} | slices={n_slices}")

    ids = [pid for pid, _, _ in rows]
    print(f"\n{case.upper()} patient IDs:")
    print(ids)

    return rows, ids
@torch.no_grad()
def evaluate_patientwise(model, loader, device, cfg, threshold):
    model.eval()

    patient_probs = defaultdict(list)
    patient_label = {}

    for x, y, pid_list in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        ys = y.detach().cpu().numpy().astype(int).tolist()

        for pid, p, yy in zip(pid_list, probs, ys):
            patient_probs[pid].append(float(p))
            if pid in patient_label:
                if patient_label[pid] != int(yy):
                    raise ValueError(f"Inconsistent labels within patient {pid}: {patient_label[pid]} vs {yy}")
            else:
                patient_label[pid] = int(yy)

    tp = fp = tn = fn = 0
    patient_pred = {}

    y_true_all = []
    y_score_all = []

    for pid, probs_list in patient_probs.items():
        score = aggregate_patient_score(probs_list, cfg)
        pred = int(score >= threshold)
        ytrue = int(patient_label[pid])

        y_true_all.append(ytrue)
        y_score_all.append(float(score))

        patient_pred[pid] = {
            "score": float(score),
            "pred": int(pred),
            "y": int(ytrue),
            "n_slices": len(probs_list),
        }

        if pred == 1 and ytrue == 1:
            tp += 1
        elif pred == 1 and ytrue == 0:
            fp += 1
        elif pred == 0 and ytrue == 0:
            tn += 1
        elif pred == 0 and ytrue == 1:
            fn += 1

    precision, recall, f1 = prf_from_counts(tp, fp, fn)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)

    pat_loss = bce_from_probs(y_true_all, y_score_all)

    if len(set(y_true_all)) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true_all, y_score_all))

    return {
        "patients": len(patient_probs),
        "threshold": float(threshold),
        "loss": pat_loss,
        "roc_auc": roc_auc,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "detail": patient_pred,
        "y_true": y_true_all,
        "y_score": y_score_all,
    }


def print_final_fp_patients(metrics):
    fp_rows = []

    for pid, info in sorted(metrics["detail"].items()):
        if info["pred"] == 1 and info["y"] == 0:
            fp_rows.append((pid, info["score"], info["n_slices"]))

    print(f"\nFinal false-positive patients: {len(fp_rows)}")
    for pid, score, n_slices in fp_rows:
        print(f"pid={pid} | score={score:.4f} | slices={n_slices}")

    fp_ids = [pid for pid, _, _ in fp_rows]
    print("\nFP patient IDs:")
    print(fp_ids)

    return fp_rows, fp_ids


def print_final_fn_patients(metrics):
    fn_rows = []

    for pid, info in sorted(metrics["detail"].items()):
        if info["pred"] == 0 and info["y"] == 1:
            fn_rows.append((pid, info["score"], info["n_slices"]))

    print(f"\nFinal false-negative patients: {len(fn_rows)}")
    for pid, score, n_slices in fn_rows:
        print(f"pid={pid} | score={score:.4f} | slices={n_slices}")

    fn_ids = [pid for pid, _, _ in fn_rows]
    print("\nFN patient IDs:")
    print(fn_ids)

    return fn_rows, fn_ids
    
def find_best_threshold_patient(model, loader, device, cfg, thresholds=None, metric_name="f1"):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    best = {"t": -1.0, "score": float("-inf"), "metric": None}

    for t in thresholds:
        m = evaluate_patientwise(model, loader, device, cfg, threshold=float(t))
        score = float(m[metric_name])

        if (score > best["score"]) or (score == best["score"] and float(t) > best["t"]):
            best = {
                "t": float(t),
                "score": score,
                "metric": m,
            }

    return best