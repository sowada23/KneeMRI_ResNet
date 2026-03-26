import argparse
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.paths import prepare_test_paths
from src.utils.history import save_json, build_split_summary, build_split_patient_report
from src.data.datamodule import build_test_loader
from src.models.resnet50_binary import Resnet50
from src.engine.evaluator import evaluate
from src.metrics.patientwise import evaluate_patientwise, print_patient_case_rows
from src.viz.confusion import print_patient_confusion_matrix


def find_latest_checkpoint(project_root: Path, prefer="best.ckpt"):
    outputs_dir = project_root / "outputs"
    if not outputs_dir.exists():
        raise FileNotFoundError(f"No outputs directory found: {outputs_dir}")

    run_dirs = sorted(
        [p for p in outputs_dir.glob("Output_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    for run_dir in run_dirs:
        ckpt_dir = run_dir / "checkpoints"
        preferred = ckpt_dir / prefer
        fallback = ckpt_dir / "last.ckpt"

        if preferred.exists():
            return preferred
        if fallback.exists():
            return fallback

    raise FileNotFoundError(f"No checkpoint found under {outputs_dir}")


def test(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg.CKPT_PATH

    test_loader = build_test_loader(cfg)
    model = Resnet50(cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)

    criterion = torch.nn.BCEWithLogitsLoss()

    extra = ckpt.get("extra", {})
    if "best_t" not in extra:
        raise ValueError(f"'best_t' not found in checkpoint: {ckpt_path}")
    best_t = float(extra["best_t"])
    print(f"Using saved validation threshold: {best_t:.2f}")

    te = evaluate(model, test_loader, device, criterion)

    te_pat = evaluate_patientwise(
        model, test_loader, device,
        cfg, threshold=best_t
    )

    save_json(
        {
            "checkpoint_used": str(cfg.CKPT_PATH),
            "threshold_used": float(best_t),
            "slice_metrics": {
                "loss": float(te["loss"]),
                "acc": float(te["acc"]),
            },
            "patient_metrics": build_split_summary("test", te_pat)["metrics"],
        },
        cfg.BASE_OUTPUT_DIR / "test"/ "test_summary.json"
    )

    save_json(
        build_split_patient_report("test", te_pat),
        cfg.BASE_OUTPUT_DIR / "test" / "test_patient_details.json"
    )

    print_patient_confusion_matrix(
        model=model,
        loader=test_loader,
        device=device,
        cfg=cfg,
        threshold=best_t,
        save_path=cfg.LOG_DIR / "test " / "confusion_matrix.png"
    )

    print_patient_case_rows(te_pat, "fp")
    print_patient_case_rows(te_pat, "fn")

    print(f"\n[TEST slice] loss={te['loss']:.4f} acc={te['acc']:.4f}")
    print(f"[TEST patient] patients={te_pat['patients']} acc={te_pat['acc']:.4f} "
          f"P={te_pat['precision']:.3f} R={te_pat['recall']:.3f} F1={te_pat['f1']:.3f} "
          f"(TP {te_pat['tp']} FP {te_pat['fp']} TN {te_pat['tn']} FN {te_pat['fn']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--ckpt", default=None,
                    help="Optional checkpoint path. If omitted, uses the newest outputs/.../checkpoints/best.ckpt")
    args = ap.parse_args()

    if args.ckpt is None:
        ckpt_path = find_latest_checkpoint(PROJECT_ROOT, prefer="best.ckpt")
    else:
        ckpt_path = Path(args.ckpt).resolve()

    cfg = load_config(args.config, PROJECT_ROOT)
    cfg = prepare_test_paths(cfg, ckpt_path)

    print(f"Using checkpoint: {cfg.CKPT_PATH}")
    test(cfg)


if __name__ == "__main__":
    main()