import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.paths import prepare_test_paths
from src.data.datamodule import build_test_loader
from src.data.transforms import build_eval_transforms
from src.models.resnet50_binary import Resnet50
from src.metrics.patientwise import evaluate_patientwise
from src.viz.gradcam import save_patient_middle3_gradcams


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


def fp_patient_ids_from_metrics(metrics):
    fp_ids = []
    for pid, info in sorted(metrics["detail"].items()):
        if info["pred"] == 1 and info["y"] == 0:
            fp_ids.append(pid)
    return fp_ids

def fn_patient_ids_from_metrics(metrics):
    fn_ids = []
    for pid, info in sorted(metrics["detail"].items()):
        if info["pred"] == 0 and info["y"] == 1:
            fn_ids.append(pid)
    return fn_ids



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument(
        "--ckpt",
        default=None,
        help="Optional checkpoint path. If omitted, uses newest outputs/Output_*/checkpoints/best.ckpt"
    )
    args = ap.parse_args()

    if args.ckpt is None:
        ckpt_path = find_latest_checkpoint(PROJECT_ROOT, prefer="best.ckpt")
    else:
        ckpt_path = Path(args.ckpt).resolve()

    cfg = load_config(args.config, PROJECT_ROOT)
    cfg = prepare_test_paths(cfg, ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using checkpoint: {cfg.CKPT_PATH}")
    print(f"Data dir:         {cfg.DATA_DIR}")
    print(f"Run output dir:   {cfg.BASE_OUTPUT_DIR}")

    # ----- Build model -----
    model = Resnet50(cfg).to(device)
    ckpt = torch.load(cfg.CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    extra = ckpt.get("extra", {})
    if "best_t" not in extra:
        raise ValueError(f"'best_t' not found in checkpoint: {cfg.CKPT_PATH}")
    best_t = float(extra["best_t"])
    print(f"Using saved validation threshold: {best_t:.2f}")
    
    # ----- Re-run patientwise test evaluation to collect FP patients -----
    test_loader = build_test_loader(cfg)
    te_pat = evaluate_patientwise(
        model=model,
        loader=test_loader,
        device=device,
        cfg=cfg,
        threshold=best_t,
    )

    fp_ids = fp_patient_ids_from_metrics(te_pat)
    fn_ids = fn_patient_ids_from_metrics(te_pat)
    
    print(f"False-positive patients in test pool: {len(fp_ids)}")
    print(fp_ids)
    print(f"False-negative patients in test pool: {len(fn_ids)}")
    print(fn_ids)

    if len(fp_ids) == 0 and len(fn_ids) == 0:
        print("No FP or FN patients found. Nothing to save.")
        return

    # ----- Grad-CAM output root -----
    gradcam_root = cfg.BASE_OUTPUT_DIR / "gradcam"
    gradcam_root.mkdir(parents=True, exist_ok=True)

    eval_transform = build_eval_transforms(cfg)

    # Last conv layer of ResNet50
    target_layer = model.layer4[-1].conv3

    saved_total = 0
    for error_type, patient_ids in [("FP", fp_ids), ("FN", fn_ids)]:
        for pid in patient_ids:
            # pid format from dataset is "class/patient"
            parts = pid.split("/", 1)
            if len(parts) != 2:
                print(f"[WARN] Unexpected pid format: {pid}")
                continue
    
            class_name, patient_name = parts
            patient_dir = cfg.DATA_DIR / "test" / class_name / patient_name
            patient_out_dir = gradcam_root / error_type / patient_name
    
            if not patient_dir.exists():
                print(f"[WARN] Patient folder does not exist: {patient_dir}")
                continue
    
            saved = save_patient_middle3_gradcams(
                model=model,
                device=device,
                patient_dir=patient_dir,
                patient_id=patient_name,
                out_dir=patient_out_dir,
                eval_transform=eval_transform,
                threshold=best_t,
                target_layer=target_layer,
            )
            saved_total += len(saved)
            print(f"[OK] {error_type} {patient_name}: saved {len(saved)} PNG(s) -> {patient_out_dir}")

    print(f"\nDone. Saved {saved_total} Grad-CAM image(s) under:")
    print(gradcam_root)


if __name__ == "__main__":
    main()