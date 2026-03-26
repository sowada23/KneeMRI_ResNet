import torch
import torch.optim as optim
import numpy as np

from src.utils.freeze import print_trainable_params, setup_layer4_layer3_fc
from src.utils.reproducibility import set_seed
from src.utils.checkpoint import save_checkpoint, EarlyStopping
from src.data.datamodule import build_train_val_loaders
from src.models.resnet50_binary import Resnet50
from src.engine.evaluator import evaluate
from src.metrics.patientwise import (
    evaluate_patientwise,
    find_best_threshold_patient,
    print_patient_case_rows,
)
from src.viz.plots import (
    plot_train_val_curves,
    plot_prf_acc_curves,
    plot_patient_val_loss,
    plot_patient_val_f1,
    plot_patient_roc_curve,
)


def train_one_epoch(model, loader, optimizer, device, scaler, criterion):
    model.train()
    total_loss, total, correct = 0.0, 0, 0

    for x, y, _pids in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == y).sum().item()
        total += y.numel()

    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}


def train(cfg):
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_train_val_loaders(cfg)
    criterion = torch.nn.BCEWithLogitsLoss()

    model = Resnet50(cfg).to(device)
    model = setup_layer4_layer3_fc(model)

    print_trainable_params(model)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCH)
    scaler = torch.amp.GradScaler("cuda") if (cfg.USE_AMP and device.type == "cuda") else None

    start_epoch = 1
    best_val = float("-inf")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_pat_loss": [],
        "val_pat_roc_auc": [],
        "val_pat_acc": [],
        "val_pat_precision": [],
        "val_pat_recall": [],
        "val_pat_f1": [],
    }
    
    early = EarlyStopping(cfg)

    for epoch in range(start_epoch, cfg.NUM_EPOCH + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device, scaler, criterion)
        va = evaluate(model, val_loader, device, criterion)

        best = find_best_threshold_patient(
            model=model,
            loader=val_loader,
            device=device,
            cfg=cfg,
            metric_name="f1", 
        )
        # Fixed-threshold monitoring during training only
        va_pat = evaluate_patientwise(
            model=model,
            loader=val_loader,
            device=device,
            cfg=cfg,
            threshold=cfg.THRESHOLD,
        )

        should_stop, improved = early.step(va["loss"], model)
        best_t = cfg.THRESHOLD

        if should_stop:
            print("Early stopping triggered. Restoring best model weights.")
            early.restore_best(model)
            break

        print(
            f"Epoch {epoch:03d}/{cfg.NUM_EPOCH} | "
            f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.4f} | "
            f"VAL(PAT@{best_t:.2f}) loss {va_pat['loss']:.4f} "
            f"AUC {va_pat['roc_auc']:.4f} acc {va_pat['acc']:.4f} "
            f"P {va_pat['precision']:.3f} R {va_pat['recall']:.3f} "
            f"F1 {va_pat['f1']:.3f} (TP {va_pat['tp']} FP {va_pat['fp']} "
            f"TN {va_pat['tn']} FN {va_pat['fn']})"
        )

        history["val_pat_loss"].append(va_pat["loss"])
        history["val_pat_roc_auc"].append(va_pat["roc_auc"])
        history["val_pat_acc"].append(va_pat["acc"])
        history["val_pat_precision"].append(va_pat["precision"])
        history["val_pat_recall"].append(va_pat["recall"])
        history["val_pat_f1"].append(va_pat["f1"])
        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])

        scheduler.step()

        save_checkpoint(
            cfg.CKPT_DIR / "last.ckpt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            epoch=epoch,
            best_val=best_val,
            cfg=cfg,
            extra={
                "val_loss": float(va["loss"]),
                "val_acc": float(va["acc"]),
                "best_t": float(best_t),
                "val_pat_acc": float(va_pat["acc"]),
                "val_pat_precision": float(va_pat["precision"]),
                "val_pat_recall": float(va_pat["recall"]),
                "val_pat_f1": float(va_pat["f1"]),
                "val_pat_loss": float(va_pat["loss"]),
                "val_pat_roc_auc": float(va_pat["roc_auc"]) if not np.isnan(va_pat["roc_auc"]) else None,
            }
        )

        current = float(va_pat["f1"])
        if epoch == 1 or current > best_val:
            best_val = current
            save_checkpoint(
                cfg.CKPT_DIR / "best.ckpt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=None,
                epoch=epoch,
                best_val=best_val,
                cfg=cfg,
                extra={
                    "val_loss": float(va["loss"]),
                    "val_acc": float(va["acc"]),
                    "best_t": float(best_t),
                    "val_pat_acc": float(va_pat["acc"]),
                    "val_pat_precision": float(va_pat["precision"]),
                    "val_pat_recall": float(va_pat["recall"]),
                    "val_pat_f1": float(va_pat["f1"]),
                    "val_pat_loss": float(va_pat["loss"]),
                    "val_pat_roc_auc": float(va_pat["roc_auc"]) if not np.isnan(va_pat["roc_auc"]) else None,
                }
            )

    best = find_best_threshold_patient(
        model=model,
        loader=val_loader,
        device=device,
        cfg=cfg,
        metric_name="f1",
    )

    final_t = best["t"]
    final_val_pat = best["metric"]

    save_checkpoint(
        cfg.CKPT_DIR / "best.ckpt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=None,
        epoch=epoch,
        best_val=best_val,
        cfg=cfg,
        extra={
            "val_loss": float(va["loss"]),
            "val_acc": float(va["acc"]),
            "best_t": float(final_t),
            "val_pat_acc": float(final_val_pat["acc"]),
            "val_pat_precision": float(final_val_pat["precision"]),
            "val_pat_recall": float(final_val_pat["recall"]),
            "val_pat_f1": float(final_val_pat["f1"]),
            "val_pat_loss": float(final_val_pat["loss"]),
            "val_pat_roc_auc": float(final_val_pat["roc_auc"]) if not np.isnan(final_val_pat["roc_auc"]) else None,
            }
    )

    print(f"Final tuned threshold from validation set: {final_t:.2f}")
            
    plot_train_val_curves(
        history,
        save_path=cfg.LOG_DIR / "train_val_loss_acc.png"
    )

    plot_prf_acc_curves(
        history,
        save_path=cfg.LOG_DIR / "val_pat_prf_acc.png",
        split="val_pat"
    )

    plot_patient_val_loss(
        history,
        save_path=cfg.LOG_DIR / "val_patient_loss.png"
    )
    
    plot_patient_val_f1(
        history,
        save_path=cfg.LOG_DIR / "val_patient_f1.png"
    )
    
    plot_patient_roc_curve(
        final_val_pat["y_true"],
        final_val_pat["y_score"],
        save_path=cfg.LOG_DIR / "val_patient_roc_curve.png"
    )

    fp_rows, fp_ids = print_patient_case_rows(final_val_pat, "fp")
    fn_rows, fn_ids = print_patient_case_rows(final_val_pat, "fn")

    print("\nTraining finished")