from torch.utils.data import DataLoader
from src.data.dataset import KneeDataset, collate_with_pid
from src.data.transforms import build_train_transforms, build_eval_transforms


def build_train_val_loaders(cfg):
    train_root = cfg.DATA_DIR / "train"
    val_root = cfg.DATA_DIR / "val"

    train_ds = KneeDataset(
        train_root, 
        build_train_transforms(cfg),  
    )
    
    val_ds = KneeDataset(
        val_root, 
        build_eval_transforms(cfg), 
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_with_pid,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_with_pid,
    )

    return train_loader, val_loader


def build_test_loader(cfg):
    test_root = cfg.DATA_DIR / "test"
    test_ds = KneeDataset(
        test_root, 
        build_eval_transforms(cfg),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_with_pid,
    )

    return test_loader