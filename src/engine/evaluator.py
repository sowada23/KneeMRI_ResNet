import torch


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0

    for x, y, _pids in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x).squeeze(1)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.numel()
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == y).sum().item()
        total += y.numel()

    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}