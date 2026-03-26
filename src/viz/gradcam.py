from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt



class GradCAM:
    """
    Minimal Grad-CAM for a single target layer.
    Assumes binary classifier output of shape [B, 1].
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple; take tensor gradient wrt layer output
        self.gradients = grad_output[0].detach()

    def remove(self):
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()

    def __call__(self, x: torch.Tensor):
        """
        x: [1, 1, H, W]
        Returns:
            cam_np: [H, W] in [0,1]
            logit: float
            prob: float
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x).squeeze(1)  # [1]
        target = logits[0]                 # positive-class logit for ACL tear
        target.backward()

        # activations/grads: [1, C, h, w]
        acts = self.activations[0]   # [C, h, w]
        grads = self.gradients[0]    # [C, h, w]

        weights = grads.mean(dim=(1, 2), keepdim=True)  # [C,1,1]
        cam = (weights * acts).sum(dim=0)               # [h,w]
        cam = torch.relu(cam)

        cam_np = cam.detach().cpu().numpy()
        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)

        h, w = x.shape[-2:]
        cam_np = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)

        logit = float(target.detach().cpu().item())
        prob = float(torch.sigmoid(target.detach()).cpu().item())
        return cam_np.astype(np.float32), logit, prob


def normalize_to_uint8_gray(arr_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr_2d, dtype=np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def cam_to_heatmap_rgb(cam: np.ndarray) -> np.ndarray:
    cam_u8 = (np.clip(cam, 0.0, 1.0) * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def overlay_cam_on_image(gray_u8: np.ndarray, cam: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    base_rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    heatmap_rgb = cam_to_heatmap_rgb(cam)
    overlay = (1.0 - alpha) * base_rgb.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def choose_middle_three(npy_paths):
    npy_paths = sorted(npy_paths)
    n = len(npy_paths)

    if n == 0:
        return []
    if n <= 3:
        return npy_paths

    mid = n // 2
    idxs = [mid - 1, mid, mid + 1]

    selected = []
    seen = set()
    for i in idxs:
        i = max(0, min(n - 1, i))
        if i not in seen:
            selected.append(npy_paths[i])
            seen.add(i)

    return selected


def save_gradcam_figure(
    arr_2d: np.ndarray,
    cam: np.ndarray,
    out_path: Path,
    *,
    patient_id: str,
    slice_name: str,
    prob: float,
    threshold: float,
):
    gray_u8 = normalize_to_uint8_gray(arr_2d)
    heatmap_rgb = cam_to_heatmap_rgb(cam)
    overlay_rgb = overlay_cam_on_image(gray_u8, cam, alpha=0.40)

    pred = 1 if prob >= threshold else 0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(gray_u8, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap_rgb)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay_rgb)
    axes[2].set_title(f"Overlay\np={prob:.3f}  thr={threshold:.2f}  pred={pred}")
    axes[2].axis("off")

    fig.suptitle(f"Patient {patient_id} | Slice {slice_name}", fontsize=12)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_patient_middle3_gradcams(
    *,
    model: torch.nn.Module,
    device: torch.device,
    patient_dir: Path,
    patient_id: str,
    out_dir: Path,
    eval_transform,
    threshold: float,
    target_layer: torch.nn.Module,
):
    npy_paths = sorted(patient_dir.glob("*.npy"))
    selected_paths = choose_middle_three(npy_paths)

    if len(selected_paths) == 0:
        print(f"[WARN] No .npy files found for patient folder: {patient_dir}")
        return []

    gradcam = GradCAM(model, target_layer)
    saved_paths = []

    try:
        model.eval()

        for npy_path in selected_paths:
            arr = np.load(npy_path)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got {arr.shape} for {npy_path}")

            arr_proc = arr.astype(np.float32)
            x = torch.from_numpy(arr_proc).unsqueeze(0)
            
            if eval_transform is not None:
                x = eval_transform(x)
            x = x.unsqueeze(0).to(device)  

            cam, logit, prob = gradcam(x)

            out_path = out_dir / f"{npy_path.stem}.png"
            save_gradcam_figure(
                arr_2d=arr,
                cam=cam,
                out_path=out_path,
                patient_id=patient_id,
                slice_name=npy_path.stem,
                prob=prob,
                threshold=threshold,
            )
            saved_paths.append(out_path)

    finally:
        gradcam.remove()

    return saved_paths