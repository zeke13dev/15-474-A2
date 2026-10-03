"""Image I/O, device selection, and PSNR."""

import numpy as np
import torch
from PIL import Image


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(path, size=None) -> torch.Tensor:
    """Read an image as an (H, W, 3) float tensor in [0, 1], optionally resized
    so its longer side is ``size`` pixels."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        scale = size / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)


def write_image(pixels, path) -> None:
    """Save an (H, W, 3) float image in [0, 1] as an 8-bit PNG."""
    arr = pixels.detach().cpu().numpy() if torch.is_tensor(pixels) else np.asarray(pixels)
    Image.fromarray(np.rint(np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


def psnr(mse: float) -> float:
    """PSNR in dB from an MSE over [0, 1]-normalized pixels (MAX = 1)."""
    return float("inf") if mse == 0 else -10 * np.log10(mse)
