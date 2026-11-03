"""Turn any uploaded picture into the model's input: one grayscale 128x128 image in [0, 1].

PneumoniaMNIST was made by center-cropping each X-ray to a square and resizing it, so uploads get
the same treatment. Upsampling to 224 px and ImageNet normalisation live inside the exported model,
which keeps this step identical between training and the app.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

SIZE = 128


def to_grayscale(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        # 16-bit PNG/DICOM exports: stretch to 8 bits instead of clipping.
        a = np.asarray(img, dtype=np.float64)
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
        return Image.fromarray(a.astype(np.uint8), mode="L")
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("L")


def center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def prepare(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 array of shape (1, 1, 128, 128) in [0, 1]."""
    g = center_square(to_grayscale(img))
    if g.size != (SIZE, SIZE):
        g = g.resize((SIZE, SIZE), Image.BICUBIC)
    a = np.asarray(g, dtype=np.float32) / 255.0
    return a[None, None, :, :]


def from_uint8(a: np.ndarray) -> np.ndarray:
    """Dataset array (128, 128) uint8 -> model input (1, 1, 128, 128)."""
    a = np.asarray(a)
    if a.shape != (SIZE, SIZE):
        raise ValueError(f"expected a {SIZE}x{SIZE} image, got {a.shape}")
    return (a.astype(np.float32) / 255.0)[None, None, :, :]
