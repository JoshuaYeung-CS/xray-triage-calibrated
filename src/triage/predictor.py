"""The app's inference path: ONNX model -> calibrated probability -> triage band -> heatmap.

The exported model returns two things per image: the raw pneumonia logit, and a 7x7 map
cam[h, w] = sum_k w_k * A_k[h, w], where A is the last ResNet feature map and w the linear head.
Because the head is a single linear layer on globally averaged features, Grad-CAM for the logit
is exactly this map (the gradient of the logit with respect to A_k is the constant w_k / 49), so
the heatmap needs no autograd at serving time: mean(cam) + bias == logit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .calibration import sigmoid, triage_band
from .preprocess import SIZE

LABELS = {
    "pneumonia": "Likely pneumonia",
    "normal": "Likely normal",
    "unsure": "Unsure: refer to a radiologist",
}


@dataclass
class Result:
    logit: float
    prob: float  # calibrated p(pneumonia)
    band: str  # "pneumonia" | "normal" | "unsure"
    heatmap: np.ndarray  # (128, 128) in [0, 1], evidence for the leaning class
    leaning: str  # class the heatmap explains

    @property
    def label(self) -> str:
        return LABELS[self.band]


def load_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    for key in ("temperature", "band_lo", "band_hi"):
        if key not in cfg:
            raise KeyError(f"calibration config is missing {key!r}")
    if not 0.0 < cfg["band_lo"] < 0.5 < cfg["band_hi"] < 1.0:
        raise ValueError("band must satisfy 0 < lo < 0.5 < hi < 1")
    if cfg["temperature"] <= 0:
        raise ValueError("temperature must be positive")
    return cfg


def heatmap_from_cam(cam: np.ndarray, toward_pneumonia: bool) -> np.ndarray:
    """7x7 class map -> 128x128 heatmap in [0, 1] for the chosen class (ReLU, then max-normalise)."""
    c = np.asarray(cam, dtype=np.float32)
    c = np.maximum(c if toward_pneumonia else -c, 0.0)
    m = float(c.max())
    if m > 0:
        c = c / m
    up = Image.fromarray(c, mode="F").resize((SIZE, SIZE), Image.BILINEAR)
    return np.clip(np.asarray(up, dtype=np.float32), 0.0, 1.0)


# A short perceptual ramp (dark blue -> teal -> yellow -> red) for overlays without matplotlib.
_RAMP = np.array(
    [[0.05, 0.03, 0.30], [0.10, 0.45, 0.70], [0.20, 0.75, 0.55], [0.98, 0.85, 0.20], [0.85, 0.15, 0.10]],
    dtype=np.float32,
)


def colorize(h: np.ndarray) -> np.ndarray:
    x = np.clip(h, 0, 1) * (len(_RAMP) - 1)
    i = np.minimum(x.astype(int), len(_RAMP) - 2)
    t = (x - i)[..., None]
    return _RAMP[i] * (1 - t) + _RAMP[i + 1] * t


def overlay(gray01: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Grayscale image (H, W) in [0, 1] + heatmap -> RGB uint8 overlay. Weak evidence stays clear."""
    g = np.repeat(np.clip(gray01, 0, 1)[..., None], 3, axis=2)
    a = (alpha * np.clip(heat, 0, 1))[..., None]
    rgb = g * (1 - a) + colorize(heat) * a
    return (rgb * 255).round().astype(np.uint8)


class Predictor:
    """Wraps any object with an onnxruntime-style ``run(None, {"image": x})`` method."""

    def __init__(self, session, config: dict):
        self.session = session
        self.temperature = float(config["temperature"])
        self.lo = float(config["band_lo"])
        self.hi = float(config["band_hi"])

    @classmethod
    def from_files(cls, model_path: str | Path, config_path: str | Path) -> "Predictor":
        import onnxruntime as ort  # imported here so tests and tools never need it

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        return cls(session, load_config(config_path))

    def calibrate(self, logit: float) -> float:
        return float(sigmoid(np.array([logit / self.temperature]))[0])

    def predict(self, x: np.ndarray) -> Result:
        if x.shape != (1, 1, SIZE, SIZE):
            raise ValueError(f"expected input of shape (1, 1, {SIZE}, {SIZE}), got {x.shape}")
        logit, cam = self.session.run(None, {"image": x.astype(np.float32)})
        logit = float(np.asarray(logit).reshape(-1)[0])
        cam = np.asarray(cam).reshape(-1, *np.asarray(cam).shape[-2:])[0]
        prob = self.calibrate(logit)
        toward = prob >= 0.5
        return Result(
            logit=logit,
            prob=prob,
            band=triage_band(prob, self.lo, self.hi),
            heatmap=heatmap_from_cam(cam, toward),
            leaning="pneumonia" if toward else "normal",
        )
