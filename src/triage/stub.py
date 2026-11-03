"""A stand-in for the ONNX session, used by the test suite and by `TRIAGE_STUB=1 streamlit run`.

It never loads the real network: the logit is a fixed linear function of image brightness and the
7x7 map is the image's own block means, so the whole app can be exercised without the model file.
"""
from __future__ import annotations

import numpy as np


class StubSession:
    def __init__(self, scale: float = 12.0, offset: float = 0.45):
        self.scale = scale
        self.offset = offset
        self.calls = 0

    def run(self, _outputs, feeds):
        self.calls += 1
        x = np.asarray(feeds["image"], dtype=np.float32)
        n = x.shape[0]
        blocks = x[:, 0, :126, :126].reshape(n, 7, 18, 7, 18).mean(axis=(2, 4))
        cam = (blocks - self.offset) * self.scale
        logit = cam.mean(axis=(1, 2))
        return [logit.astype(np.float32), cam.astype(np.float32)]


STUB_CONFIG = {"temperature": 1.5, "band_lo": 0.2, "band_hi": 0.8}
