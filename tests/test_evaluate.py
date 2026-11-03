import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate  # noqa: E402


def fake_run(tmp_path):
    rng = np.random.default_rng(0)

    def split(n):
        z = rng.normal(0, 2.5, n)
        y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(np.uint8)
        return 2.0 * z, y  # twice too confident, so the fitted temperature should be near 2

    vz, vy = split(600)
    tz, ty = split(700)
    preds = {"val_labels": vy, "test_labels": ty, "test_cam_onnx": rng.normal(size=(700, 7, 7)).astype(np.float32)}
    for name in ("frozen_lr", "finetune_s0", "finetune_s1", "onnx"):
        preds[f"val_logits_{name}"], preds[f"test_logits_{name}"] = vz, tz
    run = tmp_path / "run"
    run.mkdir()
    np.savez(run / "predictions.npz", **preds)
    rec = {"shipped": "finetune_s0", "onnx": {"max_abs_logit_diff": 1e-5}, "candidates": {"finetune_s0": {"history": [], "val_nll": 0.2}}}
    (run / "train_record.json").write_text(json.dumps(rec))
    return run


def test_evaluate_end_to_end_on_synthetic_logits(tmp_path):
    out = evaluate.main(str(fake_run(tmp_path)), tmp_path / "art")
    cfg = json.loads((tmp_path / "art" / "calibration.json").read_text())
    assert 1.6 < cfg["temperature"] < 2.5
    assert (cfg["band_lo"], cfg["band_hi"]) == (0.2, 0.8)
    s = out["test"]["shipped"]
    assert s["ece"] < s["uncalibrated"]["ece"]
    lo, hi = s["ci95"]["auroc"]
    assert lo < s["auroc"] < hi
    h = out["headline"]
    assert 0 < h["coverage"] < 1 and h["selective_accuracy"] > s["accuracy"]
    assert out["test"]["majority_class_accuracy"] >= 0.5
    assert set(out["candidates"]) == {"frozen_lr", "finetune_s0", "finetune_s1", "onnx"}
