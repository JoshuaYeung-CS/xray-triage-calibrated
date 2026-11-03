"""Score the saved validation and test logits. NumPy only; no model is loaded.

Protocol (fixed before any test number was looked at):
- Every candidate gets its own temperature, fitted on validation NLL.
- The shipped model is scored on the logits of the exported ONNX file, i.e. exactly what the app serves.
- The triage band is set in advance: answer "likely normal" at p <= 0.20, "likely pneumonia" at p >= 0.80,
  otherwise "unsure, refer". Nothing about the band is tuned.
- The high-sensitivity operating point (90% specificity) is chosen on validation and then applied to test.
- 95% intervals are percentile bootstraps over test images (2,000 resamples).

Usage: python scripts/evaluate.py runs/full-<jobid>   (writes artifacts/metrics.json and calibration.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triage import calibration as cal  # noqa: E402

BAND_LO, BAND_HI = 0.20, 0.80
TARGET_SPEC = 0.90
N_BOOT = 2000


def score(p, y):
    r = cal.classification_report(p, y, 0.5)
    return {
        "auroc": cal.auroc(p, y),
        "accuracy": r["accuracy"],
        "sensitivity": r["sensitivity"],
        "specificity": r["specificity"],
        "f1": r["f1"],
        "ece": cal.ece(p, y),
        "brier": cal.brier(p, y),
        "nll": float(-np.mean(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))),
        "confusion": {k: r[k] for k in ("tp", "fp", "tn", "fn")},
    }


def main(run_dir: str, art_dir: str | Path | None = None) -> dict:
    run = Path(run_dir)
    z = np.load(run / "predictions.npz")
    rec = json.loads((run / "train_record.json").read_text())
    yv, yt = z["val_labels"].astype(int), z["test_labels"].astype(int)

    names = sorted({k[len("val_logits_"):] for k in z.files if k.startswith("val_logits_")})
    out = {"protocol": __doc__.split("Usage:")[0].strip(), "band": [BAND_LO, BAND_HI], "candidates": {}, "test": {}}
    for name in names:
        lv, lt = z[f"val_logits_{name}"], z[f"test_logits_{name}"]
        t = cal.fit_temperature(lv, yv)
        out["candidates"][name] = {
            "temperature": t,
            "val": score(cal.sigmoid(lv / t), yv),
            "test": score(cal.sigmoid(lt / t), yt),
        }

    # Shipped model = the ONNX export of the selected fine-tune seed.
    lv, lt = z["val_logits_onnx"], z["test_logits_onnx"]
    t = cal.fit_temperature(lv, yv)
    pv, pt = cal.sigmoid(lv / t), cal.sigmoid(lt / t)
    raw_v, raw_t = cal.sigmoid(lv), cal.sigmoid(lt)
    shipped = score(pt, yt)
    shipped["uncalibrated"] = {k: score(raw_t, yt)[k] for k in ("ece", "brier", "nll")}
    sel = cal.selective_report(pt, yt, BAND_LO, BAND_HI)
    sel_val = cal.selective_report(pv, yv, BAND_LO, BAND_HI)
    thr = cal.threshold_for_specificity(pv, yv, TARGET_SPEC)
    hs = cal.classification_report(pt, yt, thr)

    ci = {}
    for key, fn in {
        "auroc": cal.auroc,
        "accuracy": lambda p, y: np.mean((p >= 0.5) == y),
        "ece": cal.ece,
        "brier": cal.brier,
        "coverage": lambda p, y: cal.selective_report(p, y, BAND_LO, BAND_HI)["coverage"],
        "selective_accuracy": lambda p, y: cal.selective_report(p, y, BAND_LO, BAND_HI)["selective_accuracy"],
    }.items():
        ci[key] = cal.bootstrap_ci(fn, pt, yt, n_boot=N_BOOT, seed=0)

    seeds = [n for n in names if n.startswith("finetune_s")]
    seed_auroc = [out["candidates"][n]["test"]["auroc"] for n in seeds]
    out["test"] = {
        "n": int(len(yt)),
        "pneumonia_share": float(yt.mean()),
        "majority_class_accuracy": float(max(yt.mean(), 1 - yt.mean())),
        "shipped": {**shipped, "temperature": t, "ci95": ci},
        "triage": sel,
        "high_sensitivity_point": {
            "threshold_from_val": thr,
            "sensitivity": hs["sensitivity"],
            "specificity": hs["specificity"],
            "target_specificity_on_val": TARGET_SPEC,
        },
        "seed_spread_auroc": {"seeds": seeds, "values": seed_auroc, "mean": float(np.mean(seed_auroc)), "sd": float(np.std(seed_auroc, ddof=1)) if len(seed_auroc) > 1 else 0.0},
    }
    out["val"] = {"n": int(len(yv)), "pneumonia_share": float(yv.mean()), "shipped": score(pv, yv), "triage": sel_val}
    out["shipped_name"] = rec["shipped"]
    out["onnx"] = rec["onnx"]
    out["training"] = {
        n: {k: v for k, v in c.items() if k != "history"} for n, c in rec["candidates"].items()
    }
    out["headline"] = {
        "test_n": int(len(yt)),
        "coverage": sel["coverage"],
        "selective_accuracy": sel["selective_accuracy"],
        "referred": sel["referred"],
        "missed_pneumonia": sel["missed_pneumonia"],
        "answered_normal": sel["answered_normal"],
        "false_alarm": sel["false_alarm"],
        "pneumonia_n": int(yt.sum()),
        "normal_n": int(len(yt) - yt.sum()),
    }

    art = Path(art_dir) if art_dir else ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    (art / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    (art / "calibration.json").write_text(
        json.dumps({"temperature": t, "band_lo": BAND_LO, "band_hi": BAND_HI, "high_sensitivity_threshold": thr, "model": rec["shipped"]}, indent=2)
    )
    np.savez_compressed(art / "eval_arrays.npz", val_prob=pv, test_prob=pt, test_prob_raw=raw_t, val_prob_raw=raw_v, val_labels=yv, test_labels=yt, test_cam=z["test_cam_onnx"])

    s = out["test"]["shipped"]
    print(f"shipped {rec['shipped']}  T={t:.3f}")
    print(f"test AUROC {s['auroc']:.4f} {ci['auroc']}  acc {s['accuracy']:.4f}  ECE {s['uncalibrated']['ece']:.4f} -> {s['ece']:.4f}  Brier {s['brier']:.4f}")
    print(f"triage: coverage {sel['coverage']:.3f} {ci['coverage']}  selective acc {sel['selective_accuracy']:.4f} {ci['selective_accuracy']}  referred {sel['referred']}  missed pneumonia {sel['missed_pneumonia']}  false alarm {sel['false_alarm']}")
    print(f"90%-spec point from val: sens {hs['sensitivity']:.3f} spec {hs['specificity']:.3f}")
    for n, c in out["candidates"].items():
        print(f"  {n:14s} val AUROC {c['val']['auroc']:.4f}  test AUROC {c['test']['auroc']:.4f}  test acc {c['test']['accuracy']:.4f}  test ECE {c['test']['ece']:.4f}")
    return out


if __name__ == "__main__":
    main(sys.argv[1])
