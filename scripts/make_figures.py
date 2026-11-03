"""Render README/app figures and the app's example images from saved evaluation outputs.

Reads artifacts/metrics.json, artifacts/eval_arrays.npz (written by evaluate.py) and the test images
from the dataset file. Plotting only; no model is run here. The heatmaps come from the ONNX model's
own outputs saved by the training job.

Usage: python scripts/make_figures.py data/pneumoniamnist_128.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triage import calibration as cal  # noqa: E402
from triage.predictor import heatmap_from_cam, overlay  # noqa: E402

DOCS, EX = ROOT / "docs", ROOT / "app" / "examples"
INK, MUTED, BLUE, ORANGE, GRID = "#1f2933", "#6b7785", "#2563eb", "#d97706", "#e5e7eb"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


def reliability(m, a):
    y, raw, cal_p = a["test_labels"], a["test_prob_raw"], a["test_prob"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
    ax.plot([0, 1], [0, 1], color=GRID, lw=2, zorder=0, label="perfect calibration")
    for p, color, name in [(raw, MUTED, "before (raw network)"), (cal_p, BLUE, "after temperature scaling")]:
        b = cal.reliability_bins(p, y, 10)
        k = b["count"] >= 5
        ax.plot(b["mean_prob"][k], b["pos_rate"][k], "o-", color=color, lw=2, ms=5, label=f"{name}: ECE {cal.ece(p, y):.3f}")
    ax.set(xlabel="Predicted chance of pneumonia", ylabel="Share that really had pneumonia", xlim=(0, 1), ylim=(0, 1))
    ax.set_title(f"Reliability on {len(y)} held-out test X-rays", color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "reliability.png")
    plt.close(fig)


def coverage(m, a):
    y, p = a["test_labels"], a["test_prob"]
    cov, acc = cal.coverage_curve(p, y)
    tri = m["test"]["triage"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=150)
    ax.plot(100 * cov, 100 * acc, color=BLUE, lw=2.2)
    ax.axhline(100 * m["test"]["shipped"]["accuracy"], color=GRID, lw=1.5)
    ax.text(97, 100 * m["test"]["shipped"]["accuracy"] - 1.2, "answer everything", ha="right", va="top", color=MUTED, fontsize=9)
    ax.plot(100 * tri["coverage"], 100 * tri["selective_accuracy"], "o", color=ORANGE, ms=8, zorder=3)
    ax.annotate(
        f"app's 20%/80% band: answers {100 * tri['coverage']:.0f}%,\n{100 * tri['selective_accuracy']:.1f}% of those correct",
        (100 * tri["coverage"], 100 * tri["selective_accuracy"]), xytext=(-12, -38), textcoords="offset points",
        ha="right", color=INK, fontsize=9,
    )
    ax.set(xlabel="Share of X-rays the app answers (%)", ylabel="Accuracy on the answered ones (%)")
    ax.set_xlim(100, 5)
    ax.set_title("Refer the least certain cases, accuracy goes up", color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "coverage.png")
    plt.close(fig)


def pick_examples(a):
    """Deterministic picks: the clearest correct cases, uncertain ones, and the most confident mistake."""
    y, p = a["test_labels"], a["test_prob"]
    idx = np.arange(len(y))
    picks = []
    for cls, name in [(0, "normal"), (1, "pneumonia")]:
        ok = idx[(y == cls) & ((p >= 0.5) == bool(cls))]
        conf = np.abs(p[ok] - 0.5)
        for j in ok[np.argsort(-conf, kind="mergesort")][[0, len(ok) // 3]]:
            picks.append((int(j), f"Confident {name}"))
    unsure = idx[(p > 0.2) & (p < 0.8)]
    for j in unsure[np.argsort(np.abs(p[unsure] - 0.5), kind="mergesort")][:2]:
        picks.append((int(j), "Unsure case"))
    wrong = idx[(p >= 0.5) != (y == 1)]
    if len(wrong):
        j = wrong[np.argmax(np.abs(p[wrong] - 0.5))]
        picks.append((int(j), "Its most confident mistake"))
    return picks


def examples(a, images):
    EX.mkdir(parents=True, exist_ok=True)
    y, p, cams = a["test_labels"], a["test_prob"], a["test_cam"]
    out, tiles = [], []
    for k, (j, kind) in enumerate(pick_examples(a)):
        fname = f"test_{j:03d}.png"
        Image.fromarray(images[j]).save(EX / fname)
        truth = "pneumonia" if y[j] else "normal"
        out.append({"title": f"{k + 1}. {kind} (test image #{j})", "file": fname, "truth": truth, "prob": float(p[j])})
        heat = heatmap_from_cam(cams[j], p[j] >= 0.5)
        tiles.append((images[j], overlay(images[j] / 255.0, heat), kind, truth, p[j]))
    (EX / "examples.json").write_text(json.dumps({"source": "PneumoniaMNIST 128 px test split (MedMNIST v3, CC BY 4.0)", "examples": out}, indent=2))

    show = [t for t in tiles if t[2] in ("Confident normal", "Confident pneumonia", "Unsure case", "Its most confident mistake")]
    seen, chosen = set(), []
    for t in show:
        if t[2] not in seen:
            seen.add(t[2])
            chosen.append(t)
    fig, axes = plt.subplots(2, len(chosen), figsize=(3.0 * len(chosen), 6.4), dpi=150)
    for c, (img, ov, kind, truth, prob) in enumerate(chosen):
        axes[0, c].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0, c].set_title(f"{kind}\ntruth: {truth}", fontsize=10, color=INK)
        axes[1, c].imshow(ov)
        axes[1, c].set_title(f"p(pneumonia) = {prob:.2f}", fontsize=10, color=INK)
        for r in (0, 1):
            axes[r, c].axis("off")
    fig.suptitle("Test X-rays and where the model's evidence is (class activation map)", color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(DOCS / "examples.png")
    plt.close(fig)


def main(npz_path: str) -> None:
    DOCS.mkdir(exist_ok=True)
    m = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    a = np.load(ROOT / "artifacts" / "eval_arrays.npz")
    images = np.load(npz_path)["test_images"]
    reliability(m, a)
    coverage(m, a)
    examples(a, images)
    print("wrote", ", ".join(sorted(p.name for p in DOCS.glob("*.png"))), "and", len(list(EX.glob("*.png"))), "examples")


if __name__ == "__main__":
    main(sys.argv[1])
