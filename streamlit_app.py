"""X-Ray Triage, Calibrated (Neural Storm 2026, HI-01).

Run locally:  streamlit run streamlit_app.py
Without the model file (UI only): TRIAGE_STUB=1 streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from triage.predictor import Predictor, overlay  # noqa: E402
from triage.preprocess import prepare  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
EXAMPLES = ROOT / "app" / "examples"
DOCS = ROOT / "docs"

st.set_page_config(page_title="X-Ray Triage, Calibrated", page_icon="🩻", layout="wide")


@st.cache_resource
def get_predictor() -> Predictor:
    if os.environ.get("TRIAGE_STUB") == "1":
        from triage.stub import STUB_CONFIG, StubSession

        return Predictor(StubSession(), STUB_CONFIG)
    return Predictor.from_files(ARTIFACTS / "triage.onnx", ARTIFACTS / "calibration.json")


@st.cache_data
def load_json(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def pct(x, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


metrics = load_json(str(ARTIFACTS / "metrics.json"))
examples = load_json(str(EXAMPLES / "examples.json")).get("examples", [])
head = metrics.get("headline", {})

st.title("X-Ray Triage, Calibrated")
st.caption("Neural Storm 2026 · Human Impact · Problem HI-01 (medical imaging)")
if head:
    st.markdown(
        f"#### On {head['test_n']} held-out pediatric chest X-rays, it answered **{pct(head['coverage'], 0)}** "
        f"of them with **{pct(head['selective_accuracy'])}** accuracy and referred the rest to a radiologist."
    )
st.warning(
    "Research demo, not a medical device. It was trained on one public pediatric dataset and has not been "
    "validated for clinical use. Never use it to make or rule out a diagnosis.",
    icon="⚠️",
)

tab_try, tab_rel, tab_lim = st.tabs(["Triage an X-ray", "How reliable is it?", "Limits and data"])

with tab_try:
    left, right = st.columns([1, 2])
    with left:
        source = st.radio("Image", ["Pick an example", "Upload my own"], horizontal=True)
        img, truth = None, None
        if source == "Pick an example" and examples:
            names = [e["title"] for e in examples]
            choice = st.selectbox("Example from the held-out test set", names)
            ex = examples[names.index(choice)]
            img, truth = Image.open(EXAMPLES / ex["file"]), ex["truth"]
        elif source == "Upload my own":
            up = st.file_uploader("Chest X-ray (PNG or JPEG, frontal view)", type=["png", "jpg", "jpeg"])
            st.caption("Images are processed in memory and never saved.")
            if up is not None:
                try:
                    img = Image.open(up)
                    img.load()
                except Exception:
                    st.error("That file could not be read as an image.")
                    img = None
        else:
            st.info("No examples are bundled with this copy.")
    with right:
        if img is None:
            st.info("Pick an example or upload an image to see a prediction.")
        else:
            x = prepare(img)
            res = get_predictor().predict(x)
            c1, c2 = st.columns(2)
            c1.image(Image.fromarray((x[0, 0] * 255).astype(np.uint8)).resize((320, 320)), caption="What the model sees (128 px)")
            c2.image(
                Image.fromarray(overlay(x[0, 0], res.heatmap)).resize((320, 320)),
                caption=f"Where the evidence for '{res.leaning}' is (class activation map)",
            )
            box = {"pneumonia": st.error, "normal": st.success, "unsure": st.warning}[res.band]
            box(f"**{res.label}**  ·  calibrated chance of pneumonia: **{pct(res.prob)}**")
            st.markdown(
                f"The model only gives an answer when it is at least {pct(get_predictor().hi, 0)} sure either way "
                f"(probability below {pct(get_predictor().lo, 0)} or above {pct(get_predictor().hi, 0)}). "
                "Anything in between goes to a radiologist."
            )
            if truth is not None:
                st.caption(f"Ground truth for this test image: **{truth}**.")
            with st.expander("Details"):
                st.write({"raw logit": round(res.logit, 3), "temperature": get_predictor().temperature, "band": res.band})

with tab_rel:
    shipped = metrics.get("test", {}).get("shipped", {})
    if not shipped:
        st.info("Metrics appear here once the model has been evaluated.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AUROC (test)", f"{shipped['auroc']:.3f}")
        m2.metric("Accuracy at 50%", pct(shipped["accuracy"]))
        m3.metric("Calibration error (ECE)", f"{shipped['ece']:.3f}")
        m4.metric("Brier score", f"{shipped['brier']:.3f}")
        st.markdown(
            "**Calibrated** means that when the app says 80%, about 80 of 100 such X-rays really show pneumonia. "
            "The reliability diagram checks that on the held-out test set; the coverage curve shows how accuracy "
            "rises as the app refers more of its least certain cases."
        )
        for name, cap in [("reliability.png", "Reliability diagram (test set)"), ("coverage.png", "Coverage vs accuracy (test set)")]:
            if (DOCS / name).exists():
                st.image(str(DOCS / name), caption=cap)

with tab_lim:
    st.markdown((ROOT / "app" / "limits.md").read_text() if (ROOT / "app" / "limits.md").exists() else "")
