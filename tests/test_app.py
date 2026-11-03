"""Drive the Streamlit page headlessly with the stub session (the real model is never loaded)."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


@pytest.fixture(autouse=True)
def stub(monkeypatch):
    monkeypatch.setenv("TRIAGE_STUB", "1")


def test_page_renders_title_tabs_and_disclaimer():
    at = st_testing.AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "X-Ray Triage, Calibrated"
    assert any("not a medical device" in w.value for w in at.warning)
    assert len(at.tabs) == 3


def test_example_gallery_gives_a_banded_prediction():
    at = st_testing.AppTest.from_file(APP, default_timeout=30).run()
    if not at.selectbox:
        pytest.skip("no bundled examples yet")
    at.selectbox[0].select_index(len(at.selectbox[0].options) - 1).run()
    assert not at.exception
    banners = [b.value for b in list(at.error) + list(at.success) + list(at.warning)]
    assert any("calibrated chance of pneumonia" in b for b in banners)
    assert any("Ground truth" in c.value for c in at.caption)


def test_upload_path_switch_shows_prompt():
    at = st_testing.AppTest.from_file(APP, default_timeout=30).run()
    at.radio[0].set_value("Upload my own").run()
    assert not at.exception
    assert any("Pick an example or upload" in i.value for i in at.info)


def test_stub_prediction_path_matches_predictor(tmp_path):
    from triage.predictor import Predictor
    from triage.preprocess import prepare
    from triage.stub import STUB_CONFIG, StubSession

    img = Image.fromarray((np.random.default_rng(0).random((200, 150)) * 255).astype(np.uint8))
    r = Predictor(StubSession(), STUB_CONFIG).predict(prepare(img))
    assert 0.0 < r.prob < 1.0 and r.heatmap.shape == (128, 128)
