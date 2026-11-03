import json

import numpy as np
import pytest
from PIL import Image

from triage.predictor import Predictor, heatmap_from_cam, load_config, overlay
from triage.preprocess import from_uint8, prepare
from triage.stub import STUB_CONFIG, StubSession


@pytest.mark.parametrize(
    "img",
    [
        Image.new("RGB", (300, 200), (120, 60, 30)),
        Image.new("RGBA", (64, 90), (255, 255, 255, 0)),
        Image.new("L", (128, 128), 200),
        Image.fromarray(np.linspace(0, 60000, 400 * 500).reshape(400, 500).astype(np.uint16)),
        Image.new("P", (50, 50)),
    ],
)
def test_prepare_gives_one_gray_128_image_in_unit_range(img):
    x = prepare(img)
    assert x.shape == (1, 1, 128, 128)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() <= x.max() <= 1.0


def test_prepare_center_crops_wide_images():
    a = np.zeros((100, 300), np.uint8)
    a[:, 100:200] = 255  # the centre square is all white, the sides black
    x = prepare(Image.fromarray(a))
    assert x.min() > 0.99


def test_16_bit_images_are_stretched_not_clipped():
    a = np.tile(np.linspace(1000, 3000, 256), (256, 1)).astype(np.uint16)
    x = prepare(Image.fromarray(a))
    assert x.max() - x.min() > 0.9


def test_from_uint8_rejects_wrong_size():
    with pytest.raises(ValueError):
        from_uint8(np.zeros((64, 64), np.uint8))
    assert from_uint8(np.full((128, 128), 255, np.uint8)).max() == 1.0


def test_heatmap_explains_the_leaning_class():
    cam = np.zeros((7, 7), np.float32)
    cam[1, 1] = 3.0  # pushes toward pneumonia
    cam[5, 5] = -2.0  # pushes toward normal
    toward_pneu = heatmap_from_cam(cam, True)
    toward_norm = heatmap_from_cam(cam, False)
    assert toward_pneu.shape == (128, 128)
    assert toward_pneu.max() == pytest.approx(1.0, abs=0.02)
    assert np.unravel_index(toward_pneu.argmax(), toward_pneu.shape)[0] < 64
    assert np.unravel_index(toward_norm.argmax(), toward_norm.shape)[0] > 64


def test_heatmap_of_all_zero_cam_is_zero():
    assert heatmap_from_cam(np.zeros((7, 7)), True).max() == 0.0


def test_overlay_shape_and_type():
    o = overlay(np.full((128, 128), 0.5), np.random.default_rng(0).random((128, 128)))
    assert o.shape == (128, 128, 3) and o.dtype == np.uint8


def test_predictor_bands_follow_calibrated_probability():
    pred = Predictor(StubSession(), STUB_CONFIG)
    dark = pred.predict(np.zeros((1, 1, 128, 128), np.float32))
    bright = pred.predict(np.ones((1, 1, 128, 128), np.float32))
    mid = pred.predict(np.full((1, 1, 128, 128), 0.45, np.float32))
    assert dark.band == "normal" and dark.leaning == "normal" and dark.prob < 0.2
    assert bright.band == "pneumonia" and bright.prob > 0.8
    assert mid.band == "unsure" and mid.prob == pytest.approx(0.5)
    assert bright.label == "Likely pneumonia"


def test_temperature_softens_probability():
    hot = Predictor(StubSession(), {**STUB_CONFIG, "temperature": 1.0})
    cool = Predictor(StubSession(), {**STUB_CONFIG, "temperature": 3.0})
    x = np.full((1, 1, 128, 128), 0.7, np.float32)
    assert 0.5 < cool.predict(x).prob < hot.predict(x).prob


def test_predictor_rejects_wrong_shape():
    with pytest.raises(ValueError):
        Predictor(StubSession(), STUB_CONFIG).predict(np.zeros((1, 3, 128, 128), np.float32))


def test_load_config_validates(tmp_path):
    good = tmp_path / "ok.json"
    good.write_text(json.dumps(STUB_CONFIG))
    assert load_config(good)["temperature"] == 1.5
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**STUB_CONFIG, "band_lo": 0.6}))
    with pytest.raises(ValueError):
        load_config(bad)
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"temperature": 1.0}))
    with pytest.raises(KeyError):
        load_config(missing)
