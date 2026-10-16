import numpy as np

from aslrec import robustness
from aslrec.imaging import resize_rgb, square_crop_around


def test_perturbations_preserve_shape_and_dtype():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    for name, fn in robustness.PERTURBATIONS.items():
        out = fn(img, np.random.default_rng(1))
        assert out.shape == img.shape and out.dtype == np.uint8, name
    assert (robustness.PERTURBATIONS["none"](img, rng) == img).all()
    assert robustness.darker(img, rng).mean() < img.mean()


def test_resize_and_crop():
    img = np.zeros((480, 640, 3), np.uint8)
    lm = np.zeros((21, 3), np.float32)
    lm[:, 0] = np.linspace(0.4, 0.6, 21)
    lm[:, 1] = np.linspace(0.3, 0.7, 21)
    crop = square_crop_around(img, lm)
    assert crop.shape[0] == crop.shape[1] and crop.shape[0] > 0
    assert resize_rgb(crop, 96).shape == (96, 96, 3)
    # hand at the very edge must not crash (border padding)
    lm[:, 0] = np.linspace(0.9, 1.05, 21)
    assert square_crop_around(img, lm).shape[0] > 0
