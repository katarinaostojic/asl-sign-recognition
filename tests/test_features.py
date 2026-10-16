import numpy as np

from aslrec import features as F


def _random_hand(rng, n=4):
    return rng.normal(size=(n, 21, 3)).astype(np.float32)


def test_shapes():
    lm = _random_hand(np.random.default_rng(0))
    assert F.raw_features(lm).shape == (4, 60)
    assert F.engineered_features(lm).shape == (4, 94)
    assert F.graph_features(lm).shape == (4, 21, 6)


def test_translation_and_scale_invariance():
    rng = np.random.default_rng(1)
    lm = _random_hand(rng, 3)
    moved = lm * 3.7 + np.array([0.4, -2.0, 1.0], dtype=np.float32)
    for name in ("raw", "engineered", "graph"):
        a = F.FEATURE_BUILDERS[name](lm)
        b = F.FEATURE_BUILDERS[name](moved)
        np.testing.assert_allclose(a, b, atol=1e-4)


def test_single_sample_matches_batch():
    lm = _random_hand(np.random.default_rng(2), 5)
    batch = F.engineered_features(lm)
    single = np.stack([F.engineered_features(x) for x in lm])
    np.testing.assert_allclose(batch, single, atol=1e-6)


def test_angles_are_cosines():
    lm = _random_hand(np.random.default_rng(3), 10)
    f = F.engineered_features(lm)
    angles = f[:, 60:79]
    assert angles.min() >= -1.0001 and angles.max() <= 1.0001


def test_degenerate_hand_has_no_nans():
    lm = np.zeros((1, 21, 3), dtype=np.float32)
    for name in ("raw", "engineered", "graph"):
        assert np.isfinite(F.FEATURE_BUILDERS[name](lm)).all()


def test_adjacency():
    a = F.adjacency_matrix(normalize=False)
    assert a.shape == (21, 21)
    assert np.allclose(a, a.T)
    assert np.all(np.diag(a) == 1)
    assert a[0, 1] == 1 and a[4, 8] == 0
    an = F.adjacency_matrix()
    assert np.allclose(an, an.T)
