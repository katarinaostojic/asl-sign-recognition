import numpy as np
import pandas as pd

from aslrec import metrics, splits
from aslrec.wordbuilder import WordBuilder


def _df(n_per_class=3000, n_classes=3):
    rows = [(c, i + 1) for c in range(n_classes) for i in range(n_per_class)]
    return pd.DataFrame(rows, columns=["label", "idx"])


def test_block_split_proportions_and_no_leak():
    df = _df()
    s = splits.block_split(df, block_size=100, seed=0)
    assert set(s.unique()) == {"train", "val", "test"}
    frac = s.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.03
    assert abs(frac["test"] - 0.15) < 0.03
    # every block of 100 consecutive frames stays in exactly one split
    df["split"] = s
    df["block"] = (df["idx"] - 1) // 100
    assert (df.groupby(["label", "block"])["split"].nunique() == 1).all()


def test_block_split_is_deterministic():
    df = _df(1000, 2)
    a = splits.block_split(df, seed=1)
    b = splits.block_split(df, seed=1)
    assert (a == b).all()


def test_random_split_stratified():
    df = _df(300, 3)
    s = splits.random_split(df, seed=0)
    for _, g in df.assign(s=s).groupby("label"):
        assert set(g["s"]) == {"train", "val", "test"}


def test_tiny_split_does_not_crash():
    df = pd.DataFrame({"label": [0, 0, 0], "idx": [1, 2, 3]})
    s = splits.block_split(df, block_size=1)
    assert len(s) == 3 and s.notna().all()


def test_metrics_and_ece():
    y = np.array([0, 1, 2, 2])
    p = np.array([0, 1, 2, 1])
    m = metrics.summarize(y, p, 3)
    assert m["accuracy"] == 0.75
    probs = np.eye(3)[p]
    ece, _ = metrics.expected_calibration_error(probs, y)
    assert abs(ece - 0.25) < 1e-6  # fully confident, 75% accurate
    cm = metrics.confusion(y, p, 3)
    assert cm[2, 1] == 1
    assert metrics.top_confusions(cm, ["a", "b", "c"], 3) == [("c", "b", 1)]
    cov, acc = metrics.risk_coverage(probs, y)
    assert len(cov) == len(acc)


def test_wordbuilder_types_and_delete():
    wb = WordBuilder(window=5, min_votes=3, min_confidence=0.5)

    def show(label, n=3):
        for _ in range(n):
            wb.update(label, 0.9)

    show("H"); show("nothing"); show("I"); show("space"); show("nothing")
    show("X"); show("del"); show("nothing")
    assert wb.text == "HI "


def test_wordbuilder_hold_does_not_repeat_and_ignores_low_confidence():
    wb = WordBuilder(window=5, min_votes=3, min_confidence=0.6)
    for _ in range(20):
        wb.update("A", 0.95)
    assert wb.text == "A"
    wb2 = WordBuilder(window=5, min_votes=3, min_confidence=0.6)
    for _ in range(20):
        wb2.update("B", 0.3)
    assert wb2.text == ""


def test_mcnemar_and_bootstrap():
    a = np.array([True] * 90 + [False] * 10)
    b = np.array([True] * 50 + [False] * 50)
    assert metrics.mcnemar_pvalue(a, b) < 0.001
    assert metrics.mcnemar_pvalue(a, a) == 1.0
    groups = np.repeat(np.arange(10), 10)
    lo, hi = metrics.cluster_bootstrap_ci(a, groups, n_boot=200)
    assert lo <= 0.9 <= hi
