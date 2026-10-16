"""Step 9: compare all trained models on the held-out test split.

Reads results/predictions/*.npz (written by the training scripts) and produces
  results/summary.csv                  metrics per model
  results/pairwise_mcnemar.csv         significance of accuracy differences
  results/top_confusions.csv           most frequent mistakes per model
  results/REPORT.md                    all tables in Markdown (paste into the README)
  results/figures/*.png                confusion matrices, per-class F1, reliability, coverage, ...

Two views are reported for the landmark models:
  end-to-end  - every test image counts; a missed hand detection is predicted as 'nothing'
                (correct for the class 'nothing', a mistake for every other class)
  common      - only test images where MediaPipe found a hand, for ALL models
                (fair comparison of landmarks vs. full image on identical inputs)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from aslrec import config, data, metrics

ORDER = ["knn_raw", "knn", "mlp_raw", "mlp", "gnn", "cnn", "mobilenet"]
NAMES = config.CLASS_NAMES


def load_predictions():
    preds = {}
    for p in sorted(config.PRED_DIR.glob("*.npz")):
        z = np.load(p)
        preds[p.stem] = {k: z[k] for k in z.files}
    ordered = [n for n in ORDER if n in preds] + sorted(n for n in preds if n not in ORDER)
    if not ordered:
        raise SystemExit("No predictions found. Train at least one model first.")
    ref = preds[ordered[0]]["test_rows"]
    for n in ordered:
        assert np.array_equal(preds[n]["test_rows"], ref), f"{n} was evaluated on a different test set"
    return {n: preds[n] for n in ordered}


def end_to_end_probs(p):
    probs = p["probs"].copy()
    miss = ~p["detected"]
    probs[miss] = 0.0
    probs[miss, config.NOTHING_INDEX] = 1.0
    return probs


def md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                         else ("" if pd.isna(v) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    preds = load_predictions()
    models_ = list(preds)
    df = data.load_index()
    rows = preds[models_[0]]["test_rows"]
    y = preds[models_[0]]["y_true"]
    n = len(y)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    common = np.ones(n, dtype=bool)
    for m in models_:
        common &= preds[m]["detected"]
    groups = (df["label"].to_numpy()[rows] * 100000 + (df["idx"].to_numpy()[rows] - 1) // config.BLOCK_SIZE)
    print(f"{n} test images, {common.sum()} ({common.mean():.1%}) detected by every landmark model")

    runs = {}
    for m in models_:
        f = config.RUNS_DIR / m / "metrics.json"
        runs[m] = json.loads(f.read_text()) if f.exists() else {}

    records, e2e_probs, correct_e2e = [], {}, {}
    for m in models_:
        p = preds[m]
        probs = end_to_end_probs(p)
        e2e_probs[m] = probs
        pred = probs.argmax(1)
        correct_e2e[m] = pred == y
        for view, mask in (("end-to-end", np.ones(n, bool)), ("common", common)):
            s = metrics.summarize(y[mask], pred[mask], config.NUM_CLASSES)
            ece, _ = metrics.expected_calibration_error(probs[mask], y[mask])
            conf = metrics.confidence_summary(probs[mask], y[mask])
            lo, hi = metrics.cluster_bootstrap_ci(correct_e2e[m][mask], groups[mask])
            records.append({
                "model": m, "view": view, "n_test": int(mask.sum()),
                "accuracy": s["accuracy"], "acc_ci95_low": lo, "acc_ci95_high": hi,
                "precision_macro": s["precision_macro"], "recall_macro": s["recall_macro"],
                "f1_macro": s["f1_macro"], "ece": ece, **conf,
                "parameters": runs[m].get("parameters"), "model_file_mb": runs[m].get("model_file_mb"),
                "train_seconds": runs[m].get("train_seconds"),
                "detected_fraction": float(p["detected"].mean()),
            })
    summary = pd.DataFrame(records)
    summary.to_csv(config.RESULTS_DIR / "summary.csv", index=False)

    # ---- significance: paired McNemar on the end-to-end correctness
    pw = [{"model_a": a, "model_b": b, "acc_a": correct_e2e[a].mean(), "acc_b": correct_e2e[b].mean(),
           "p_value": metrics.mcnemar_pvalue(correct_e2e[a], correct_e2e[b])}
          for i, a in enumerate(models_) for b in models_[i + 1:]]
    pairwise = pd.DataFrame(pw)
    pairwise.to_csv(config.RESULTS_DIR / "pairwise_mcnemar.csv", index=False)

    # ---- confusion matrices + top confusions
    tops, cms = [], {}
    for m in models_:
        cm = metrics.confusion(y, e2e_probs[m].argmax(1), config.NUM_CLASSES)
        cms[m] = cm
        for t, p_, c in metrics.top_confusions(cm, NAMES, 10):
            tops.append({"model": m, "true": t, "predicted": p_, "count": c})
    pd.DataFrame(tops).to_csv(config.RESULTS_DIR / "top_confusions.csv", index=False)

    ncols = min(3, len(models_))
    nrows = int(np.ceil(len(models_) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6.3 * nrows), squeeze=False)
    for ax, m in zip(axes.ravel(), models_):
        cmn = cms[m] / np.maximum(cms[m].sum(1, keepdims=True), 1)
        sns.heatmap(cmn, ax=ax, cmap="Blues", vmin=0, vmax=1, cbar=False, xticklabels=NAMES,
                    yticklabels=NAMES, square=True)
        ax.set_title(f"{m}  (acc {correct_e2e[m].mean():.3f})")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(models_):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "confusion_matrices.png", dpi=130)
    plt.close()

    # ---- per-class F1 heat-map (models x classes)
    f1 = np.stack([metrics.summarize(y, e2e_probs[m].argmax(1), config.NUM_CLASSES)["per_class"]["f1"]
                   for m in models_])
    plt.figure(figsize=(14, 0.55 * len(models_) + 1.5))
    sns.heatmap(f1, annot=False, cmap="RdYlGn", vmin=0.5, vmax=1, xticklabels=NAMES, yticklabels=models_,
                cbar_kws={"label": "F1"})
    plt.title("Per-class F1 (end-to-end)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "per_class_f1.png", dpi=150)
    plt.close()

    # ---- accuracy bar chart with CI
    e2e = summary[summary["view"] == "end-to-end"].set_index("model").loc[models_]
    com = summary[summary["view"] == "common"].set_index("model").loc[models_]
    x = np.arange(len(models_))
    plt.figure(figsize=(1.3 * len(models_) + 3, 4))
    plt.bar(x - 0.2, e2e["accuracy"], 0.4, label="end-to-end (all test images)",
            yerr=[e2e["accuracy"] - e2e["acc_ci95_low"], e2e["acc_ci95_high"] - e2e["accuracy"]], capsize=3)
    plt.bar(x + 0.2, com["accuracy"], 0.4, label="images where a hand was detected",
            yerr=[com["accuracy"] - com["acc_ci95_low"], com["acc_ci95_high"] - com["accuracy"]], capsize=3)
    plt.xticks(x, models_, rotation=20)
    plt.ylim(max(0, min(e2e["accuracy"].min(), com["accuracy"].min()) - 0.1), 1.0)
    plt.ylabel("test accuracy (95% block-bootstrap CI)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "accuracy_comparison.png", dpi=150)
    plt.close()

    # ---- confidence analysis: reliability diagrams, risk-coverage, confidence histograms
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for m in models_:
        _, b = metrics.expected_calibration_error(e2e_probs[m], y)
        axes[0].plot(b["conf"], b["acc"], "o-", ms=3, label=m)
        cov, acc = metrics.risk_coverage(e2e_probs[m], y)
        axes[1].plot(cov, acc, label=m)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="mean confidence", ylabel="accuracy", title="Reliability diagram")
    axes[1].set(xlabel="coverage (fraction of most-confident predictions kept)", ylabel="accuracy",
                title="Accuracy vs. coverage")
    axes[1].legend(fontsize=7)
    bins = np.linspace(0, 1, 21)
    for m in models_:
        conf = e2e_probs[m].max(1)
        axes[2].hist(conf[~correct_e2e[m]], bins=bins, histtype="step", label=m, density=True)
    axes[2].set(xlabel="confidence", title="Confidence of WRONG predictions")
    for a in axes:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "confidence_analysis.png", dpi=150)
    plt.close()

    # ---- report
    lat = config.RESULTS_DIR / "latency.csv"
    cols = ["model", "accuracy", "acc_ci95_low", "acc_ci95_high", "precision_macro", "recall_macro",
            "f1_macro", "ece"]
    parts = [
        "# Results\n",
        f"Test set: {n} images from blocks of frames never seen in training "
        f"({common.mean():.1%} detected by MediaPipe).\n",
        "## End-to-end (every test image counts)\n", md_table(e2e.reset_index()[cols]), "",
        "## Common subset (hand detected, identical inputs for all models)\n",
        md_table(com.reset_index()[cols]), "",
        "## Size and training cost\n",
        md_table(e2e.reset_index()[["model", "parameters", "model_file_mb", "train_seconds",
                                    "detected_fraction"]], "{:.2f}"), "",
        "## Confidence\n",
        md_table(e2e.reset_index()[["model", "mean_conf_correct", "mean_conf_wrong",
                                    "frac_wrong_below_0.9", "frac_correct_above_0.9"]]), "",
        "## Most frequent mistakes (end-to-end)\n",
        md_table(pd.DataFrame(tops), "{:.0f}"), "",
        "## Pairwise significance (McNemar)\n", md_table(pairwise, "{:.4g}"), "",
    ]
    if lat.exists():
        parts += ["## Prediction time per image (ms)\n", md_table(pd.read_csv(lat), "{:.2f}"), ""]
    (config.RESULTS_DIR / "REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    print("\n" + md_table(e2e.reset_index()[cols]))
    print("\nWrote results/summary.csv, REPORT.md and figures/*.png")


if __name__ == "__main__":
    main()
