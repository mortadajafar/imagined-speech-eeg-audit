"""Imbalance-aware category metrics for the zero-shot 39-way classification: accuracy, balanced accuracy,
macro-F1 and a label-permutation null, computed from the per-trial prediction files (fields 'cat' and
'pred_cat'). Used for the numbers quoted in the Results (EEGNet, three seeds)."""

import glob
import json
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def metrics(preds, n_perm=2000, seed=0):
    y = np.array([p["cat"] for p in preds])
    yhat = np.array([p["pred_cat"] for p in preds])
    keep = y >= 0
    y, yhat = y[keep], yhat[keep]
    rng = np.random.default_rng(seed)
    null = np.array([(rng.permutation(y) == yhat).mean() for _ in range(n_perm)])
    return dict(
        n=int(len(y)),
        accuracy=float((y == yhat).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y, yhat)),
        macro_f1=float(f1_score(y, yhat, average="macro")),
        null_mean=float(null.mean()),
        null_p95=float(np.percentile(null, 95)),
    )


def main(
    patterns=(
        "kaggle/logs/e11/results/e11_aligner_sub0?_k0_sub0_preds.json",
        "kaggle/logs/e12c/results/e12c_insub_sub0?_s?_k0_sub0_preds.json",
    ),
    out="notes/category_metrics_eegnet.json",
):
    files = sorted(f for pat in patterns for f in glob.glob(pat))
    per = {f.split("/")[-1]: metrics(json.load(open(f))) for f in files}
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "null_mean", "null_p95"]
    mean = {k: float(np.mean([v[k] for v in per.values()])) for k in keys}
    print(f"{len(per)} files | " + " ".join(f"{k} {100*v:.2f}" for k, v in mean.items()))
    json.dump({"per_file": per, "mean": mean}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*(tuple(sys.argv[1:2]) if len(sys.argv) > 1 else ()))
