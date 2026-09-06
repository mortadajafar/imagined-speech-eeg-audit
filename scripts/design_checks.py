"""Design checks on the stimulus lists: order identity across participants and category-imbalance baselines.

Runs unchanged on a Kaggle script kernel (see scripts/kaggle_registry.json for the data sources) or locally
with the same command-line layout once the caches exist.
"""

import os
import sys

try:
    import eegsem  # noqa: F401
except ImportError:  # Kaggle: the package wheel is attached as a dataset
    import glob
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            glob.glob("/kaggle/input/**/eegsem-*.whl", recursive=True)[0],
        ],
        check=True,
    )

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eegsem.sweep import Summary, kaggle_inputs, retrieval_fields  # noqa: E402
import json
from collections import Counter

import numpy as np

from eegsem import paths
from eegsem.data.splits import split_of

caches = paths.chisco_caches()
out = "/kaggle/working/results"
os.makedirs(out, exist_ok=True)
report = {"order_identity": {}, "category": {}}
seq = {}
for cache in caches:
    meta = json.load(open(os.path.join(cache, "meta_imagine.json"), encoding="utf-8"))
    seq[os.path.basename(cache)] = {
        (m.get("run", 0), m.get("pos", i)): m["text"] for i, m in enumerate(meta)
    }
subs = sorted(seq)
for i in range(len(subs)):
    for j in range(i + 1, len(subs)):
        shared = set(seq[subs[i]]) & set(seq[subs[j]])
        same = sum(seq[subs[i]][k] == seq[subs[j]][k] for k in shared)
        report["order_identity"][f"{subs[i]}-{subs[j]}"] = dict(
            shared_positions=len(shared), identical=same, frac=same / max(len(shared), 1)
        )
for cache in caches:
    meta = json.load(open(os.path.join(cache, "meta_imagine.json"), encoding="utf-8"))
    cats = np.array([m["cat"] for m in meta])
    runs = np.array([m.get("run", 0) for m in meta])
    split = np.array([split_of(m["text"]) for m in meta])
    train, test = split == "train", split == "test"
    majority = Counter(cats[test & (cats >= 0)]).most_common(1)[0][1] / (test & (cats >= 0)).sum()
    run_majority = {
        r: Counter(cats[train & (runs == r) & (cats >= 0)]).most_common(1)[0][0]
        for r in np.unique(runs)
        if (train & (runs == r) & (cats >= 0)).any()
    }
    pred = np.array([run_majority.get(r, -1) for r in runs[test]])
    true = cats[test]
    per_class = [np.mean(pred[true == c] == c) for c in np.unique(true) if c >= 0]
    report["category"][os.path.basename(cache)] = dict(
        majority_class_acc=float(majority),
        run_majority_acc=float(np.mean(pred == true)),
        run_majority_balanced_acc=float(np.mean(per_class)),
        per_run=[
            dict(
                run=int(r),
                n_cat=len(set(cats[train & (runs == r)])),
                majority_share=float(
                    Counter(cats[train & (runs == r)]).most_common(1)[0][1]
                    / (train & (runs == r)).sum()
                ),
            )
            for r in np.unique(runs)
        ],
    )
json.dump(
    report,
    open(os.path.join(out, "summary_design_checks.json"), "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)
print(json.dumps(report["category"], indent=1)[:2000])
