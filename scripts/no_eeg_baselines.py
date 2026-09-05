"""No-EEG design baselines: run-mean, position +-K, run-majority category and the run-level permutation null, evaluated exactly like the decoders.

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
from _common import Summary, kaggle_inputs, retrieval_fields  # noqa: E402
import json

import numpy as np

from eegsem import paths
from eegsem.data.splits import split_of
from eegsem.data.text_bank import TextBank
from eegsem.evaluation.metrics import retrieval_metrics, zero_shot_category

bank = TextBank(paths.text_bank(), "labse")
out = "/kaggle/working/results"
os.makedirs(out, exist_ok=True)
summary = Summary(os.path.join(out, "summary_no_eeg_baselines.json"))


def evaluate(Z, sid_test, ses_test, cats_test, centroids):
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    uniq, target = np.unique(sid_test, return_inverse=True)
    res = retrieval_metrics(Z, bank.emb[uniq], target, n_draws=5)
    res.update(zero_shot_category(Z, centroids, cats_test))
    S = Z @ bank.emb[uniq].T
    within = []
    for i in range(len(Z)):
        cand = np.where(np.isin(uniq, sid_test[ses_test == ses_test[i]]))[0]
        if len(cand) >= 5:
            within.append(((S[i][cand] > S[i][target[i]]).sum() + 1, len(cand)))
    if within:
        rank = np.array([r for r, _ in within])
        n = np.array([n for _, n in within], float)
        res.update(ws_top1=float((rank <= 1).mean()), ws_chance_top1=float((1 / n).mean()))
    return res


for cache in paths.chisco_caches() + paths.cofett_caches():
    name = os.path.basename(cache)
    meta = json.load(open(os.path.join(cache, "meta_imagine.json"), encoding="utf-8"))
    texts = [m["text"] for m in meta]
    sid = bank.ids(texts)
    cats = np.array([m["cat"] for m in meta])
    ses = np.array([m.get("ses", m.get("run", 0)) for m in meta])
    pos = np.array([m.get("pos", i) for i, m in enumerate(meta)])
    split = np.array([split_of(t) for t in texts])
    train, test = np.where(split == "train")[0], np.where(split == "test")[0]
    centroids = np.zeros((int(cats.max()) + 1, bank.dim), np.float32)
    for k in range(len(centroids)):
        members = np.unique(sid[cats == k])
        if len(members):
            centroids[k] = bank.emb[members].mean(0)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
    run_mean = {r: bank.emb[np.unique(sid[train][ses[train] == r])].mean(0) for r in np.unique(ses)}
    predictors = {"run_mean": np.stack([run_mean[r] for r in ses[test]])}
    for K in (3, 10):
        rows = []
        for i in test:
            near = (ses[train] == ses[i]) & (np.abs(pos[train] - pos[i]) <= K)
            rows.append(bank.emb[sid[train][near]].mean(0) if near.any() else run_mean[ses[i]])
        predictors[f"position_k{K}"] = np.stack(rows)
    for pname, Z in predictors.items():
        summary.record(
            dict(tag=f"{name}_{pname}", **evaluate(Z, sid[test], ses[test], cats[test], centroids))
        )
    majority = {
        r: np.bincount(cats[train][(ses[train] == r) & (cats[train] >= 0)]).argmax()
        for r in np.unique(ses)
        if ((ses[train] == r) & (cats[train] >= 0)).any()
    }
    summary.record(
        dict(
            tag=f"{name}_run_majority_category",
            cat_acc=float(
                np.mean([majority.get(r, -1) == c for r, c in zip(ses[test], cats[test]) if c >= 0])
            ),
        )
    )
    rng = np.random.default_rng(0)
    runs = np.unique(ses)
    null = []
    for _ in range(2000):
        perm = dict(zip(runs, rng.permutation(runs)))
        shuffled = np.array([perm[r] for r in ses])
        means = {r: bank.emb[np.unique(sid[train][shuffled[train] == r])].mean(0) for r in runs}
        Z = np.stack([means[r] for r in ses[test]])
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        uniq, target = np.unique(sid[test], return_inverse=True)
        null.append(retrieval_metrics(Z, bank.emb[uniq], target, n_draws=1).get("top1_pool100"))
    summary.record(
        dict(
            tag=f"{name}_run_permutation_null",
            null_mean=float(np.mean(null)),
            null_p95=float(np.percentile(null, 95)),
            n_perm=len(null),
        )
    )
