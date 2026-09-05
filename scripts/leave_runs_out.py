"""Leave-runs-out evaluation: random run folds and true leave-one-day-out on Chisco, held-out day on COFETT, plus extra seeds with within-run metrics.

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
import argparse
import json

import numpy as np

from eegsem import training

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["random_runs", "day", "cofett_day", "seeds"], default="day")
opts = parser.parse_args()

inp = kaggle_inputs(need_cofett=(opts.mode == "cofett_day"))
summary = Summary(os.path.join(inp["out"], f"summary_leave_runs_out_{opts.mode}.json"))
base = [
    "--bank_dir",
    inp["bank"],
    "--phase",
    "imagine",
    "--encoder",
    "eegnet",
    "--epochs",
    "25",
    "--patience",
    "6",
    "--bs",
    "128",
    "--out",
    inp["out"],
    "--channels_json",
    inp["channels"],
    "--save_embeddings",
]


def runs_of(cache):
    meta = json.load(open(os.path.join(cache, "meta_imagine.json"), encoding="utf-8"))
    return sorted({m.get("ses", m.get("run", 0)) for m in meta})


def fit(tag, cache, test_runs, val_runs, seed=0):
    args = ["--cache_dirs", cache] + base + ["--seed", str(seed), "--tag", tag]
    if test_runs:
        args += ["--test_runs", *map(str, test_runs), "--val_runs", *map(str, val_runs)]
    summary.run(tag, lambda: retrieval_fields(training.main(args)), seed=seed)


if opts.mode == "random_runs":
    rng = np.random.default_rng(0)
    for cache in inp["chisco"]:
        runs = np.array(runs_of(cache))
        folds = np.array_split(rng.permutation(runs), 5)
        for k, test in enumerate(folds):
            rest = [r for r in runs if r not in set(test)]
            fit(
                f"random_runs_{os.path.basename(cache)}_fold{k}",
                cache,
                list(map(int, test)),
                list(map(int, rest[:2])),
            )
elif opts.mode == "day":  # blocks are numbered consecutively, nine per recording day
    for cache in inp["chisco"]:
        runs = runs_of(cache)
        days = [runs[i : i + 9] for i in range(0, len(runs), 9)]
        for d, test in enumerate(days):
            rest = [r for r in runs if r not in set(test)]
            val = rest[:2] if d > 0 else rest[-2:]
            fit(
                f"leave_one_day_out_{os.path.basename(cache)}_day{d + 1}",
                cache,
                list(map(int, test)),
                list(map(int, val)),
            )
elif opts.mode == "cofett_day":
    for cache in inp["cofett"]:
        days = runs_of(cache)
        for i, d in enumerate(days):
            fit(
                f"held_out_day_{os.path.basename(cache)}_day{d}",
                cache,
                [int(d)],
                [int(days[(i + 1) % len(days)])],
            )
else:
    for cache in inp["chisco"]:
        for seed in (1, 2):
            fit(f"in_subject_{os.path.basename(cache)}_s{seed}", cache, [], [], seed=seed)
