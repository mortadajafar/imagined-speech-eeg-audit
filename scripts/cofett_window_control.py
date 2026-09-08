"""COFETT recall-window control (E16): in-subject and held-out-day EEGNet evaluations on caches cut with a 0-2 s window.

Runs unchanged on a Kaggle script kernel (see scripts/kaggle_registry.json for the data sources) or locally
with the same command-line layout once the caches exist.
"""

import json
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

from eegsem.sweep import Summary, kaggle_inputs, retrieval_fields  # noqa: E402
from eegsem import training

# E16. COFETT recall-window control: the same in-subject (within-day metrics) and held-out-day evaluations as
# E7/E12b, run on caches cut with a shorter window (0-2 s after recall onset, always inside the recall period)
# instead of the 3.3 s Chisco-compatible window. Attach only the short-window caches and the text bank.
inp = kaggle_inputs(need_cofett=True)
summary = Summary(os.path.join(inp["out"], "summary_cofett_window.json"))
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
]


def runs_of(cache):
    meta = json.load(open(os.path.join(cache, "meta_imagine.json"), encoding="utf-8"))
    return sorted({m.get("ses", m.get("run", 0)) for m in meta})


for cache in inp["cofett"]:
    sub = os.path.basename(cache)
    args = (
        ["--cache_dirs", cache]
        + base
        + ["--seed", "0", "--tag", f"win_within_run_{sub}", "--save_embeddings"]
    )
    summary.run(
        f"win_within_run_{sub}",
        lambda a=args: retrieval_fields(training.main(a)),
        sub=sub,
        kind="in_subject",
    )
    days = runs_of(cache)
    for i, d in enumerate(days):
        args = (
            ["--cache_dirs", cache]
            + base
            + [
                "--seed",
                "0",
                "--tag",
                f"win_held_out_day_{sub}_day{d}",
                "--test_runs",
                str(int(d)),
                "--val_runs",
                str(int(days[(i + 1) % len(days)])),
            ]
        )
        summary.run(
            f"win_held_out_day_{sub}_day{d}",
            lambda a=args: retrieval_fields(training.main(a)),
            sub=sub,
            day=int(d),
            kind="held_out_day",
        )
