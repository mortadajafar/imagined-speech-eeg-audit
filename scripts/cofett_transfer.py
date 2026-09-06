"""COFETT in-subject decoding and Chisco-to-COFETT transfer (zero-shot and few-shot adapter calibration).

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
from eegsem import training

inp = kaggle_inputs(need_cofett=True)
summary = Summary(os.path.join(inp["out"], "summary_cofett.json"))
caches = inp["chisco"] + inp["cofett"]
n_chisco = len(inp["chisco"])
base = [
    "--cache_dirs",
    *caches,
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
    "--seed",
    "0",
    "--out",
    inp["out"],
    "--channels_json",
    inp["channels"],
]
for j in range(len(inp["cofett"])):
    args = base + [
        "--train_subjects",
        str(n_chisco + j),
        "--test_subjects",
        str(n_chisco + j),
        "--tag",
        f"cofett_in_subject_{j}",
    ]
    summary.run(
        f"cofett_in_subject_{j}",
        lambda a=args, j=j: retrieval_fields(training.main(a), f"sub{n_chisco + j}"),
    )
args = base + [
    "--train_subjects",
    *map(str, range(n_chisco)),
    "--test_subjects",
    *map(str, range(n_chisco, len(caches))),
    "--few_shot_list",
    "0",
    "100",
    "500",
    "1000",
    "--epochs",
    "20",
    "--tag",
    "chisco_to_cofett",
]


def run(a=args):
    r = training.main(a)
    return {key: r["test"][key]["eeg"] for key in r["test"]}


summary.run("chisco_to_cofett", run)
