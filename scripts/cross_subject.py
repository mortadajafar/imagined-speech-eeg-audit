"""Leave-one-subject-out decoding with few-shot calibration of the spatial adapter on the held-out participant.

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

inp = kaggle_inputs()
summary = Summary(os.path.join(inp["out"], "summary_cross_subject.json"))
n = len(inp["chisco"])
for held in range(n):
    train_subjects = [i for i in range(n) if i != held]
    args = [
        "--cache_dirs",
        *inp["chisco"],
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
        "--train_subjects",
        *map(str, train_subjects),
        "--test_subjects",
        str(held),
        "--few_shot_list",
        "0",
        "50",
        "200",
        "1000",
        "--seed",
        "0",
        "--out",
        inp["out"],
        "--channels_json",
        inp["channels"],
        "--tag",
        f"loso_held{held}",
    ]

    def run(a=args, held=held):
        r = training.main(a)
        return {key: r["test"][key]["eeg"] for key in r["test"]}

    summary.run(f"loso_held{held}", run, held=held)
