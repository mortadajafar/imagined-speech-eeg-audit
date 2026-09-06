"""Repeat the decoder and the no-EEG baselines in the bge-m3 embedding space.

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
summary = Summary(os.path.join(inp["out"], "summary_embedding_robustness.json"))
for cache in inp["chisco"]:
    sub = os.path.basename(cache)
    args = [
        "--cache_dirs",
        cache,
        "--bank_dir",
        inp["bank"],
        "--emb",
        "bge-m3",
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
        "--tag",
        f"bgem3_{sub}",
    ]
    summary.run(f"bgem3_{sub}", lambda a=args: retrieval_fields(training.main(a)), sub=sub)
print(
    "no-EEG baselines in the bge-m3 space: run scripts/no_eeg_baselines.py with the bank name changed to bge-m3"
)
