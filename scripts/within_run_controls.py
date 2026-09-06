"""Within-run and matched-pool retrieval for EEGNet on Chisco and COFETT, plus occlusion maps.

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
summary = Summary(os.path.join(inp["out"], "summary_within_run.json"))
for cache in inp["chisco"] + inp["cofett"]:
    sub = os.path.basename(cache)
    args = [
        "--cache_dirs",
        cache,
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
        "--tag",
        f"within_run_{sub}",
        "--occlusion",
        "--save_embeddings",
    ]
    summary.run(f"within_run_{sub}", lambda a=args: retrieval_fields(training.main(a)), sub=sub)
