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

from _common import Summary, kaggle_inputs, retrieval_fields  # noqa: E402
from eegsem import ridge, training

# E15a. (i) EEGNet trained with sentence labels permuted WITHIN each run: the pairing between epoch and
# sentence is destroyed but every run keeps its own set of sentences, so any remaining full-pool accuracy
# is attributable to run structure alone. Three permutations per participant. (ii) Ridge with within-run
# metrics, which the original ridge run did not report.
inp = kaggle_inputs()
summary = Summary(os.path.join(inp["out"], "summary_within_run_permutation.json"))
for cache in inp["chisco"]:
    sub = os.path.basename(cache)
    for seed in (0, 1, 2):
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
            str(seed),
            "--out",
            inp["out"],
            "--channels_json",
            inp["channels"],
            "--tag",
            f"permrun_{sub}_s{seed}",
            "--permute_within_run",
        ]
        summary.run(
            f"permrun_{sub}_s{seed}",
            lambda a=args: retrieval_fields(training.main(a)),
            sub=sub,
            seed=seed,
            kind="permute_within_run",
        )
    args = [
        "--cache_dirs",
        cache,
        "--bank_dir",
        inp["bank"],
        "--phase",
        "imagine",
        "--out",
        inp["out"],
        "--tag",
        f"ridge_{sub}",
    ]
    summary.run(
        f"ridge_{sub}",
        lambda a=args: retrieval_fields(ridge.main(a)),
        sub=sub,
        kind="ridge_within_run",
    )
