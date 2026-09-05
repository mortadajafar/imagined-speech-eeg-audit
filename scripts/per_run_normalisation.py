"""Per-run z-scoring as a mitigation, and the reading-trained model scored within run.

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
from eegsem import training

inp = kaggle_inputs(need_cofett=True)
summary = Summary(os.path.join(inp["out"], "summary_per_run_normalisation.json"))
base = [
    "--bank_dir",
    inp["bank"],
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
for cache in inp["chisco"]:
    sub = os.path.basename(cache)
    for name, extra in (
        ("imagined_only_per_run_normalised", ["--phase", "imagine", "--norm", "session"]),
        (
            "reading_trained_per_run_normalised",
            ["--phase", "read", "--test_phase", "imagine", "--norm", "session"],
        ),
        ("reading_trained", ["--phase", "read", "--test_phase", "imagine"]),
    ):
        args = ["--cache_dirs", cache] + base + extra + ["--tag", f"{name}_{sub}"]
        summary.run(
            f"{name}_{sub}",
            lambda a=args: retrieval_fields(training.main(a)),
            condition=name,
            sub=sub,
        )
for cache in inp["cofett"]:
    sub = os.path.basename(cache)
    args = (
        ["--cache_dirs", cache]
        + base
        + [
            "--phase",
            "imagine",
            "--norm",
            "session",
            "--tag",
            f"imagined_only_per_run_normalised_{sub}",
        ]
    )
    summary.run(
        f"imagined_only_per_run_normalised_{sub}",
        lambda a=args: retrieval_fields(training.main(a)),
        condition="imagined_only_per_run_normalised",
        sub=sub,
    )
