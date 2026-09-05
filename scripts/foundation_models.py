"""Fine-tuned and frozen LaBraM and CBraMod encoders in-subject on Chisco, with within-run metrics.

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

inp = kaggle_inputs(need_weights=True)
summary = Summary(os.path.join(inp["out"], "summary_foundation_models.json"))
conditions = [
    ("labram", ["--lr", "5e-4", "--fm_lr_mult", "0.2"], "finetuned"),
    ("labram", ["--lr", "1e-3", "--fm_lr_mult", "0.0"], "frozen"),
    ("cbramod62", ["--lr", "5e-4", "--fm_lr_mult", "0.2"], "finetuned"),
]
for cache in inp["chisco"]:
    sub = os.path.basename(cache)
    for encoder, extra, mode in conditions:
        args = [
            "--cache_dirs",
            cache,
            "--bank_dir",
            inp["bank"],
            "--phase",
            "imagine",
            "--encoder",
            encoder,
            "--epochs",
            "25",
            "--patience",
            "6",
            "--bs",
            "64",
            "--seed",
            "0",
            "--out",
            inp["out"],
            "--weights_dir",
            inp["weights"],
            "--channels_json",
            inp["channels"],
            "--tag",
            f"{encoder}_{mode}_{sub}",
        ] + extra
        summary.run(
            f"{encoder}_{mode}_{sub}",
            lambda a=args: retrieval_fields(training.main(a)),
            encoder=encoder,
            mode=mode,
            sub=sub,
        )
