"""In-subject imagined-speech decoding on all Chisco participants: ridge, EEGNet and Conformer (three seeds), with the standard sentence-disjoint split.

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
from eegsem import ridge, training

inp = kaggle_inputs()
summary = Summary(os.path.join(inp["out"], "summary_in_subject.json"))
for cache in inp["chisco"]:
    sub = os.path.basename(cache)
    summary.run(
        f"ridge_{sub}",
        lambda: retrieval_fields(
            ridge.main(
                [
                    "--cache_dirs",
                    cache,
                    "--bank_dir",
                    inp["bank"],
                    "--phase",
                    "imagine",
                    "--subject",
                    "0",
                    "--out",
                    inp["out"],
                    "--tag",
                    f"ridge_{sub}",
                ]
            )
        ),
        encoder="ridge",
        sub=sub,
    )
    for encoder in ("eegnet", "conformer"):
        for seed in (0, 1, 2):
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
                "128",
                "--seed",
                str(seed),
                "--out",
                inp["out"],
                "--channels_json",
                inp["channels"],
                "--tag",
                f"{encoder}_{sub}_s{seed}",
                "--save_embeddings",
            ]
            summary.run(
                f"{encoder}_{sub}_s{seed}",
                lambda a=args: retrieval_fields(training.main(a)),
                encoder=encoder,
                sub=sub,
                seed=seed,
            )
