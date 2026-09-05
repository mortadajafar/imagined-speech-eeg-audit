"""Reading-phase to imagined-speech transfer (zero-shot, pre-training, data scaling) for all Chisco participants.

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

from eegsem import training

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
parser.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5])
opts = parser.parse_args()

inp = kaggle_inputs()
summary = Summary(os.path.join(inp["out"], "summary_reading_to_imagining.json"))
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
    "--out",
    inp["out"],
    "--channels_json",
    inp["channels"],
]
conditions = {
    "imagined_only": ["--phase", "imagine"],
    "reading_trained_tested_on_imagined": ["--phase", "read", "--test_phase", "imagine"],
    "reading_pretrained_then_finetuned": ["--phase", "imagine", "--pretrain_phase", "read"],
    "reading_only": ["--phase", "read"],
}
for cache in inp["chisco"]:
    sub = os.path.basename(cache)
    for seed in opts.seeds:
        for name, extra in conditions.items():
            args = (
                ["--cache_dirs", cache]
                + base
                + extra
                + ["--seed", str(seed), "--tag", f"{name}_{sub}_s{seed}", "--occlusion"]
            )
            summary.run(
                f"{name}_{sub}_s{seed}",
                lambda a=args: retrieval_fields(training.main(a)),
                condition=name,
                sub=sub,
                seed=seed,
            )
        for frac in opts.fractions:
            for name, extra in (
                ("imagined_only", ["--phase", "imagine"]),
                (
                    "reading_pretrained_then_finetuned",
                    ["--phase", "imagine", "--pretrain_phase", "read"],
                ),
            ):
                args = (
                    ["--cache_dirs", cache]
                    + base
                    + extra
                    + [
                        "--train_frac",
                        str(frac),
                        "--seed",
                        str(seed),
                        "--tag",
                        f"{name}_frac{frac}_{sub}_s{seed}",
                    ]
                )
                summary.run(
                    f"{name}_frac{frac}_{sub}_s{seed}",
                    lambda a=args: retrieval_fields(training.main(a)),
                    condition=name,
                    fraction=frac,
                    sub=sub,
                    seed=seed,
                )
