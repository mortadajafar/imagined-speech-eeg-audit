"""Direct tests of the shortcut: run classification and position regression from EEG, linear probes on EEGNet embeddings, matched pools.

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
from eegsem.analysis.probes import run_probe

inp = kaggle_inputs()
summary = Summary(os.path.join(inp["out"], "summary_mechanism_probes.json"))
for cache in inp["chisco"]:
    sub = os.path.basename(cache)
    tag = f"aligner_{sub}"

    def run():
        r = training.main(
            [
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
                tag,
                "--save_embeddings",
            ]
        )
        probes = run_probe(
            cache,
            inp["bank"],
            inp["out"],
            seed=0,
            epochs=20,
            emb=os.path.join(inp["out"], f"{tag}_k0_sub0_emb.npz"),
        )
        return dict(**retrieval_fields(r), **probes)

    summary.run(f"probes_{sub}", run, sub=sub)
