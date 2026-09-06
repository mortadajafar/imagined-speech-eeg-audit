"""Build the 250 Hz float16 cache (imagined and reading epochs) for one Chisco participant from the preprocessed pickles.

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
import argparse

from eegsem import paths
from eegsem.data.chisco import build_cache

parser = argparse.ArgumentParser()
parser.add_argument("--subject", required=True, help="two-digit participant id, e.g. 02")
parser.add_argument("--roots", nargs="+", default=["/kaggle/input"])
parser.add_argument("--out", default=None)
args = parser.parse_args()

out = args.out or f"/kaggle/working/sub{args.subject}"
textmaps = os.path.join(paths.meta_dir(args.roots[0]), "textmaps.json")
for phase in ("imagine", "read"):
    build_cache(args.roots, args.subject, phase, out, fs_out=250, textmaps_path=textmaps)
