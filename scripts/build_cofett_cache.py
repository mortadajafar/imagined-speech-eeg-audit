"""Stream the raw COFETT EDF runs of one participant from OpenNeuro and cut Chisco-compatible recall epochs.

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
import json

from eegsem import paths
from eegsem.data.cofett import build_cofett_cache

parser = argparse.ArgumentParser()
parser.add_argument("--subject", required=True)
parser.add_argument("--out", default=None)
parser.add_argument("--win_s", type=float, default=3.3, help="epoch length after recall onset (s)")
args = parser.parse_args()

meta = paths.meta_dir()
channels = json.load(open(os.path.join(meta, "chisco_channels.json")))["eeg122"]
sentences = json.load(open(os.path.join(meta, "cofett_sentences.json"), encoding="utf-8"))
textmaps = {
    " ".join(k.split()): v
    for k, v in json.load(open(os.path.join(meta, "textmaps.json"), encoding="utf-8")).items()
}
build_cofett_cache(
    args.subject,
    ["01", "02", "03", "04"],
    ["01", "02", "03", "04"],
    channels,
    os.path.join(meta, "chisco_montage.csv"),
    sentences,
    args.out or f"/kaggle/working/cofett_sub{args.subject}",
    "/kaggle/tmp",
    fs_out=250,
    textmaps=textmaps,
    win_s=args.win_s,
)
