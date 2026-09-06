"""Embed every Chisco sentence once with LaBSE and bge-m3 and store the banks used by all decoders.

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
import json

from eegsem import paths
from eegsem.data.text_bank import embed_sentences, save_bank

textmaps = json.load(open(os.path.join(paths.meta_dir(), "textmaps.json"), encoding="utf-8"))
sentences = sorted({" ".join(k.split()) for k in textmaps})
for name in ("labse", "bge-m3"):
    save_bank(sentences, embed_sentences(sentences, name), "/kaggle/working/bank", name)
