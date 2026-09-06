"""BrainMosaic (authors public code, open text assets) trained and evaluated on our sentence-disjoint split.

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
import shutil
import subprocess

from eegsem import paths
from eegsem.brainmosaic import (
    build_text_assets,
    build_token_bank,
    convert_cache,
    eval_retrieval,
    train_subject,
)

REPO = "/kaggle/tmp/BrainMosaic_ICLR26"
source = [
    p
    for p in __import__("glob").glob("/kaggle/input/**/config_args.py", recursive=True)
    if os.path.exists(os.path.join(os.path.dirname(p), "models", "detr.py"))
]
shutil.copytree(os.path.dirname(source[0]), REPO, dirs_exist_ok=True)
subprocess.run(
    [
        sys.executable,
        "-c",
        open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "brainmosaic_patch.py")
        ).read(),
    ],
    check=True,
    cwd=REPO,
)
meta = paths.meta_dir()
textmaps = {
    " ".join(k.split()): v
    for k, v in json.load(open(os.path.join(meta, "textmaps.json"), encoding="utf-8")).items()
}
work = "/kaggle/working/bm"
os.makedirs(work, exist_ok=True)
caches = paths.chisco_caches()
sentences = sorted(
    set(textmaps)
    | {
        m["text"]
        for c in caches
        for m in json.load(open(os.path.join(c, "meta_imagine.json"), encoding="utf-8"))
    }
)
segmentation, vocab = build_text_assets(sentences, f"{work}/assets", pooling="last", center=True)
build_token_bank(REPO, f"{work}/assets", f"{work}/token_bank")
summary = Summary(os.path.join(work, "summary_brainmosaic.json"))
for cache in caches:
    sub = os.path.basename(cache)

    def run():
        data = f"/kaggle/tmp/bm_{sub}"
        convert_cache(cache, segmentation, data)
        cfg = train_subject(
            REPO,
            data,
            f"{work}/token_bank",
            f"{work}/assets/sentence_embeddings.pt",
            f"{work}/assets/segmentation.json",
            f"{work}/{sub}",
            epochs=30,
            batch_size=32,
        )
        best = json.load(open(f"{work}/{sub}/best_summary.json"))
        retrieval = eval_retrieval(
            REPO,
            cfg,
            f"{work}/{sub}/eval_embeddings/best_by_matching_acc.pth",
            f"{data}/test.pt",
            f"{work}/assets/sentence_embeddings.pt",
            textmaps,
        )
        os.chdir("/kaggle/working")
        return dict(best_val=best, test_retrieval=retrieval)

    summary.run(f"brainmosaic_{sub}", run, sub=sub)
