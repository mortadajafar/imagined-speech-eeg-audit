"""Push one experiment script to Kaggle as a script kernel.

Example:  python scripts/push_to_kaggle.py leave_runs_out.py --name lodo-chisco -- --mode day
The data sources come from kaggle_registry.json; the script itself is copied unchanged.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("--name", required=True, help="kernel slug, e.g. in-subject-decoding")
    parser.add_argument("--subject", default=None, help="fills {S} in the registry entry")
    parser.add_argument(
        "script_args", nargs="*", help="arguments appended to the script (after --)"
    )
    opts = parser.parse_args()

    registry = json.load(open(os.path.join(HERE, "kaggle_registry.json")))
    entry = {**registry["defaults"], **registry[opts.script]}
    fill = lambda s: s.replace("{S}", opts.subject or "")
    owner = entry["owner"]
    datasets = [f"{owner}/{d}" for d in entry["datasets"]] + [
        fill(x) for x in entry.get("external", [])
    ]
    kernels = [f"{owner}/{k}" for k in entry["caches"]]

    work = tempfile.mkdtemp()
    shutil.copy(os.path.join(HERE, opts.script), work)
    shutil.copy(os.path.join(HERE, "_common.py"), work)
    metadata = {
        "id": f"{owner}/{opts.name}",
        "title": opts.name,
        "code_file": opts.script,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": entry["gpu"],
        "enable_tpu": False,
        "enable_internet": entry["internet"],
        "dataset_sources": datasets,
        "kernel_sources": kernels,
        "competition_sources": [],
        "model_sources": [],
    }
    if entry["gpu"]:
        metadata["machine_shape"] = entry["accelerator"]
    json.dump(metadata, open(os.path.join(work, "kernel-metadata.json"), "w"), indent=1)
    cmd = ["kaggle", "kernels", "push", "-p", work]
    if entry["gpu"]:
        cmd += ["--accelerator", entry["accelerator"]]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    shutil.rmtree(work)


if __name__ == "__main__":
    main()
