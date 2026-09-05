"""Locate inputs on a Kaggle kernel (or any mounted layout) without hard-coding dataset slugs.

Kaggle mounts datasets and notebook outputs somewhere under /kaggle/input; the helpers below
search recursively for the files each experiment needs and return their directories.
"""

import glob
import os

INPUT_ROOT = "/kaggle/input"


def _first(pattern, root=INPUT_ROOT):
    hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"nothing matches {pattern} under {root}")
    return hits[0]


def chisco_caches(root=INPUT_ROOT):
    """Directories of the per-participant Chisco caches (sub01 ... sub05)."""
    return sorted(
        {
            os.path.dirname(p)
            for p in glob.glob(os.path.join(root, "**", "sub0?", "X_imagine.npy"), recursive=True)
        }
    )


def cofett_caches(root=INPUT_ROOT):
    return sorted(
        {
            os.path.dirname(p)
            for p in glob.glob(
                os.path.join(root, "**", "cofett_sub0?", "X_imagine.npy"), recursive=True
            )
        }
    )


def text_bank(root=INPUT_ROOT):
    return os.path.dirname(_first(os.path.join("bank", "sentences.json"), root))


def weights_dir(root=INPUT_ROOT):
    return os.path.dirname(_first("labram-base.pth", root))


def channels_json(root=INPUT_ROOT):
    return _first("chisco_channels.json", root)


def meta_dir(root=INPUT_ROOT):
    return os.path.dirname(_first("textmaps.json", root))


def ensure_package_installed():
    """On Kaggle the package is shipped as a wheel inside a dataset; install it if it is not importable."""
    try:
        import eegsem  # noqa: F401

        return
    except ImportError:
        pass
    import subprocess
    import sys

    wheel = _first("eegsem-*.whl")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", wheel], check=True)
