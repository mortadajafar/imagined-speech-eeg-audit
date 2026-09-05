"""Build compact per-subject caches from Chisco preprocessed pickles.
Input  : derivatives/preprocessed_pkl/sub-XX/eeg/sub-XX_task-{read,imagine}_run-NNN_eeg.pkl
         each = list of {'text': str, 'input_features': ndarray (1, 125, T) float64 [Volts]}
Output : X_{phase}.npy  float16 [N, 122, T'] in microvolts at `fs_out` Hz
         meta_{phase}.json  [{'text','cat','run','pos'}]   (cat = 39-class id from textmaps.json, -1 if unknown)
"""

import glob, json, os, pickle, re, sys, time
import numpy as np
from scipy.signal import resample_poly

N_CH = 122  # official Chisco code keeps channels [:122] (drops VEO/HEO/mastoid refs)
FS_IN = 500
PHASE_SEC = {"imagine": 3.3, "read": 5.0}


def run_id(path):
    m = re.search(r"run-(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def find_pkls(roots, subject, phase):
    out = []
    for r in roots:
        out += glob.glob(
            os.path.join(r, "**", f"sub-{subject}_task-{phase}_run-*_eeg.pkl"), recursive=True
        )
    out = sorted(set(out), key=run_id)
    return out


def load_textmaps(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            tm = json.load(f)
        return {" ".join(k.strip().split()): int(v) for k, v in tm.items()}
    return {}


def build_cache(
    roots, subject, phase, out_dir, fs_out=250, textmaps_path=None, max_files=None, log=print
):
    files = find_pkls(roots, subject, phase)
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no pkl for sub-{subject} {phase} under {roots}")
    tm = load_textmaps(textmaps_path)
    T_out = int(round(PHASE_SEC[phase] * fs_out))
    up, down = fs_out, FS_IN
    from math import gcd

    g = gcd(up, down)
    up //= g
    down //= g
    X_parts, meta = [], []
    t0 = time.time()
    for fi, f in enumerate(files):
        with open(f, "rb") as fh:
            trials = pickle.load(fh)
        rid = run_id(f)
        arrs = []
        for pos, tr in enumerate(trials):
            a = np.asarray(tr["input_features"])
            a = (
                a.reshape(a.shape[-2], a.shape[-1])[:N_CH].astype(np.float32) * 1e6
            )  # Volts -> microvolts
            if (up, down) != (1, 1):
                a = resample_poly(a, up, down, axis=1).astype(np.float32)
            if a.shape[1] < T_out:
                a = np.pad(a, ((0, 0), (0, T_out - a.shape[1])))
            a = a[:, :T_out]
            arrs.append(a.astype(np.float16))
            text = " ".join(str(tr["text"]).strip().split())
            meta.append({"text": text, "cat": tm.get(text, -1), "run": rid, "pos": pos})
        X_parts.append(np.stack(arrs))
        log(f"[{fi+1}/{len(files)}] run {rid}: {len(trials)} trials  ({time.time()-t0:.0f}s)")
        del trials
    X = np.concatenate(X_parts)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"X_{phase}.npy"), X)
    with open(os.path.join(out_dir, f"meta_{phase}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    stats = {
        "subject": subject,
        "phase": phase,
        "n": int(X.shape[0]),
        "shape": list(X.shape),
        "fs": fs_out,
        "n_unique_text": len({m["text"] for m in meta}),
        "n_unknown_cat": int(sum(m["cat"] < 0 for m in meta)),
        "abs_mean_uV": float(np.abs(X.astype(np.float32)).mean()),
        "std_uV": float(X.astype(np.float32).std()),
        "files": [os.path.basename(f) for f in files],
    }
    with open(os.path.join(out_dir, f"stats_{phase}.json"), "w") as f:
        json.dump(stats, f, indent=1)
    log(json.dumps({k: v for k, v in stats.items() if k != "files"}))
    return stats


class SubjectCache:
    """Lazy reader for one subject's cache dir."""

    def __init__(self, cache_dir, phase):
        self.dir, self.phase = cache_dir, phase
        self.X = np.load(os.path.join(cache_dir, f"X_{phase}.npy"), mmap_mode="r")
        with open(os.path.join(cache_dir, f"meta_{phase}.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.texts = [m["text"] for m in self.meta]
        self.cats = np.array([m["cat"] for m in self.meta])

    def __len__(self):
        return self.X.shape[0]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--phases", nargs="+", default=["imagine", "read"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fs", type=int, default=250)
    ap.add_argument("--textmaps", default=None)
    ap.add_argument("--max_files", type=int, default=None)
    a = ap.parse_args()
    for ph in a.phases:
        try:
            build_cache(a.roots, a.subject, ph, a.out, a.fs, a.textmaps, a.max_files)
        except FileNotFoundError as e:
            print("SKIP:", e)
