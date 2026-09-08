"""COFETT (OpenNeuro ds006317) -> Chisco-compatible cache. Streams raw EDF runs from the public S3 bucket, epochs the
recall (inner-speech) phase into fixed 3.3 s windows (same as Chisco 'imagine'), reorders channels to Chisco's 122-channel set.
Preprocessing: drop junk channels, 1-75 Hz band-pass + 50 Hz notch, EOG regression (VEO/HEO), average reference, bad-channel
interpolation (std outliers, Chisco montage), resample to fs_out. Output: X_imagine.npy float16 [N,122,T] uV + meta_imagine.json.
"""

import json, os, re, sys, time, urllib.request
import numpy as np

S3 = "https://s3.amazonaws.com/openneuro.org/ds006317"
CODE_READ, CODE_RECALL, CODE_REST = 65329, 65379, 65381
JUNK = ["10", "111"]


def fetch(url, dst, log=print):
    if os.path.exists(dst):
        return dst
    t0 = time.time()
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(dst + ".part", "wb") as f:
        n = 0
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
            n += len(b)
    os.replace(dst + ".part", dst)
    log(f"fetched {os.path.basename(dst)} {n/1e9:.2f} GB in {time.time()-t0:.0f}s")
    return dst


def load_events(path):
    import csv

    ev = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ev.append((float(row["onset"]), int(float(row["value"]))))
    return ev


def trials_from_events(ev):
    """Return list of (read_onset, recall_onset, rest_onset) triplets in order."""
    out, cur = [], {}
    for onset, code in ev:
        if code == CODE_READ:
            cur = {"read": onset}
        elif code == CODE_RECALL and "read" in cur:
            cur["recall"] = onset
        elif code == CODE_REST and "recall" in cur:
            cur["rest"] = onset
            out.append((cur["read"], cur["recall"], cur["rest"]))
            cur = {}
    return out


def process_run(
    edf_path,
    events_path,
    sentences,
    ch_order,
    montage_csv,
    fs_out=250,
    win_s=3.3,
    textmaps=None,
    log=print,
    meta_extra=None,
):
    import mne

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    drop = [c for c in raw.ch_names if c in JUNK or c in ("EDF Annotations", "EKG", "EMG")]
    raw.drop_channels([c for c in drop if c in raw.ch_names])
    eog = [c for c in ("VEO", "HEO") if c in raw.ch_names]
    raw.set_channel_types({c: "eog" for c in eog})
    raw.set_channel_types({c: "stim" for c in raw.ch_names if c == "Trigger"})
    eeg = [c for c in raw.ch_names if c not in eog and c != "Trigger"]
    missing = [c for c in ch_order if c not in eeg]
    assert not missing, f"missing channels {missing}"
    # montage (Chisco cap coordinates, metres)
    import csv

    pos = {}
    with open(montage_csv) as f:
        for row in csv.DictReader(f):
            pos[row["label"]] = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
    mont = mne.channels.make_dig_montage(
        ch_pos={c: pos[c] for c in eeg if c in pos}, coord_frame="head"
    )
    raw.set_montage(mont, on_missing="ignore")
    raw.resample(500)  # first decimation for speed
    raw.filter(1.0, 75.0, picks="eeg")
    raw.notch_filter(50.0, picks="eeg")
    # bad channels: robust z of log-variance
    data = raw.get_data(picks=eeg)
    lv = np.log(data.var(axis=1) + 1e-30)
    z = (lv - np.median(lv)) / (1.4826 * np.median(np.abs(lv - np.median(lv))) + 1e-9)
    bads = [eeg[i] for i in np.where(np.abs(z) > 4)[0]]
    raw.info["bads"] = bads
    if bads:
        raw.interpolate_bads(reset_bads=True)
    raw.set_eeg_reference("average", projection=False)
    if eog:
        try:
            from mne.preprocessing import EOGRegression

            raw = EOGRegression(picks="eeg", picks_artifact="eog").fit(raw).apply(raw)
            log("EOG regression applied")
        except Exception as e:
            raise RuntimeError(f"EOG regression failed for {events_path}: {e}") from e
    raw.resample(fs_out)
    tr = trials_from_events(load_events(events_path))
    if len(tr) != len(
        sentences
    ):  # a missing event would shift every later label: refuse rather than truncate
        raise ValueError(f"{events_path}: {len(tr)} recall events vs {len(sentences)} sentences")
    n = min(len(tr), len(sentences))
    T = int(round(win_s * fs_out))
    X = np.zeros((n, len(ch_order), T), np.float16)
    picks = [raw.ch_names.index(c) for c in ch_order]
    sf = raw.info["sfreq"]
    meta = []
    for i in range(n):
        rd, rc, rs = tr[i]
        s0 = int(round(rc * sf))
        seg = raw._data[picks, s0 : s0 + T] * 1e6
        X[i, :, : seg.shape[1]] = seg.astype(np.float16)
        text = sentences[i]
        meta.append(
            {
                "text": text,
                "cat": (textmaps or {}).get(text, -1),
                "pos": i,
                "read_dur": round(rc - rd, 3),
                "recall_dur": round(rs - rc, 3),
                **(meta_extra or {}),
            }
        )
    log(f"run done: {n} trials, bads={bads}, X {X.shape}")
    return X, meta


def build_cofett_cache(
    subject,
    sessions,
    runs,
    ch_order,
    montage_csv,
    sent_lists,
    out_dir,
    tmp_dir,
    fs_out=250,
    textmaps=None,
    log=print,
    max_runs=None,
    win_s=3.3,
):
    Xs, meta = [], []
    k = 0
    for ses in sessions:
        for run in runs:
            if max_runs and k >= max_runs:
                break
            base = f"sub-{subject}_ses-{ses}_task-para1_run-{run}"
            edf = fetch(
                f"{S3}/sub-{subject}/ses-{ses}/eeg/{base}_eeg.edf",
                os.path.join(tmp_dir, base + "_eeg.edf"),
                log,
            )
            evp = fetch(
                f"{S3}/sub-{subject}/ses-{ses}/eeg/{base}_events.tsv",
                os.path.join(tmp_dir, base + "_events.tsv"),
                log,
            )
            sents = sent_lists[f"text1-{int(ses)}"]
            X, m = process_run(
                edf,
                evp,
                sents,
                ch_order,
                montage_csv,
                fs_out,
                textmaps=textmaps,
                log=log,
                meta_extra={"ses": int(ses), "run": int(run)},
                win_s=win_s,
            )
            Xs.append(X)
            meta += m
            k += 1
            os.remove(edf)
    X = np.concatenate(Xs)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_imagine.npy"), X)
    with open(os.path.join(out_dir, "meta_imagine.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    stats = {
        "subject": subject,
        "phase": "imagine",
        "n": int(X.shape[0]),
        "shape": list(X.shape),
        "fs": fs_out,
        "n_unique_text": len({m["text"] for m in meta}),
        "abs_mean_uV": float(np.abs(X.astype(np.float32)).mean()),
        "std_uV": float(X.astype(np.float32).std()),
    }
    with open(os.path.join(out_dir, "stats_imagine.json"), "w") as f:
        json.dump(stats, f, indent=1)
    log(json.dumps(stats))
    return stats
