"""Statistics for the paper: paired tests across subjects (Wilcoxon signed-rank, exact for n=5) with Holm correction,
per-condition means with across-subject SD, and bootstrap CIs from per-trial ranks. Usage: python -m eegsem.stats --csv notes/results_all.csv
"""

import argparse, csv, itertools, json, os, re
from collections import defaultdict
import numpy as np
from scipy import stats as st


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        for k in (
            "top1",
            "top5",
            "top10",
            "mrr",
            "top1_p100",
            "cat",
            "noise_top1_p100",
            "train_frac",
        ):
            r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else np.nan
        r["seed"] = int(float(r["seed"]))
        r["few_shot"] = int(float(r["few_shot"]))
        r["permute"] = r["permute"] == "True"
    return rows


def cond_e3(r):
    if r["exp"] not in ("e3_transfer", "e3b_seeds", "e3c_dataeff"):
        return None
    if r["train_frac"] < 1.0:
        return None
    if r["phase"] == "read" and r["test_phase"] == "imagine":
        return "B"
    if r["phase"] == "read":
        return "R"
    if r["pretrain"] not in (None, "", "None"):
        return "C"
    return "A"


def per_subject_means(rows, key_fn, metric):
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = key_fn(r)
        if k is None or np.isnan(r[metric]):
            continue
        d[k][r["subject"]].append(r[metric])
    return {k: {s: float(np.mean(v)) for s, v in subs.items()} for k, subs in d.items()}


def paired(a, b):
    subs = sorted(set(a) & set(b))
    x = np.array([a[s] for s in subs])
    y = np.array([b[s] for s in subs])
    if len(subs) < 2:
        return dict(n=len(subs))
    w = st.wilcoxon(x, y, alternative="two-sided", method="exact") if len(subs) >= 5 else None
    t = st.ttest_rel(x, y)
    return dict(
        n=len(subs),
        mean_a=float(x.mean()),
        mean_b=float(y.mean()),
        diff=float((x - y).mean()),
        wins=int((x > y).sum()),
        wilcoxon_p=float(w.pvalue) if w else None,
        ttest_p=float(t.pvalue),
        cohen_dz=float((x - y).mean() / ((x - y).std(ddof=1) + 1e-12)),
    )


def holm(pvals):
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0
    for rank, i in enumerate(idx):
        running = max(running, pvals[i] * (m - rank))
        adj[i] = min(1.0, running)
    return adj


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="notes/results_all.csv")
    ap.add_argument("--out", default="notes/stats.md")
    a = ap.parse_args(argv)
    rows = load(a.csv)
    md = ["# Statistics (auto-generated)\n"]
    # ---- E1: encoder vs ridge and vs chance, per subject means over seeds
    for metric, chance in (("top1_p100", 0.01), ("cat", 1 / 39), ("mrr", None)):
        e1 = per_subject_means(
            [r for r in rows if r["exp"] == "e1_all" and not r["permute"]],
            lambda r: r["encoder"],
            metric,
        )
        md.append(f"\n## E1 in-subject — {metric}\n")
        for enc, subs in e1.items():
            v = np.array(list(subs.values()))
            line = f"- {enc}: mean {v.mean():.4f} ± {v.std(ddof=1):.4f} (n={len(v)} subjects)"
            if chance is not None and len(v) >= 2:
                t = st.ttest_1samp(v, chance)
                line += f"; vs chance {chance:.4f}: t={t.statistic:.2f}, p={t.pvalue:.4f}"
            md.append(line)
        tests = []
        for x, y in (("eegnet", "ridge"), ("conformer", "ridge"), ("conformer", "eegnet")):
            if x in e1 and y in e1:
                tests.append(((x, y), paired(e1[x], e1[y])))
        if tests:
            p = holm([t[1].get("ttest_p", 1.0) for t in tests])
            for (pair, res), pa in zip(tests, p):
                md.append(
                    f"- {pair[0]} vs {pair[1]}: diff {res['diff']:+.4f}, wins {res['wins']}/{res['n']}, Wilcoxon p={res['wilcoxon_p']}, paired-t p={res['ttest_p']:.4f} (Holm {pa:.4f}), dz={res['cohen_dz']:.2f}"
                )
    # ---- E3: A vs B vs C vs R
    md.append("\n## E3 reading→imagining (per-subject means over seeds)\n")
    for metric in ("top1_p100", "mrr", "cat", "top10"):
        e3 = per_subject_means(rows, cond_e3, metric)
        md.append(f"\n**{metric}**")
        for c in ("A", "B", "C", "R"):
            if c in e3:
                v = np.array(list(e3[c].values()))
                md.append(f"- {c}: {v.mean():.4f} ± {v.std(ddof=1):.4f} (n={len(v)})")
        tests = [
            (pair, paired(e3[pair[0]], e3[pair[1]]))
            for pair in (("C", "A"), ("A", "B"), ("R", "A"))
            if pair[0] in e3 and pair[1] in e3
        ]
        if tests:
            p = holm([t[1].get("ttest_p", 1.0) for t in tests])
            for (pair, res), pa in zip(tests, p):
                md.append(
                    f"- {pair[0]} vs {pair[1]}: diff {res['diff']:+.4f}, wins {res['wins']}/{res['n']}, Wilcoxon p={res['wilcoxon_p']}, paired-t p={res['ttest_p']:.4f} (Holm {pa:.4f}), dz={res['cohen_dz']:.2f}"
                )
        if "B" in e3 and metric == "top1_p100":
            v = np.array(list(e3["B"].values()))
            t = st.ttest_1samp(v, 0.01)
            md.append(f"- B vs chance (1%): t={t.statistic:.2f}, p={t.pvalue:.5f}")
    # ---- E2: few-shot k vs chance and vs in-subject
    e2 = defaultdict(dict)
    for r in rows:
        if r["exp"] == "e2_loso":
            e2[r["few_shot"]][r["subject"]] = r["top1_p100"]
    if e2:
        md.append("\n## E2 LOSO few-shot (pool-100 top-1)\n")
        for k in sorted(e2):
            v = np.array(list(e2[k].values()))
            t = st.ttest_1samp(v, 0.01)
            md.append(
                f"- k={k}: {v.mean():.4f} ± {v.std(ddof=1):.4f}; vs chance t={t.statistic:.2f} p={t.pvalue:.4f}"
            )
    open(a.out, "w", encoding="utf-8").write("\n".join(md))
    print("wrote", a.out)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- equivalence testing and hierarchical bootstrap
def tost_equivalence(diffs, margin):
    """Two one-sided tests on paired differences (e.g. EEG within-run top-1 minus chance, one value per participant).
    Returns p-values for H0: diff <= -margin and H0: diff >= +margin; equivalence is declared if both < 0.05.
    """
    d = np.asarray(diffs, float)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n) + 1e-12
    p_low = 1 - st.t.cdf((m + margin) / se, n - 1)
    p_high = st.t.cdf((m - margin) / se, n - 1)
    ci = st.t.ppf(0.95, n - 1) * se
    return dict(
        n=n,
        mean_diff=float(m),
        ci90=(float(m - ci), float(m + ci)),
        margin=margin,
        p_lower=float(p_low),
        p_upper=float(p_high),
        equivalent=bool(max(p_low, p_high) < 0.05),
    )


def hierarchical_bootstrap(preds_files, metric="mrr", n_boot=2000, seed=0, rank_key="rank"):
    """Hierarchical bootstrap CI of a retrieval metric: resample participants, then runs within participant,
    then trials within run. preds_files: {participant: path to *_preds.json} with per-trial ``rank_key`` and
    'run'. metric: 'top1', 'top5', 'top10' or 'mrr' (of the stored rank; pass rank_key='ws_rank' for within-run
    ranks if the file stores them). Participants are summarised first so that each carries equal weight.
    """
    if metric not in ("top1", "top5", "top10", "mrr"):
        raise ValueError(f"unknown metric {metric!r}")
    k = {"top1": 1, "top5": 5, "top10": 10}.get(metric)
    score = (lambda r: 1.0 / r) if metric == "mrr" else (lambda r: 1.0 if r <= k else 0.0)
    rng = np.random.default_rng(seed)
    data = {}
    for p, f in preds_files.items():
        by_run = defaultdict(list)
        for x in json.load(open(f)):
            if rank_key in x:
                by_run[x.get("run", 0)].append(score(x[rank_key]))
        data[p] = {r: np.array(v) for r, v in by_run.items() if len(v)}
    parts = list(data)
    point = float(np.mean([np.mean([v.mean() for v in data[p].values()]) for p in parts]))
    boots = []
    for _ in range(n_boot):
        ps = rng.choice(parts, len(parts), replace=True)
        vals = []
        for p in ps:
            runs = list(data[p])
            rs = rng.choice(runs, len(runs), replace=True)
            vals.append(
                np.mean([rng.choice(data[p][r], len(data[p][r]), replace=True).mean() for r in rs])
            )
        boots.append(np.mean(vals))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(
        metric=metric,
        mean=point,
        ci95=(float(lo), float(hi)),
        n_participants=len(parts),
        n_boot=n_boot,
    )
