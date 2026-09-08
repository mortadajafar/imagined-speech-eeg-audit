"""Aggregate results JSONs (kaggle/logs/*/results/*.json) into one table with mean +- sd over seeds, chance levels, and
bootstrap CIs from per-trial predictions when available. Usage: python -m eegsem.report --logs kaggle/logs --out notes/results_all.md
"""

import argparse, glob, json, os, re
import numpy as np


def load_runs(logs_dir):
    rows = []
    for p in glob.glob(os.path.join(logs_dir, "*", "results", "*.json")):
        if os.path.basename(p).startswith("summary_") or p.endswith("_preds.json"):
            continue
        d = json.load(open(p))
        a = d.get("args", {})
        for key, v in d.get("test", {}).items():
            e = v.get("eeg", {})
            n = v.get("noise", {})
            base = os.path.basename(p)
            m_sub = re.search(r"(sub\d\d)", base) or re.search(
                r"(sub\d\d)", os.path.basename(os.path.dirname(os.path.dirname(p)))
            )
            m_idx = re.search(r"_sub(\d)_", base)
            single = a.get("train_subjects") in (None, [0]) and key.startswith("sub0")
            if single and m_sub:
                subject = m_sub.group(1)
            elif single and m_idx:
                subject = f"sub0{int(m_idx.group(1)) + 1}"
            else:
                subject = key.split("_")[0]
            rows.append(
                dict(
                    exp=os.path.basename(os.path.dirname(os.path.dirname(p))),
                    run=os.path.basename(p)[:-5],
                    test=key,
                    subject=subject,
                    encoder=a.get("encoder", "ridge"),
                    phase=a.get("phase"),
                    test_phase=a.get("test_phase") or a.get("phase"),
                    pretrain=a.get("pretrain_phase"),
                    train_subjects=str(a.get("train_subjects")),
                    seed=a.get("seed", 0),
                    permute=bool(a.get("permute")),
                    permute_within_run=bool(a.get("permute_within_run")),
                    fm_lr_mult=a.get("fm_lr_mult"),
                    norm=a.get("norm"),
                    train_frac=a.get("train_frac", 1.0),
                    few_shot=(
                        int(re.search(r"_k(\d+)", key).group(1))
                        if "_k" in key
                        else a.get("few_shot", 0)
                    ),
                    n_train=d.get("n_train"),
                    pool=e.get("pool_size_full"),
                    top1=e.get("top1_full"),
                    top5=e.get("top5_full"),
                    top10=e.get("top10_full"),
                    mrr=e.get("mrr_full"),
                    top1_p100=e.get("top1_pool100"),
                    top5_p100=e.get("top5_pool100"),
                    cat=e.get("cat_acc"),
                    sem_sim=e.get("sem_sim_top1"),
                    noise_top1_p100=n.get("top1_pool100"),
                    noise_cat=n.get("cat_acc"),
                )
            )
    # ridge rows from summary files (per-run ridge JSONs may have been overwritten before tags existed)
    for p in glob.glob(os.path.join(logs_dir, "*", "results", "summary_*.json")):
        exp = os.path.basename(os.path.dirname(os.path.dirname(p)))
        rows_in = json.load(open(p))
        if not isinstance(rows_in, list):
            continue
        for r in rows_in:
            if not isinstance(r, dict):
                continue
            if r.get("enc") == "ridge" and not r.get("error") and r.get("sub"):
                rows.append(
                    dict(
                        exp=exp,
                        run=f"ridge_{r['sub']}",
                        test="sub0",
                        subject=r["sub"],
                        encoder="ridge",
                        phase="imagine",
                        test_phase="imagine",
                        pretrain=None,
                        train_subjects="[0]",
                        seed=r.get("seed", 0),
                        permute=False,
                        train_frac=1.0,
                        few_shot=0,
                        n_train=None,
                        pool=r.get("pool"),
                        top1=r.get("top1"),
                        top5=r.get("top5"),
                        top10=r.get("top10"),
                        mrr=r.get("mrr"),
                        top1_p100=r.get("top1_pool100"),
                        top5_p100=None,
                        cat=r.get("cat"),
                        sem_sim=None,
                        noise_top1_p100=None,
                        noise_cat=None,
                    )
                )
    rows = [r for r in rows if not (r["encoder"] == "ridge" and r["subject"] == "sub0")]
    return rows


def bootstrap_ci(hits, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    hits = np.asarray(hits, float)
    if len(hits) == 0:
        return (float("nan"), float("nan"))
    bs = rng.choice(hits, (n_boot, len(hits)), replace=True).mean(1)
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def preds_ci(logs_dir):
    """per-run top-1/top-5 bootstrap CIs from *_preds.json"""
    out = {}
    for p in glob.glob(os.path.join(logs_dir, "*", "results", "*_preds.json")):
        d = json.load(open(p))
        t1 = [x["true"] == x["top5"][0] for x in d]
        t5 = [x["true"] in x["top5"] for x in d]
        out[os.path.basename(p)[:-11]] = dict(
            n=len(d), top1_ci=bootstrap_ci(t1), top5_ci=bootstrap_ci(t5)
        )
    return out


def table(
    rows,
    group_keys,
    metrics=("top1", "top5", "top10", "mrr", "top1_p100", "cat", "noise_top1_p100"),
):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in group_keys), []).append(r)
    lines = [
        "| " + " | ".join(group_keys) + " | n | " + " | ".join(metrics) + " |",
        "|" + "---|" * (len(group_keys) + 1 + len(metrics)),
    ]
    for g, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        cells = []
        for m in metrics:
            v = np.array([r[m] for r in rs if r.get(m) is not None], float)
            cells.append(
                ""
                if len(v) == 0
                else (
                    f"{100*v.mean():.2f}±{100*v.std(ddof=0):.2f}"
                    if m != "mrr"
                    else f"{v.mean():.3f}±{v.std(ddof=0):.3f}"
                )
            )
        lines.append(
            "| " + " | ".join(str(x) for x in g) + f" | {len(rs)} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="kaggle/logs")
    ap.add_argument("--out", default="notes/results_all.md")
    a = ap.parse_args(argv)
    rows = load_runs(a.logs)
    ci = preds_ci(a.logs)
    import csv

    with open(a.out.replace(".md", ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = [
        "# All results (auto-generated; percentages except MRR; mean±sd over runs in the group)\n",
        "Chance: top-1 = 1/pool, top-5 = 5/pool, pool-100 top-1 = 1%, 39-way category = 2.56%.\n",
        "## By experiment / encoder / test subject / few-shot / permute\n",
        table(
            rows,
            (
                "exp",
                "encoder",
                "phase",
                "test_phase",
                "pretrain",
                "train_frac",
                "few_shot",
                "permute",
                "test",
            ),
        ),
        "\n## Pooled over subjects and seeds\n",
        table(
            rows,
            (
                "exp",
                "encoder",
                "phase",
                "test_phase",
                "pretrain",
                "train_frac",
                "few_shot",
                "permute",
            ),
        ),
        "\n## Bootstrap 95% CIs (per run, from per-trial predictions)\n",
        "| run | n | top-1 CI | top-5 CI |",
        "|---|---|---|---|",
    ]
    for k, v in sorted(ci.items()):
        md.append(
            f"| {k} | {v['n']} | {100*v['top1_ci'][0]:.2f}–{100*v['top1_ci'][1]:.2f} | {100*v['top5_ci'][0]:.2f}–{100*v['top5_ci'][1]:.2f} |"
        )
    open(a.out, "w", encoding="utf-8").write("\n".join(md))
    print("wrote", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
