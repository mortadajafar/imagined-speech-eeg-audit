"""Figures for the audit paper. Sources: kaggle/logs summaries (E1-all, E8, E8b, E7, E6 v4). Output: paper/figures/audit_fig*.pdf/png"""

import json, glob, os, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {
    "eeg": "#2a78d6",
    "conf": "#eb6834",
    "run": "#eda100",
    "pos3": "#1baf7a",
    "pos10": "#e87ba4",
    "bm": "#4a3aa7",
    "noise": "#52514e",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


def _style(ax, ylabel, chance=None, label="chance"):
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK)
    if chance is not None:
        ax.axhline(chance, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(
            ax.get_xlim()[1], chance, f" {label}", va="center", ha="left", fontsize=8, color=INK2
        )


def load():
    d = {}
    d["e1"] = json.load(open("kaggle/logs/e1_all/results/summary_e1_all.json"))
    d["e8"] = json.load(
        open(glob.glob("kaggle/logs/e8/**/summary_e8_runonly.json", recursive=True)[0])
    )
    d["e8b"] = json.load(open(glob.glob("kaggle/logs/e8b/**/summary_e8b.json", recursive=True)[0]))
    d["e7"] = json.load(
        open(glob.glob("kaggle/logs/e7_controls/**/summary_e7.json", recursive=True)[0])
    )
    d["e6"] = json.load(
        open(glob.glob("kaggle/logs/e6_v4/**/summary_e6_v4.json", recursive=True)[0])
    )
    return d


SUBS = ["sub01", "sub02", "sub03", "sub04", "sub05"]


def fig_eeg_vs_noeeg(d, out):
    """Fig A: per participant, EEG decoders vs no-EEG predictors (pool-100 top-1 and category)."""
    eeg = {s: [r for r in d["e1"] if r["enc"] == "eegnet" and r["sub"] == s] for s in SUBS}
    e8 = {r["cache"]: r for r in d["e8"]}
    e8b = {r["tag"]: r for r in d["e8b"]}
    series = [
        ("EEGNet (EEG)", C["eeg"], lambda s, m: np.mean([r[m] for r in eeg[s]])),
        (
            "Run identity only (no EEG)",
            C["run"],
            lambda s, m: e8[s][{"top1_pool100": "top1_pool100", "cat": "cat_acc"}[m]],
        ),
        (
            "Position ±10 (no EEG)",
            C["pos10"],
            lambda s, m: e8b[f"{s}_position_k10"][
                {"top1_pool100": "top1_pool100", "cat": "cat_acc"}[m]
            ],
        ),
        (
            "Position ±3 (no EEG)",
            C["pos3"],
            lambda s, m: e8b[f"{s}_position_k3"][
                {"top1_pool100": "top1_pool100", "cat": "cat_acc"}[m]
            ],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    w = 0.8 / len(series)
    for ax, m, yl, ch in (
        (axes[0], "top1_pool100", "Top-1, 100-sentence pool (%)", 1.0),
        (axes[1], "cat", "Zero-shot 39-way category (%)", 100 / 39),
    ):
        for j, (lab, col, f) in enumerate(series):
            ax.bar(
                np.arange(5) + (j - 1.5) * w,
                [100 * f(s, m) for s in SUBS],
                width=w * 0.92,
                color=col,
                label=lab,
            )
        ax.set_xticks(range(5))
        ax.set_xticklabels([s.replace("sub", "S") for s in SUBS])
        _style(ax, yl, ch)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def fig_within_run(d, out):
    """Fig B: full-pool vs within-run retrieval (top-1 over chance ratio) for EEGNet, noise, BrainMosaic, run-mean, position ±3."""
    e7 = {r["tag"].replace("e7_chisco_", ""): r for r in d["e7"] if "chisco" in r["tag"]}
    e8 = {r["cache"]: r for r in d["e8"]}
    e8b = {r["tag"]: r for r in d["e8b"]}
    e6 = {r["sub"]: r for r in d["e6"]}
    rows = [
        (
            "EEGNet",
            C["eeg"],
            lambda s: (e7[s]["top1_pool100"] / 0.01, e7[s]["ws_top1"] / e7[s]["ws_chance_top1"]),
        ),
        (
            "EEGNet, noise input",
            C["noise"],
            lambda s: (
                e7[s]["noise_top1_pool100"] / 0.01,
                e7[s]["noise_ws_top1"] / e7[s]["ws_chance_top1"],
            ),
        ),
        (
            "BrainMosaic (public code)",
            C["bm"],
            lambda s: (
                e6[s]["test_retrieval"]["top1_pool100"] / 0.01,
                e6[s]["test_retrieval"]["ws_top1"] / e6[s]["test_retrieval"]["ws_chance_top1"],
            ),
        ),
        (
            "Run identity (no EEG)",
            C["run"],
            lambda s: (e8[s]["top1_pool100"] / 0.01, e8[s]["ws_top1"] / e8[s]["ws_chance_top1"]),
        ),
        (
            "Position ±3 (no EEG)",
            C["pos3"],
            lambda s: (
                e8b[f"{s}_position_k3"]["top1_pool100"] / 0.01,
                e8b[f"{s}_position_k3"]["ws_top1"] / e8b[f"{s}_position_k3"]["ws_chance_top1"],
            ),
        ),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    w = 0.8 / len(rows)
    x = np.arange(2)
    rng = np.random.default_rng(0)
    for j, (lab, col, f) in enumerate(rows):
        vals = np.array([f(s) for s in SUBS])
        m = vals.mean(0)
        sd = vals.std(0, ddof=1)
        ax.bar(
            x + (j - 2) * w,
            m,
            width=w * 0.92,
            color=col,
            label=lab,
            yerr=sd,
            error_kw=dict(ecolor=INK2, lw=0.8, capsize=2),
            zorder=2,
        )
        for k in range(2):
            ax.scatter(
                x[k] + (j - 2) * w + rng.uniform(-w * 0.25, w * 0.25, len(SUBS)),
                vals[:, k],
                s=9,
                color="white",
                edgecolor=INK,
                linewidth=0.6,
                zorder=3,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Full pool (100-sentence subsets)", "Within-run (same-run candidates only)"]
    )
    _style(ax, "Top-1 accuracy / chance", 1.0, "chance (×1)")
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def fig_order_structure(out):
    """Fig C: stimulus-order structure — category entropy per Chisco block and COFETT day list; position-baseline curve."""
    import pandas as pd, collections

    tm = {
        " ".join(k.split()): v
        for k, v in json.load(open("data/chisco_meta/textmaps.json", encoding="utf-8")).items()
    }

    def H(cats):
        c = collections.Counter(cats)
        p = np.array(list(c.values()), float)
        p /= p.sum()
        return -(p * np.log2(p)).sum()

    ch = []
    for f in sorted(
        glob.glob("data/chisco_meta/textdataset/split_data_*.xlsx"),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    ):
        df = pd.read_excel(f)
        ch.append(H([tm.get(" ".join(str(s).split()), -1) for s in df[df.columns[0]]]))
    cof = json.load(open("data/chisco_meta/cofett_sentences.json", encoding="utf-8"))
    cf = [H([tm.get(s, -1) for s in cof[k]]) for k in ["text1-1", "text1-2", "text1-3", "text1-4"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    ax.bar(np.arange(len(ch)), ch, color=C["eeg"], width=0.8, label="Chisco block lists")
    ax.bar(len(ch) + 1 + np.arange(4), cf, color=C["run"], width=0.8, label="COFETT day lists")
    ax.axhline(np.log2(39), color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(0, np.log2(39) + 0.15, "uniform over 39 categories (5.3 bits)", fontsize=7, color=INK2)
    ax.set_xticks([])
    ax.set_xlabel("stimulus list (Chisco blocks, then COFETT days)", color=INK)
    _style(ax, "Category entropy of list (bits)")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.set_ylim(0, 6)
    ax = axes[1]
    e8b = json.load(open(glob.glob("kaggle/logs/e8b/**/summary_e8b.json", recursive=True)[0]))
    e8 = json.load(open(glob.glob("kaggle/logs/e8/**/summary_e8_runonly.json", recursive=True)[0]))
    ks = ["run mean", "±10", "±3"]
    for s in SUBS:
        v = [
            100 * next(r["top1_pool100"] for r in e8 if r["cache"] == s),
            100 * next(r["top1_pool100"] for r in e8b if r["tag"] == f"{s}_position_k10"),
            100 * next(r["top1_pool100"] for r in e8b if r["tag"] == f"{s}_position_k3"),
        ]
        ax.plot(range(3), v, color=C["pos3"], alpha=0.35, lw=1.2, marker="o", ms=3)
    e1 = json.load(open("kaggle/logs/e1_all/results/summary_e1_all.json"))
    eeg = 100 * np.mean([r["top1_pool100"] for r in e1 if r["enc"] == "eegnet"])
    ax.axhline(eeg, color=C["eeg"], lw=1.6, ls=(0, (6, 3)), label="EEGNet (EEG)")
    ax.set_xticks(range(3))
    ax.set_xticklabels(ks)
    ax.set_xlabel("no-EEG predictor: neighbourhood in presentation order", color=INK)
    _style(ax, "Top-1, 100-sentence pool (%)", 1.0)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main():
    d = load()
    os.makedirs("paper/figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig_eeg_vs_noeeg(d, f"paper/figures/audit_fig1_eeg_vs_noeeg.{ext}")
        fig_within_run(d, f"paper/figures/audit_fig2_within_run.{ext}")
        fig_order_structure(f"paper/figures/audit_fig3_order_structure.{ext}")


if __name__ == "__main__":
    main()


def fig_occlusion(
    logs=("kaggle/logs/e3b_seeds/results", "kaggle/logs/e7_controls/results"),
    out="paper/figures/audit_fig4_occlusion",
):
    """Fig 4: relative MRR drop when 0.5 s windows or scalp regions of the imagined epoch are occluded."""
    import re
    from collections import defaultdict

    occ = defaultdict(lambda: defaultdict(list))
    for d in logs:
        for f in glob.glob(os.path.join(d, "*.json")):
            if f.endswith("_preds.json") or "summary" in f:
                continue
            run = json.load(open(f))
            o = run.get("occlusion", {}).get("sub0")
            if not o:
                continue
            tag = run["args"]["tag"]
            cond = (
                "A"
                if ("A_imagine" in tag or "e7_chisco" in tag)
                else ("B" if "B_read" in tag else ("C" if "C_read" in tag else None))
            )
            if cond is None:
                continue
            base = o["baseline"]["mrr"]
            for t in o["temporal"]:
                occ[cond][("t", t["t_start"])].append(100 * (base - t["mrr"]) / base)
            for k, v in o["regions"].items():
                occ[cond][("r", k)].append(100 * (base - v["mrr"]) / base)
    cols = {"A": C["eeg"], "B": C["run"], "C": C["pos3"]}
    labels = {
        "A": "imagined-only decoder",
        "B": "reading-trained, tested on imagined",
        "C": "reading-pretrained + fine-tuned",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    handles = []
    for c in "ABC":
        ks = sorted(k for k in occ[c] if k[0] == "t")
        m = [np.mean(occ[c][k]) for k in ks]
        se = [np.std(occ[c][k], ddof=1) / np.sqrt(len(occ[c][k])) for k in ks]
        handles.append(
            ax.errorbar(
                [k[1] + 0.25 for k in ks],
                m,
                yerr=se,
                color=cols[c],
                marker="o",
                ms=4,
                lw=1.6,
                capsize=2,
                label=labels[c],
            )
        )
    ax.set_xlabel("occluded 0.5 s window (centre, s after recall onset)", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("MRR drop when occluded (%)", color=INK)
    ax.set_ylim(0, None)
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=1,
    )
    ax = axes[1]
    regs = [
        "frontal",
        "fronto-central",
        "central",
        "centro-parietal",
        "temporal",
        "parietal",
        "parieto-occipital",
    ]
    w = 0.27
    for j, c in enumerate("ABC"):
        m = [np.mean(occ[c][("r", r)]) for r in regs]
        se = [np.std(occ[c][("r", r)], ddof=1) / np.sqrt(len(occ[c][("r", r)])) for r in regs]
        ax.bar(
            np.arange(len(regs)) + (j - 1) * w,
            m,
            width=w * 0.92,
            color=cols[c],
            yerr=se,
            error_kw=dict(ecolor=INK2, lw=0.8, capsize=2),
        )
    ax.set_xticks(range(len(regs)))
    ax.set_xticklabels(["F", "FC", "C", "CP", "T", "P", "PO"])
    ax.set_xlabel("occluded scalp region", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("MRR drop when occluded (%)", color=INK)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_mechanism(out="paper/figures/audit_fig5_mechanism"):
    """Fig 5: run identity is decodable and drives retrieval; matched pools and leave-one-day-out."""
    e11 = json.load(open(glob.glob("kaggle/logs/e11/**/summary_e11.json", recursive=True)[0]))
    e12 = [
        r
        for r in json.load(
            open(glob.glob("kaggle/logs/e12/**/summary_e12.json", recursive=True)[0])
        )
        if not r.get("error")
    ]
    e12c = [
        r
        for r in json.load(
            open(glob.glob("kaggle/logs/e12c/**/summary_e12c.json", recursive=True)[0])
        )
        if not r.get("error")
    ]
    subs = [r["sub"] for r in e11]
    # three-seed averages for the in-subject conditions (seed 0 from E11, seeds 1-2 from E12c)
    extra = {
        s: [r for r in e12c if "insub" in r["tag"] and r["tag"].split("_")[2] == s] for s in subs
    }
    p100 = {
        s: np.mean([e11[i]["aligner_p100"]] + [r["top1_pool100"] for r in extra[s]])
        for i, s in enumerate(subs)
    }
    matched = {
        s: np.mean([e11[i]["xs_matched_top1"]] + [r["xs_matched_top1"] for r in extra[s]])
        for i, s in enumerate(subs)
    }
    within = {
        s: np.mean([e11[i]["aligner_ws_top1"]] + [r["ws_top1"] for r in extra[s]])
        for i, s in enumerate(subs)
    }
    chance = {s: e11[i]["aligner_ws_chance"] for i, s in enumerate(subs)}
    loro = {
        s: np.mean([r["top1_pool100"] for r in e12 if r["tag"].split("_")[1] == s]) for s in subs
    }
    e15a = [
        r
        for r in json.load(
            open(
                glob.glob(
                    "kaggle/logs/e15a/**/summary_within_run_permutation.json", recursive=True
                )[0]
            )
        )
        if r.get("kind") == "permute_within_run"
    ]
    permrun = {s: np.mean([r["top1_pool100"] for r in e15a if r["sub"] == s]) for s in subs}
    lodo = {
        s: np.mean(
            [r["top1_pool100"] for r in e12c if "lodo" in r["tag"] and r["tag"].split("_")[2] == s]
        )
        for s in subs
    }
    x = np.arange(5)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1.0, 1.5]})
    ax = axes[0]
    w = 0.27
    b1 = ax.bar(
        x - w,
        [100 * r["run_acc"] for r in e11],
        w * 0.92,
        color=C["eeg"],
        label="run identity decoded from EEG (45-way)",
    )
    b2 = ax.bar(
        x,
        [100 * r["emb_run_acc"] for r in e11],
        w * 0.92,
        color="#eb6834",
        label="run identity from EEGNet embedding (linear probe)",
    )
    b3 = ax.bar(
        x + w,
        [100 * max(r["pos_within_run_spearman"], 0) for r in e11],
        w * 0.92,
        color="#1baf7a",
        label="position within run (Spearman ρ × 100)",
    )
    ax.axhline(100 / 45, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(4.45, 100 / 45 + 2.5, "chance 2.2%", fontsize=7, color=INK2, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("sub", "S") for s in subs])
    ax.set_ylabel("accuracy (%) / ρ × 100", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 105)
    ax.legend(
        handles=[b1, b2, b3],
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
    )
    ax = axes[1]
    conds = [
        "full pool",
        "labels\npermuted\nwithin run",
        "matched,\nother-run",
        "matched,\nsame-run",
        "leave-\nruns-out",
        "leave-one-\nday-out",
    ]
    vals = np.array(
        [
            [p100[s] / 0.01 for s in subs],
            [permrun[s] / 0.01 for s in subs],
            [matched[s] / chance[s] for s in subs],
            [within[s] / chance[s] for s in subs],
            [loro[s] / 0.01 for s in subs],
            [lodo[s] / 0.01 for s in subs],
        ]
    )
    m = vals.mean(1)
    sd = vals.std(1, ddof=1)
    xx = np.arange(6)
    ax.bar(
        xx,
        m,
        0.6,
        color=[C["eeg"], "#8fb8ea", C["eeg"], "#a8a7a3", "#a8a7a3", "#a8a7a3"],
        yerr=sd,
        error_kw=dict(ecolor=INK2, lw=0.8, capsize=2),
        zorder=2,
    )
    rng = np.random.default_rng(0)
    for k in range(6):
        ax.scatter(
            xx[k] + rng.uniform(-0.15, 0.15, 5),
            vals[k],
            s=9,
            color="white",
            edgecolor=INK,
            linewidth=0.6,
            zorder=3,
        )
    ax.axhline(1, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(-0.4, 1.12, "chance", fontsize=7, color=INK2, ha="left")
    ax.set_xticks(xx)
    ax.set_xticklabels(conds, fontsize=7)
    ax.set_xlabel("EEGNet evaluation condition", color=INK, fontsize=8)
    ax.set_ylabel("top-1 accuracy / chance", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
