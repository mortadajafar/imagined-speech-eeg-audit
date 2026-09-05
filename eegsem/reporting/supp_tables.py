"""Generate supplementary LaTeX tables from kaggle/logs summaries -> paper/tex/supp_tables.tex"""

import json, glob, os, re, numpy as np


def load(pat):
    f = glob.glob(pat, recursive=True)
    return json.load(open(f[0], encoding="utf-8")) if f else []


def tab(caption, header, rows, label, colspec=None):
    if not rows:
        return "% table %s omitted (no data)\n" % label
    rows = [[c.replace("_", "\\_") if "$" not in c else c for c in r] for r in rows]
    cols = colspec or "l" + "c" * (len(header) - 1)
    return (
        "\\begin{table*}[t]\\centering\\caption{%s}\\label{%s}\\footnotesize\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{%s}\\toprule\n%s \\\\\\midrule\n%s\n\\bottomrule\\end{tabular}\\end{table*}\n"
        % (
            caption,
            label,
            cols,
            " & ".join(header),
            " \\\\\n".join(" & ".join(r) for r in rows) + " \\\\",
        )
    )


def pct(x, d=1):
    return "--" if x is None else f"{100*x:.{d}f}"


def main(logs_dir="kaggle/logs", out_path="paper/tex/supp_tables.tex"):
    """Write all supplementary tables as LaTeX."""
    out = []
    # S1: E1-all per subject per encoder (mean over seeds)
    e1 = load(logs_dir + "/e1_all/results/summary_e1_all.json")
    rows = []
    for enc in ("ridge", "eegnet", "conformer"):
        for s in sorted({r["sub"] for r in e1}):
            rs = [r for r in e1 if r["enc"] == enc and r["sub"] == s and not r.get("error")]
            if rs:
                rows.append(
                    [
                        enc,
                        s.replace("sub", "S"),
                        str(len(rs)),
                        pct(np.mean([r["top1_pool100"] for r in rs])),
                        pct(np.mean([r["top5"] for r in rs])),
                        pct(np.mean([r["top10"] for r in rs])),
                        f"{np.mean([r['mrr'] for r in rs]):.3f}",
                        pct(np.mean([r["cat"] for r in rs])),
                    ]
                )
    out.append(
        tab(
            "In-subject imagined-speech decoding on Chisco (E1), per participant; mean over seeds. p100 = top-1 in 100-sentence pools.",
            ["encoder", "participant", "seeds", "p100", "top-5", "top-10", "MRR", "cat"],
            rows,
            "tab:s1",
        )
    )
    # S2: foundation models per subject
    fm = load(logs_dir + "/e1fm_all/results/summary_e1fm_all.json")
    rows = [
        [
            r["enc"] + ("-frozen" if r["mode"] == "frozen" else ""),
            r["sub"].replace("sub", "S"),
            pct(r["top1_pool100"]),
            pct(r["top5"]),
            pct(r["top10"]),
            f"{r['mrr']:.3f}",
            pct(r["cat"]),
        ]
        for r in fm
        if not r.get("error")
    ]
    out.append(
        tab(
            "Foundation-model encoders in-subject (E1-FM), seed 0.",
            ["encoder", "participant", "p100", "top-5", "top-10", "MRR", "cat"],
            rows,
            "tab:s2",
        )
    )
    # S3: within-run / cross-run (E7, E10, E6 v4)
    rows = []
    for r in load(logs_dir + "/e7_controls/**/summary_e7.json"):
        if "chisco" in r["tag"]:
            rows.append(
                [
                    "EEGNet",
                    r["tag"].split("_")[-1].replace("sub", "S"),
                    pct(r["top1_pool100"]),
                    pct(r["ws_top1"]),
                    pct(r["ws_chance_top1"]),
                    pct(r["ws_top5"]),
                    pct(r["ws_chance_top5"]),
                    pct(r["noise_ws_top1"]),
                ]
            )
    for r in load(logs_dir + "/e10/**/summary_e10.json"):
        rows.append(
            [
                r["enc"],
                r["tag"].split("_")[-1].replace("sub", "S"),
                pct(r["top1_pool100"]),
                pct(r["ws_top1"]),
                pct(r["ws_chance_top1"]),
                pct(r["ws_top5"]),
                pct(r["ws_chance_top5"]),
                pct(r["noise_ws_top1"]),
            ]
        )
    for r in load(logs_dir + "/e6_v4/**/summary_e6_v4.json"):
        t = r["test_retrieval"]
        rows.append(
            [
                "BrainMosaic (public)",
                r["sub"].replace("sub", "S"),
                pct(t["top1_pool100"]),
                pct(t["ws_top1"]),
                pct(t["ws_chance_top1"]),
                pct(t["ws_top5"]),
                pct(t["ws_chance_top5"]),
                "--",
            ]
        )
    out.append(
        tab(
            "Full-pool vs.\\ within-run retrieval per participant and model (E7, E10, E6).",
            [
                "model",
                "participant",
                "p100",
                "within-run top-1",
                "chance",
                "within-run top-5",
                "chance",
                "noise within-run top-1",
            ],
            rows,
            "tab:s3",
        )
    )
    # S4: no-EEG baselines per subject (E8, E8b v2)
    e8 = {r["cache"]: r for r in load(logs_dir + "/e8/**/summary_e8_runonly.json")}
    e8b = {r["tag"]: r for r in load(logs_dir + "/e8b_v2/**/summary_e8b_v2.json")}
    rows = []
    for s in sorted(e8):
        rows.append(
            [
                "run-mean",
                s.replace("sub", "S").replace("cofett_", "COFETT "),
                pct(e8[s]["top1_pool100"]),
                pct(e8[s]["top10_full"]),
                f"{e8[s]['mrr_full']:.3f}",
                pct(e8[s]["cat_acc"]),
                pct(e8[s]["run_majority_cat_acc"]),
            ]
        )
        for k in (10, 3):
            r = e8b.get(f"{s}_position_k{k}")
            if r:
                rows.append(
                    [
                        f"position $\\pm{k}$",
                        s.replace("sub", "S").replace("cofett_", "COFETT "),
                        pct(r["top1_pool100"]),
                        pct(r["top10_full"]),
                        f"{r['mrr_full']:.3f}",
                        pct(r["cat_acc"]),
                        "--",
                    ]
                )
    out.append(
        tab(
            "No-EEG design baselines per participant (E8, E8b).",
            ["predictor", "participant", "p100", "top-10", "MRR", "cat", "run-majority cat"],
            rows,
            "tab:s4",
        )
    )
    # S5: cross-subject few-shot (E2) and transfer (E3/E3b), E9
    rows = [
        [
            f"held-out S0{r['held']+1}",
            r["key"].split("_")[-1],
            pct(r["top1_pool100"]),
            pct(r["top10"]),
            f"{r['mrr']:.3f}",
            pct(r["cat"]),
        ]
        for r in load(logs_dir + "/e2_loso/**/summary_e2.json")
        if not r.get("error")
    ]
    out.append(
        tab(
            "Leave-one-subject-out with adapter calibration on $k$ trials (E2, EEGNet).",
            ["held-out", "$k$", "p100", "top-10", "MRR", "cat"],
            rows,
            "tab:s5",
        )
    )
    LAB9 = {
        "e9_sesnorm_A": "imagined-only, per-run normalised",
        "e9_sesnorm_B": "reading-trained, tested on imagined, per-run normalised",
        "e9_plain_B": "reading-trained, tested on imagined",
    }
    rows = [
        [
            LAB9[re.sub(r"_(sub\d+|cofett_sub\d+)$", "", r["tag"])],
            re.search(r"(cofett_sub\d+|sub\d+)$", r["tag"])
            .group(1)
            .replace("cofett_sub", "COFETT C")
            .replace("sub", "S"),
            pct(r["top1_pool100"]),
            pct(r["top10"]),
            f"{r['mrr']:.3f}",
            pct(r["cat"]),
            pct(r.get("ws_top1")),
            pct(r.get("ws_chance_top1")),
        ]
        for r in load(logs_dir + "/e9/**/summary_e9.json")
        if not r.get("error")
    ]
    out.append(
        tab(
            "Per-run normalisation and reading$\\rightarrow$imagining transfer scored within run (E9; EEGNet, seed 0).",
            [
                "condition",
                "participant",
                "p100",
                "top-10",
                "MRR",
                "cat",
                "within-run top-1",
                "chance",
            ],
            rows,
            "tab:s6",
        )
    )

    # S7: embedding robustness (E13, bge-m3)
    rows = [
        [
            r["kind"]
            .replace("eegnet_bgem3", "EEGNet (EEG)")
            .replace("noeeg_bgem3_runmean", "run-mean")
            .replace("noeeg_bgem3_position_k", "position $\\pm$"),
            r["sub"].replace("sub", "S"),
            pct(r.get("top1_pool100")),
            pct(r.get("top10")),
            f"{r.get('mrr') or 0:.3f}",
            pct(r.get("cat")),
            pct(r.get("ws_top1")) if r.get("ws_top1") is not None else "--",
            pct(r.get("xs_matched_top1")) if r.get("xs_matched_top1") is not None else "--",
        ]
        for r in load(logs_dir + "/e13/**/summary_e13.json")
        if not r.get("error")
    ]
    out.append(
        tab(
            "Embedding robustness: decoder and no-EEG predictors in the bge-m3 space (E13); matched = top-1 in same-size pools drawn from other runs.",
            [
                "predictor",
                "participant",
                "p100",
                "top-10",
                "MRR",
                "cat",
                "within-run top-1",
                "matched other-run top-1",
            ],
            rows,
            "tab:s7",
        )
    )

    # S8a/S8b: mechanism probes (E11), split to fit the column width
    e11 = [r for r in load(logs_dir + "/e11/**/summary_e11.json") if not r.get("error")]
    rows = [
        [
            r["sub"].replace("sub", "S"),
            pct(r["run_acc"]),
            pct(r["run_top5"]),
            f"{r['pos_r2']:.2f}",
            f"{r['pos_spearman']:.2f}",
            f"{r['pos_within_run_spearman']:.2f}",
            pct(r.get("emb_run_acc")),
            f"{r.get('emb_pos_spearman', 0):.2f}",
        ]
        for r in e11
    ]
    out.append(
        tab(
            "Mechanism probes (E11, seed 0): 45-way run classification and within-run position regression from EEG (EEGNet heads), and linear probes for run and position on the contrastive decoder embeddings (5-fold CV on the test partition).",
            [
                "participant",
                "run acc",
                "run top-5",
                "position $R^2$",
                "position $\\rho$",
                "position $\\rho$ (within run)",
                "embedding: run acc",
                "embedding: position $\\rho$",
            ],
            rows,
            "tab:s8a",
        )
    )
    rows = [
        [
            r["sub"].replace("sub", "S"),
            pct(r["aligner_p100"]),
            pct(r.get("xs_matched_top1")),
            pct(r["aligner_ws_top1"]),
            pct(r["aligner_ws_chance"]),
            f"{r.get('run_conf_vs_top10_spearman', 0):.3f}",
            pct(r.get("top10_when_run_correct")),
            pct(r.get("top10_when_run_wrong")),
        ]
        for r in e11
    ]
    out.append(
        tab(
            "Matched-pool and within-run retrieval of the EEGNet decoder (E11, seed 0; three-seed values in Table~\\ref{tab:s12}), and the relation between run-probe confidence and retrieval success.",
            [
                "participant",
                "p100",
                "matched other-run top-1",
                "within-run top-1",
                "chance",
                "$\\rho$(run conf., hit)",
                "top-10 $\\mid$ run correct",
                "top-10 $\\mid$ run wrong",
            ],
            rows,
            "tab:s8b",
        )
    )
    rows = [
        [
            r["tag"].replace("e12_", "").replace("sub", "S").replace("_fold", " fold "),
            str(r["n_test"]),
            str(r["pool"]),
            pct(r["top1"], 2),
            pct(r["top10"]),
            f"{r['mrr']:.3f}",
            pct(r["top1_pool100"]),
            f"{r['rank_pct']:.3f}",
            pct(r["cat"]),
            pct(r["noise_top1_pool100"]),
            f"{r['noise_rank_pct']:.3f}",
        ]
        for r in load(logs_dir + "/e12/**/summary_e12.json")
        if not r.get("error")
    ]
    out.append(
        tab(
            "Leave-runs-out evaluation on Chisco (E12): five folds of nine held-out runs per participant.",
            [
                "fold",
                "test trials",
                "pool",
                "top-1",
                "top-10",
                "MRR",
                "p100",
                "rank pct",
                "cat",
                "noise p100",
                "noise rank pct",
            ],
            rows,
            "tab:s9",
        )
    )

    # S10: COFETT held-out day (E12b)
    rows = [
        [
            r["tag"]
            .replace("e12b_cofett_", "COFETT ")
            .replace("sub", "C")
            .replace("_day", " day "),
            str(r["n_test"]),
            str(r["pool"]),
            pct(r["top1"], 2),
            pct(r["top10"]),
            f"{r['mrr']:.3f}",
            pct(r["top1_pool100"]),
            f"{r['rank_pct']:.3f}",
            pct(r["cat"]),
            pct(r["noise_top1_pool100"]),
            f"{r['noise_rank_pct']:.3f}",
        ]
        for r in load(logs_dir + "/e12b/summary_e12b.json")
        if not r.get("error")
    ]
    out.append(
        tab(
            "COFETT held-out-day evaluation (E12b): train on two days, validate on one, test on the fourth.",
            [
                "fold",
                "test trials",
                "pool",
                "top-1",
                "top-10",
                "MRR",
                "p100",
                "rank pct",
                "cat",
                "noise p100",
                "noise rank pct",
            ],
            rows,
            "tab:s10",
        )
    )

    # S0: reproducibility map of experiment identifiers
    rows = [
        [
            "E1 / E1-all",
            "in-subject decoders, sentence split",
            "kaggle/e1_all, kaggle/e1fm_all",
            "0,1,2 (FM: 0)",
        ],
        ["E2", "leave-one-subject-out + few-shot adapter", "kaggle/e2_loso", "0"],
        [
            "E3 / E3b / E3c",
            "reading$\\rightarrow$imagining transfer, data scaling",
            "kaggle/e3_transfer, e3b_seeds, e3c_dataeff",
            "0,1,2",
        ],
        [
            "E5 / E5a",
            "COFETT in-subject and Chisco$\\rightarrow$COFETT",
            "kaggle/e5_cofett_eval, e5a_cofett_eegnet",
            "0",
        ],
        [
            "E6",
            "BrainMosaic public-code re-implementation",
            "kaggle/e6_brainmosaic (+e6_textassets_v2)",
            "42 (authors\\textquotesingle{} default)",
        ],
        ["E7", "within-run controls", "kaggle/e7_controls", "0"],
        [
            "E8 / E8b",
            "no-EEG run-mean and position baselines, run-permutation null",
            "kaggle/e8_runonly, e8b_positionbaseline",
            "deterministic; null seed 0",
        ],
        [
            "E9",
            "per-run normalisation; within-run reading$\\rightarrow$imagining",
            "kaggle/e9_sessionnorm",
            "0",
        ],
        ["E10", "foundation models within run", "kaggle/e10_fm_withinrun", "0"],
        ["E11", "run/position probes, embedding probes, matched pools", "kaggle/e11_probe", "0"],
        [
            "E12 / E12b / E12c",
            "leave-runs-out (random folds), COFETT held-out day, leave-one-day-out + seeds",
            "kaggle/e12_loro, e12b_cofett_day, e12c_lodo_seeds",
            "0 (E12c: 0,1,2)",
        ],
        ["E13", "bge-m3 robustness", "kaggle/e13_embedding", "0"],
        [
            "E14",
            "design checks (order identity, category baselines)",
            "kaggle/e14_design",
            "deterministic",
        ],
    ]
    out.insert(
        0,
        tab(
            "Experiment identifiers used in the supplementary tables, the corresponding scripts in the code repository, and seeds.",
            ["identifier", "content", "script folder", "seeds"],
            rows,
            "tab:s0",
            colspec="l>{\\raggedright\\arraybackslash}p{5cm}>{\\raggedright\\arraybackslash}p{6cm}>{\\raggedright\\arraybackslash}p{2.8cm}",
        ),
    )
    # S11/S12: leave-one-day-out (E12c) and three-seed within-run metrics
    e12c = [r for r in load(logs_dir + "/e12c/**/summary_e12c.json") if not r.get("error")]
    rows = [
        [
            r["tag"].replace("e12c_lodo_", "").replace("sub", "S").replace("_day", " day "),
            str(r["n_test"]),
            str(r["pool"]),
            pct(r["top10"]),
            pct(r["top1_pool100"]),
            f"{r['rank_pct']:.3f}",
            pct(r["cat"]),
            pct(r["noise_top1_pool100"]),
            f"{r['noise_rank_pct']:.3f}",
        ]
        for r in e12c
        if "lodo" in r["tag"]
    ]
    out.append(
        tab(
            "Leave-one-day-out evaluation on Chisco (E12c, EEGNet seed 0): all nine blocks of one recording day held out per fold.",
            [
                "fold",
                "test trials",
                "pool",
                "top-10",
                "p100",
                "rank pct",
                "cat",
                "noise p100",
                "noise rank pct",
            ],
            rows,
            "tab:s11",
        )
    )
    e11 = {r["sub"]: r for r in load(logs_dir + "/e11/**/summary_e11.json") if not r.get("error")}
    rows = []
    for s_ in sorted(e11):
        seeds = [
            (0, e11[s_]["aligner_p100"], e11[s_]["aligner_ws_top1"], e11[s_]["xs_matched_top1"])
        ] + [
            (int(r["tag"][-1]), r["top1_pool100"], r["ws_top1"], r["xs_matched_top1"])
            for r in e12c
            if "insub" in r["tag"] and r["tag"].split("_")[2] == s_
        ]
        for sd, a, b, c in sorted(seeds):
            rows.append(
                [
                    s_.replace("sub", "S"),
                    str(sd),
                    pct(a),
                    pct(b),
                    pct(e11[s_]["aligner_ws_chance"]),
                    pct(c),
                ]
            )
    out.append(
        tab(
            "Within-run and matched-pool retrieval of the EEGNet decoder for three seeds (E11, E12c).",
            [
                "participant",
                "seed",
                "p100",
                "within-run top-1",
                "chance",
                "matched other-run top-1",
            ],
            rows,
            "tab:s12",
        )
    )
    open(out_path, "w", encoding="utf-8").write("\n".join(out))
    print("supp_tables.tex written with", len(out), "tables")


if __name__ == "__main__":
    main()
