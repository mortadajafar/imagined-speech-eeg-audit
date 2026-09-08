# Experiment manifest

Every table and figure of the paper is generated from the JSON summaries and per-trial prediction files that
the experiment scripts write (`summary_*.json`, `*_preds.json`, `*_emb.npz`). The table below maps each
reported item to the script that produced its inputs and to the log folder expected by the reporting code
(`kaggle/logs/<folder>/`). Tags inside the summaries follow `<experiment>_<condition>_<participant>[_s<seed>]`.

| Reported item | Script | Log folder(s) | Seeds |
|---|---|---|---|
| Table I, Fig. 1, Fig. 2 (EEGNet, Conformer, ridge) | `in_subject_decoding.py` | `e1_all` | 0, 1, 2 |
| Table I (LaBraM, CBraMod) | `foundation_models.py` | `e1fm_all` | 0 |
| Table I (labels permuted within run), Table S18, Fig. 5 | `within_run_permutation.py` | `e15a` | 0, 1, 2 (permutations) |
| Table II, Fig. 1, Fig. 3 right, Table S4 | `no_eeg_baselines.py` | `e8`, `e8b` | deterministic; null seed 0 |
| Table III, Fig. 2, Table S8 | `within_run_controls.py` | `e7_controls` | 0 |
| Three-seed within-run and matched pools (Tables S14, S17) | `leave_runs_out.py --mode seeds` | `e12c` | 1, 2 |
| Fig. 3 left (category entropy of presented blocks), Tables S15 | `leave_runs_out.py --mode day` (prediction files cover every trial) | `e12c` | 0 |
| Fig. 4 (occlusion) | `within_run_controls.py`, `reading_to_imagining.py` | `e7_controls`, `e3b_seeds` | 0; 1, 2 |
| Fig. 5 left, Tables S9, S10, S16 | `mechanism_probes.py`, `eegsem/analysis/matched_pools.py` | `e11` | 0 |
| Leave-runs-out and leave-one-day-out (Tables S11, S13) | `leave_runs_out.py --mode random_runs`, `--mode day` | `e12`, `e12c` | 0 |
| COFETT (Tables S7, S12) | `cofett_transfer.py`, `leave_runs_out.py --mode cofett_day` | `e5_cofett`, `e5a`, `e12b` | 0 |
| Cross-participant transfer (Table S5) | `cross_subject.py` | `e2_loso` | 0 |
| Reading-to-imagining transfer and data efficiency (Table S6) | `reading_to_imagining.py` | `e3_transfer`, `e3b_seeds`, `e3c` | 0, 1, 2 |
| Per-run normalisation (Table S8) | `per_run_normalisation.py` | `e9` | 0 |
| Foundation models within run (Table S17) | `foundation_models.py` | `e10`, `e15b` | 0 |
| bge-m3 robustness (Table S8) | `embedding_robustness.py` | `e13` | 0 |
| Design checks | `design_checks.py` | `e14` | deterministic |
| BrainMosaic re-implementation | `brainmosaic_reimplementation.py` | `e6_v4`, `e6_textassets` | 42 |

Regenerate everything from the logs without retraining:

```bash
python -m eegsem.reporting.supp_tables            # paper/tex/supp_tables.tex (Tables S1-S19)
python -m eegsem.reporting.figures                # paper/figures/audit_fig1-5
python eegsem/analysis/matched_pools.py           # notes/matched_pools.json (Table S16)
python eegsem/analysis/category_metrics.py        # balanced accuracy, macro-F1, permutation null
python -m eegsem.analysis.stats                   # participant-level CIs and the three-seed TOST
```

Known data issue: in OpenNeuro ds005170 the file `sub-05_task-imagine_run-011_eeg.pkl` is a byte-identical copy
of `run-010` (same MD5). The loader now warns when two run files have identical content; the paper reports the
headline statistics with and without S05 (Table S19).
