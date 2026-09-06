# Auditing sentence-level imagined-speech EEG decoding

Code for *Auditing Sentence-Level Imagined-Speech EEG Decoding: Run Identity and Stimulus-Order Confounds*. The package trains contrastive EEG-to-sentence-embedding decoders on the Chisco and COFETT imagined-speech corpora, evaluates them with run-controlled metrics, and compares them with predictors that never see EEG.

## Layout

```
eegsem/                 the Python package
  data/                 caches for Chisco (preprocessed pickles) and COFETT (raw EDF), hash-based sentence splits, text banks
  models/               EEGNet, Conformer-lite, LaBraM/CBraMod wrappers (vendored model code under models/vendor/)
  training.py           contrastive training and all evaluation metrics (full pool, within-run, cross-run, matched pools)
  ridge.py              linear baseline
  analysis/             occlusion maps, run/position probes, statistics (TOST, hierarchical bootstrap)
  reporting/            result tables, figures, supplementary tables
  brainmosaic.py        harness around the public BrainMosaic code
  paths.py              locating inputs on a Kaggle kernel
scripts/                one script per experiment (see below) and a Kaggle push tool
data/chisco_meta/       category map, channel names, COFETT sentence lists, cap montage
```

## Experiments

| script | what it does |
|---|---|
| `build_chisco_cache.py`, `build_cofett_cache.py`, `build_text_bank.py` | data preparation |
| `in_subject_decoding.py` | ridge / EEGNet / Conformer, sentence-disjoint split, three seeds |
| `foundation_models.py` | LaBraM (fine-tuned, frozen) and CBraMod |
| `cross_subject.py` | leave-one-subject-out with few-shot adapter calibration |
| `reading_to_imagining.py` | reading→imagining transfer, pre-training, data scaling, occlusion |
| `cofett_transfer.py` | COFETT in-subject and Chisco→COFETT |
| `within_run_controls.py`, `per_run_normalisation.py` | run-controlled evaluation and the normalisation mitigation |
| `leave_runs_out.py` | random run folds, leave-one-day-out, COFETT held-out day |
| `mechanism_probes.py` | run classification and position regression from EEG; probes on decoder embeddings |
| `no_eeg_baselines.py`, `design_checks.py` | run-mean and position ±K predictors, permutation null, stimulus-list checks |
| `embedding_robustness.py` | everything again in the bge-m3 space |
| `brainmosaic_reimplementation.py` | BrainMosaic public code on our split |

Each script runs unchanged on a Kaggle script kernel (`python scripts/push_to_kaggle.py <script> --name <slug>`; sources are listed in `scripts/kaggle_registry.json`) or locally with the same arguments. Results are written as JSON and aggregated with `python -m eegsem.reporting.report`.

## Installation

```
pip install -e .          # or: python -m build && pip install dist/eegsem-*.whl
```
Python ≥ 3.10, PyTorch ≥ 2.x. Pre-trained weights: LaBraM-base (github.com/935963004/LaBraM) and CBraMod (huggingface.co/weighting666/CBraMod), placed in a directory passed with `--weights_dir`.

## Data

Chisco (OpenNeuro ds005170) and COFETT (OpenNeuro ds006317) are public. The Chisco caches are built from the authors' preprocessed pickles; the COFETT caches are cut from the raw EDF files (see `eegsem/data/cofett.py` for the preprocessing). `data/chisco_meta/` contains the small metadata files needed by the scripts.

## Licence

MIT. Vendored model code: LaBraM and CBraMod (MIT). BrainMosaic is used from its public repository with the fixes in `scripts/brainmosaic_patch.py`.
