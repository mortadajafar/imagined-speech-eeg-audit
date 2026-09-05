"""Shared helpers for the experiment scripts: Kaggle setup, result bookkeeping."""

import json
import os
import time
import traceback

from eegsem import paths


def kaggle_inputs(need_cofett=False, need_weights=False):
    """Resolve the standard inputs on a Kaggle kernel and prepare a results directory."""
    from eegsem.data.text_bank import ensure_bank

    caches = paths.chisco_caches()
    cofett = paths.cofett_caches() if need_cofett else []
    bank = ensure_bank(
        paths.text_bank(), "labse", caches + cofett, ["imagine", "read"], "/kaggle/working/bank"
    )
    out = "/kaggle/working/results"
    os.makedirs(out, exist_ok=True)
    return {
        "chisco": caches,
        "cofett": cofett,
        "bank": bank,
        "channels": paths.channels_json(),
        "weights": paths.weights_dir() if need_weights else None,
        "out": out,
    }


class Summary:
    """Collects one dictionary per run and keeps a JSON file up to date after every run."""

    def __init__(self, path):
        self.path = path
        self.rows = []

    def record(self, row):
        self.rows.append(row)
        print("SUMMARY", json.dumps(row), flush=True)
        with open(self.path, "w") as f:
            json.dump(self.rows, f, indent=1)

    def run(self, tag, fn, **extra):
        """Call fn(), store its result together with the elapsed time; never let one failure stop the sweep."""
        t0 = time.time()
        try:
            result = fn()
            self.record(dict(tag=tag, minutes=(time.time() - t0) / 60, **extra, **result))
        except Exception:
            traceback.print_exc()
            self.record(dict(tag=tag, error=True, **extra))


def retrieval_fields(r, key="sub0"):
    """Pick the headline metrics out of a training.main() result."""
    e = r["test"][key]["eeg"]
    n = r["test"][key]["noise"]
    return dict(
        top1=e["top1_full"],
        top10=e["top10_full"],
        mrr=e["mrr_full"],
        pool=e["pool_size_full"],
        top1_pool100=e.get("top1_pool100"),
        rank_pct=e.get("rank_percentile_full"),
        cat=e.get("cat_acc"),
        ws_top1=e.get("ws_top1"),
        ws_chance_top1=e.get("ws_chance_top1"),
        xs_matched_top1=e.get("xs_matched_top1"),
        noise_top1_pool100=n.get("top1_pool100"),
        n_train=r.get("n_train"),
    )
