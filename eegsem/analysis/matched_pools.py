"""Same-run vs cross-run retrieval with distractors matched on semantic similarity to the target.

Uses the test-set EEG embeddings saved by the probe experiment (E11) and the frozen LaBSE bank, so it
runs offline. The rules follow ``eegsem.training.evaluate``: pools with fewer than five candidates are
skipped, a tie never counts against the target, and chance is the per-trial 1/pool averaged over trials.

For every test trial the same-run pool is the set of test sentences of that run; the size-matched
cross-run pool draws the same number of sentences from other runs at random (20 draws); the
similarity-matched cross-run pool picks, for each same-run distractor, the other-run test sentence whose
cosine similarity to the target is closest (greedy, without replacement).
"""

import glob
import json
import sys

import numpy as np

MIN_POOL = 5


def evaluate(emb_file, bank_emb, n_rep=20, seed=0):
    z = np.load(emb_file)
    Z, sid, run = z["Z"], z["sid"], z["run"]
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    T = bank_emb / np.linalg.norm(bank_emb, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    scores = Z @ T.T  # trial x bank
    sim = T @ T.T  # sentence x sentence semantic similarity
    test_sids = np.unique(sid)
    keys = ["same", "size", "sem", "sim_same", "sim_size", "sim_sem", "chance"]
    out = {k: [] for k in keys}
    for i in range(len(sid)):
        t = sid[i]
        same = np.unique(sid[(run == run[i]) & (sid != t)])
        if len(same) + 1 < MIN_POOL:
            continue
        other = np.setdiff1d(np.unique(sid[run != run[i]]), np.append(same, t))
        if len(other) < len(same):
            continue
        hit = lambda pool: float((scores[i, pool] > scores[i, t]).sum() == 0)
        out["same"].append(hit(same))
        out["sim_same"].append(sim[t, same].mean())
        out["chance"].append(1 / (len(same) + 1))
        picks = [rng.choice(other, size=len(same), replace=False) for _ in range(n_rep)]
        out["size"].append(np.mean([hit(p) for p in picks]))
        out["sim_size"].append(np.mean([sim[t, p].mean() for p in picks]))
        cand_sim = sim[t, other].copy()
        chosen = []
        for s_ in sim[t, same]:
            j = int(np.argmin(np.abs(cand_sim - s_)))
            chosen.append(other[j])
            cand_sim[j] = np.inf
        chosen = np.array(chosen)
        out["sem"].append(hit(chosen))
        out["sim_sem"].append(sim[t, chosen].mean())
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["n"] = len(out["same"])
    res["pool"] = float(np.mean([1 / c for c in out["chance"]]))
    return res


def main(logs="kaggle/logs/e11", out="notes/matched_pools.json"):
    bank = np.load(f"{logs}/bank/emb_labse.npy")
    res = {}
    for f in sorted(glob.glob(f"{logs}/results/e11_aligner_sub0?_k0_sub0_emb.npz")):
        s = f.split("aligner_")[1][:5]
        res[s] = r = evaluate(f, bank)
        print(
            f"{s}: n={r['n']} pool~{r['pool']:.1f} chance {100*r['chance']:.1f}% | top-1 same-run {100*r['same']:.1f}"
            f"  size-matched other-run {100*r['size']:.1f}  similarity-matched other-run {100*r['sem']:.1f}"
            f" | mean sim to target: same {r['sim_same']:.3f} size {r['sim_size']:.3f} sem {r['sim_sem']:.3f}"
        )
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
