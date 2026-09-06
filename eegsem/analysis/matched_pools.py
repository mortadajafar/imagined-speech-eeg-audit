"""Same-run vs cross-run retrieval with distractors matched on semantic similarity to the target.

Uses the test-set EEG embeddings saved by the probe experiment (E11) and the frozen LaBSE bank, so it
runs offline. For every test trial the same-run pool is the set of test sentences of that run; the
size-matched cross-run pool draws the same number of sentences from other runs at random; the
similarity-matched cross-run pool picks, for each same-run distractor, the other-run test sentence whose
cosine similarity to the target is closest (without replacement).
"""

import glob
import json
import sys

import numpy as np


def evaluate(emb_file, bank_emb, n_rep=20, seed=0):
    z = np.load(emb_file)
    Z, sid, run = z["Z"], z["sid"], z["run"]
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    T = bank_emb / np.linalg.norm(bank_emb, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    scores = Z @ T.T  # trial x bank
    sim = T @ T.T  # sentence x sentence semantic similarity
    test_sids = np.unique(sid)
    out = {"same": [], "size": [], "sem": [], "sim_same": [], "sim_size": [], "sim_sem": []}
    for i in range(len(sid)):
        t = sid[i]
        same = np.unique(sid[(run == run[i]) & (sid != t)])
        if len(same) == 0:
            continue
        other = np.setdiff1d(test_sids[np.isin(test_sids, sid[run != run[i]])], same)
        other = other[other != t]
        out["same"].append(scores[i, t] > scores[i, same].max())
        out["sim_same"].append(sim[t, same].mean())
        acc_size, acc_sem, s_size, s_sem = [], [], [], []
        for _ in range(n_rep):
            pick = rng.choice(other, size=len(same), replace=False)
            acc_size.append(scores[i, t] > scores[i, pick].max())
            s_size.append(sim[t, pick].mean())
        # similarity-matched: greedy nearest similarity to each same-run distractor
        cand = other.copy()
        cand_sim = sim[t, cand]
        chosen = []
        for s_ in sim[t, same]:
            j = np.argmin(np.abs(cand_sim - s_))
            chosen.append(cand[j])
            cand_sim[j] = np.inf
        chosen = np.array(chosen)
        out["sem"].append(scores[i, t] > scores[i, chosen].max())
        out["sim_sem"].append(sim[t, chosen].mean())
        out["size"].append(np.mean(acc_size))
        out["sim_size"].append(np.mean(s_size))
    return {k: float(np.mean(v)) for k, v in out.items()} | {
        "n": len(out["same"]),
        "pool": float(np.mean([len(np.unique(sid[run == r])) for r in np.unique(run)])),
    }


def main(logs="kaggle/logs/e11", out="notes/matched_pools.json"):
    bank = np.load(f"{logs}/bank/emb_labse.npy")
    res = {}
    for f in sorted(glob.glob(f"{logs}/results/e11_aligner_sub0?_k0_sub0_emb.npz")):
        s = f.split("aligner_")[1][:5]
        res[s] = evaluate(f, bank)
        r = res[s]
        print(
            f"{s}: n={r['n']} pool~{r['pool']:.1f} chance~{100/r['pool']:.1f}% | top-1 same-run {100*r['same']:.1f}  size-matched other-run {100*r['size']:.1f}  similarity-matched other-run {100*r['sem']:.1f} | mean sim to target: same {r['sim_same']:.3f} size {r['sim_size']:.3f} sem {r['sim_sem']:.3f}"
        )
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
