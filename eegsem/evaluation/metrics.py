"""Teacher-forcing-free evaluation: retrieval in sentence pools, zero-shot category, semantic similarity."""

import numpy as np


def retrieval_metrics(z_eeg, z_pool, target_idx, ks=(1, 5, 10), sub_pool=100, n_draws=20, rng=None):
    """z_eeg [N,d] (unit), z_pool [P,d] (unit), target_idx [N] index into pool.
    Full-pool top-k / MRR, and average over random sub-pools of size `sub_pool` containing the target.
    """
    rng = rng or np.random.default_rng(0)
    S = z_eeg @ z_pool.T  # [N,P]
    N, P = S.shape
    tgt = S[np.arange(N), target_idx]
    rank = (S > tgt[:, None]).sum(1) + 1  # 1 = best
    out = {f"top{k}_full": float((rank <= k).mean()) for k in ks}
    out["mrr_full"] = float((1.0 / rank).mean())
    out["pool_size_full"] = int(P)
    out["median_rank_full"] = float(np.median(rank))
    if P > sub_pool:
        accs = {k: [] for k in ks}
        mrrs = []
        for _ in range(n_draws):
            for i in range(N):
                others = rng.choice(P - 1, sub_pool - 1, replace=False)
                others = others + (others >= target_idx[i])  # skip target
                r = 1 + (S[i, others] > tgt[i]).sum()
                for k in ks:
                    accs[k].append(r <= k)
                mrrs.append(1.0 / r)
        for k in ks:
            out[f"top{k}_pool{sub_pool}"] = float(np.mean(accs[k]))
        out[f"mrr_pool{sub_pool}"] = float(np.mean(mrrs))
        for k in ks:
            out[f"chance_top{k}_pool{sub_pool}"] = k / sub_pool
    for k in ks:
        out[f"chance_top{k}_full"] = k / P
    return out


def zero_shot_category(z_eeg, cat_centroids, cats):
    """cat_centroids [K,d] from TEXT embeddings only; cats [N] true category (-1 = unknown, skipped)."""
    m = cats >= 0
    if m.sum() == 0:
        return {}
    pred = (z_eeg[m] @ cat_centroids.T).argmax(1)
    return {
        "cat_acc": float((pred == cats[m]).mean()),
        "cat_n": int(m.sum()),
        "cat_chance": float(1.0 / cat_centroids.shape[0]),
    }


def semantic_similarity(z_eeg, z_pool, target_idx):
    """cos(text emb of retrieved top-1, text emb of true sentence) — 1.0 means exact hit."""
    top1 = (z_eeg @ z_pool.T).argmax(1)
    return {"sem_sim_top1": float((z_pool[top1] * z_pool[target_idx]).sum(1).mean())}


def within_run_metrics(S, target, ses, min_pool=5):
    """Retrieval restricted to the test sentences of the trial's own run.

    S [N, U]: scores of each trial against the U unique test sentences; target [N]: column of the true
    sentence; ses [N]: run id. Pools with fewer than ``min_pool`` candidates are skipped, a tie never counts
    against the target, and chance is the per-trial 1/pool averaged over the retained trials.
    """
    ranks = []
    for i in range(len(target)):
        cand = np.unique(target[ses == ses[i]])
        if len(cand) < min_pool:
            continue
        r = int((S[i, cand] > S[i, target[i]]).sum()) + 1
        ranks.append((r, len(cand)))
    if not ranks:
        return {}
    rk = np.array([r for r, _ in ranks], float)
    n = np.array([m for _, m in ranks], float)
    return {
        "ws_top1": float((rk <= 1).mean()),
        "ws_top5": float((rk <= 5).mean()),
        "ws_mrr": float((1 / rk).mean()),
        "ws_chance_top1": float((1 / n).mean()),
        "ws_chance_top5": float(np.minimum(5 / n, 1).mean()),
        "ws_mean_pool": float(n.mean()),
        "ws_n": int(len(rk)),
    }
