"""Linear baseline: z-scored EEG -> 25 Hz average pooling -> PCA -> RidgeCV to sentence embedding. Same metrics."""

import argparse, json, os, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from .training import Pool, channel_stats, category_centroids, log
from .data.text_bank import TextBank
from .evaluation.metrics import retrieval_metrics, zero_shot_category, semantic_similarity


def feats(X, mu, sd, pool_len=10):
    x = (X.astype(np.float32) - mu[None, :, None]) / sd[None, :, None]
    x = np.clip(x, -10, 10)
    T = (x.shape[2] // pool_len) * pool_len
    x = x[:, :, :T].reshape(x.shape[0], x.shape[1], T // pool_len, pool_len).mean(-1)
    return x.reshape(x.shape[0], -1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dirs", nargs="+", required=True)
    ap.add_argument("--bank_dir", required=True)
    ap.add_argument("--emb", default="labse")
    ap.add_argument("--phase", default="imagine")
    ap.add_argument("--subject", type=int, default=0)
    ap.add_argument("--n_pca", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    bank = TextBank(args.bank_dir, args.emb)
    pool = Pool(args.cache_dirs, args.phase, bank, subject_ids=[args.subject])
    tr, va, te = pool.idx("train"), pool.idx("val"), pool.idx("test")
    mu, sd = channel_stats(pool.X, tr)
    t0 = time.time()
    Ftr = feats(pool.X[tr], mu, sd)
    Fte = feats(pool.X[te], mu, sd)
    pca = PCA(min(args.n_pca, len(Ftr) - 1, Ftr.shape[1]), random_state=args.seed).fit(Ftr)
    Ptr, Pte = pca.transform(Ftr), pca.transform(Fte)
    Y = bank.emb[pool.sid[tr]]
    ridge = RidgeCV(alphas=np.logspace(0, 6, 13)).fit(Ptr, Y)
    Z = ridge.predict(Pte)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    uniq, target = np.unique(pool.sid[te], return_inverse=True)
    res = retrieval_metrics(Z, bank.emb[uniq], target)
    res.update(semantic_similarity(Z, bank.emb[uniq], target))
    cent = category_centroids(bank, pool)
    if cent is not None:
        res.update(zero_shot_category(Z, cent, pool.cat[te]))
    # noise control: same ridge applied to Gaussian features
    Zn = ridge.predict(
        pca.transform(np.random.default_rng(0).standard_normal(Fte.shape).astype(np.float32))
    )
    Zn /= np.linalg.norm(Zn, axis=1, keepdims=True) + 1e-8
    noise = retrieval_metrics(Zn, bank.emb[uniq], target)
    out = {
        "args": vars(args),
        "alpha": float(ridge.alpha_),
        "n_train": int(len(tr)),
        "test": {f"sub{args.subject}": {"eeg": res, "noise": noise}},
        "time_s": time.time() - t0,
    }
    os.makedirs(args.out, exist_ok=True)
    name = args.tag or f"ridge_{args.phase}_sub{args.subject}_s{args.seed}"
    with open(os.path.join(args.out, f"{name}.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(
        f"RIDGE sub{args.subject}: top1 {res['top1_full']:.4f} top5 {res['top5_full']:.4f} top10 {res['top10_full']:.4f} mrr {res['mrr_full']:.4f} "
        f"pool {res['pool_size_full']} | pool100 top1 {res.get('top1_pool100', float('nan')):.4f} | cat {res.get('cat_acc', float('nan')):.4f} | noise top1 {noise['top1_full']:.4f} | alpha {ridge.alpha_:.3g}"
    )
    return out


if __name__ == "__main__":
    main()
