"""Direct tests of the shortcut mechanism (E11): (a) classify the recording run and regress the within-run position from EEG
with an EEGNet head trained from scratch; (b) linear probes for run / position on the contrastive decoder's test embeddings;
(c) per-trial relation between run-probe confidence and retrieval success."""

import argparse, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from ..training import Pool, Normalizer, channel_stats, augment, log
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


from ..models.encoders import EEGNetEncoder
from ..data.text_bank import TextBank


class RunHead(nn.Module):
    def __init__(self, C, T, n_runs):
        super().__init__()
        self.enc = EEGNetEncoder(C, T, out_dim=128)
        self.cls = nn.Linear(128, n_runs)
        self.pos = nn.Linear(128, 1)

    def forward(self, x):
        h = F.gelu(self.enc(x))
        return self.cls(h), self.pos(h).squeeze(-1)


def run_probe(cache_dir, bank_dir, out, seed=0, epochs=20, bs=128, lr=1e-3, device=None, emb=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(seed)
    bank = TextBank(bank_dir, "labse")
    pool = Pool([cache_dir], "imagine", bank)
    runs = np.unique(pool.sess)
    rid = {r: i for i, r in enumerate(runs)}
    y_run = np.array([rid[r] for r in pool.sess])
    # position within run, normalised to [0,1]
    pos = np.zeros(len(pool.sess), np.float32)
    for r in runs:
        m = np.where(pool.sess == r)[0]
        pos[m] = np.linspace(0, 1, len(m))
    tr, va, te = pool.idx("train", [0]), pool.idx("val", [0]), pool.idx("test", [0])
    stats = {0: channel_stats(pool.X, tr)}
    norm = Normalizer(stats, device)
    model = RunHead(pool.C, pool.T, len(runs)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * (len(tr) // bs + 1)
    )

    def batches(idx, shuffle):
        idx = np.random.permutation(idx) if shuffle else idx
        for i in range(0, len(idx), bs):
            b = idx[i : i + bs]
            x = torch.from_numpy(np.asarray(pool.X[b])).to(device).float()
            s = torch.from_numpy(pool.sub[b]).to(device)
            yield b, norm(x, s)

    @torch.no_grad()
    def predict(idx):
        model.eval()
        L, P = [], []
        for b, x in batches(idx, False):
            l, p = model(x)
            L.append(l.float().cpu().numpy())
            P.append(p.float().cpu().numpy())
        return np.concatenate(L), np.concatenate(P)

    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for b, x in batches(tr, True):
            x = augment(x, 250)
            l, p = model(x)
            loss = F.cross_entropy(l, torch.from_numpy(y_run[b]).to(device)) + F.mse_loss(
                p, torch.from_numpy(pos[b]).to(device)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        L, P = predict(va)
        acc = float((L.argmax(1) == y_run[va]).mean())
        if acc > best:
            best, best_state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"probe ep {ep+1}/{epochs} val run-acc {acc:.3f}")
    model.load_state_dict(best_state)
    L, P = predict(te)
    from scipy.stats import spearmanr

    res = {
        "n_runs": int(len(runs)),
        "run_acc": float((L.argmax(1) == y_run[te]).mean()),
        "run_chance": float(1 / len(runs)),
        "run_top5": float(np.mean([y_run[te][i] in np.argsort(-L[i])[:5] for i in range(len(te))])),
        "pos_r2": float(1 - ((P - pos[te]) ** 2).sum() / ((pos[te] - pos[te].mean()) ** 2).sum()),
        "pos_spearman": float(spearmanr(P, pos[te])[0]),
        "pos_within_run_spearman": float(
            np.nanmean(
                [
                    spearmanr(P[pool.sess[te] == r], pos[te][pool.sess[te] == r])[0]
                    for r in runs
                    if (pool.sess[te] == r).sum() >= 5
                ]
            )
        ),
    }
    # (b) linear probes on the decoder's test embeddings (5-fold CV within the test set), if provided
    if emb and os.path.exists(emb) and len(np.unique(np.load(emb)["run"])) >= 2:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import cross_val_predict, KFold

        d = np.load(emb)
        Z = d["Z"]
        r_emb = np.array([rid[r] for r in d["run"]])
        rank = d["rank"]
        posz = np.zeros(len(Z), np.float32)
        for r in np.unique(d["run"]):
            m = np.where(d["run"] == r)[0]
            posz[m] = np.linspace(0, 1, len(m))
        kf = KFold(5, shuffle=True, random_state=0)
        pr = cross_val_predict(
            LogisticRegression(max_iter=2000, C=1.0), Z, r_emb, cv=kf, method="predict_proba"
        )
        res["emb_run_acc"] = float((pr.argmax(1) == r_emb).mean())
        res["emb_run_chance"] = float(1 / len(runs))
        pp = cross_val_predict(Ridge(alpha=1.0), Z, posz, cv=kf)
        res["emb_pos_spearman"] = float(spearmanr(pp, posz)[0])
        conf = pr[np.arange(len(Z)), r_emb]
        hit = (rank <= 10).astype(float)
        res["run_conf_vs_top10_spearman"] = (
            float(spearmanr(conf, hit)[0]) if hit.std() > 0 else float("nan")
        )
        res["top10_when_run_correct"] = (
            float(hit[pr.argmax(1) == r_emb].mean())
            if (pr.argmax(1) == r_emb).any()
            else float("nan")
        )
        res["top10_when_run_wrong"] = (
            float(hit[pr.argmax(1) != r_emb].mean())
            if (pr.argmax(1) != r_emb).any()
            else float("nan")
        )
    os.makedirs(out, exist_ok=True)
    name = f"probe_{os.path.basename(cache_dir)}_s{seed}"
    json.dump(res, open(os.path.join(out, name + ".json"), "w"), indent=1)
    log("PROBE " + json.dumps(res))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--bank_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--emb", default=None)
    a = ap.parse_args(argv)
    return run_probe(a.cache_dir, a.bank_dir, a.out, a.seed, a.epochs, emb=a.emb)


if __name__ == "__main__":
    main()
