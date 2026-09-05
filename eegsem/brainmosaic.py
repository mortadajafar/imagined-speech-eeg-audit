"""Re-implementation harness for BrainMosaic (Li et al., ICLR 2026) on OUR sentence-disjoint split, using the authors' public code
(github.com/Erikaqvq/BrainMosaic_ICLR26) with open text assets: jieba segmentation (semantic units = content words) and
Qwen3-Embedding sentence/token embeddings (truncated to 256-d, as in their config). Metrics: their UMA/MUS on the test split,
plus OUR retrieval metrics computed from their slot-0 sentence embedding in the Qwen3 space."""

import argparse, json, os, re, subprocess, sys, time
import numpy as np, torch
from .data.splits import split_of

STOP = set(
    "的 了 吗 呢 吧 啊 呀 哦 嗯 地 得 着 过 ， 。 ？ ！ 、 ； ： “ ” ‘ ’ （ ） , . ? ! ; : ( ) 一下 一些".split()
)
PUNCT = re.compile(r"^[\W_]+$")


def segment(sentences):
    import jieba

    out = {}
    for s in sentences:
        toks = [
            t.strip()
            for t in jieba.cut(s)
            if t.strip() and t.strip() not in STOP and not PUNCT.match(t.strip())
        ]
        out[s] = toks if toks else [s]
    return out


def build_text_assets(
    sentences,
    out_dir,
    model_name="Qwen/Qwen3-Embedding-0.6B",
    truncate_dim=256,
    device="cuda",
    fake=False,
    pooling="mean",
    center=False,
):
    """pooling='mean' reproduces the authors' public script; pooling='last' is the pooling Qwen3-Embedding was trained with.
    center=True subtracts the mean embedding (over sentences+tokens) before normalisation to remove the common anisotropic component.
    """
    """Writes sentence_embeddings.pt, word_embeddings.pt (+ token bank via their emb_preprocessing) and segmentation.json."""
    os.makedirs(out_dir, exist_ok=True)
    seg = segment(sentences)
    vocab = sorted({t for toks in seg.values() for t in toks})
    json.dump(
        [{"sentence": s, "tokens": toks} for s, toks in seg.items()],
        open(os.path.join(out_dir, "segmentation.json"), "w", encoding="utf-8"),
        ensure_ascii=False,
    )
    if fake:
        g = torch.Generator().manual_seed(0)
        semb = torch.nn.functional.normalize(
            torch.randn(len(sentences), truncate_dim, generator=g), dim=-1
        )
        wemb = torch.nn.functional.normalize(
            torch.randn(len(vocab), truncate_dim, generator=g), dim=-1
        )
    else:
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval().to(device)

        def enc(texts, bs=64):
            outs = []
            for i in range(0, len(texts), bs):
                b = tok(
                    texts[i : i + bs],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    h = mdl(**b).last_hidden_state
                    m = b["attention_mask"].unsqueeze(-1).float()
                    if pooling == "last":
                        if tok.padding_side == "left":
                            pooled = h[:, -1, :]
                        else:
                            idx = b["attention_mask"].sum(1) - 1
                            pooled = h[torch.arange(h.shape[0], device=h.device), idx]
                    else:
                        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
                outs.append(pooled.float().cpu())
            return torch.cat(outs)

        semb_raw, wemb_raw = enc(list(sentences)), enc(vocab)
        if center:
            mu = torch.cat([semb_raw, wemb_raw]).mean(0, keepdim=True)
            semb_raw, wemb_raw = semb_raw - mu, wemb_raw - mu
        semb = torch.nn.functional.normalize(semb_raw[:, :truncate_dim], dim=-1)
        wemb = torch.nn.functional.normalize(wemb_raw[:, :truncate_dim], dim=-1)
        rs = (wemb[:500] @ wemb[500:1000].T).mean().item() if len(vocab) > 1000 else float("nan")
        print(f"text assets: pooling={pooling} center={center}; mean random token cosine {rs:.3f}")
    torch.save(
        {"sentences": list(sentences), "embeddings": semb},
        os.path.join(out_dir, "sentence_embeddings.pt"),
    )
    torch.save({"keys": vocab, "embeddings": wemb}, os.path.join(out_dir, "word_embeddings.pt"))
    print(f"text assets: {len(sentences)} sentences, {len(vocab)} tokens, dim {semb.shape[1]}")
    return seg, vocab


def build_token_bank(repo, assets_dir, bank_dir, sim_threshold=0.78, truncate_dim=256):
    cfg = {
        "input": {"word_embeddings_pt": os.path.join(assets_dir, "word_embeddings.pt")},
        "output": {"token_bank_dir": bank_dir},
        "truncate_dim": truncate_dim,
        "cluster_sim_threshold": sim_threshold,
    }
    p = os.path.join(assets_dir, "token_bank.json")
    json.dump(cfg, open(p, "w"))
    subprocess.run(
        [sys.executable, os.path.join(repo, "labels", "emb_preprocessing.py"), "--config", p],
        check=True,
        cwd=repo,
    )


def convert_cache(cache_dir, seg, out_dir, phase="imagine", scale=1.0):
    """Our cache -> their unified records (train/val from hash split; test written separately)."""
    from .data.chisco import SubjectCache

    c = SubjectCache(cache_dir, phase)
    os.makedirs(out_dir, exist_ok=True)
    recs = {"train": [], "val": [], "test": []}
    for i, t in enumerate(c.texts):
        recs[split_of(t)].append(
            {
                "eeg": torch.from_numpy(np.asarray(c.X[i]).astype(np.float32) * scale),
                "sentence": t,
                "words": seg.get(t, [t]),
                "cat": int(c.cats[i]),
                "ses": int(c.meta[i].get("ses", c.meta[i].get("run", 0))),
            }
        )
    for k, v in recs.items():
        torch.save(v, os.path.join(out_dir, f"{k}.pt"))
    print({k: len(v) for k, v in recs.items()})
    return {k: len(v) for k, v in recs.items()}


def train_subject(
    repo,
    data_dir,
    bank_dir,
    sent_emb,
    seg_json,
    out_dir,
    epochs=30,
    batch_size=32,
    device="cuda",
    extra=None,
):
    base = json.load(open(os.path.join(repo, "configs", "train.example.json")))
    base["data"].update(
        {
            "in_channels": 122,
            "eeg_scale": 1.0,
            "token_path": bank_dir,
            "sent_emb_path": sent_emb,
            "segmentation_path": seg_json,
            "eeg_path": data_dir,
        }
    )
    base["runtime"].update(
        {
            "output_dir": out_dir,
            "device": device,
            "batch_size": batch_size,
            "num_workers": 2,
            "eval": False,
        }
    )
    base["train"].update({"epochs": epochs, "lr_drop": max(1, epochs - 5)})
    if extra:
        base.update(extra)
    cfg = os.path.join(out_dir, "train.json")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(base, open(cfg, "w"), indent=1)
    t0 = time.time()
    subprocess.run([sys.executable, "main.py", "--config", cfg], check=True, cwd=repo)
    print(f"train done {(time.time()-t0)/60:.1f} min")
    return cfg


@torch.no_grad()
def eval_retrieval(repo, cfg_path, ckpt, test_pt, sent_emb_pt, cats_by_sentence, device="cuda"):
    """Load their best checkpoint, embed test trials (slot 0), retrieval + zero-shot category in the Qwen3 space."""
    sys.path.insert(0, repo)
    os.chdir(repo)
    import importlib

    sys.argv = ["main.py", "--config", cfg_path]
    from config_args import parse_args

    args = parse_args()
    from datasets import load_token_bank
    from models import build_model

    token_bank = load_token_bank(
        args.token_path, dim=args.hidden_dim, normalize=bool(args.normalize_token_emb)
    )
    model, criterion, post = build_model(args, token_bank)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model"])
    model.to(device).eval()
    recs = torch.load(test_pt)
    se = torch.load(sent_emb_pt)
    s2i = {s: i for i, s in enumerate(se["sentences"])}
    E = torch.nn.functional.normalize(se["embeddings"].float(), dim=-1).numpy()
    Z, sid, cats, ses = [], [], [], []
    for i in range(0, len(recs), 64):
        b = recs[i : i + 64]
        x = torch.stack([r["eeg"] for r in b]).to(device)
        from util.misc import NestedTensor

        out = model(NestedTensor(x, None))
        z = (
            torch.nn.functional.normalize(out["pred_embeddings"][:, 0, :].float(), dim=-1)
            .cpu()
            .numpy()
        )
        Z.append(z)
        sid += [s2i[r["sentence"]] for r in b]
        cats += [r.get("cat", -1) for r in b]
        ses += [r.get("ses", 0) for r in b]
    Z = np.concatenate(Z)
    sid = np.array(sid)
    cats = np.array(cats)
    ses = np.array(ses)
    from .evaluation.metrics import retrieval_metrics, zero_shot_category, semantic_similarity

    uniq, target = np.unique(sid, return_inverse=True)
    zp = E[uniq]
    res = retrieval_metrics(Z, zp, target)
    res.update(semantic_similarity(Z, zp, target))
    # within-session and cross-session (same-session candidates excluded) retrieval — the confound-controlled metrics
    S = Z @ zp.T
    ws, xs = [], []
    for i in range(len(Z)):
        same = np.isin(uniq, sid[ses == ses[i]])
        cand = np.where(same)[0]
        if len(cand) >= 5:
            sc = S[i][cand]
            ws.append(((sc > S[i][target[i]]).sum() + 1, len(cand)))
        other = ~same
        other[target[i]] = True
        sc = S[i][other]
        xs.append(((sc > S[i][target[i]]).sum() + 1, int(other.sum())))
    if ws:
        rk = np.array([r for r, _ in ws])
        n = np.array([n for _, n in ws], float)
        res.update(
            ws_top1=float((rk <= 1).mean()),
            ws_top5=float((rk <= 5).mean()),
            ws_mrr=float((1 / rk).mean()),
            ws_chance_top1=float((1 / n).mean()),
            ws_chance_top5=float(np.minimum(5 / n, 1).mean()),
        )
    rk = np.array([r for r, _ in xs])
    n = np.array([n for _, n in xs], float)
    res.update(
        xs_top10=float((rk <= 10).mean()),
        xs_chance_top10=float((10 / n).mean()),
        xs_mrr=float((1 / rk).mean()),
    )
    np.savez_compressed(
        os.path.join(os.path.dirname(ckpt), "test_embeddings.npz"), Z=Z, sid=sid, ses=ses, cats=cats
    )
    K = int(cats.max()) + 1
    cent = np.zeros((K, E.shape[1]), np.float32)
    for k in range(K):
        ss = [s2i[s] for s, c in cats_by_sentence.items() if c == k and s in s2i]
        if ss:
            cent[k] = E[ss].mean(0)
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8
    res.update(zero_shot_category(Z, cent, cats))
    noise = retrieval_metrics(
        np.random.default_rng(0).standard_normal(Z.shape).astype(np.float32) / np.sqrt(Z.shape[1]),
        zp,
        target,
    )
    res["noise_top1_pool100"] = noise.get("top1_pool100")
    return res
