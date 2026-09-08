"""Train / evaluate EEG->sentence-embedding aligners. Teacher-forcing-free evaluation only.
Settings: in-subject | cross-subject (LOSO) | few-shot calibration | phase transfer (read->imagine) | controls.
"""

import argparse, json, os, sys, time, math, copy
import numpy as np
import torch, torch.nn.functional as F
from .data.splits import split_of
from .data.chisco import SubjectCache
from .data.text_bank import TextBank
from .models.encoders import build_encoder, Aligner, info_nce
from .evaluation.metrics import retrieval_metrics, zero_shot_category, semantic_similarity


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- data
class Pool:
    """All trials of the requested subjects/phase in RAM (float16), with sentence ids into the text bank."""

    def __init__(self, cache_dirs, phase, bank, subject_ids=None):
        Xs, sids, subs, cats, splits, texts, sess = [], [], [], [], [], [], []
        for si, d in enumerate(cache_dirs):
            if subject_ids is not None and si not in subject_ids:
                continue
            c = SubjectCache(d, phase)
            missing = [t for t in c.texts if t not in bank.index]
            if missing:
                raise KeyError(
                    f"{len(missing)} sentences of {d} not in text bank, e.g. {missing[:3]}"
                )
            Xs.append(np.asarray(c.X))
            sids.append(bank.ids(c.texts))
            subs.append(np.full(len(c), si))
            cats.append(c.cats)
            splits += [split_of(t) for t in c.texts]
            texts += c.texts
            sess.append(np.array([m.get("ses", m.get("run", 0)) for m in c.meta]))
            log(f"loaded {d} {phase}: {c.X.shape}")
        self.X = np.concatenate(Xs)
        self.sid = np.concatenate(sids)
        self.sub = np.concatenate(subs)
        self.cat = np.concatenate(cats)
        self.split = np.array(splits)
        self.texts = texts
        self.sess = np.concatenate(sess)
        self.C, self.T = self.X.shape[1], self.X.shape[2]

    def idx(self, split, subjects=None):
        if getattr(
            self, "run_split", None
        ):  # leave-runs-out: partition by run instead of by sentence
            held = (
                np.concatenate([v for v in self.run_split.values()])
                if self.run_split
                else np.array([])
            )
            m = (
                np.isin(self.sess, self.run_split[split])
                if split in self.run_split
                else ~np.isin(self.sess, held)
            )
        else:
            m = self.split == split
        if subjects is not None:
            m &= np.isin(self.sub, subjects)
        return np.where(m)[0]


def channel_stats(X, idx):
    """per-channel mean/std over training trials (float32), std floor for flat channels."""
    x = X[idx].astype(np.float32)
    mu = x.mean(axis=(0, 2))
    sd = x.std(axis=(0, 2))
    sd = np.maximum(sd, 1e-3)
    return mu.astype(np.float32), sd.astype(np.float32)


def session_stats(pool):
    out = {}
    for s in np.unique(pool.sub):
        for r in np.unique(pool.sess[pool.sub == s]):
            idx = np.where((pool.sub == s) & (pool.sess == r))[0]
            out[(int(s), int(r))] = channel_stats(pool.X, idx)
    return out


class Normalizer:
    """mode 'zscore': per-subject per-channel z-score (task-specific encoders). mode 'fm': raw microvolts clipped at +-1000 uV
    (foundation-model wrappers apply their own /100 scaling)."""

    def __init__(self, stats_by_subject, device, mode="zscore", session_stats=None):
        self.mode = mode
        if session_stats:
            self.ses_mu = {
                k: torch.tensor(m, device=device)[None, :, None]
                for k, (m, _) in session_stats.items()
            }
            self.ses_sd = {
                k: torch.tensor(v, device=device)[None, :, None]
                for k, (_, v) in session_stats.items()
            }
        self.mu = {
            s: torch.tensor(m, device=device)[None, :, None]
            for s, (m, _) in stats_by_subject.items()
        }
        self.sd = {
            s: torch.tensor(v, device=device)[None, :, None]
            for s, (_, v) in stats_by_subject.items()
        }

    def __call__(self, x, sub, ses=None):
        if self.mode == "fm":
            return x.clamp(-1000, 1000)
        if self.mode == "session" and ses is not None:
            out = torch.empty_like(x)
            for s in torch.unique(sub).tolist():
                for r in torch.unique(ses[sub == s]).tolist():
                    m = (sub == s) & (ses == r)
                    key = (s, r)
                    mu, sd = self.ses_mu[key], self.ses_sd[key]
                    out[m] = (x[m] - mu) / sd
            return out.clamp_(-10, 10)
        out = torch.empty_like(x)
        for s in torch.unique(sub).tolist():
            m = sub == s
            out[m] = (x[m] - self.mu[s]) / self.sd[s]
        return out.clamp_(-10, 10)


def augment(x, fs, channel_drop=0.1):
    B, C, T = x.shape
    shift = int(0.05 * fs)
    k = torch.randint(-shift, shift + 1, (1,)).item()
    x = torch.roll(x, k, dims=2)
    x = x * (1 + 0.1 * torch.randn(B, 1, 1, device=x.device))
    if channel_drop > 0:
        x = x * (torch.rand(B, C, 1, device=x.device) > channel_drop).float()
    return x


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def embed_all(model, X, sub, normalizer, device, bs=256, transform=None, ses_all=None):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(np.asarray(X[i : i + bs])).to(device).float()
        s = torch.from_numpy(sub[i : i + bs]).to(device)
        x = normalizer(
            x, s, torch.from_numpy(ses_all[i : i + bs]).to(device) if ses_all is not None else None
        )
        if transform == "noise":
            x = torch.randn_like(x) * (100.0 if normalizer.mode == "fm" else 1.0)
        elif transform == "timeshuffle":
            x = x[:, :, torch.randperm(x.shape[2], device=device)]
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")
        ):
            z = model(x, s)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def evaluate(model, pool, idx, bank, normalizer, device, cat_centroids, transform=None, tag="test"):
    z = embed_all(
        model,
        pool.X[idx],
        pool.sub[idx],
        normalizer,
        device,
        transform=transform,
        ses_all=pool.sess[idx],
    )
    sid = pool.sid[idx]
    uniq, target = np.unique(sid, return_inverse=True)  # pool = unique sentences of this split
    z_pool = bank.emb[uniq]
    res = retrieval_metrics(z, z_pool, target)
    res.update(semantic_similarity(z, z_pool, target))
    # matched-size pools: same-run pool (size n_i) vs. random pools of the same size from OTHER runs (20 draws); rank percentile in full pool
    ses = pool.sess[idx]
    S_full = z @ z_pool.T
    rng_m = np.random.default_rng(0)
    mt = []
    for i in range(len(idx)):
        same = np.isin(uniq, sid[ses == ses[i]])
        n_same = int(same.sum())
        others = np.where(~same)[0]
        if n_same < 5 or len(others) < n_same - 1:
            continue
        mt.append(
            float(
                np.mean(
                    [
                        (
                            S_full[i][rng_m.choice(others, n_same - 1, replace=False)]
                            > S_full[i][target[i]]
                        ).sum()
                        == 0
                        for _ in range(20)
                    ]
                )
            )
        )
    if mt:
        res["xs_matched_top1"] = float(np.mean(mt))
    rank_full = (S_full > S_full[np.arange(len(idx)), target][:, None]).sum(1) + 1
    res["rank_percentile_full"] = float(((rank_full - 1) / max(len(uniq) - 1, 1)).mean())
    # session-controlled retrieval: candidates restricted to test sentences of the SAME session/run as the trial
    ranks = []
    for i in range(len(idx)):
        cand = np.where(np.isin(uniq, sid[ses == ses[i]]))[0]
        if len(cand) < 5:
            continue
        sc = z[i] @ z_pool[cand].T
        r = (sc > sc[cand == target[i]][0]).sum() + 1
        ranks.append((r, len(cand)))
    if ranks:
        rk = np.array([r for r, _ in ranks])
        n = np.array([n for _, n in ranks], float)
        res.update(
            {
                "ws_top1": float((rk <= 1).mean()),
                "ws_top5": float((rk <= 5).mean()),
                "ws_mrr": float((1 / rk).mean()),
                "ws_chance_top1": float((1 / n).mean()),
                "ws_chance_top5": float(np.minimum(5 / n, 1).mean()),
                "ws_mean_pool": float(n.mean()),
                "ws_n": int(len(rk)),
            }
        )
    # cross-session retrieval: same-session candidates (other than the target) EXCLUDED -> rank among sentences from OTHER runs + target
    S_all = z @ z_pool.T
    xr = []
    for i in range(len(idx)):
        other = ~np.isin(uniq, sid[ses == ses[i]])
        other[target[i]] = True
        sc = S_all[i][other]
        r = (sc > S_all[i][target[i]]).sum() + 1
        xr.append((r, other.sum()))
    rk = np.array([r for r, _ in xr])
    n = np.array([n for _, n in xr], float)
    res.update(
        {
            "xs_top1": float((rk <= 1).mean()),
            "xs_top10": float((rk <= 10).mean()),
            "xs_mrr": float((1 / rk).mean()),
            "xs_chance_top10": float((10 / n).mean()),
            "xs_mean_pool": float(n.mean()),
        }
    )
    if cat_centroids is not None:
        res.update(zero_shot_category(z, cat_centroids, pool.cat[idx]))
    res["n_trials"] = int(len(idx))
    res["n_unique_sentences"] = int(len(uniq))
    res["transform"] = transform or "none"
    S = z @ z_pool.T
    top5 = np.argsort(-S, axis=1)[:, :5]
    rank = (S > S[np.arange(len(idx)), target][:, None]).sum(1) + 1
    pred_cat = (
        (z @ cat_centroids.T).argmax(1) if cat_centroids is not None else np.full(len(idx), -1)
    )
    preds = [
        {
            "true": int(sid[i]),
            "top5": [int(uniq[j]) for j in top5[i]],
            "rank": int(rank[i]),
            "run": int(ses[i]),
            "cat": int(pool.cat[idx][i]),
            "pred_cat": int(pred_cat[i]),
        }
        for i in range(len(idx))
    ]
    return res, preds


def category_centroids(bank, pool):
    """Centroids of TEXT embeddings per category using every sentence with a known category (text side only)."""
    cats = pool.cat
    sid = pool.sid
    K = int(cats.max()) + 1 if (cats >= 0).any() else 0
    if K == 0:
        return None
    cent = np.zeros((K, bank.dim), np.float32)
    for k in range(K):
        s = np.unique(sid[cats == k])
        if len(s):
            cent[k] = bank.emb[s].mean(0)
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8
    return cent


# ----------------------------------------------------------------------------- train
def train_one(
    args,
    pool,
    train_idx,
    val_idx,
    bank,
    device,
    normalizer,
    n_subjects,
    init_state=None,
    train_adapter_only=False,
):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    enc = build_encoder(
        args.encoder,
        pool.C,
        pool.T,
        bank.dim,
        ch_names=args.ch_names,
        weights_dir=args.weights_dir,
        fs=args.fs,
    )
    model = Aligner(enc, n_subjects=n_subjects, C=pool.C, use_adapter=not args.no_adapter).to(
        device
    )
    if init_state is not None:
        model.load_state_dict(init_state, strict=True)
    named = [
        (n, p)
        for n, p in model.named_parameters()
        if (n.startswith("adapter") or not train_adapter_only)
    ]
    fm_backbone = [p for n, p in named if n.startswith("encoder.net.")]
    rest = [p for n, p in named if not n.startswith("encoder.net.")]
    groups = [{"params": rest, "lr": args.lr}] + (
        [{"params": fm_backbone, "lr": args.lr * args.fm_lr_mult}] if fm_backbone else []
    )
    params = [p for _, p in named]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, math.ceil(len(train_idx) / args.bs))
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))),
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    emb_t = torch.from_numpy(bank.emb).to(device)
    sid_all = pool.sid.copy()
    if args.permute:  # control: break EEG<->sentence pairing in TRAIN only
        rng = np.random.default_rng(args.seed + 1000)
        sid_all[train_idx] = rng.permutation(sid_all[train_idx])
    if (
        args.permute_within_run
    ):  # control: break the pairing but keep every run's set of sentences (TRAIN only)
        rng = np.random.default_rng(args.seed + 2000)
        for r in np.unique(pool.sess[train_idx]):
            ii = train_idx[pool.sess[train_idx] == r]
            sid_all[ii] = rng.permutation(sid_all[ii])
    best, best_state, bad = -1, None, 0
    for ep in range(args.epochs):
        model.train()
        if train_adapter_only and getattr(args, "freeze_bn_in_calibration", False):
            for (
                m
            ) in (
                model.modules()
            ):  # adapter-only calibration: keep the backbone's BatchNorm statistics fixed
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        perm = np.random.permutation(train_idx)
        tl, nb = 0.0, 0
        for i in range(0, len(perm), args.bs):
            b = perm[i : i + args.bs]
            if len(b) < 2:
                continue
            x = torch.from_numpy(np.asarray(pool.X[b])).to(device).float()
            s = torch.from_numpy(pool.sub[b]).to(device)
            x = augment(
                normalizer(x, s, torch.from_numpy(pool.sess[b]).to(device)),
                args.fs,
                channel_drop=0.0 if normalizer.mode == "fm" else 0.1,
            )
            sid = torch.from_numpy(sid_all[b]).to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")
            ):
                z = model(x, s)
                loss = info_nce(z.float(), emb_t[sid], sid, model.logit_scale)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item()
            nb += 1
        vres, _ = evaluate(model, pool, val_idx, bank, normalizer, device, None, tag="val")
        score = vres["mrr_pool100"] if "mrr_pool100" in vres else vres["mrr_full"]
        log(
            f"ep {ep+1}/{args.epochs} loss {tl/max(nb,1):.4f} | val top1 {vres['top1_full']:.4f} top5 {vres['top5_full']:.4f} "
            f"mrr {vres['mrr_full']:.4f} (pool {vres['pool_size_full']}) tau {1/model.logit_scale.exp().item():.3f}"
        )
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                log("early stop")
                break
    model.load_state_dict(best_state)
    return model, best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache_dirs",
        nargs="+",
        required=True,
        help="one dir per subject (order defines subject index)",
    )
    ap.add_argument("--bank_dir", required=True)
    ap.add_argument("--emb", default="labse")
    ap.add_argument("--phase", default="imagine")
    ap.add_argument("--test_phase", default=None)
    ap.add_argument(
        "--pretrain_phase", default=None, help="stage-1 phase before fine-tuning on --phase"
    )
    ap.add_argument("--encoder", default="eegnet")
    ap.add_argument("--no_adapter", action="store_true")
    ap.add_argument("--train_subjects", nargs="*", type=int, default=None)
    ap.add_argument("--test_subjects", nargs="*", type=int, default=None)
    ap.add_argument(
        "--few_shot",
        type=int,
        default=0,
        help="calibration trials from test subject's TRAIN sentences (adapter only)",
    )
    ap.add_argument(
        "--few_shot_list",
        nargs="*",
        type=int,
        default=None,
        help="evaluate several k from ONE base model (cross-subject only)",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--fs", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--permute_within_run", action="store_true")
    ap.add_argument(
        "--freeze_bn_in_calibration",
        action="store_true",
        help="keep BatchNorm statistics fixed while only the adapter is trained",
    )
    ap.add_argument(
        "--train_frac", type=float, default=1.0, help="data-scaling: fraction of train trials"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--norm",
        default=None,
        choices=["zscore", "fm", "session"],
        help="default: fm for foundation models else zscore; 'session' = per-run z-scoring (drift removal)",
    )
    ap.add_argument("--weights_dir", default=None)
    ap.add_argument("--channels_json", default=None, help="json with 'eeg122' channel names")
    ap.add_argument(
        "--fm_lr_mult", type=float, default=0.1, help="LR multiplier for pretrained FM backbone"
    )
    ap.add_argument(
        "--save_model",
        action="store_true",
        help="save best model state + normaliser stats to <out>/<name>.pt",
    )
    ap.add_argument(
        "--test_runs",
        nargs="*",
        type=int,
        default=None,
        help="leave-runs-out: these run ids form the test set (all their trials)",
    )
    ap.add_argument(
        "--val_runs", nargs="*", type=int, default=None, help="leave-runs-out: validation runs"
    )
    ap.add_argument(
        "--save_embeddings",
        action="store_true",
        help="save test-trial embeddings, sentence ids, runs, ranks to <out>/<name>_emb.npz",
    )
    ap.add_argument(
        "--occlusion",
        action="store_true",
        help="temporal (500 ms) and channel-region occlusion analysis on test set",
    )
    args = ap.parse_args(argv)
    from .models.encoders import FM_NAMES

    if args.norm is None:
        args.norm = "fm" if args.encoder in FM_NAMES else "zscore"
    args.ch_names = None
    if args.channels_json:
        with open(args.channels_json) as f:
            args.ch_names = json.load(f)["eeg122"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device", device, "args", vars(args))
    bank = TextBank(args.bank_dir, args.emb)
    n_sub = len(args.cache_dirs)
    all_subjects = list(range(n_sub))
    train_subjects = args.train_subjects if args.train_subjects is not None else all_subjects
    test_subjects = args.test_subjects if args.test_subjects is not None else train_subjects
    test_phase = args.test_phase or args.phase
    cross_subject = set(test_subjects) - set(train_subjects)

    pool = Pool(args.cache_dirs, args.phase, bank)
    if args.test_runs is not None:
        pool.run_split = {"test": np.array(args.test_runs), "val": np.array(args.val_runs or [])}
        log(f"leave-runs-out: test runs {args.test_runs} val runs {args.val_runs}")
    pool_test = pool if test_phase == args.phase else Pool(args.cache_dirs, test_phase, bank)
    train_idx = pool.idx("train", train_subjects)
    val_idx = pool.idx("val", train_subjects)
    if args.train_frac < 1.0:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(train_idx, int(len(train_idx) * args.train_frac), replace=False)
    stats = {s: channel_stats(pool.X, pool.idx("train", [s])) for s in all_subjects}
    normalizer = Normalizer(
        stats,
        device,
        args.norm,
        session_stats=session_stats(pool) if args.norm == "session" else None,
    )
    log(
        f"train {len(train_idx)} val {len(val_idx)} trials | train_subjects {train_subjects} test_subjects {test_subjects} | cross_subject {sorted(cross_subject)}"
    )

    init_state = None
    if args.pretrain_phase:
        pre = Pool(args.cache_dirs, args.pretrain_phase, bank)
        pre_norm = Normalizer(
            {s: channel_stats(pre.X, pre.idx("train", [s])) for s in all_subjects},
            device,
            args.norm,
            session_stats=session_stats(pre) if args.norm == "session" else None,
        )
        a2 = copy.copy(args)
        a2.epochs = max(5, args.epochs)
        log(f"stage-1 pretraining on phase {args.pretrain_phase}")
        m_pre, _ = train_one(
            a2,
            pre,
            pre.idx("train", train_subjects),
            pre.idx("val", train_subjects),
            bank,
            device,
            pre_norm,
            n_sub,
        )
        init_state = m_pre.state_dict()
        # temporal length may differ between phases: drop shape-mismatched params
        enc_tmp = Aligner(
            build_encoder(args.encoder, pool.C, pool.T, bank.dim),
            n_sub,
            pool.C,
            not args.no_adapter,
        )
        tgt = enc_tmp.state_dict()
        init_state = {k: v for k, v in init_state.items() if k in tgt and v.shape == tgt[k].shape}
        missing = [k for k in tgt if k not in init_state]
        init_state = {**tgt, **init_state}
        log(f"transferred {len(init_state)-len(missing)} tensors, re-initialised {missing}")

    model, best_val = train_one(
        args, pool, train_idx, val_idx, bank, device, normalizer, n_sub, init_state=init_state
    )

    cent = category_centroids(bank, pool)
    test_norm = (
        normalizer
        if pool_test is pool
        else Normalizer(
            {s: channel_stats(pool_test.X, pool_test.idx("train", [s])) for s in all_subjects},
            device,
            args.norm,
            session_stats=session_stats(pool_test) if args.norm == "session" else None,
        )
    )
    results = {"args": vars(args), "best_val": best_val, "n_train": int(len(train_idx)), "test": {}}
    os.makedirs(args.out, exist_ok=True)
    name = (
        args.tag
        or f"{args.encoder}_{args.phase}->{test_phase}_tr{''.join(map(str,train_subjects))}_te{''.join(map(str,test_subjects))}_fs{args.few_shot}_s{args.seed}{'_perm' if args.permute else ''}{'_permrun' if args.permute_within_run else ''}"
    )
    ks = args.few_shot_list if (cross_subject and args.few_shot_list) else [args.few_shot]
    base_state = copy.deepcopy(model.state_dict())
    for k in ks:
        if (
            cross_subject and k > 0
        ):  # calibrate adapter of held-out subject on k train-sentence trials
            model.load_state_dict(base_state)
            rng = np.random.default_rng(args.seed)
            cal = np.concatenate(
                [
                    rng.choice(
                        pool.idx("train", [s]), min(k, len(pool.idx("train", [s]))), replace=False
                    )
                    for s in cross_subject
                ]
            )
            a3 = copy.copy(args)
            a3.epochs = 20
            a3.lr = 1e-3
            a3.patience = 5
            log(f"few-shot calibration k={k}: {len(cal)} trials (adapter only)")
            model, _ = train_one(
                a3,
                pool,
                cal,
                pool.idx("val", list(cross_subject)),
                bank,
                device,
                normalizer,
                n_sub,
                init_state=base_state,
                train_adapter_only=True,
            )
        for s in test_subjects:
            idx = pool_test.idx("test", [s])
            if len(idx) == 0:
                continue
            r = {}
            for tf in (None, "noise", "timeshuffle"):
                res, preds = evaluate(
                    model, pool_test, idx, bank, test_norm, device, cent, transform=tf
                )
                r[tf or "eeg"] = res
                if tf is None:
                    with open(os.path.join(args.out, f"{name}_k{k}_sub{s}_preds.json"), "w") as f:
                        json.dump(preds, f)
                    if args.save_embeddings:
                        zz = embed_all(
                            model,
                            pool_test.X[idx],
                            pool_test.sub[idx],
                            test_norm,
                            device,
                            ses_all=pool_test.sess[idx],
                        )
                        np.savez_compressed(
                            os.path.join(args.out, f"{name}_k{k}_sub{s}_emb.npz"),
                            Z=zz,
                            sid=pool_test.sid[idx],
                            run=pool_test.sess[idx],
                            cat=pool_test.cat[idx],
                            rank=np.array([p["rank"] for p in preds]),
                        )
            key = f"sub{s}" if len(ks) == 1 else f"sub{s}_k{k}"
            results["test"][key] = r
            log(
                f"TEST {key}: top1 {r['eeg']['top1_full']:.4f} top5 {r['eeg']['top5_full']:.4f} top10 {r['eeg']['top10_full']:.4f} "
                f"mrr {r['eeg']['mrr_full']:.4f} pool {r['eeg']['pool_size_full']} | pool100 top1 {r['eeg'].get('top1_pool100', float('nan')):.4f} "
                f"| cat {r['eeg'].get('cat_acc', float('nan')):.4f} | noise top1 {r['noise']['top1_full']:.4f} pool100 {r['noise'].get('top1_pool100', float('nan')):.4f}"
            )
    if args.save_model:
        torch.save(
            {"state": base_state, "stats": stats, "args": vars(args), "ch_names": args.ch_names},
            os.path.join(args.out, f"{name}.pt"),
        )
    if args.occlusion:
        from .analysis.occlusion import occlusion_analysis

        model.load_state_dict(base_state)
        results["occlusion"] = {
            f"sub{s}": occlusion_analysis(
                model,
                pool_test,
                pool_test.idx("test", [s]),
                bank,
                test_norm,
                device,
                args.fs,
                args.ch_names,
            )
            for s in test_subjects
            if len(pool_test.idx("test", [s]))
        }
    with open(os.path.join(args.out, f"{name}.json"), "w") as f:
        json.dump(results, f, indent=1)
    log("saved", name)
    return results


if __name__ == "__main__":
    main()
