"""Post-hoc analyses on a trained aligner: temporal occlusion (which 500 ms windows carry semantic information) and
channel-region occlusion (which scalp regions). Occlusion = replace the window/region with Gaussian noise of matched scale
and measure the drop in retrieval MRR / pool-100 top-1 on the test set."""

import numpy as np, torch
from ..evaluation.metrics import retrieval_metrics

REGIONS = {  # by 10-5 name prefix on the Chisco 122-channel set
    "frontal": lambda c: c.upper().startswith(("FP", "AF", "F"))
    and not c.upper().startswith(("FC", "FT", "FCC", "FFC", "FFT", "FTT")),
    "fronto-central": lambda c: c.upper().startswith(("FC", "FT", "FCC", "FFC", "FFT", "FTT")),
    "central": lambda c: c.upper().startswith(("C", "CC"))
    and not c.upper().startswith(("CP", "CPP", "CCP")),
    "centro-parietal": lambda c: c.upper().startswith(("CP", "CPP", "CCP", "TP", "TPP", "TTP")),
    "temporal": lambda c: c.upper().startswith("T")
    and not c.upper().startswith(("TP", "TPP", "TTP")),
    "parietal": lambda c: c.upper().startswith(("P", "PP"))
    and not c.upper().startswith(("PO", "POO", "PPO")),
    "parieto-occipital": lambda c: c.upper().startswith(("PO", "POO", "PPO", "O", "I")),
}


@torch.no_grad()
def _embed(model, X, sub, normalizer, device, mask_fn=None, bs=256, ses=None):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(np.asarray(X[i : i + bs])).to(device).float()
        s = torch.from_numpy(sub[i : i + bs]).to(device)
        x = normalizer(
            x, s, torch.from_numpy(ses[i : i + bs]).to(device) if ses is not None else None
        )
        if mask_fn is not None:
            x = mask_fn(x)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")
        ):
            z = model(x, s)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def occlusion_analysis(
    model, pool, idx, bank, normalizer, device, fs=250, ch_names=None, win_s=0.5
):
    sid = pool.sid[idx]
    uniq, target = np.unique(sid, return_inverse=True)
    z_pool = bank.emb[uniq]
    X, sub = pool.X[idx], pool.sub[idx]
    ses = pool.sess[idx]
    scale = 100.0 if getattr(normalizer, "mode", "zscore") == "fm" else 1.0

    def metric(z):
        r = retrieval_metrics(z, z_pool, target, n_draws=5)
        return {
            "mrr": r["mrr_full"],
            "top1_p100": r.get("top1_pool100", r["top1_full"]),
            "top10": r["top10_full"],
        }

    base = metric(_embed(model, X, sub, normalizer, device, ses=ses))
    T = X.shape[2]
    w = int(win_s * fs)
    out = {"baseline": base, "temporal": [], "regions": {}}
    for t0 in range(0, T - w + 1, w):

        def mk(x, t0=t0):
            x = x.clone()
            x[:, :, t0 : t0 + w] = torch.randn_like(x[:, :, t0 : t0 + w]) * scale
            return x

        out["temporal"].append(
            {
                "t_start": t0 / fs,
                "t_end": (t0 + w) / fs,
                **metric(_embed(model, X, sub, normalizer, device, mk, ses=ses)),
            }
        )
    if ch_names:
        for name, f in REGIONS.items():
            ch = [i for i, c in enumerate(ch_names) if f(c)]
            if not ch:
                continue

            def mk(x, ch=ch):
                x = x.clone()
                x[:, ch, :] = torch.randn_like(x[:, ch, :]) * scale
                return x

            out["regions"][name] = {
                "n_channels": len(ch),
                **metric(_embed(model, X, sub, normalizer, device, mk, ses=ses)),
            }
    return out
