"""Brain foundation models as EEG encoders (LaBraM, CBraMod). Input: raw microvolt EEG [B, C, T] at `fs_in` Hz.
Wrapper does: FIR low-pass (75 Hz) -> resample to 200 Hz -> zero-pad to whole 1-s patches -> /100 (uV scaling used by both models).
"""

import math, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import firwin

LABRAM_STD_1020 = [
    "FP1",
    "FPZ",
    "FP2",
    "AF9",
    "AF7",
    "AF5",
    "AF3",
    "AF1",
    "AFZ",
    "AF2",
    "AF4",
    "AF6",
    "AF8",
    "AF10",
    "F9",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "F10",
    "FT9",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "FT10",
    "T9",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "T10",
    "TP9",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "TP10",
    "P9",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO9",
    "PO7",
    "PO5",
    "PO3",
    "PO1",
    "POZ",
    "PO2",
    "PO4",
    "PO6",
    "PO8",
    "PO10",
    "O1",
    "OZ",
    "O2",
    "O9",
    "CB1",
    "CB2",
    "IZ",
    "O10",
    "T3",
    "T5",
    "T4",
    "T6",
    "M1",
    "M2",
    "A1",
    "A2",
    "CFC1",
    "CFC2",
    "CFC3",
    "CFC4",
    "CFC5",
    "CFC6",
    "CFC7",
    "CFC8",
    "CCP1",
    "CCP2",
    "CCP3",
    "CCP4",
    "CCP5",
    "CCP6",
    "CCP7",
    "CCP8",
    "T1",
    "T2",
    "FTT9h",
    "TTP7h",
    "TPP9h",
    "FTT10h",
    "TPP8h",
    "TPP10h",
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]


class FMPreproc(nn.Module):
    """[B,C,T]@fs_in uV -> [B,C,N,200]@200Hz scaled by 1/100."""

    def __init__(self, fs_in=250, fs_out=200, lp=75.0, numtaps=101):
        super().__init__()
        self.fs_in, self.fs_out = fs_in, fs_out
        h = firwin(numtaps, lp, fs=fs_in).astype(np.float32)
        self.register_buffer("fir", torch.tensor(h)[None, None, :])

    def forward(self, x):
        B, C, T = x.shape
        x = F.conv1d(x.reshape(B * C, 1, T), self.fir, padding=self.fir.shape[-1] // 2).reshape(
            B, C, -1
        )[:, :, :T]
        T2 = int(round(T * self.fs_out / self.fs_in))
        x = F.interpolate(x, size=T2, mode="linear", align_corners=False)
        n = math.ceil(T2 / self.fs_out)
        x = F.pad(x, (0, n * self.fs_out - T2))
        return (x / 100.0).reshape(B, C, n, self.fs_out)


def _strip_prefix(sd, prefix):
    return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}


class LaBraMEncoder(nn.Module):
    """LaBraM-base (5.8M) on the subset of channels present in its 10-20 vocabulary. Returns [B, out_dim]."""

    def __init__(self, ch_names, weights_path, out_dim=768, fs_in=250, drop=0.1):
        super().__init__()
        from .vendor.labram import NeuralTransformer

        up = [c.upper() for c in ch_names]
        self.keep = [i for i, c in enumerate(up) if c in LABRAM_STD_1020]
        self.register_buffer(
            "input_chans", torch.tensor([0] + [LABRAM_STD_1020.index(up[i]) + 1 for i in self.keep])
        )
        self.pre = FMPreproc(fs_in)
        self.net = NeuralTransformer(
            patch_size=200,
            embed_dim=200,
            depth=12,
            num_heads=10,
            mlp_ratio=4,
            qkv_bias=False,
            qk_norm=nn.LayerNorm,
            norm_layer=lambda d: nn.LayerNorm(d, eps=1e-6),
            init_values=0.1,
            use_mean_pooling=True,
            num_classes=0,
        )
        if weights_path and os.path.exists(weights_path):
            ck = torch.load(weights_path, map_location="cpu", weights_only=False)
            sd = _strip_prefix(ck.get("model", ck), "student.")
            sd = {k: v for k, v in sd.items() if not k.startswith("head")}
            msg = self.net.load_state_dict(sd, strict=False)
            n_loaded = len(set(sd) & set(self.net.state_dict()))
            if n_loaded < 0.9 * len(self.net.state_dict()):
                raise RuntimeError(
                    f"LaBraM checkpoint covers only {n_loaded} of {len(self.net.state_dict())} tensors: {msg}"
                )
            print(f"LaBraM weights: {n_loaded}/{len(self.net.state_dict())} tensors loaded; {msg}")
            print("LaBraM weights:", msg)
        else:
            raise FileNotFoundError(f"LaBraM pretrained weights not found at {weights_path}")
        self.out_dim = out_dim
        self.n_channels_used = len(self.keep)

    def forward(self, x):  # x: uV [B,C,T]
        x = self.pre(x[:, self.keep])
        h = self.net.forward_features(
            x, input_chans=self.input_chans, return_patch_tokens=True
        )  # [B, C*N, 200]
        return self.head(h.mean(1))


class CBraModEncoder(nn.Module):
    """CBraMod (ICLR 2025), channel-agnostic criss-cross transformer; uses all channels by default. Returns [B, out_dim]."""

    def __init__(self, ch_names, weights_path, out_dim=768, fs_in=250, drop=0.1, keep=None):
        super().__init__()
        from .vendor.cbramod import CBraMod

        self.keep = keep
        self.pre = FMPreproc(fs_in)
        self.net = CBraMod(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=30,
            n_layer=12,
            nhead=8,
        )
        if weights_path and os.path.exists(weights_path):
            sd = torch.load(weights_path, map_location="cpu", weights_only=False)
            msg = self.net.load_state_dict(sd, strict=False)
            print("CBraMod weights:", msg)
        else:
            raise FileNotFoundError(f"CBraMod pretrained weights not found at {weights_path}")
        self.head = nn.Sequential(nn.LayerNorm(200), nn.Dropout(drop), nn.Linear(200, out_dim))
        self.out_dim = out_dim

    def forward(self, x):
        if self.keep is not None:
            x = x[:, self.keep]
        x = self.pre(x)
        h = self.net(x)  # [B, C, N, 200]
        return self.head(h.mean((1, 2)))


def build_fm(name, ch_names, weights_dir, out_dim, fs_in=250):
    weights_dir = weights_dir or ""
    if name == "labram":
        return LaBraMEncoder(ch_names, os.path.join(weights_dir, "labram-base.pth"), out_dim, fs_in)
    if name == "cbramod":
        return CBraModEncoder(
            ch_names, os.path.join(weights_dir, "cbramod_pretrained_weights.pth"), out_dim, fs_in
        )
    if name == "cbramod62":  # same 62-channel subset as LaBraM, for a channel-matched comparison
        up = [c.upper() for c in ch_names]
        keep = [i for i, c in enumerate(up) if c in LABRAM_STD_1020]
        return CBraModEncoder(
            ch_names,
            os.path.join(weights_dir, "cbramod_pretrained_weights.pth"),
            out_dim,
            fs_in,
            keep=keep,
        )
    raise ValueError(name)
