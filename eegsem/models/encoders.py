"""EEG encoders -> fixed-size embedding aligned to a frozen text space."""

import torch, torch.nn as nn, torch.nn.functional as F


class SubjectAdapter(nn.Module):
    """Per-subject linear spatial re-mixing (C x C, identity init). The only thing tuned in few-shot calibration."""

    def __init__(self, n_subjects, C):
        super().__init__()
        self.W = nn.Parameter(torch.eye(C).unsqueeze(0).repeat(n_subjects, 1, 1))
        self.b = nn.Parameter(torch.zeros(n_subjects, C, 1))

    def forward(self, x, sid):  # x [B,C,T], sid [B]
        return torch.einsum("bij,bjt->bit", self.W[sid], x) + self.b[sid]


class EEGNetEncoder(nn.Module):
    """EEGNet-style (Lawhern 2018) temporal -> depthwise spatial -> separable; then attention pooling."""

    def __init__(
        self, C=122, T=825, F1=16, D=2, F2=32, k1=64, k2=16, p1=4, p2=8, drop=0.25, out_dim=768
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, p1)),
            nn.Dropout(drop),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, k2), groups=F1 * D, padding="same", bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, p2)),
            nn.Dropout(drop),
        )
        Tp = T // p1 // p2
        self.pos = nn.Parameter(torch.zeros(1, F2, Tp))
        self.att = nn.Linear(F2, 1)
        self.head = nn.Sequential(
            nn.Linear(F2 * 2, 512), nn.GELU(), nn.Dropout(drop), nn.Linear(512, out_dim)
        )
        self.out_dim = out_dim

    def forward(self, x):  # [B,C,T]
        h = self.block2(self.block1(x.unsqueeze(1))).squeeze(2)  # [B,F2,Tp]
        h = h + self.pos[..., : h.shape[-1]]
        a = torch.softmax(self.att(h.transpose(1, 2)), dim=1)  # [B,Tp,1]
        pooled = torch.cat([(h.transpose(1, 2) * a).sum(1), h.mean(-1)], dim=-1)
        return self.head(pooled)


class ConformerLiteEncoder(nn.Module):
    """EEG-Conformer-like: EEGNet front-end + 2 transformer layers."""

    def __init__(
        self,
        C=122,
        T=825,
        F1=16,
        D=2,
        F2=64,
        k1=64,
        p1=4,
        p2=4,
        d_model=64,
        n_layers=2,
        drop=0.25,
        out_dim=768,
    ):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, p1)),
            nn.Dropout(drop),
            nn.Conv2d(F1 * D, F2, (1, 16), padding="same", bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, p2)),
            nn.Dropout(drop),
        )
        Tp = T // p1 // p2
        self.proj = nn.Linear(F2, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, Tp + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, 4, d_model * 4, drop, batch_first=True, activation="gelu"
        )
        self.tr = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim))
        self.out_dim = out_dim

    def forward(self, x):
        h = self.front(x.unsqueeze(1)).squeeze(2).transpose(1, 2)  # [B,Tp,F2]
        h = self.proj(h)
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], 1) + self.pos[:, : h.shape[1] + 1]
        h = self.tr(h)
        return self.head(h[:, 0])


FM_NAMES = ("labram", "cbramod", "cbramod62")


def build_encoder(name, C, T, out_dim, ch_names=None, weights_dir=None, fs=250):
    if name in FM_NAMES:
        from .foundation import build_fm

        return build_fm(name, ch_names, weights_dir, out_dim, fs)
    if name == "eegnet":
        return EEGNetEncoder(C, T, out_dim=out_dim)
    if name == "conformer":
        return ConformerLiteEncoder(C, T, out_dim=out_dim)
    raise ValueError(name)


class Aligner(nn.Module):
    """adapter(optional) -> encoder -> L2-normalised embedding; learnable temperature."""

    def __init__(self, encoder, n_subjects=1, C=122, use_adapter=True, init_tau=0.07):
        super().__init__()
        self.adapter = SubjectAdapter(n_subjects, C) if use_adapter else None
        self.encoder = encoder
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / init_tau).log())

    def forward(self, x, sid=None):
        if self.adapter is not None and sid is not None:
            x = self.adapter(x, sid)
        return F.normalize(self.encoder(x), dim=-1)


def info_nce(z_eeg, z_txt, sent_ids, logit_scale):
    """Symmetric InfoNCE where all pairs sharing a sentence id are positives (handles repeated sentences in a batch)."""
    logits = logit_scale.exp().clamp(max=100) * z_eeg @ z_txt.t()  # [B,B]
    pos = (sent_ids[:, None] == sent_ids[None, :]).float()
    pos = pos / pos.sum(1, keepdim=True)
    l1 = -(F.log_softmax(logits, 1) * pos).sum(1).mean()
    l2 = -(F.log_softmax(logits.t(), 1) * pos).sum(1).mean()
    return 0.5 * (l1 + l2)
