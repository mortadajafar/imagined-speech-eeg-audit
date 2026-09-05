"""Frozen sentence embeddings. Output: emb_<name>.npy [S, d] (L2-normalised) + sentences.json (same order)."""

import json, os
import numpy as np

MODELS = {
    "labse": "sentence-transformers/LaBSE",
    "bge-m3": "BAAI/bge-m3",
    "e5": "intfloat/multilingual-e5-large",
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}


def embed_sentences(sentences, name="labse", batch_size=128, device=None):
    from sentence_transformers import SentenceTransformer
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(MODELS[name], device=device, trust_remote_code=True)
    emb = model.encode(
        list(sentences),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def save_bank(sentences, emb, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"emb_{name}.npy"), emb)
    with open(os.path.join(out_dir, "sentences.json"), "w", encoding="utf-8") as f:
        json.dump(list(sentences), f, ensure_ascii=False)


class TextBank:
    def __init__(self, bank_dir, name):
        with open(os.path.join(bank_dir, "sentences.json"), encoding="utf-8") as f:
            self.sentences = json.load(f)
        self.emb = np.load(os.path.join(bank_dir, f"emb_{name}.npy")).astype(np.float32)
        self.index = {s: i for i, s in enumerate(self.sentences)}
        self.dim = self.emb.shape[1]

    def ids(self, texts):
        return np.array([self.index[t] for t in texts])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sentences_json",
        nargs="+",
        required=True,
        help="json list OR textmaps dict OR meta_*.json",
    )
    ap.add_argument("--models", nargs="+", default=["labse"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = set()
    for p in a.sentences_json:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            S |= {" ".join(k.strip().split()) for k in d}
        else:
            S |= {" ".join((x["text"] if isinstance(x, dict) else x).strip().split()) for x in d}
    S = sorted(S)
    print(len(S), "unique sentences")
    for m in a.models:
        e = embed_sentences(S, m)
        save_bank(S, e, a.out, m)
        print(m, e.shape)


def ensure_bank(bank_dir, name, cache_dirs, phases, out_dir):
    """Copy bank to out_dir and append embeddings for any cached sentence missing from it (needs internet for the model)."""
    import shutil, glob

    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(bank_dir, "*")):
        shutil.copy(f, out_dir)
    with open(os.path.join(out_dir, "sentences.json"), encoding="utf-8") as f:
        sentences = json.load(f)
    emb = np.load(os.path.join(out_dir, f"emb_{name}.npy"))
    have = set(sentences)
    missing = set()
    for d in cache_dirs:
        for ph in phases:
            p = os.path.join(d, f"meta_{ph}.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    missing |= {m["text"] for m in json.load(f)} - have
    if missing:
        missing = sorted(missing)
        print(f"bank: embedding {len(missing)} missing sentences with {name}")
        e = embed_sentences(missing, name)
        sentences += missing
        emb = np.concatenate([emb, e])
        save_bank(sentences, emb, out_dir, name)
    else:
        print("bank: complete")
    return out_dir
