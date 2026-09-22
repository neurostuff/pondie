"""Sentence encoders, chosen by input length rather than by domain.

SapBERT is trained on UMLS synonym pairs and wins on short entity strings; MiniLM wins on
paragraphs, and the two invert completely on this corpus. `for_phrases` and `for_prose`
name the choice so a caller states the input's shape instead of guessing a model.

Encodings are cached on disk keyed by model and content. Measurements in
docs/normalization-layer.md.
"""

from __future__ import annotations

import functools
import hashlib
import os

from pondie import paths

PHRASE_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
PROSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = paths.CACHE / "embeddings"


@functools.lru_cache(maxsize=1)
def device() -> str:
    """The accelerator if there is one. `PONDIE_EMBED_DEVICE` overrides.

    Detected, not pinned: the machines this runs on differ and a hardcoded `cpu` left the
    GPU host idle. Deliberately not part of the cache key, so a run on one machine may
    reuse another's cache.
    """
    override = os.environ.get("PONDIE_EMBED_DEVICE")
    if override:
        return override
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@functools.lru_cache(maxsize=4)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=device())


def encode(texts: list[str], model: str, cache: bool = True):
    """L2-normalized embeddings, from disk when the same texts were encoded before."""
    import numpy as np

    if not texts:
        return np.zeros((0, 1), dtype="float32")
    key = hashlib.sha256(("\x00".join([model, *texts])).encode()).hexdigest()[:24]
    path = CACHE / f"{model.split('/')[-1]}-{key}.npy"
    if cache and path.is_file():
        return np.load(path)
    out = _model(model).encode(
        texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False
    )
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, out)
    return out


def for_phrases(texts: list[str], **kw):
    """Entity strings: a disease name, a group label, a condition."""
    return encode(texts, PHRASE_MODEL, **kw)


def for_prose(texts: list[str], **kw):
    """Descriptions, instructions, anything with sentences in it."""
    return encode(texts, PROSE_MODEL, **kw)
