"""Sentence encoders, chosen by input length rather than by domain.

Measured on this corpus, the same two models invert completely:

    short entity strings (17-30 chars)   SapBERT R@1 66.3% / 62.9%   MiniLM 50.6%
    task descriptions (~400 words)       SapBERT R@1 24.5% (last)    MiniLM 58.5%

SapBERT is trained on UMLS synonym pairs, so a paragraph is off-distribution for it and a
30-character string is off-distribution for a sentence encoder. Neither is "the biomedical
model"; `for_phrases` and `for_prose` name the choice so a caller states the input's shape
instead of guessing a model.

Encodings are cached on disk keyed by model and content, because the corpus side of a
retrieval is the same on every run and re-encoding 32k MONDO labels is minutes each time.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

PHRASE_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
PROSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = Path("data/eval/embedding-cache")


@functools.lru_cache(maxsize=4)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


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
