"""Does this passage support this claim?

`quote.py` decides which passages *warrant* a value -- the model's own citation, and the
cross-encoder's second opinion. Neither judges entailment: a retriever returns its best match
whether or not the match says anything, and on one field that was the acknowledgements
section offered as the warrant for a model term's type.

A grounding model answers the question the locators cannot. Measured on that case: 0.041 for
the acknowledgements sentence, 0.919 for a sentence that really says it, and 0.025 for a
sentence naming the wrong term against 0.952 for the right one. Wide enough that the
retriever can stay permissive, which is why it recovers the warrant on fields the extracting
model never quoted (5 of 14 sampled) without those recoveries being taken on trust.

`Checker` is a protocol, and the pass that uses it takes `None`, because the weights are a
heavyweight optional dependency and the rest of a repair is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Claim:
    """One thing the record asserts, and the text it is to be judged against."""

    #: The assertion in prose, e.g. "In the model estimation, the term X type is continuous."
    claim: str
    #: What it is checked against: the span that warrants it, or the methods and results
    #: when asking whether an entity exists at all.
    premise: str


class Checker(Protocol):
    """Scores each claim against its own premise, 0 to 1."""

    def score(self, claims: Sequence[Claim]) -> Sequence[float]: ...


class MiniCheck:
    """`bespokelabs/minicheck` behind the protocol.

    Two settings are measurements rather than defaults. The batch is small because sixteen
    claims against a methods-and-results premise exhausts an 8 GB card, and MiniCheck chunks
    the document per claim, so the batch is the only thing that can shrink. The device is set
    before construction because MiniCheck takes no device argument -- it loads with
    `device_map="auto"` and reads the visible devices, which its own logs ask callers to set.
    """

    def __init__(self, model_name: str = "flan-t5-large", batch_size: int = 4,
                 device: str = "", cache_dir: str | None = None) -> None:
        import os

        if device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        from minicheck.minicheck import MiniCheck as _MiniCheck

        self._batch = batch_size
        self._model = _MiniCheck(model_name=model_name, enable_prefix_caching=False,
                                 batch_size=batch_size, cache_dir=cache_dir)

    def score(self, claims: Sequence[Claim]) -> Sequence[float]:
        """Scored in slices, freeing between them, so one long premise cannot end a run."""
        import torch

        out: list[float] = []
        for start in range(0, len(claims), self._batch):
            batch = claims[start:start + self._batch]
            _, probabilities, _, _ = self._model.score(
                docs=[c.premise for c in batch], claims=[c.claim for c in batch]
            )
            out.extend(probabilities)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return out
