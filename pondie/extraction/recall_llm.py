"""The extraction model as the repair proposer, to separate the design from the model.

Repair was wrong about 11 of the 19 fields it changed across four hand-read papers -- a
Wilson 95% interval of 36% to 77%, so a direction and a kind rather than a rate -- and every
one of those errors was a real fact from the paper in the wrong place -- an excluded patient's drug recorded as
the cohort's medication, a subgroup's mean age as a group's, the analysed count in the
enrolled slot. None of it is invention and no grounding check can see any of it.

That is either a limit of a 3B model answering a flat template about a whole paper, or a
property of asking the question that way at all. The two have different fixes and the
measurement cannot tell them apart, so this offers the same interface backed by the model
the extraction pass itself uses.

Read it for direction, not for a difference: four papers cannot separate two arms less than
about twenty points apart, which is most of the range this could plausibly move.

Deliberately the same `ask` contract as `NuExtract`, so `propose` is inherited unchanged
from `_Proposes` and the only variable between arms is which model answers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pondie.extraction.models import Cost, ModelCall
from pondie.extraction.recall import INSTRUCTION, _NOUN, _Proposes, directive

#: What a served or local NuExtract gets from its chat template and a chat model does not:
#: the template is the shape of the answer, not an example of one.
SHAPE = """\
Below is a JSON template and a paper. Return ONE JSON object matching the template exactly.

  * Every key you emit must appear in the template. Emit no others.
  * Omit a key entirely rather than guessing it. An omitted field is a fact the paper did
    not state; a wrong one is a fact it contradicts.
  * `local_id` addresses an entity the record already holds. Copy one you were shown under
    "Already extracted"; do not invent one.
"""
# Says nothing about which entity a value belongs to, deliberately. That instruction is
# `recall.SCOPED`, added to BOTH proposers by `PONDIE_TEMPLATE=scoped` or to neither:
# carried here it would make this arm differ from the local one in prompt as well as in
# model, and the arm exists to vary the model alone. Kept as a comment and not as part of
# the literal -- written inside it once, and the model was sent our reasoning about the
# experiment along with its instructions.


class ModelProposer(_Proposes):
    """A `Proposer` backed by the network model rather than by weights on a card."""

    #: Not local: `repair` bounds concurrency over the cards, and throttling this one would
    #: serialise a wait on the network for a GPU it never touches. See `Proposer.local`.
    local = False

    def __init__(self, caller: Any, model: str, *, study_id: str = "",
                 service_tier: str = "", effort: str = "low",
                 max_chars: int = 120_000) -> None:
        self._caller = caller
        self._model = model
        self._study_id = study_id
        self._service_tier = service_tier
        self._effort = effort
        self._max_chars = max_chars
        #: What this proposer spent. A local proposer costs a card and nothing a ledger can
        #: see, so `repair` only ever recorded the adjudication's cost -- and a network
        #: proposer put its whole spend outside the run's accounting. The first Luna arm
        #: reported 0 calls and 0 tokens for a stage that had made hundreds.
        self.cost = Cost()

    def ask(self, template: Mapping[str, Any], instruction: str, premise: str,
            what: str = "") -> Mapping[str, Any]:
        """One templated generation. Same contract as `NuExtract.ask`, different transport."""
        prompt = (
            (directive(what) if what in _NOUN or what == "Analysis" else "")
            + INSTRUCTION + instruction
            + "\n\n# Template\n\n" + json.dumps(template, indent=1)
            + "\n\n# Paper\n\n" + premise[:self._max_chars]
            + "\n\nEmit the JSON object now."
        )
        reply = self._caller(
            ModelCall(model=self._model, system=SHAPE, prompt=prompt,
                      max_output_tokens=8_000, effort=self._effort,
                      service_tier=self._service_tier, attempts=2),
            paper=self._study_id, stage=f"repair:propose:{what or 'any'}")
        self.cost = self.cost + reply.cost
        payload = getattr(reply, "payload", None)
        return payload if isinstance(payload, Mapping) else {}
