"""A `recall.Proposer` backed by a vLLM OpenAI-compatible server.

Drop-in for `recall.NuExtract` -- same `propose`/`ask` -- with no model in this process. The
NuExtract `template=` jinja variable travels in `chat_template_kwargs`, which vLLM forwards
verbatim to `apply_chat_template`; verified to render a prompt token-for-token identical to
the local `AutoProcessor` (3,884 tokens on both).

Measured against transformers + bitsandbytes NF4 over the same eighteen calls: 185.7s ->
31.5s at concurrency four, 5.9x. Not the order of magnitude a first look suggested -- the
work is prefill-bound (81,755 prompt tokens against 3,046 generated) and the card runs out
of SMs before it runs out of memory.

The reason to prefer it is not the speed. Constrained decoding ends a failure the in-process
path cannot address: on 16962339 the free-running decoder repeats a *completed* `tasks`
object until `max_tokens`, burning 2,048 tokens and 27.5s to produce something unparseable.
Under a schema the same call answers in 70 tokens and 2.4s.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence

from pondie.extraction.recall import INSTRUCTION, _NOUN, Starved, directive, template_for

_KNOWN = {"string", "verbatim-string", "integer", "number", "boolean", "date-time"}
_PRIM = {"string": "string", "verbatim-string": "string", "integer": "integer",
         "number": "number", "boolean": "boolean", "date-time": "string"}


def json_schema_for(node: Any) -> dict:
    """A NuExtract template as a JSON schema the server can compile.

    Every leaf admits `null`, and that is the whole of what keeps this honest. A grammar
    with no way to say "nothing here" does not decline -- it emits the best string it can,
    and the model duly filled `Group.arm` with 'smoking' and 'non-smoking' on 16759342, a
    paper that declares no arms at all and whose every group the extractor left empty. An
    `Arm` is something participants were *assigned to receive*; smokers were not assigned to
    smoke, and the slot is a reference, so the value did not even name a declared entity.

    Nullability is what fixes that, not `strict`. Measured on that paper: nullable answers
    `null` under `strict: true` with every property required, and non-nullable invents the
    same two values under `strict: true` and `strict: false` alike.
    """
    if isinstance(node, str):
        return {"type": [_PRIM.get(node, "string"), "null"]}
    if isinstance(node, dict):
        return {"type": "object", "additionalProperties": False, "required": [],
                "properties": {k: json_schema_for(v) for k, v in node.items()}}
    if isinstance(node, list):
        if len(node) == 1 and (isinstance(node[0], dict) or node[0] in _KNOWN):
            return {"type": "array", "items": json_schema_for(node[0])}
        # An enum projected from the schema. `template_for` renders single- and multi-valued
        # enums identically, so accept either shape -- and carry `None` in the enum itself,
        # because a bare `enum` is a closed set that a type union cannot widen.
        return {"anyOf": [{"enum": list(node) + [None]},
                          {"type": "array", "items": {"enum": list(node)}}]}
    return {"type": ["string", "null"]}


class TooLong(RuntimeError):
    """The server refused the prompt as longer than its context."""


class NuExtractServer:
    """The proposer, over HTTP.

    `strict` is on. It was off while the invention above was blamed on it; the schema was
    the cause, and a strict grammar over a nullable schema is the combination that both
    guarantees parseable output and lets the model answer nothing.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8311/v1", model: str = "nu",
                 max_premise_chars: int = 45_000, max_new_tokens: int = 2_048,
                 structured: bool = True, strict: bool = True,
                 timeout: float = 1_800.0) -> None:
        self._base = base_url.rstrip("/")
        self._url = self._base + "/chat/completions"
        self._model, self._max_chars, self._max_new = model, max_premise_chars, max_new_tokens
        self._structured, self._strict, self._timeout = structured, strict, timeout

    def reachable(self, timeout: float = 3.0) -> bool:
        """Whether a server is answering here, asked before a run commits to one.

        Cheap and non-committal: `GET /models` allocates nothing and returns in milliseconds
        or not at all. The alternative is discovering the server is down on the first paper
        of a long run, by which point the fallback has already cost a model load.
        """
        # Built from the base, not by unpicking the completions URL: `rsplit` left
        # `/v1/chat/models`, which 404s, so a live server read as absent and the run fell
        # back to loading five gigabytes it did not need.
        url = self._base + "/models"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as reply:
                return reply.status == 200
        except Exception:  # noqa: BLE001 -- any failure to answer means "not there"
            return False

    def propose(self, sch, class_name: str, premise: str,
                instruction: str) -> Sequence[Mapping[str, Any]]:
        template = template_for(sch, class_name)
        key = next(iter(template))
        payload = self.ask(template, instruction, premise, what=class_name)
        proposed = payload.get(key) if isinstance(payload, Mapping) else None
        return [p for p in (proposed or []) if isinstance(p, Mapping)]

    def ask(self, template: Mapping[str, Any], instruction: str, premise: str,
            what: str = "") -> Mapping[str, Any]:
        limit = self._max_chars
        while True:
            body = ((directive(what) if what in _NOUN or what == "Analysis" else "")
                    + INSTRUCTION + instruction + premise[:limit])
            request = {
                "model": self._model,
                "messages": [{"role": "user", "content": [{"type": "text", "text": body}]}],
                "chat_template_kwargs": {"template": json.dumps(template, indent=2),
                                         "enable_thinking": False},
                "temperature": 0.0,
                "max_tokens": self._max_new,
            }
            if self._structured:
                request["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "nuextract", "strict": self._strict,
                    "schema": json_schema_for(template)}}
            try:
                text = self._post(request)
                break
            except TooLong:
                # Halving mirrors what the in-process proposer does on a CUDA OOM: a skipped
                # call loses the whole sweep for that class, which reads in the report as a
                # pass with nothing to add.
                if limit <= 6_000:
                    raise Starved(
                        f"{what or 'proposal'}: prompt too long at a {limit}-character "
                        f"premise, the smallest this pass will try") from None
                limit //= 2
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, Mapping) else {}

    def _post(self, request: dict) -> str:
        posted = urllib.request.Request(
            self._url, data=json.dumps(request).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(posted, timeout=self._timeout) as reply:
                body = json.load(reply)
        except urllib.error.HTTPError as error:
            detail = error.read().decode()[:500]
            if "maximum context length" in detail or "longer than the maximum" in detail:
                raise TooLong(detail) from None
            raise RuntimeError(f"vLLM {error.code}: {detail}") from None
        return body["choices"][0]["message"]["content"].strip()
