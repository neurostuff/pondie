"""Ask a second model what the first one missed.

`satisfy` fills the slots the demands stage asked for. What it cannot do is notice an entity
nobody asked about, or a link between two entities it filled separately -- and those are most
of what a built record lacks: a scanner named in the methods with no `Acquisition.device`
pointing at it, four ROIs an analysis searched with `regions` empty.

The sweep asks per class, with a template projected from the schema, and gives the model the
entities it may point at. Two properties of that are load-bearing:

  * **Targets are swept before the classes that reference them.** Analyses came first, being
    the central object, and that is exactly wrong for linking: four correctly named regions
    were refused because no Region existed yet, and the regions sweep ran afterwards.
  * **Candidates are listed per reference slot**, read from the schema. A fixed list of five
    classes offered the same five however the sweep was aimed, so sweeping groups -- whose
    `diagnostic_instrument` targets an Assessment -- offered no assessments, and the model
    named the instrument in prose instead. A second copy of it was then created.

`Proposer` is a protocol and the pass takes `None`: the weights are optional, and the
deterministic half of a repair is most of its value.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Protocol, Sequence

from pondie.extraction.record.edit import label_of
from pondie.formats import values
from pondie.schema.reader import Schema


class Proposer(Protocol):
    """Returns entities of `class_name` the paper describes, as flat dicts.

    `local` says whether proposing occupies this process's GPU. `repair` bounds concurrency
    over the local models, and a served proposer belongs outside that bound: throttling it
    serialises a wait on the network for the sake of a card it never touches.

    `ask` is on the protocol and not an implementation detail of `NuExtract`, because
    `evidence.relocate` calls it with its own template. A stub carrying only `propose`
    satisfied the type and then failed at the second caller -- the shape of stub that has
    twice let a real fault reach a live run here.
    """

    local: bool

    def propose(self, sch: Schema, class_name: str, premise: str,
                instruction: str) -> Sequence[Mapping[str, Any]]: ...

    def ask(self, template: Mapping[str, Any], instruction: str, premise: str,
            what: str = "") -> Mapping[str, Any]: ...


class Starved(RuntimeError):
    """The proposer ran out of memory at the smallest premise it is willing to send.

    Raised rather than returning nothing, because the two are indistinguishable in a report:
    eight workers sharing one card starved every sweep on every full-length paper for an
    hour, and each of those papers recorded `0 written, 0 refused` -- which reads as a pass
    with nothing to do. Whoever catches this turns it into a refusal a reviewer can see.
    """


#: LinkML range -> the type NuExtract templates use. Anything unmapped becomes a string,
#: which is the safe default: NuExtract validates its own output against the template, so a
#: wrong type costs a field and a wrong *shape* costs the reply.
_TYPES = {"string": "string", "integer": "integer", "float": "number",
          "double": "number", "decimal": "number", "boolean": "boolean",
          "date": "date-time", "datetime": "date-time", "uriorcurie": "string"}

#: Slots a proposal has no business setting. `id` is the schema's identifier and `local_id`
#: is added back explicitly at the front; the rest are minted by other stages or hold text
#: this pass cannot check.
_SKIP = frozenset({"id", "mirror_of", "source_table_analysis", "defines_regions",
                   "model_representation_notes"})

INSTRUCTION = """\
`local_id` is an ADDRESS, not a description. To CORRECT an entity already listed, copy its
`local_id` exactly; the reply is then an edit of that entity rather than a new one. Leave it
out for an entity you are adding. Never invent an id for one that is not listed.

A cross-reference names another entity. Give the name exactly as listed, or omit the field
entirely when there is nothing to point at: a reference has no "not reported" form, so an
empty string or a guess is worse than an absent field.

Do not invent a value to fill a field. If the paper does not state it, leave the field out.

An analysis restricted to a region names that region; one run over the whole brain names
none. The volume searched and the volume corrected are different claims, and gray or white
matter masking is not a restriction to a region at all.
"""


#: What each class is called when asking for it. Read from the container name rather than
#: the class so the phrasing matches the schema's own plural.
_NOUN = {"Analysis": "statistical analysis", "Region": "brain region",
         "Group": "participant group", "Task": "task", "Measure": "measured quantity",
         "Acquisition": "imaging acquisition", "Device": "scanner",
         "Preprocessing": "preprocessing pipeline", "ModelEstimation": "statistical model",
         "InferenceSettings": "thresholding and correction scheme",
         "Table": "table", "Assessment": "assessment instrument"}


def directive(class_name: str) -> str:
    """What to enumerate, which the model will not do unasked.

    Measured, not assumed: on 16508348 the same template and premise returned nothing
    without this and three correct regions -- hippocampus, parahippocampal, medial temporal
    lobe -- with it. A template says what the answer must look like and not what question it
    answers, so the directive is part of the call rather than the caller's business.
    """
    noun = _NOUN.get(class_name, class_name.lower())
    if class_name == "Analysis":
        # Not "used by one of its statistical analyses", which is circular for this sweep.
        return ("List every statistical analysis this paper reports on brain data. An "
                "analysis is one tested comparison or association that produces a "
                "statistical map or a set of regional results -- not a method or a piece "
                "of software.\n\n")
    return (f"List every {noun} in this paper that is used by, or reported for, one of its "
            f"statistical analyses. Ignore anything not tied to an analysis.\n\n")


def nu_type(sch: Schema, slot: Any) -> Any:
    """The template type for one slot, or None for a slot a proposal should not carry."""
    # Through the wrapper. `Region.name` declares `ExtractedString`, which is a class, so
    # every native slot read as a reference and dropped out -- leaving a template of
    # `{"local_id": "string"}` and a proposer that could only ever echo an id.
    ranges = sch.value_ranges(slot)
    for candidate in ranges:
        if candidate in sch.enums:
            # `any_of: [SomeEnum, string]` is how the schema keeps a vocabulary open. Taking
            # the enum branch keeps the closed values in the template; without it the slot
            # degrades to a free string and the model stops being told what it may say.
            permissible = list(getattr(sch.enums[candidate], "permissible_values", {}) or {})
            return permissible or "string"
    if any(candidate in sch.classes for candidate in ranges):
        return None                      # a reference; `candidates` offers those by name
    return _TYPES.get(str(ranges[0] if ranges else "string").lower(), "string")


def flat(sch: Schema, class_name: str) -> bool:
    """Is every slot of this class a leaf, so a template can carry a list of them?

    `Analysis.effect` nests a structure a flat reply cannot express, which is why nested
    slots are skipped at all. `Task.conditions` is not that: a Condition is an id, a name, a
    kind and a description, and a list of those is as expressible as a list of regions.
    """
    return all(kind != "nested" for _n, _s, kind in sch.iter_slots(class_name))


def vocabulary(sch: Schema, class_name: str, limit: int = 1_400) -> str:
    """What the enum tokens in this class's template mean, for the instruction beside it.

    A NuExtract template is a type skeleton: `condition_kind` arrives as
    `["task_state", "rest", "fixation", "control_state"]` and the schema's careful prose
    about each never reaches the model. Asked to classify four picture-viewing conditions
    from those four words, it answered `fixation` for three of them.

    Follows `template_for` into flat nested classes, and must: `condition_kind` lives on
    `Condition`, reachable only through `Task.conditions`, so a sweep that skipped nested
    slots documented every enum except the one this docstring was written about. Across 610
    papers that slot was filled 0 times out of 1,571.

    Only the enums a class actually offers, first sentence each, and capped -- the
    instruction shares a context with the paper, and an entity sweep that spends it on
    documentation has less left for the document.
    """
    lines = _vocabulary_lines(sch, class_name, "")
    if not lines:
        return ""
    block, total = [], 0
    for line in lines:
        total += len(line) + 1
        if total > limit:
            # Said rather than silently cut: the tokens that fall off are still in the
            # template, and a model that sees four values documented and a fifth not
            # otherwise reads the omission as the fifth being unavailable.
            block.append(f"- ... {len(lines) - len(block)} further values not listed here")
            break
        block.append(line)
    return ("\n\nWhat the listed values mean. Choose one only if the paper describes that; "
            "leave the field out otherwise.\n" + "\n".join(block) + "\n\n")


def _vocabulary_lines(sch: Schema, class_name: str, prefix: str) -> list[str]:
    lines: list[str] = []
    for name, slot, kind in sch.iter_slots(class_name):
        if name in _SKIP or kind == "reference":
            continue
        if kind == "nested":
            inner = str(slot.range or "")
            if slot.multivalued and inner in sch.classes and flat(sch, inner):
                lines += _vocabulary_lines(sch, inner, f"{prefix}{name}.")
            continue
        for candidate in sch.value_ranges(slot):
            enum = sch.enums.get(candidate)
            if enum is None:
                continue
            for token, value in (getattr(enum, "permissible_values", None) or {}).items():
                said = " ".join(str(getattr(value, "description", "") or "").split())
                said = said.split(". ")[0].rstrip(".")
                if said:
                    lines.append(f"- `{prefix}{name}` {token}: {said}")
    return lines


def template_for(sch: Schema, class_name: str) -> dict:
    """The NuExtract template for one class, projected from the schema.

    `local_id` first, so the reply reads as an edit list. It is offered on every class and
    not only on Analysis: without it the model can name an entity but never address one, so
    every correction to a region or a group had to be matched by label -- the path that
    minted a second copy of an instrument the record already held.
    """
    fields: dict[str, Any] = {"local_id": "string"}
    for name, slot, kind in sch.iter_slots(class_name):
        if name in _SKIP or name == "local_id":
            continue
        if kind == "reference":
            fields[name] = ["verbatim-string"] if slot.multivalued else "verbatim-string"
            continue
        if kind == "nested":
            # A nested class whose own slots are all leaves projects as a list of objects.
            # `Task.conditions` is the case that matters: the condition's kind and
            # description were unreachable, so the one field that says a state was a control
            # or a rest could only ever be filled by the extraction pass.
            inner = str(slot.range or "")
            if slot.multivalued and inner in sch.classes and flat(sch, inner):
                fields[name] = [dict(template_for(sch, inner).popitem()[1][0])]
            continue
        projected = nu_type(sch, slot)
        if projected is None:
            continue
        fields[name] = [projected] if slot.multivalued and isinstance(projected, str) \
            else projected
    return {sch.containers().get(class_name, class_name.lower()): [fields]}


class _Proposes:
    """`propose` for any proposer that can `ask`.

    The two proposers differ only in transport, and this method was byte-identical between
    them but for one type annotation. It is the duplication that has already cost something:
    wiring `vocabulary` into proposing meant making the same edit twice by hand, and a fix
    applied to one copy and not the other is invisible until a run disagrees with itself.
    """

    def propose(self, sch: Schema, class_name: str, premise: str,
                instruction: str) -> Sequence[Mapping[str, Any]]:
        template = template_for(sch, class_name)
        key = next(iter(template))
        # The vocabulary rides with the instruction, not the template: a template is a type
        # skeleton and has nowhere to say what `fixation` means.
        payload = self.ask(template, vocabulary(sch, class_name) + instruction, premise,
                           what=class_name)
        proposed = payload.get(key) if isinstance(payload, Mapping) else None
        return [p for p in (proposed or []) if isinstance(p, Mapping)]


class NuExtract(_Proposes):
    """NuExtract 3 behind the protocol, in this process and on this card.

    `device_map={"": device}` and never `"auto"`: auto placed 9.3 GB of weights on the CPU
    rather than splitting them over two cards, said nothing, and the only symptom was a run
    that never finished -- which reads as a slow card, not a misplaced model.
    """

    #: Weights in this process, on this card. See `Proposer.local`.
    local = True

    def __init__(self, model_name: str = "numind/NuExtract3", device: int = 0,
                 max_premise_chars: int = 45_000, max_new_tokens: int = 2_048,
                 load_4bit: bool = True) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._max_chars, self._max_new = max_premise_chars, max_new_tokens
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        options: dict[str, Any] = {"trust_remote_code": True,
                                   "device_map": {"": device}}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        else:
            options["dtype"] = torch.bfloat16
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_name, **options).eval()

    def ask(self, template: Mapping[str, Any], instruction: str, premise: str,
            what: str = "") -> Mapping[str, Any]:
        """One templated generation, halving the premise until it fits.

        Split out of `propose` so a caller with its own template -- `evidence.relocate`, which
        asks for the sentences that support a value rather than for entities -- reuses this
        ladder rather than writing a second one that OOMs differently.
        """
        import json

        import torch

        limit = self._max_chars
        while True:
            messages = [{"role": "user", "content": [
                {"type": "text",
                 "text": (directive(what) if what in _NOUN or what == "Analysis" else "")
                         + INSTRUCTION + instruction
                         + premise[:limit]}]}]
            inputs = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt", template=json.dumps(template, indent=2),
                enable_thinking=False).to(self._model.device)
            try:
                with torch.inference_mode():
                    ids = self._model.generate(**inputs, max_new_tokens=self._max_new,
                                               do_sample=False)
                break
            except torch.OutOfMemoryError:
                # Halve and retry rather than drop. A skipped call loses the whole sweep for
                # that class: one paper lost all ten of its calls and got no entity pass at
                # all, because the section finder returned the whole document as the premise.
                del inputs
                torch.cuda.empty_cache()
                if limit <= 6_000:
                    raise Starved(
                        f"{what or 'proposal'}: out of memory at a {limit}-character "
                        f"premise, the smallest this pass will try")
                limit //= 2
        text = self._processor.batch_decode(
            ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, Mapping) else {}


def sweep_order(sch: Schema, keys: Sequence[str]) -> list[str]:
    """`keys` reordered so a class comes after everything it points at.

    Read from the schema's own slot kinds, so a reference added later orders itself. A cycle
    -- `Analysis.mirror_of` and the other two self-referential slots -- means neither side can
    go first, and the caller's order stands for it.
    """
    containers = sch.containers()
    class_of = sch.classes_by_container()
    targets = {
        key: {containers.get(slot.range) for _n, slot, kind in sch.iter_slots(class_of[key])
              if kind == "reference" and isinstance(slot.range, str)}
        for key in keys if key in class_of
    }
    out: list[str] = []
    seen: set[str] = set()

    def visit(key: str, path: frozenset[str]) -> None:
        if key in seen or key in path:
            return
        for target in sorted(t for t in (targets.get(key) or set()) if t in keys):
            visit(target, path | {key})
        if key not in seen:
            seen.add(key)
            out.append(key)

    for key in keys:
        visit(key, frozenset())
    return out


def existing(sch: Schema, record: Mapping[str, Any], class_name: str) -> str:
    """The entities of this class already held, with their ids and current values.

    Without the ids the sweep is append-only: the model can propose something new but has no
    handle on what is there, so nothing can be corrected and nothing can be linked to. That
    is not a prediction -- a version listing labels alone added 0 links across three papers
    where the same models under the same thresholds added 6, 14 and 3, and created entities
    instead, because every proposal looked new.

    The current values go with the id for the same reason: a model asked to correct a value
    it cannot see will either repeat it or invent one.
    """
    container = sch.containers().get(class_name)
    held = [e for e in (record.get(container) or []) if isinstance(e, Mapping)]
    head = f"## {class_name} already extracted\n\n"
    if not held:
        return head + "NONE.\n\n"
    lines = []
    for entity in held:
        lines.append(f"- local_id `{entity.get('local_id')}` -- {label_of(entity)}")
        summary = _slot_summary(sch, entity, class_name)
        if summary:
            lines.append(f"    current values: {summary}")
    return (head + "\n".join(lines) + "\n\n"
            "Return the COMPLETE list. To correct one of these, reuse its `local_id` and "
            "give only the values the paper contradicts. For one the paper describes that "
            "is missing above, leave `local_id` out.\n\n")


def _slot_summary(sch: Schema, entity: Mapping[str, Any], class_name: str,
                  limit: int = 6) -> str:
    """A few of the entity's filled slots, so a correction has something to correct."""
    parts = []
    for name, _slot, kind in sch.iter_slots(class_name):
        if name in ("local_id", "name") or kind == "nested" or len(parts) >= limit:
            continue
        # `is_field`, not the schema's kind: the schema here is storage, where a slot holds
        # a plain value, while the record is extraction-shaped and wraps it. Classifying
        # against the wrong one printed whole `ExtractedValue` dicts, evidence spans and all,
        # into a prompt meant to show the model a value.
        raw = entity.get(name)
        value = values.read(raw) if values.is_field(raw) else raw
        if value not in (None, "", []) and not isinstance(value, (dict, list)):
            parts.append(f"{name}={value}")
    return "; ".join(parts)


def candidates(sch: Schema, record: Mapping[str, Any], class_name: str) -> str:
    """What this class may point at, per slot, with what the slot means.

    The candidate list alone is not enough to get a link drawn. The script this replaces
    added a hand-written sentence per slot -- "name the regions its search space was
    restricted to, and leave that list empty when it was run over the whole brain" -- and
    that sentence is what produced the links; without it the model listed entities and
    connected nothing.

    The sentence does not need writing, because the schema already says it, and says it
    better: `Analysis.regions` reads "the regions the analysis ran over -- its search space
    ... Empty for whole-brain and searchlight, where the emptiness asserts that inference was
    not restricted", and `Group.diagnostic_instrument` reads "naming one here claims it
    classified this cohort, which is narrower than having been administered to it" -- which
    is exactly the over-inclusion that put a depression inventory in a PTSD group's
    diagnostic slot. Using the schema's own prose means a slot whose meaning is sharpened
    tomorrow is asked for correctly tomorrow.
    """
    blocks = []
    for name, slot, kind in sorted(sch.iter_slots(class_name)):
        if kind != "reference" or not isinstance(slot.range, str):
            continue
        key = sch.containers().get(slot.range)
        listed = "; ".join(
            label for entity in (record.get(key) or []) if isinstance(entity, Mapping)
            for label in [label_of(entity)] if label
        )
        blocks.append(f"- `{name}` ({slot.range}): {_meaning(slot)}\n"
                      f"  {'already in the record: ' + listed if listed else 'none in the record yet'}")
    if not blocks:
        return ""
    return ("## Links this record can carry\n\n" + "\n".join(blocks) +
            "\n\nName an entity in the slot that describes it, exactly as listed where it is "
            "listed. Name one the paper describes even if it is not listed. Leave a slot out "
            "when the paper gives nothing to put there -- an empty slot is a claim in itself "
            "where the description above says so.\n\n")


def _meaning(slot: Any, sentences: int = 2) -> str:
    """The slot's own description, trimmed to what fits in a prompt."""
    text = " ".join((getattr(slot, "description", "") or "").split())
    parts = re.split(r"(?<=[.?!]) ", text)
    return " ".join(parts[:sentences]) or "an entity of this class"


