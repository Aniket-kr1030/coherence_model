from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Literal

import torch

from .modeling import model_device, DreamCoherenceModel
from .state import ProposedUpdate


STATE_PATTERN = re.compile(
    r"^\s*state:\s*entity=(?P<entity>[^;]+);\s*type=(?P<type>[^;]+);\s*attr=(?P<attr>[^;]*);\s*relation=(?P<relation>[^;]*);\s*note=(?P<note>.*)$",
    re.IGNORECASE,
)
LOOSE_STATE_PATTERN = re.compile(
    r"entity=(?P<entity>[^;]+);\s*type=(?P<type>[^;]+);\s*attr=(?P<attr>[^;]*);\s*relation=(?P<relation>[^;]*);\s*note=(?P<note>.*)$",
    re.IGNORECASE,
)
EVENT_PATTERN = re.compile(
    r"^\s*event:\s*actor=(?P<actor>[^;]+);\s*verb=(?P<verb>[^;]+);\s*target=(?P<target>[^;]+);\s*note=(?P<note>.*)$",
    re.IGNORECASE,
)
LOOSE_EVENT_PATTERN = re.compile(
    r"actor=(?P<actor>[^;]+);\s*verb=(?P<verb>[^;]+);\s*target=(?P<target>[^;]+);\s*note=(?P<note>.*)$",
    re.IGNORECASE,
)


def build_state_prompt(state) -> str:
    return (
        "You MUST output exactly ONE line, no bullets, no extra text.\n"
        "Strict format:\n"
        "state: entity=<name>; type=<kind>; attr=k1:v1, k2:v2; relation=verb1:target1; note=<reason>\n"
        "Examples:\n"
        "state: entity=cat; type=animal; attr=color:gray, mood:curious; relation=observes:mouse; note=because it heard a noise\n"
        "state: entity=forest; type=place; attr=weather:rainy; relation=contains:river; note=calm morning\n"
        "If no attributes or relations, leave them empty like attr=; relation=;\n"
        f"Current world: {state.summary()}\n"
        "Output the next plausible state line now:"
    )


def build_event_prompt(state) -> str:
    last_events = "; ".join(f"{ev.actor}->{ev.verb}->{ev.target}" for ev in state.events[-3:])
    return (
        "You MUST output exactly ONE line, no bullets, no lists.\n"
        "Strict format:\n"
        "event: actor=<name>; verb=<action>; target=<name_or_none>; note=<reason>\n"
        "Examples:\n"
        "event: actor=cat; verb=chases; target=mouse; note=because it is hungry\n"
        "event: actor=rain; verb=falls; target=forest; note=one year later the forest revived\n"
        f"Recent events: {last_events or 'none'}\n"
        f"Current world: {state.summary()}\n"
        "Output the next plausible event line now:"
    )


def parse_state_update_line(text: str) -> Optional[ProposedUpdate]:
    m = STATE_PATTERN.match(text.strip()) or LOOSE_STATE_PATTERN.match(text.strip())
    if not m:
        return None
    name = m.group("entity").strip()
    new_type = m.group("type").strip()
    attr_block = m.group("attr").strip()
    rel_block = m.group("relation").strip()
    note = m.group("note").strip()

    attributes: Dict[str, str] = {}
    for pair in attr_block.split(","):
        if ":" in pair:
            k, v = pair.split(":", 1)
            attributes[k.strip()] = v.strip()

    relations: List[Tuple[str, str]] = []
    for pair in rel_block.split(","):
        if ":" in pair:
            r, tgt = pair.split(":", 1)
            relations.append((r.strip(), tgt.strip()))

    allow_type_change = any(word in note.lower() for word in ("transform", "became", "reborn", "promotion"))
    allow_overwrite = "because" in note.lower() or "one year later" in note.lower()

    return ProposedUpdate(
        kind="state",
        name=name,
        new_type=new_type,
        attributes=attributes,
        relations=relations,
        note=note,
        allow_type_change=allow_type_change,
        allow_overwrite=allow_overwrite,
    )


def parse_event_update_line(text: str) -> Optional[ProposedUpdate]:
    txt = text.strip()
    m = EVENT_PATTERN.match(txt) or LOOSE_EVENT_PATTERN.match(txt)
    if not m:
        return None
    actor = m.group("actor").strip()
    verb = m.group("verb").strip()
    target_raw = m.group("target").strip()
    note = m.group("note").strip()

    target = None if target_raw.lower() in {"none", "null", ""} else target_raw
    attributes: Dict[str, str] = {}
    relations: List[Tuple[str, str]] = []

    allow_type_change = any(word in note.lower() for word in ("promotion", "became", "transform"))
    allow_overwrite = "because" in note.lower() or "one year later" in note.lower()

    return ProposedUpdate(
        kind="event",
        actor=actor,
        verb=verb,
        target=target,
        attributes=attributes,
        relations=relations,
        note=note,
        allow_type_change=allow_type_change,
        allow_overwrite=allow_overwrite,
    )


def parse_generated_line(text: str) -> ProposedUpdate:
    line = text.strip().splitlines()[0]
    line_lower = line.lower()
    if line_lower.startswith("state:"):
        parsed = parse_state_update_line(line)
        if parsed:
            return parsed
        raise ValueError("Unparseable state line")
    if line_lower.startswith("event:"):
        parsed = parse_event_update_line(line)
        if parsed:
            return parsed
        raise ValueError("Unparseable event line")

    raise ValueError("Unrecognized line format")


def generate_text(tokenizer, model, prompt: str, max_new_tokens: int = 96) -> str:
    device = model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def generate_line(tokenizer, model, prompt: str, max_new_tokens: int = 96) -> str:
    full = generate_text(tokenizer, model, prompt, max_new_tokens=max_new_tokens)
    lines = [ln for ln in full.splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[0].strip()


def predict_coherence(
    tokenizer, model, prompt: str, raw_line: str
) -> Optional[float]:
    if not isinstance(model, DreamCoherenceModel):
        return None
    device = model_device(model)
    full_text = prompt + "\n" + raw_line
    encoded = tokenizer(full_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids=encoded["input_ids"], attention_mask=encoded.get("attention_mask"))
    coh = out.get("coherence")
    if coh is None:
        return None
    return float(coh.squeeze().item())
