from __future__ import annotations

import logging
from typing import Literal

from .state import InternalWorldState
from .rules import RuleEngine
from .generation import (
    build_state_prompt,
    build_event_prompt,
    generate_line,
    predict_coherence,
    parse_generated_line,
)

log = logging.getLogger("dream")


def dream_step(
    tokenizer,
    model,
    state: InternalWorldState,
    rule_engine: RuleEngine,
    mode: Literal["state", "event"],
) -> None:
    prompt = build_state_prompt(state) if mode == "state" else build_event_prompt(state)
    attempts = 0
    while True:
        attempts += 1
        raw_line = generate_line(tokenizer, model, prompt)
        log.info("Generated (%s) attempt %d: %s", mode, attempts, raw_line)
        coh_pred = predict_coherence(tokenizer, model, prompt, raw_line)
        if coh_pred is not None:
            log.info("Coherence prediction: %.3f", coh_pred)
        if not raw_line.strip():
            state.record_rejection(["empty generation"])
            log.info("REJECTED: empty generation")
            if attempts >= 3:
                return
            continue
        try:
            update = parse_generated_line(raw_line)
            update.kind = mode  # enforce mode to avoid model drift
        except ValueError as exc:
            state.record_rejection([str(exc)])
            log.info("REJECTED: %s", exc)
            log.info("State unchanged -> %s", state.debug_summary())
            if attempts >= 3:
                return
            continue

        log.info("Parsed update: %s", update)

        ok, reasons = rule_engine.aggregate_checks(state, update)
        if ok:
            state.apply_update(update)
            # Use model-predicted coherence as a soft bonus only.
            if coh_pred is not None:
                state.coherence_score += coh_pred * 0.2
                log.info("Coherence bonus applied: %.3f", coh_pred * 0.2)
            log.info("ACCEPTED -> %s", state.debug_summary())
            return
        else:
            state.record_rejection(reasons)
            log.info("REJECTED: %s", "; ".join(reasons))
            log.info("State unchanged -> %s", state.debug_summary())
            if attempts >= 3:
                return
            # retry generation for another attempt


def dream_loop(
    tokenizer,
    model,
    state: InternalWorldState,
    rule_engine: RuleEngine,
    steps: int = 10,
) -> None:
    modes = ["state", "event"]
    for i in range(steps):
        mode = modes[i % len(modes)]
        dream_step(tokenizer, model, state, rule_engine, mode=mode)


def initialize_world_with_model(
    tokenizer,
    model,
    state: InternalWorldState,
    rule_engine: RuleEngine,
    max_lines: int = 4,
) -> None:
    prompt = (
        "Initialize a tiny fictional world with 2-4 entities.\n"
        "Output one line per entity, formatted as:\n"
        "state: entity=<name>; type=<kind>; attr=k1:v1; relation=verb:target; note=short context.\n"
        "Do not add explanations or extra text."
    )
    from .generation import generate_text
    raw_text = generate_text(tokenizer, model, prompt, max_new_tokens=196)
    lines = [ln.strip() for ln in raw_text.splitlines() if "entity=" in ln or ln.lower().startswith("state:")]
    for line in lines[:max_lines]:
        try:
            update = parse_generated_line(line)
        except ValueError as exc:
            state.record_rejection([str(exc)])
            log.info("Init rejected: %s", exc)
            continue
        ok, reasons = rule_engine.aggregate_checks(state, update)
        if ok:
            state.apply_update(update)
            log.info("Init accepted: %s", update)
        else:
            state.record_rejection(reasons)
            log.info("Init rejected: %s", "; ".join(reasons))


def run_dream_session(tokenizer, model, num_steps: int = 10) -> InternalWorldState:
    state = InternalWorldState()
    rule_engine = RuleEngine()
    initialize_world_with_model(tokenizer, model, state, rule_engine)
    # If previous loop had valid generations, gently increase steps.
    effective_steps = num_steps
    dream_loop(tokenizer, model, state, rule_engine, steps=effective_steps)
    return state
