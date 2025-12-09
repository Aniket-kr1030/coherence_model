from __future__ import annotations

import logging
import re
from threading import Thread
from typing import List, Tuple, Optional

from transformers import TextIteratorStreamer

from .dream import run_dream_session
from .modeling import DreamCoherenceModel, model_device
from .training import train_dream_coherence_head
from .dgm import generate_dreams_from_conversation, sleep_cycle_finetune_on_dreams
from .expansion import expand_model_if_needed

log = logging.getLogger("dream")

_SLEEP_RE = re.compile(
    r"^sleep(?:\s+for)?\s+(?P<num>\d+)\s*(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?)\s*$",
    re.IGNORECASE,
)


def parse_sleep_duration(text: str) -> Optional[float]:
    """Parse commands like 'sleep for 30 seconds' into a duration in seconds."""
    m = _SLEEP_RE.match(text.strip())
    if not m:
        return None
    num = int(m.group("num"))
    unit = m.group("unit").lower()
    if unit.startswith("sec"):
        return float(num)
    if unit.startswith("min"):
        return float(num) * 60.0
    if unit.startswith("hour") or unit.startswith("hr"):
        return float(num) * 3600.0
    return None


def build_conversation_text(history: List[Tuple[str, str]]) -> str:
    """Turn the accumulated (user, assistant) turns into a plain-text transcript."""
    lines: List[str] = []
    for u, a in history:
        lines.append(f"User: {u}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines)


def chat_with_model(tokenizer, model) -> None:
    device = model_device(model)
    log.info(
        "Chat mode: type 'quit' or 'exit' to stop. "
        "Type 'sleep' or 'sleep for 30 seconds' to trigger world simulation + coherence training."
    )
    history: List[Tuple[str, str]] = []

    if not isinstance(model, DreamCoherenceModel):
        model = DreamCoherenceModel(model, num_diag=6)
        model.to(device)
    dream_model: DreamCoherenceModel = model

    while True:
        try:
            user = input("You> ").strip()
        except EOFError:
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            break

        if user.lower().startswith("sleep"):
            duration = parse_sleep_duration(user)
            if duration is None:
                duration = 10.0
            intensity = max(1, min(10, int(duration // 10) or 1))

            log.info("[sleep] Simulating world with %d steps...", intensity * 2)
            state = run_dream_session(tokenizer, dream_model, num_steps=intensity * 2)
            log.info("[sleep] World summary after simulation: %s", state.summary())

            log.info("[sleep] Training Dream Coherence head for %d steps...", intensity * 2)
            coh_loss = train_dream_coherence_head(tokenizer, dream_model, steps=intensity * 2)

            metrics = {
                "rejection_rate": state.rejected_updates / max(1, state.accepted_updates + state.rejected_updates),
                "semantic_rejects": state.rejection_reasons.get("semantic: name/type mismatch", 0)
                + state.rejection_reasons.get("semantic: cluster jump", 0),
                "contradiction_rejects": state.rejection_reasons.get("contradiction: k overwrite", 0),
                "entities": len(state.entities),
                "loss_plateau": coh_loss is not None and coh_loss < 0.05,
                "user_expand": "grow" in user.lower() or "expand" in user.lower(),
            }
            dream_model = expand_model_if_needed(dream_model, metrics)

            convo = build_conversation_text(history)
            dreams = generate_dreams_from_conversation(
                tokenizer,
                dream_model,
                convo or "(no prior conversation)",
                num_dreams=max(1, intensity // 2),
            )
            grounding_qa: List[Tuple[str, str]] = []
            if dreams:
                log.info("[sleep] Running DGM sleep-cycle fine-tuning on %d dreams...", len(dreams))
                sleep_cycle_finetune_on_dreams(
                    tokenizer,
                    dream_model,
                    dreams,
                    grounding_qa,
                    lr=1e-4,
                )
            else:
                log.info("[sleep] Skipping DGM fine-tuning due to lack of dreams.")
            
            # Save the trained model!
            from .expansion import save_model_with_metadata
            metadata = {
                "expansions": {
                    "extra_layers": len(getattr(dream_model, "extra_layers", [])),
                    "extra_attn_blocks": len(getattr(dream_model, "aux_attention", [])),
                    "mlp_expansions": len(getattr(dream_model, "mlp_expansions", [])),
                    "lora_expansions": len(getattr(dream_model, "lora_adapters", [])),
                },
                "run_metrics": metrics,
            }
            save_model_with_metadata(dream_model, metadata)
            log.info("Saved updated model to disk.")

            print("Assistant: I have slept (simulated world + trained coherence + dreamed on our conversation).")
            continue

        sys_prompt = (
            "You are a concise, helpful assistant. Answer briefly. "
            "Do not restate these instructions. If unsure, say you don't know."
        )
        # Only keep the most recent 4 turns to reduce prompt echoing.
        recent_history = history[-4:]
        convo_lines = [
            "### System",
            sys_prompt,
            "### Conversation",
        ]
        for u, a in recent_history:
            convo_lines.append(f"User: {u}")
            convo_lines.append(f"Assistant: {a}")
        convo_lines.append("### User")
        convo_lines.append(user)
        convo_lines.append("### Assistant")
        prompt = "\n".join(convo_lines)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            **inputs,
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "streamer": streamer,
        }
        from threading import Thread

        thread = Thread(target=dream_model.generate, kwargs=gen_kwargs)
        thread.start()
        print("Assistant:", end=" ", flush=True)
        resp_parts: List[str] = []
        for token in streamer:
            resp_parts.append(token)
            full_resp = "".join(resp_parts)
            if full_resp.strip().endswith("User:") or "###" in full_resp:
                break
            print(token, end="", flush=True)
        thread.join()
        print()
        assistant_reply = "".join(resp_parts).replace("User:", "").strip()
        history.append((user, assistant_reply))
