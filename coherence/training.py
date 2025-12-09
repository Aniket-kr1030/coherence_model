from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.optim as optim

from .rules import DIAG_LABELS, RuleEngine
from .state import InternalWorldState, ProposedUpdate, CoherenceTargets
from .generation import build_state_prompt, build_event_prompt, generate_line
from .modeling import load_dream_model
from .expansion import save_model_with_metadata, load_model_with_metadata

import os
import time
import json
import glob
import random
from datasets import load_dataset, load_from_disk

log = logging.getLogger("dream")


def coherence_targets_from_rule(ok: bool, reasons: List[str]) -> CoherenceTargets:
    diag = {label: 0 for label in DIAG_LABELS}
    if not ok:
        for reason in reasons:
            head = reason.split(":", 1)[0].strip()
            if head in diag:
                diag[head] = 1
    return CoherenceTargets(coherence=1.0 if ok else 0.0, diagnostics=diag)


def _diag_tensor(diag: Dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([[diag.get(label, 0) for label in DIAG_LABELS]], device=device, dtype=torch.float32)


def train_dream_coherence_head(
    tokenizer,
    dream_model,
    steps: int = 10,
    lr: float = 1e-4,
    ) -> float | None:
    dream_model.train()
    
    # Freeze base model to prevent catastrophic forgetting/instability
    # Only train the head and any expansion modules
    for name, param in dream_model.base_model.named_parameters():
        param.requires_grad = False
        
    trainable_params = list(dream_model.head.parameters())
    if hasattr(dream_model, "new_parameters"):
        trainable_params.extend(dream_model.new_parameters)
        
    # Filter to be safe
    trainable_params = [p for p in trainable_params if p.requires_grad]
    
    optimizer = optim.AdamW(trainable_params, lr=lr)
    state = InternalWorldState()
    rule_engine = RuleEngine()
    state.apply_update(
        ProposedUpdate(kind="state", name="person", new_type="human", attributes={"age": "18"}, note="seed")
    )

    modes = ["state", "event"]
    last_loss: float | None = None
    for step in range(steps):
        mode = modes[step % len(modes)]
        prompt = build_state_prompt(state) if mode == "state" else build_event_prompt(state)
        raw_line = generate_line(tokenizer, dream_model, prompt)
        log.info("Train gen (%s) %d: %s", mode, step + 1, raw_line)
        if not raw_line.strip():
            state.record_rejection(["empty generation"])
            log.info("Train reject parse: empty generation")
            continue
        try:
            from .generation import parse_generated_line

            update = parse_generated_line(raw_line)
            update.kind = mode
        except ValueError as exc:
            state.record_rejection([str(exc)])
            log.info("Train reject parse: %s", exc)
            continue

        ok, reasons = rule_engine.aggregate_checks(state, update)
        targets = coherence_targets_from_rule(ok, reasons)
        coh_target = torch.tensor([targets.coherence], device=dream_model.device)
        diag_target = _diag_tensor(targets.diagnostics, dream_model.device)

        full_text = prompt + "\n" + raw_line
        encoded = tokenizer(full_text, return_tensors="pt").to(dream_model.device)
        out = dream_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            labels=encoded["input_ids"],
            coherence_target=coh_target,
            diag_target=diag_target,
        )
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
        log.info("Train loss: %.4f (coh %.4f diag %.4f)", loss.item(), out["coh_loss"].item(), out["diag_loss"].item())

        if ok:
            state.apply_update(update)
        else:
            state.record_rejection(reasons)
    return last_loss


def run_coherence_training(steps: int = 10) -> None:
    # Use metadata loader to respect existing training
    tokenizer, dream_model, metadata = load_model_with_metadata()
    
    train_dream_coherence_head(tokenizer, dream_model, steps=steps)
    
    # Save the progress
    save_model_with_metadata(dream_model, metadata)
    log.info("Finished training Dream Coherence head and saved to disk.")


def _extract_strings_from_json_obj(obj) -> List[str]:
    result: List[str] = []
    if isinstance(obj, str):
        result.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            result.extend(_extract_strings_from_json_obj(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            result.extend(_extract_strings_from_json_obj(v))
    return result


def _load_lines_from_file(path: str, max_lines: int = 5000) -> List[str]:
    lines: List[str] = []
    try:
        if path.endswith(".jsonl") or path.endswith(".json"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, ln in enumerate(f):
                    if i >= max_lines:
                        break
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                        strings = _extract_strings_from_json_obj(obj)
                        for s in strings:
                            if s:
                                lines.append(str(s))
                    except Exception:
                        lines.append(ln)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, ln in enumerate(f):
                    if i >= max_lines:
                        break
                    ln = ln.strip()
                    if ln:
                        lines.append(ln)
    except Exception as e:
        log.info("Skipping %s due to error: %s", path, e)
    return lines


def train_on_datasets(
    datasets_dir: str = "datasets",
    batch_size: int = 2,
    lr: float = 5e-5,
    epochs: int = 1,
    auto_save: bool = True,
    dataset_filter: list = None,
) -> None:
    """Train DreamCoherenceModel on local HuggingFace-style datasets with ETA logging."""
    # Load persistent model
    tokenizer, model, metadata = load_model_with_metadata()
    device = model.device
    
    # Freeze base model parameters
    for param in model.base_model.parameters():
        param.requires_grad = False
        
    # Only train adapters/expansions
    trainable_params = []
    if hasattr(model, "new_parameters"):
        trainable_params.extend(model.new_parameters)
    trainable_params = [p for p in trainable_params if p.requires_grad]
    
    if not trainable_params:
         log.warning("No trainable adapters/expansions found. Dataset training requires adapters to be added first (run 'sleep' in chat to auto-expand).")
         return

    optimizer = optim.AdamW(trainable_params, lr=lr)

    all_datasets = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
    
    # Apply filter if provided
    if dataset_filter:
        datasets = [d for d in all_datasets if d in dataset_filter]
        log.info("Filtering datasets: %s (from %d available)", datasets, len(all_datasets))
    else:
        datasets = all_datasets
    
    try:
        for dataset in datasets:
            dataset_path = os.path.join(datasets_dir, dataset)
            try:
                ds = load_from_disk(dataset_path)
            except Exception:
                try:
                    ds = load_dataset(dataset_path)
                except Exception as e:
                    log.info("Skipping %s (load error: %s)", dataset, e)
                    continue
            for split in ds.keys():
                split_ds = ds[split]
                total_samples = len(split_ds)
                if total_samples == 0:
                    continue
                for epoch in range(epochs):
                    start = time.time()
                    total_batches = max(1, (total_samples + batch_size - 1) // batch_size)
                    log.info("Training %s/%s epoch %d (%d samples, %d batches)...", dataset, split, epoch + 1, total_samples, total_batches)
                    for bi in range(total_batches):
                        start_idx = bi * batch_size
                        end_idx = min(total_samples, (bi + 1) * batch_size)
                        batch = split_ds.select(range(start_idx, end_idx))
                        batch_texts = []
                        for row in batch:
                            strings = _extract_strings_from_json_obj(row)
                            if strings:
                                batch_texts.append(" ".join(map(str, strings)))
                        if not batch_texts:
                            continue
                        batch_tok = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                        outputs = model(
                            input_ids=batch_tok["input_ids"],
                            attention_mask=batch_tok.get("attention_mask"),
                            labels=batch_tok["input_ids"],
                        )
                        loss = outputs["loss"]
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        elapsed = time.time() - start
                        remaining = (elapsed / (bi + 1)) * (total_batches - bi - 1)
                        # Log more frequently (every 20 batches) so user sees progress
                        if (bi + 1) % 20 == 0 or bi == total_batches - 1:
                            log.info(
                                "[%s/%s|ep%d] batch %d/%d loss=%.4f ETA=%.1fs",
                                dataset,
                                split,
                                epoch + 1,
                                bi + 1,
                                total_batches,
                                loss.item(),
                                remaining,
                            )
                    log.info("Finished %s/%s epoch %d in %.1fs", dataset, split, epoch + 1, time.time() - start)
            # Checkpoint: Save after each dataset completes
            if auto_save:
                metadata["expansions"] = {
                    "extra_layers": len(getattr(model, "extra_layers", [])),
                    "extra_attn_blocks": len(getattr(model, "aux_attention", [])),
                    "mlp_expansions": len(getattr(model, "mlp_expansions", [])),
                    "lora_expansions": len(getattr(model, "lora_adapters", [])),
                }
                save_model_with_metadata(model, metadata)
                log.info("Checkpoint saved after dataset: %s", dataset)
    except KeyboardInterrupt:
        log.info("\nTraining interrupted by user. Saving progress...")
    except Exception as e:
        log.error("\nTraining failed with error: %s. Saving progress...", e)
    finally:
        if auto_save:
            metadata["expansions"] = {
                "extra_layers": len(getattr(model, "extra_layers", [])),
                "extra_attn_blocks": len(getattr(model, "aux_attention", [])),
                "mlp_expansions": len(getattr(model, "mlp_expansions", [])),
                "lora_expansions": len(getattr(model, "lora_adapters", [])),
            }
            save_model_with_metadata(model, metadata)
            log.info("Model saved successfully.")
