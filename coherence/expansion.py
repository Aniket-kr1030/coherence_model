from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .modeling import DreamCoherenceModel, load_dream_model, model_device, _hidden_dim
from .rules import DIAG_LABELS

log = logging.getLogger("dream")
METADATA_PATH = "model_metadata.json"


def expand_lora_rank(model: DreamCoherenceModel, new_rank: int) -> List[nn.Parameter]:
    """Add a simple low-rank adapter on top of the final hidden states."""
    hidden = _hidden_dim(model.base_model)
    device = model_device(model)
    down = nn.Linear(hidden, new_rank, bias=False)
    up = nn.Linear(new_rank, hidden, bias=False)
    nn.init.xavier_uniform_(down.weight)
    nn.init.zeros_(up.weight)
    adapter = nn.Sequential(down, nn.ReLU(), up).to(device)
    if not hasattr(model, "lora_adapters"):
        model.lora_adapters = nn.ModuleList()
    model.lora_adapters.append(adapter)
    model.new_parameters += list(adapter.parameters())
    log.info("Expanded LoRA rank adapter by %d", new_rank)
    return list(adapter.parameters())


def widen_mlp_block(model: DreamCoherenceModel, delta_neurons: int) -> List[nn.Parameter]:
    """Attach an extra MLP expansion block applied residually after base model."""
    hidden = _hidden_dim(model.base_model)
    device = model_device(model)
    mlp = nn.Sequential(
        nn.LayerNorm(hidden),
        nn.Linear(hidden, hidden + delta_neurons),
        nn.ReLU(),
        nn.Linear(hidden + delta_neurons, hidden),
    ).to(device)
    if not hasattr(model, "mlp_expansions"):
        model.mlp_expansions = nn.ModuleList()
    model.mlp_expansions.append(mlp)
    model.new_parameters += list(mlp.parameters())
    log.info("Widened MLP with +%d neurons", delta_neurons)
    return list(mlp.parameters())


def add_attention_heads(model: DreamCoherenceModel, delta_heads: int) -> List[nn.Parameter]:
    """Add an auxiliary attention block with extra heads, applied residually."""
    hidden = _hidden_dim(model.base_model)
    device = model_device(model)
    current_heads = getattr(model.base_model.config, "num_attention_heads", 4)
    nhead = current_heads + max(1, delta_heads)
    attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=nhead, batch_first=True).to(device)
    if not hasattr(model, "aux_attention"):
        model.aux_attention = nn.ModuleList()
    model.aux_attention.append(attn)
    model.new_parameters += list(attn.parameters())
    log.info("Added auxiliary attention with %d heads", nhead)
    return list(attn.parameters())


def add_transformer_layer(model: DreamCoherenceModel) -> List[nn.Parameter]:
    """Append an extra TransformerEncoderLayer applied after base hidden states."""
    hidden = _hidden_dim(model.base_model)
    nhead = getattr(model.base_model.config, "num_attention_heads", 4)
    device = model_device(model)
    layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, batch_first=True).to(device)
    if not hasattr(model, "extra_layers"):
        model.extra_layers = nn.ModuleList()
    model.extra_layers.append(layer)
    model.new_parameters += list(layer.parameters())
    log.info("Added 1 Transformer layer (d_model=%d nhead=%d)", hidden, nhead)
    return list(layer.parameters())


def expand_coherence_head(model: DreamCoherenceModel, new_hidden_dim: int) -> List[nn.Parameter]:
    """Rebuild coherence head if hidden dim changes."""
    device = model_device(model)
    model.head = model.head.__class__(hidden_dim=new_hidden_dim, num_diag=len(DIAG_LABELS)).to(device)
    model.new_parameters += list(model.head.parameters())
    log.info("Expanded coherence head to hidden_dim=%d", new_hidden_dim)
    return list(model.head.parameters())


def save_model_with_metadata(model: DreamCoherenceModel, metadata: Dict, path: str = METADATA_PATH) -> None:
    # Save only the diff (head, adapters, etc.) to save space and avoid meta-tensor issues
    full_state = model.state_dict()
    filtered_state = {k: v for k, v in full_state.items() if not k.startswith("base_model.")}
    torch.save(filtered_state, "model_weights.pt")
    
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved model weights (diff only) and metadata to %s", path)


def load_model_with_metadata(path: str = METADATA_PATH) -> Tuple[object, DreamCoherenceModel, Dict]:
    metadata: Dict = {}
    tokenizer, model = load_dream_model(num_diag=len(DIAG_LABELS))
    if os.path.exists(path):
        with open(path, "r") as f:
            metadata = json.load(f)
        log.info("Loaded metadata: %s", metadata)
    # Rebuild expansions first, so that we have the layers to load weights into
    expansions = metadata.get("expansions", {})
    for _ in range(expansions.get("extra_layers", 0)):
        add_transformer_layer(model)
    for _ in range(expansions.get("extra_attn_blocks", 0)):
        add_attention_heads(model, delta_heads=1)
    for _ in range(expansions.get("mlp_expansions", 0)):
        widen_mlp_block(model, delta_neurons=16)
    for _ in range(expansions.get("lora_expansions", 0)):
        expand_lora_rank(model, new_rank=8)

    if os.path.exists("model_weights.pt"):
        state = torch.load("model_weights.pt", map_location=model_device(model))
        
        # Filter out mismatched shapes AND base_model weights (to recover from corrupted saves)
        model_state = model.state_dict()
        keys_to_remove = []
        for k, v in state.items():
            if k.startswith("base_model."):
                # Always ignore base model weights from checkpoint. 
                # We want the clean pre-trained weights from load_model().
                keys_to_remove.append(k)
                continue
                
            if k in model_state:
                if v.shape != model_state[k].shape:
                    check_s = list(v.shape)
                    model_s = list(model_state[k].shape)
                    log.info("Adapting layer '%s' from old shape %s to new shape %s (resetting weights).", k, check_s, model_s)
                    keys_to_remove.append(k)
        
        for k in keys_to_remove:
            del state[k]

        model.load_state_dict(state, strict=False)
        log.info("Restored model weights from checkpoint (base model kept frozen/clean).")
        
    return tokenizer, model, metadata


def freeze_params(params: List[nn.Parameter], freeze: bool = True) -> None:
    for p in params:
        p.requires_grad = not freeze


def expand_model_if_needed(model: DreamCoherenceModel, metrics: Dict) -> DreamCoherenceModel:
    """Decide and perform model expansion based on provided metrics."""
    if not hasattr(model, "new_parameters"):
        model.new_parameters: List[nn.Parameter] = []
    expansions = []
    reasons = []
    rejection_rate = metrics.get("rejection_rate", 0.0)
    semantic_rejects = metrics.get("semantic_rejects", 0)
    contradiction_rejects = metrics.get("contradiction_rejects", 0)
    entities = metrics.get("entities", 0)
    plateau = metrics.get("loss_plateau", False)
    user_expand = metrics.get("user_expand", False)

    # Require evidence before growing: either explicit user request or strong signals.
    quality_flag = rejection_rate > 0.5 or (semantic_rejects + contradiction_rejects) > 5
    complexity_flag = entities > 10

    if user_expand or quality_flag:
        expansions.append(lambda m: add_transformer_layer(m))
        reasons.append("quality")
    if plateau and user_expand:
        expansions.append(lambda m: widen_mlp_block(m, delta_neurons=32))
        reasons.append("plateau")
    if complexity_flag:
        expansions.append(lambda m: add_attention_heads(m, delta_heads=2))
        reasons.append("complexity")

    if not expansions:
        log.info("No expansion needed (rejection_rate=%.2f, entities=%d, plateau=%s, user_expand=%s)", rejection_rate, entities, plateau, user_expand)
        return model

    # Perform at most one expansion per category to avoid runaway growth.
    seen = set()
    for fn, tag in zip(expansions, reasons):
        if tag in seen:
            continue
        fn(model)
        seen.add(tag)
    log.info("Expanded model due to %s", ",".join(seen))
    return model
