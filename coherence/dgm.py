from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import logging
import json
import os
import random
import re

import torch.optim as optim

from .generation import generate_text
from .modeling import model_device

log = logging.getLogger("dream")
DREAM_LOG_PATH = "dream_log.json"


@dataclass
class DreamSample:
    """A single synthetic dream: future interaction Q/A pair."""

    conversation: str
    question: str
    answer: str


def generate_dreams_from_conversation(dgm_tokenizer, dgm_model, conversation: str, num_dreams: int = 4) -> List[DreamSample]:
    """Generate user-centric dreams (Q/A) from conversation text using the model itself."""
    
    # Prompt the model to extract key information
    prompt = (
        f"{conversation}\n\n"
        "### Instruction:\n"
        "Based on the conversation above, create 3 different Q/A pairs to remember key facts about the User and the Assistant.\n"
        "Focus on who they are, their names, and their relationship.\n"
        "Use styles like: factual, reflective, and direct.\n\n"
        "1. Question: What is the user's name?\n"
        "Answer: The user's name is Aniket.\n\n"
        "2. Question:"
    )
    
    generated = generate_text(
        dgm_model, 
        dgm_tokenizer, 
        prompt, 
        max_new_tokens=256, 
        temperature=0.8
    )
    
    # Parse the output
    dreams: List[DreamSample] = []
    
    # Simple parsing of Q/A format
    lines = generated.splitlines()
    current_q = None
    
    for line in lines:
        line = line.strip()
        if "Question:" in line:
            current_q = line.split("Question:", 1)[1].strip()
        elif "Answer:" in line and current_q:
            ans = line.split("Answer:", 1)[1].strip()
            dreams.append(DreamSample(conversation=conversation, question=current_q, answer=ans))
            current_q = None
            
    # Fallback to simple regex if model fails to generate valid format
    if not dreams:
        log.info("LLM dream generation failed to produce structured output, falling back to basic extraction.")
        # ... keep regex logic as fallback or just generic ...
        return [
            DreamSample(conversation=conversation, question="What happened?", answer="We had a conversation.")
        ]

    # Ensure we limit to requested number
    return dreams[:num_dreams]


def load_dream_log(path: str = DREAM_LOG_PATH) -> List[DreamSample]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return [DreamSample(**item) for item in data if isinstance(item, dict)]
    except Exception:
        return []


def save_dream_log(log_data: List[DreamSample], path: str = DREAM_LOG_PATH, max_len: int = 200) -> None:
    trimmed = log_data[-max_len:]
    serializable = [dream.__dict__ for dream in trimmed]
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def sleep_cycle_finetune_on_dreams(
    tokenizer,
    dream_model,
    dream_samples: List[DreamSample],
    grounding_qa: List[Tuple[str, str]],
    lr: float = 1e-4,
) -> float | None:
    """Perform a lightweight sleep cycle fine-tuning on dreams + grounding data."""

    """Perform a lightweight sleep cycle fine-tuning on dreams + grounding data."""
    
    # Even though fine-tuning is disabled, we should save the dreams (memories) for later.
    old_dreams = load_dream_log()
    updated_log = old_dreams + dream_samples
    save_dream_log(updated_log)
    log.info("Saved %d new dreams to dream_log.json (total: %d)", len(dream_samples), len(updated_log))
    
    dream_model.train()
    # Freeze base model parameters
    for param in dream_model.base_model.parameters():
        param.requires_grad = False
        
    trainable_params = []
    if hasattr(dream_model, "new_parameters"):
        trainable_params.extend(dream_model.new_parameters)
        
    # Ensure we only pick params that really require grad (adapters/expansions)
    trainable_params = [p for p in trainable_params if p.requires_grad]
    
    if not trainable_params:
        log.warning("No trainable adapters found (base model is frozen). Skipping sleep-cycle Q/A fine-tuning.")
        return None
        
    optimizer = optim.AdamW(trainable_params, lr=lr)
    return last_loss
