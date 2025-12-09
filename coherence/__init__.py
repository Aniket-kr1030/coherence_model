"""
Coherence-oriented tiny world simulator modules.
"""

from .state import Entity, Event, ProposedUpdate, CoherenceTargets, InternalWorldState
from .rules import RuleEngine, DIAG_LABELS, CheckLabel
from .modeling import (
    load_model,
    load_dream_model,
    DreamCoherenceHead,
    DreamCoherenceModel,
    model_device,
)
from .generation import (
    build_state_prompt,
    build_event_prompt,
    parse_state_update_line,
    parse_event_update_line,
    parse_generated_line,
    generate_text,
    generate_line,
    predict_coherence,
)
from .dream import dream_step, dream_loop, initialize_world_with_model, run_dream_session
from .training import coherence_targets_from_rule, train_dream_coherence_head, run_coherence_training
from .dgm import DreamSample, generate_dreams_from_conversation, sleep_cycle_finetune_on_dreams
from .dgm import DreamSample, generate_dreams_from_conversation, sleep_cycle_finetune_on_dreams
from .chat import chat_with_model
from .data_augmentation import DataAugmenter, AugmentedSample, CorruptionType, generate_training_data
from .expansion import save_model_with_metadata, load_model_with_metadata

__all__ = [
    "Entity",
    "Event",
    "ProposedUpdate",
    "CoherenceTargets",
    "InternalWorldState",
    "RuleEngine",
    "DIAG_LABELS",
    "CheckLabel",
    "load_model",
    "load_dream_model",
    "DreamCoherenceHead",
    "DreamCoherenceModel",
    "model_device",
    "build_state_prompt",
    "build_event_prompt",
    "parse_state_update_line",
    "parse_event_update_line",
    "parse_generated_line",
    "generate_text",
    "generate_line",
    "predict_coherence",
    "dream_step",
    "dream_loop",
    "initialize_world_with_model",
    "run_dream_session",
    "coherence_targets_from_rule",
    "train_dream_coherence_head",
    "run_coherence_training",
    "train_on_datasets",
    "DreamSample",
    "generate_dreams_from_conversation",
    "sleep_cycle_finetune_on_dreams",
    "chat_with_model",
    "DataAugmenter",
    "AugmentedSample",
    "CorruptionType",
    "generate_training_data",
    "save_model_with_metadata",
    "load_model_with_metadata",
]
