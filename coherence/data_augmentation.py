"""
Data augmentation utilities for coherence training.

Generates synthetic coherence training data by:
- Creating positive samples from valid world state transitions
- Corrupting positive samples to create negative samples
- Paraphrasing samples for diversity
"""

from __future__ import annotations

import random
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum

from .state import InternalWorldState, ProposedUpdate, Entity, Event
from .rules import RuleEngine, DIAG_LABELS


class CorruptionType(Enum):
    """Types of corruptions that can be applied to create negative samples."""
    TEMPORAL = "temporal"          # Violate temporal consistency
    PHYSICS = "physics"            # Violate physics constraints
    IDENTITY = "identity"          # Type flip without transform
    CAUSALITY = "causality"        # Reference non-existent entities
    CONTRADICTION = "contradiction" # Overwrite attributes
    SEMANTIC = "semantic"          # Cluster jump
    RELATIONAL = "relational"      # Self-relation or missing target


@dataclass
class AugmentedSample:
    """A single augmented training sample."""
    prompt: str
    update: ProposedUpdate
    is_coherent: bool
    corruption_type: Optional[CorruptionType] = None
    original_update: Optional[ProposedUpdate] = None
    diagnostics: Dict[str, int] = field(default_factory=dict)


class DataAugmenter:
    """Generates augmented training data for coherence model."""
    
    def __init__(
        self,
        rule_engine: Optional[RuleEngine] = None,
        seed: Optional[int] = None,
    ):
        self.rule_engine = rule_engine or RuleEngine()
        if seed is not None:
            random.seed(seed)
        
        # Import extensive vocabulary
        from .vocabulary import ENTITY_VOCABULARY, ACTION_VOCABULARY, TYPE_ATTRIBUTES
        self.entity_vocab = ENTITY_VOCABULARY
        self.action_vocab = ACTION_VOCABULARY
        self.type_attributes = TYPE_ATTRIBUTES
        
        # Flat list of all available entities for sampling
        self.all_entity_types = list(self.entity_vocab.keys())
        
        # update rule engine clusters if possible
        self._update_rule_engine_clusters()
        
    def _update_rule_engine_clusters(self):
        """Update the rule engine's semantic clusters with our expanded vocabulary."""
        for type_name, names in self.entity_vocab.items():
            # In RuleEngine, clusters are often just the type name
            # We add all nouns to the corresponding cluster
            self.rule_engine.add_semantic_cluster(type_name, set(names))

    def _generate_random_entity(self, state: InternalWorldState) -> Entity:
        """Generate a random entity with appropriate attributes."""
        type_name = random.choice(self.all_entity_types)
        name = random.choice(self.entity_vocab[type_name])
        
        # Ensure unique name
        base_name = name
        counter = 1
        while name in state.entities:
            name = f"{base_name}_{counter}"
            counter += 1
            
        # Generate attributes
        attrs = {}
        if type_name in self.type_attributes:
            for attr_name, possible_values in self.type_attributes[type_name].items():
                if random.random() < 0.7:  # 70% chance to have an attribute
                    attrs[attr_name] = random.choice(possible_values)
        
        return Entity(
            id=state._next_entity_id(),
            name=name,
            type=type_name,
            attributes=attrs,
            relations=[],
            created_at=state.time_step,
        )
    
    def generate_valid_state(self) -> Tuple[InternalWorldState, ProposedUpdate]:
        """Generate a valid world state and a coherent update."""
        state = InternalWorldState()
        
        # Add 3-5 initial entities
        num_entities = random.randint(3, 5)
        
        for _ in range(num_entities):
            entity = self._generate_random_entity(state)
            state.entities[entity.name] = entity
            state.time_step += 1
        
        # Generate a valid state update
        existing = random.choice(list(state.entities.values()))
        new_attrs = {}
        
        # Try to modify an existing attribute, or add a new one from valid attributes
        if existing.type in self.type_attributes and self.type_attributes[existing.type]:
            attr_options = self.type_attributes[existing.type]
            attr_name = random.choice(list(attr_options.keys()))
            attr_val = random.choice(attr_options[attr_name])
            new_attrs[attr_name] = attr_val
        else:
            new_attrs["status"] = "changed"
        
        update = ProposedUpdate(
            kind="state",
            name=existing.name,
            new_type=existing.type,  # Keep same type for basic update
            attributes=new_attrs,
            note="natural state change",
        )
        
        return state, update
    
    def generate_valid_event(self) -> Tuple[InternalWorldState, ProposedUpdate]:
        """Generate a valid world state and a coherent event update."""
        state = InternalWorldState()
        
        # Add entities - ensure we generate enough for various types
        num_entities = random.randint(4, 7)
        for _ in range(num_entities):
            entity = self._generate_random_entity(state)
            state.entities[entity.name] = entity
            state.time_step += 1
        
        # Find valid actor-action-target combination
        entities = list(state.entities.values())
        random.shuffle(self.action_vocab)
        
        for action in self.action_vocab:
            valid_actors = [e for e in entities if e.type in action["valid_actors"]]
            valid_targets = [e for e in entities if e.type in action["valid_targets"]]
            
            if valid_actors and valid_targets:
                actor = random.choice(valid_actors)
                # Exclude actor from targets
                available_targets = [t for t in valid_targets if t != actor]
                if available_targets:
                    target = random.choice(available_targets)
                    update = ProposedUpdate(
                        kind="event",
                        actor=actor.name,
                        verb=action["verb"],
                        target=target.name,
                        note="natural event",
                    )
                    return state, update
        
        # Fallback should be rare with big vocab, but just in case
        actor = random.choice(entities)
        # Find any other entity
        others = [e for e in entities if e != actor]
        if others:
            target = random.choice(others)
        else:
            # Create a target if none exist (shouldn't happen with num_entities >= 4)
            target = self._generate_random_entity(state)
            state.entities[target.name] = target
        
        update = ProposedUpdate(
            kind="event",
            actor=actor.name,
            verb="observes",
            target=target.name,
            note="fallback event",
        )
        
        return state, update
    
    def corrupt_temporal(
        self, state: InternalWorldState, update: ProposedUpdate
    ) -> ProposedUpdate:
        """Create a temporal violation by referencing future entities."""
        corrupted = copy.deepcopy(update)
        
        # Reference an entity that doesn't exist yet
        corrupted.note = "before it existed"
        
        # Or reference a non-existent entity in event
        if corrupted.kind == "event":
            corrupted.target = "future_entity"
        
        return corrupted
    
    def corrupt_physics(
        self, state: InternalWorldState, update: ProposedUpdate
    ) -> ProposedUpdate:
        """Create a physics violation."""
        corrupted = copy.deepcopy(update)
        
        if corrupted.kind == "event":
            # Make an impossible action
            if state.entities:
                # Try to make a rock speak or tree walk
                for entity in state.entities.values():
                    if entity.type == "object":
                        corrupted.actor = entity.name
                        corrupted.verb = "speaks"
                        corrupted.note = "impossible action"
                        break
                else:
                    # Make entity fly without ability
                    corrupted.verb = "flies"
                    corrupted.note = "physics violation"
        
        return corrupted
    
    def corrupt_identity(
        self, state: InternalWorldState, update: ProposedUpdate
    ) -> ProposedUpdate:
        """Create an identity violation by changing type without transform."""
        corrupted = copy.deepcopy(update)
        
        if corrupted.kind == "state" and state.entities.get(corrupted.name):
            entity = state.entities[corrupted.name]
            # Change type without transform keyword
            if entity.type == "animal":
                corrupted.new_type = "person"
            elif entity.type == "person":
                corrupted.new_type = "animal"
            else:
                corrupted.new_type = "animal"
            corrupted.allow_type_change = False
            corrupted.note = "sudden change"
        
        return corrupted
    
    def corrupt_causality(
        self, state: InternalWorldState, update: ProposedUpdate
    ) -> ProposedUpdate:
        """Create a causality violation by referencing non-existent entities."""
        corrupted = copy.deepcopy(update)
        
        if corrupted.kind == "event":
            # Reference non-existent actor or target
            corrupted.actor = "nonexistent_actor"
            corrupted.target = "nonexistent_target"
            corrupted.verb = "attacks"  # Not a creation verb
        
        return corrupted
    
    def corrupt_contradiction(
        self, state: InternalWorldState, update: ProposedUpdate
    ) -> ProposedUpdate:
        """Create a contradiction by overwriting attributes."""
        corrupted = copy.deepcopy(update)
        
        if corrupted.kind == "state" and state.entities.get(corrupted.name):
            entity = state.entities[corrupted.name]
            # Overwrite existing attribute with contradictory value
            if "age" in entity.attributes:
                try:
                    current_age = int(entity.attributes["age"])
                    corrupted.attributes["age"] = str(current_age - 10)  # Age decrease
                except ValueError:
                    corrupted.attributes["age"] = "0"
            elif entity.attributes:
                key = list(entity.attributes.keys())[0]
                corrupted.attributes[key] = "CONTRADICTORY_VALUE"
            corrupted.allow_overwrite = False
        
        return corrupted
    
    def corrupt_semantic(
        self, state: InternalWorldState, update: ProposedUpdate
    ) -> ProposedUpdate:
        """Create a semantic violation with cluster jump."""
        corrupted = copy.deepcopy(update)
        
        if corrupted.kind == "state":
            # Jump between incompatible clusters
            name = corrupted.name.lower()
            current_type = corrupted.new_type
            
            # Pick a type that is definitely different from current
            available_types = [t for t in self.all_entity_types if t != current_type]
            if available_types:
                corrupted.new_type = random.choice(available_types)
            else:
                corrupted.new_type = "unknown_type"
                
            corrupted.note = "regular update"  # No transform keyword
        
        return corrupted
    
    def corrupt_relational(
        self, state: InternalWorldState, update: ProposedUpdate
    ) -> ProposedUpdate:
        """Create a relational violation."""
        corrupted = copy.deepcopy(update)
        
        if corrupted.kind == "event":
            # Actor equals target
            if corrupted.actor:
                corrupted.target = corrupted.actor
        elif corrupted.kind == "state":
            # Self-relation
            corrupted.relations = [(corrupted.name, corrupted.name)]
        
        return corrupted
    
    def generate_corrupted_sample(
        self,
        corruption_type: CorruptionType,
    ) -> AugmentedSample:
        """Generate a single corrupted (negative) sample."""
        # Generate base valid state
        if random.random() < 0.5:
            state, original_update = self.generate_valid_state()
        else:
            state, original_update = self.generate_valid_event()
        
        # Apply corruption
        corruption_map = {
            CorruptionType.TEMPORAL: self.corrupt_temporal,
            CorruptionType.PHYSICS: self.corrupt_physics,
            CorruptionType.IDENTITY: self.corrupt_identity,
            CorruptionType.CAUSALITY: self.corrupt_causality,
            CorruptionType.CONTRADICTION: self.corrupt_contradiction,
            CorruptionType.SEMANTIC: self.corrupt_semantic,
            CorruptionType.RELATIONAL: self.corrupt_relational,
        }
        
        corrupted_update = corruption_map[corruption_type](state, original_update)
        
        # Build prompt
        prompt = f"World: {state.summary()}\nProposed: {self._update_to_text(corrupted_update)}"
        
        # Get diagnostics from rule engine
        ok, reasons = self.rule_engine.aggregate_checks(state, corrupted_update)
        diagnostics = {label: 0 for label in DIAG_LABELS}
        for reason in reasons:
            for label in DIAG_LABELS:
                if label in reason.lower():
                    diagnostics[label] = 1
                    break
        
        return AugmentedSample(
            prompt=prompt,
            update=corrupted_update,
            is_coherent=False,
            corruption_type=corruption_type,
            original_update=original_update,
            diagnostics=diagnostics,
        )
    
    def generate_positive_sample(self) -> AugmentedSample:
        """Generate a single positive (coherent) sample."""
        if random.random() < 0.5:
            state, update = self.generate_valid_state()
        else:
            state, update = self.generate_valid_event()
        
        prompt = f"World: {state.summary()}\nProposed: {self._update_to_text(update)}"
        
        # Verify it's actually valid
        ok, reasons = self.rule_engine.aggregate_checks(state, update)
        
        diagnostics = {label: 0 for label in DIAG_LABELS}
        
        return AugmentedSample(
            prompt=prompt,
            update=update,
            is_coherent=ok,
            corruption_type=None,
            original_update=None,
            diagnostics=diagnostics,
        )
    
    def generate_batch(
        self,
        batch_size: int = 32,
        positive_ratio: float = 0.5,
    ) -> List[AugmentedSample]:
        """Generate a batch of augmented samples."""
        samples = []
        num_positive = int(batch_size * positive_ratio)
        num_negative = batch_size - num_positive
        
        # Generate positive samples
        for _ in range(num_positive):
            samples.append(self.generate_positive_sample())
        
        # Generate negative samples with different corruption types
        corruption_types = list(CorruptionType)
        for i in range(num_negative):
            corruption = corruption_types[i % len(corruption_types)]
            samples.append(self.generate_corrupted_sample(corruption))
        
        random.shuffle(samples)
        return samples
    
    def _update_to_text(self, update: ProposedUpdate) -> str:
        """Convert update to text representation."""
        if update.kind == "state":
            attrs = ", ".join(f"{k}:{v}" for k, v in update.attributes.items()) or "none"
            rels = ", ".join(f"{r}:{t}" for r, t in update.relations) or "none"
            return f"state: entity={update.name}; type={update.new_type}; attr={attrs}; relation={rels}; note={update.note}"
        else:
            return f"event: actor={update.actor}; verb={update.verb}; target={update.target}; note={update.note}"
    
    def paraphrase_update(self, update: ProposedUpdate) -> ProposedUpdate:
        """Create a paraphrased version of the update for diversity."""
        paraphrased = copy.deepcopy(update)
        
        # Synonym mappings for common verbs
        verb_synonyms = {
            "observes": ["watches", "looks at", "sees", "notices", "spots", "witnesses", "spies"],
            "chases": ["pursues", "runs after", "follows", "hunts", "tracks", "tails"],
            "walks_to": ["moves to", "goes to", "travels to", "strolls to", "wanders to", "marches to"],
            "runs_to": ["Sprints to", "dashes to", "rushes to", "hurries to"],
            "speaks_to": ["talks to", "says to", "tells", "chats with", "whispers to", "shouts at"],
            "attacks": ["fights", "assaults", "strikes", "hits", "engages"],
            "eats": ["consumes", "devours", "swallows", "nibbles", "feeds on"],
            "takes": ["grabs", "seizes", "picks up", "snatches", "collects"],
        }
        
        if paraphrased.verb and paraphrased.verb in verb_synonyms:
            paraphrased.verb = random.choice(verb_synonyms[paraphrased.verb])
        
        # Vary note phrasing
        note_templates = [
            "because {reason}",
            "as a result of {reason}",
            "due to {reason}",
            "{reason} happened",
        ]
        
        if paraphrased.note and "because" not in paraphrased.note.lower():
            template = random.choice(note_templates)
            paraphrased.note = template.format(reason=paraphrased.note)
        
        return paraphrased


def generate_training_data(
    num_samples: int = 1000,
    positive_ratio: float = 0.5,
    seed: Optional[int] = 42,
) -> List[AugmentedSample]:
    """
    Generate a full training dataset.
    
    Args:
        num_samples: Total number of samples to generate
        positive_ratio: Ratio of positive (coherent) samples
        seed: Random seed for reproducibility
        
    Returns:
        List of augmented samples
    """
    augmenter = DataAugmenter(seed=seed)
    return augmenter.generate_batch(batch_size=num_samples, positive_ratio=positive_ratio)
