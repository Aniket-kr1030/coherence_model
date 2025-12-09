from __future__ import annotations

import os
import json
from typing import Dict, List, Literal, Tuple, Optional, Set

from .state import InternalWorldState, ProposedUpdate

# Extended diagnostic labels with new temporal and physics checks
DIAG_LABELS: List[str] = [
    "unknown", "identity", "causality", "contradiction",
    "semantic", "relational", "temporal", "physics"
]
CheckLabel = Literal[
    "unknown", "identity", "causality", "contradiction",
    "semantic", "relational", "temporal", "physics"
]

# Default semantic clusters
DEFAULT_SEMANTIC_CLUSTERS: Dict[str, Set[str]] = {
    "animal": {"cat", "dog", "mouse", "wolf", "bear", "bird", "fish", "rabbit"},
    "object": {"rock", "tree", "car", "chair", "stone", "stick", "table", "book"},
    "number": {"number", "integer", "digit", "zero", "one", "two", "three"},
    "person": {"john", "mary", "alice", "bob", "human", "man", "woman", "child"},
    "place": {"forest", "city", "house", "room", "garden", "mountain", "river"},
    "vehicle": {"car", "truck", "bike", "plane", "boat", "train"},
}

# Default rule weights (1.0 = hard constraint, <1.0 = soft constraint)
DEFAULT_RULE_WEIGHTS: Dict[str, float] = {
    "unknown": 1.0,
    "identity": 1.0,
    "causality": 0.9,
    "contradiction": 1.0,
    "semantic": 0.7,
    "relational": 1.0,
    "temporal": 0.8,
    "physics": 0.6,
}

# Physics constraints: which entities can be at which location types
DEFAULT_LOCATION_CONSTRAINTS: Dict[str, Set[str]] = {
    "water": {"fish", "boat", "river"},
    "air": {"bird", "plane"},
    "ground": {"person", "animal", "car", "tree", "rock", "house"},
}

# Verbs that imply movement/location change
MOVEMENT_VERBS: Set[str] = {"goes", "moves", "travels", "walks", "runs", "flies", "swims", "drives", "enters", "leaves"}


def load_clusters_from_config(path: str) -> Optional[Dict[str, Set[str]]]:
    """Load semantic clusters from a JSON or YAML config file."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            if path.endswith(".json"):
                data = json.load(f)
            elif path.endswith((".yaml", ".yml")):
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    return None
            else:
                return None
        return {k: set(v) for k, v in data.items()}
    except Exception:
        return None


class RuleEngine:
    def __init__(
        self,
        semantic_clusters: Optional[Dict[str, Set[str]]] = None,
        rule_weights: Optional[Dict[str, float]] = None,
        location_constraints: Optional[Dict[str, Set[str]]] = None,
        config_path: Optional[str] = None,
        soft_threshold: float = 0.5,
    ) -> None:
        # Load from config file if provided
        if config_path:
            loaded = load_clusters_from_config(config_path)
            if loaded:
                semantic_clusters = loaded

        self.semantic_clusters = semantic_clusters or DEFAULT_SEMANTIC_CLUSTERS.copy()
        self.rule_weights = rule_weights or DEFAULT_RULE_WEIGHTS.copy()
        self.location_constraints = location_constraints or DEFAULT_LOCATION_CONSTRAINTS.copy()
        self.soft_threshold = soft_threshold

        # Track entity locations for physics checks
        self.entity_locations: Dict[str, str] = {}

    def add_semantic_cluster(self, cluster_name: str, members: Set[str]) -> None:
        """Add or update a semantic cluster."""
        if cluster_name in self.semantic_clusters:
            self.semantic_clusters[cluster_name].update(members)
        else:
            self.semantic_clusters[cluster_name] = members

    def set_rule_weight(self, rule_name: str, weight: float) -> None:
        """Set weight for a specific rule (0.0-1.0)."""
        if rule_name in self.rule_weights:
            self.rule_weights[rule_name] = max(0.0, min(1.0, weight))

    def check_known(self, state: InternalWorldState, update: ProposedUpdate) -> Tuple[bool, str]:
        if update.kind == "state" and update.name.lower() == "unknown":
            return False, "unknown: unnamed entity"
        if update.kind == "event":
            if not update.actor or update.actor.lower() == "unknown":
                return False, "unknown: missing actor"
            if not update.verb or update.verb.lower() == "unknown":
                return False, "unknown: missing verb"
        return True, ""

    def check_identity(self, state: InternalWorldState, update: ProposedUpdate) -> Tuple[bool, str]:
        if update.kind != "state":
            return True, ""
        entity = state.get_entity(update.name)
        if entity and update.new_type and entity.type != update.new_type and not update.allow_type_change:
            return False, "identity: type flip without transform"
        return True, ""

    def check_causality(self, state: InternalWorldState, update: ProposedUpdate) -> Tuple[bool, str]:
        if update.kind == "event":
            if update.actor and not state.get_entity(update.actor):
                return False, "causality: actor_missing"
            if update.target and not state.get_entity(update.target):
                creatable = {"buys", "finds", "builds", "creates", "makes", "spawns", "generates"}
                if (update.verb or "").lower() not in creatable:
                    return False, "causality: target_missing"
        return True, ""

    def check_contradiction(self, state: InternalWorldState, update: ProposedUpdate) -> Tuple[bool, str]:
        if update.kind != "state":
            return True, ""
        entity = state.get_entity(update.name)
        if not entity:
            return True, ""
        for k, v in update.attributes.items():
            current = entity.attributes.get(k)
            if current and current != v:
                if k.lower() == "age" and any(trigger in (update.note or "").lower() for trigger in ("one year later", "birthday", "promotion")):
                    continue
                if k.lower() == "age":
                    try:
                        if int(v) >= int(current):
                            continue
                    except ValueError:
                        pass
                if not update.allow_overwrite:
                    return False, f"contradiction: {k} overwrite"
        return True, ""

    def check_semantic(self, state: InternalWorldState, update: ProposedUpdate) -> Tuple[bool, str]:
        target_type = update.new_type
        name = update.name if update.kind == "state" else update.actor or ""
        entity = state.get_entity(name) if name else None
        current_type = entity.type if entity else None
        target_type = target_type or current_type
        if not target_type:
            return True, ""

        def clusters_for(label: str) -> List[str]:
            if not label:
                return []
            label_lower = label.lower()
            return [cluster for cluster, members in self.semantic_clusters.items() if label_lower in members]

        type_clusters = clusters_for(target_type) if target_type else []
        name_clusters = clusters_for(name) if name else []
        existing_clusters = clusters_for(current_type) if current_type else []
        allow_transform = any(word in (update.note or "").lower() for word in ("transform", "became", "reborn"))

        if entity and existing_clusters and type_clusters and existing_clusters[0] != type_clusters[0]:
            if not allow_transform:
                return False, "semantic: cluster jump"
        if name_clusters:
            name_cluster = name_clusters[0]
            if type_clusters and type_clusters[0] != name_cluster and not allow_transform:
                return False, "semantic: name/type mismatch"
        return True, ""

    def check_relational_consistency(self, state: InternalWorldState, update: ProposedUpdate) -> Tuple[bool, str]:
        relations = update.relations if update.kind == "state" else []
        actor = update.actor if update.kind == "event" else None
        target = update.target if update.kind == "event" else None

        for rel, tgt in relations:
            if update.name == tgt:
                return False, "relational: self-relation"
            if not tgt:
                return False, "relational: missing target"
        if actor and target and actor == target:
            return False, "relational: actor==target"
        return True, ""

    def check_temporal(self, state: InternalWorldState, update: ProposedUpdate) -> Tuple[bool, str]:
        """Check temporal consistency: events can't reference entities before they exist."""
        if update.kind != "event":
            return True, ""

        # Check if actor exists and was created before current time
        if update.actor:
            actor_entity = state.get_entity(update.actor)
            if actor_entity and actor_entity.created_at is not None:
                # Actor must have been created before current time step
                if actor_entity.created_at > state.time_step:
                    return False, "temporal: actor not yet created"

        # Check if target exists and was created before current time
        if update.target:
            target_entity = state.get_entity(update.target)
            if target_entity and target_entity.created_at is not None:
                # For non-creation verbs, target must exist
                creatable = {"buys", "finds", "builds", "creates", "makes", "spawns"}
                if (update.verb or "").lower() not in creatable:
                    if target_entity.created_at > state.time_step:
                        return False, "temporal: target not yet created"

        # Check for backward time references in note
        note_lower = (update.note or "").lower()
        if any(phrase in note_lower for phrase in ("before it existed", "before creation", "retroactively")):
            return False, "temporal: backward time reference"

        return True, ""

    def check_physics(self, state: InternalWorldState, update: ProposedUpdate) -> Tuple[bool, str]:
        """Check physics/location consistency: entities should be in valid locations."""
        if update.kind == "event":
            verb = (update.verb or "").lower()

            # Check for impossible movement
            if verb in MOVEMENT_VERBS and update.target:
                actor = update.actor
                target = update.target
                actor_entity = state.get_entity(actor) if actor else None
                target_entity = state.get_entity(target) if target else None

                # Check if actor type can be at target location type
                if actor_entity and target_entity:
                    actor_type = actor_entity.type.lower()
                    target_type = target_entity.type.lower()

                    # Fish can't go to non-water locations
                    if actor_type == "fish" and target_type not in {"river", "lake", "ocean", "pond", "water"}:
                        return False, "physics: fish can't go to non-water location"

                    # Ground entities can't fly without vehicle
                    if verb == "flies" and actor_type not in {"bird", "plane", "helicopter", "insect"}:
                        return False, "physics: non-flying entity can't fly"

            # Check for impossible actions
            impossible_actions = {
                ("rock", "speaks"): "physics: rocks can't speak",
                ("tree", "walks"): "physics: trees can't walk",
                ("dead", "runs"): "physics: dead entities can't run",
            }

            if update.actor:
                actor_entity = state.get_entity(update.actor)
                if actor_entity:
                    actor_type = actor_entity.type.lower()
                    status = actor_entity.attributes.get("status", "").lower()

                    for (entity_key, action), reason in impossible_actions.items():
                        if entity_key in (actor_type, status) and action in verb:
                            return False, reason

        # For state updates, check location validity
        if update.kind == "state":
            location = update.attributes.get("location", "").lower()
            if location:
                entity = state.get_entity(update.name)
                entity_type = (update.new_type or (entity.type if entity else "")).lower()

                # Track entity location
                self.entity_locations[update.name] = location

                # Check location constraints
                for loc_type, valid_entities in self.location_constraints.items():
                    if loc_type in location:
                        if entity_type and entity_type not in valid_entities:
                            # Check if any cluster contains this entity type
                            entity_clusters = [c for c, m in self.semantic_clusters.items() if entity_type in m]
                            valid_clusters = [c for c, m in self.semantic_clusters.items() if m & valid_entities]
                            if entity_clusters and valid_clusters and not (set(entity_clusters) & set(valid_clusters)):
                                return False, f"physics: {entity_type} can't be in {location}"

        return True, ""

    def aggregate_checks(self, state: InternalWorldState, update: ProposedUpdate) -> Tuple[bool, List[str]]:
        checks: List[Tuple[CheckLabel, callable]] = [
            ("unknown", self.check_known),
            ("identity", self.check_identity),
            ("causality", self.check_causality),
            ("contradiction", self.check_contradiction),
            ("semantic", self.check_semantic),
            ("relational", self.check_relational_consistency),
            ("temporal", self.check_temporal),
            ("physics", self.check_physics),
        ]
        failures: List[str] = []
        soft_failures: List[str] = []

        for label, fn in checks:
            ok, reason = fn(state, update)
            if not ok and reason:
                weight = self.rule_weights.get(label, 1.0)
                if weight >= 1.0:
                    # Hard constraint - always fails
                    failures.append(reason)
                elif weight >= self.soft_threshold:
                    # Soft constraint above threshold - treat as failure
                    failures.append(reason)
                else:
                    # Soft constraint below threshold - log but don't fail
                    soft_failures.append(f"[soft] {reason}")

        return (len(failures) == 0), failures + soft_failures

    def vet_update(self, state: InternalWorldState, update: ProposedUpdate) -> Tuple[bool, List[str]]:
        ok, reasons = self.aggregate_checks(state, update)
        return ok, reasons

    def get_rule_summary(self) -> Dict[str, any]:
        """Return a summary of current rule configuration."""
        return {
            "semantic_clusters": {k: list(v) for k, v in self.semantic_clusters.items()},
            "rule_weights": self.rule_weights.copy(),
            "location_constraints": {k: list(v) for k, v in self.location_constraints.items()},
            "soft_threshold": self.soft_threshold,
        }

