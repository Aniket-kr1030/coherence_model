from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import uuid
from collections import Counter


@dataclass
class Entity:
    id: str
    name: str
    type: str
    attributes: Dict[str, str] = field(default_factory=dict)
    relations: List[Tuple[str, str]] = field(default_factory=list)  # (relation, target)
    created_at: Optional[int] = None


@dataclass
class Event:
    id: str
    time: int
    actor: Optional[str]
    verb: str
    target: Optional[str]
    note: str = ""


@dataclass
class ProposedUpdate:
    kind: Literal["state", "event"] = "state"
    name: str = "unknown"  # for state updates
    new_type: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    relations: List[Tuple[str, str]] = field(default_factory=list)
    note: str = ""
    allow_type_change: bool = False
    allow_overwrite: bool = False
    actor: Optional[str] = None  # for event updates
    verb: Optional[str] = None
    target: Optional[str] = None


@dataclass
class CoherenceTargets:
    coherence: float
    diagnostics: Dict[str, int]


class InternalWorldState:
    def __init__(self) -> None:
        self.entities: Dict[str, Entity] = {}
        self.events: List[Event] = []
        self.time_step: int = 0
        self.coherence_score: float = 0.0
        self.accepted_updates: int = 0
        self.rejected_updates: int = 0
        self.rejection_reasons: Counter[str] = Counter()
        self._entity_seq: int = 0

    def _next_entity_id(self) -> str:
        self._entity_seq += 1
        return f"ent-{self._entity_seq}"

    def summary(self) -> str:
        parts = []
        for e in self.entities.values():
            attrs = ", ".join(f"{k}={v}" for k, v in e.attributes.items()) or "no-attrs"
            rels = ", ".join(f"{r}->{t}" for r, t in e.relations) or "no-relations"
            parts.append(f"{e.name}({e.type}) [{attrs}; {rels}]")
        return "; ".join(parts) or "empty"

    def debug_summary(self, recent_events: int = 5) -> str:
        events = self.events[-recent_events:]
        events_desc = "; ".join(f"{ev.time}:{ev.actor or '_'} {ev.verb}->{ev.target or '_'}" for ev in events)
        return f"Entities: {self.summary()} | Events: {events_desc or 'none'} | t={self.time_step} | coherence={self.coherence_score:.2f}"

    def get_entity(self, name: str) -> Optional[Entity]:
        return self.entities.get(name)

    def find_relations(self, name: str) -> List[Tuple[str, str]]:
        entity = self.entities.get(name)
        return list(entity.relations) if entity else []

    def add_event(self, event: Event) -> None:
        self.events.append(event)

    def snapshot(self) -> Dict[str, object]:
        return {
            "time_step": self.time_step,
            "coherence_score": self.coherence_score,
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type,
                    "attributes": e.attributes,
                    "relations": e.relations,
                    "created_at": e.created_at,
                }
                for e in self.entities.values()
            ],
            "events": [
                {"id": ev.id, "time": ev.time, "actor": ev.actor, "verb": ev.verb, "target": ev.target, "note": ev.note}
                for ev in self.events
            ],
            "accepted_updates": self.accepted_updates,
            "rejected_updates": self.rejected_updates,
            "rejection_reasons": dict(self.rejection_reasons),
        }

    def _ensure_entity(self, name: str, new_type: Optional[str], created_at: Optional[int]) -> Entity:
        entity = self.entities.get(name)
        if entity is None:
            entity = Entity(
                id=self._next_entity_id(),
                name=name,
                type=new_type or "unknown",
                created_at=created_at,
            )
            self.entities[name] = entity
        return entity

    def apply_update(self, update: ProposedUpdate) -> None:
        self.time_step += 1
        if update.kind == "state":
            entity = self._ensure_entity(update.name, update.new_type, created_at=self.time_step)
            if update.new_type:
                entity.type = update.new_type
            for k, v in update.attributes.items():
                entity.attributes[k] = v
            for rel in update.relations:
                if rel not in entity.relations:
                    entity.relations.append(rel)
        elif update.kind == "event":
            actor_name = update.actor or "unknown"
            actor_entity = self._ensure_entity(actor_name, None, created_at=self.time_step)
            for k, v in update.attributes.items():
                actor_entity.attributes[k] = v
            for rel in update.relations:
                if rel not in actor_entity.relations:
                    actor_entity.relations.append(rel)
            if update.target:
                self._ensure_entity(update.target, None, created_at=self.time_step)
                relation = (update.verb or "interacts_with", update.target)
                if relation not in actor_entity.relations:
                    actor_entity.relations.append(relation)
            event = Event(
                id=str(uuid.uuid4()),
                time=self.time_step,
                actor=update.actor,
                verb=update.verb or "does",
                target=update.target,
                note=update.note,
            )
            self.add_event(event)
        self.accepted_updates += 1
        # Coherence_score will be adjusted externally (e.g. in dream_step) based on
        # model-predicted coherence and rule satisfaction.

    def record_rejection(self, reasons: List[str]) -> None:
        self.rejected_updates += 1
        for reason in reasons:
            self.rejection_reasons[reason] += 1
        self.coherence_score -= 0.5 + 0.2 * len(reasons)

    def coherence_report(self) -> str:
        reason_parts = [f"{reason}:{count}" for reason, count in self.rejection_reasons.most_common()]
        return (
            f"coherence={self.coherence_score:.2f}; accepted={self.accepted_updates}; "
            f"rejected={self.rejected_updates}; reasons={' | '.join(reason_parts) or 'none'}"
        )
