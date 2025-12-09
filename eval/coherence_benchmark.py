"""
Coherence Benchmark Module

Standard benchmark for evaluating coherence model performance with:
- Hand-crafted coherent and incoherent test scenarios
- Automated scoring and reporting
- Comparison against rule engine baseline
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherence.state import InternalWorldState, ProposedUpdate, Entity
from coherence.rules import RuleEngine, DIAG_LABELS


class ScenarioCategory(Enum):
    """Categories of benchmark scenarios."""
    TEMPORAL = "temporal"
    PHYSICS = "physics"
    IDENTITY = "identity"
    CAUSALITY = "causality"
    CONTRADICTION = "contradiction"
    SEMANTIC = "semantic"
    RELATIONAL = "relational"
    COMPLEX = "complex"  # Multiple violations


@dataclass
class BenchmarkScenario:
    """A single benchmark test scenario."""
    id: str
    name: str
    description: str
    category: ScenarioCategory
    state: InternalWorldState
    update: ProposedUpdate
    expected_coherent: bool
    expected_violations: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class BenchmarkResult:
    """Result for a single scenario."""
    scenario_id: str
    predicted_coherent: bool
    expected_coherent: bool
    is_correct: bool
    predicted_violations: List[str]
    expected_violations: List[str]
    coherence_score: Optional[float] = None


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    total_scenarios: int
    correct: int
    accuracy: float
    results_by_category: Dict[str, Dict[str, float]]
    results: List[BenchmarkResult]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_scenarios": self.total_scenarios,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "results_by_category": self.results_by_category,
            "results": [
                {
                    "scenario_id": r.scenario_id,
                    "predicted_coherent": r.predicted_coherent,
                    "expected_coherent": r.expected_coherent,
                    "is_correct": r.is_correct,
                    "coherence_score": r.coherence_score,
                }
                for r in self.results
            ],
        }


def create_benchmark_scenarios() -> List[BenchmarkScenario]:
    """Create hand-crafted benchmark scenarios."""
    scenarios = []
    
    # === TEMPORAL SCENARIOS ===
    
    # Scenario 1: Valid temporal - event after entity creation
    state1 = InternalWorldState()
    state1.entities["cat"] = Entity(id="e1", name="cat", type="animal", created_at=1)
    state1.entities["mouse"] = Entity(id="e2", name="mouse", type="animal", created_at=2)
    state1.time_step = 5
    update1 = ProposedUpdate(kind="event", actor="cat", verb="chases", target="mouse", note="hungry cat")
    scenarios.append(BenchmarkScenario(
        id="temporal_001", name="Valid event after creation",
        description="Cat chases mouse - both entities exist before event",
        category=ScenarioCategory.TEMPORAL,
        state=state1, update=update1,
        expected_coherent=True, expected_violations=[],
        difficulty="easy"
    ))
    
    # Scenario 2: Invalid temporal - backward time reference
    state2 = InternalWorldState()
    state2.entities["cat"] = Entity(id="e1", name="cat", type="animal", created_at=5)
    state2.time_step = 3
    update2 = ProposedUpdate(kind="event", actor="cat", verb="chases", target="mouse", note="before it existed")
    scenarios.append(BenchmarkScenario(
        id="temporal_002", name="Backward time reference",
        description="Event references entity before it existed",
        category=ScenarioCategory.TEMPORAL,
        state=state2, update=update2,
        expected_coherent=False, expected_violations=["temporal"],
        difficulty="medium"
    ))
    
    # === PHYSICS SCENARIOS ===
    
    # Scenario 3: Invalid physics - rock speaks
    state3 = InternalWorldState()
    state3.entities["rock"] = Entity(id="e1", name="rock", type="object", created_at=1)
    state3.entities["alice"] = Entity(id="e2", name="alice", type="person", created_at=2)
    state3.time_step = 5
    update3 = ProposedUpdate(kind="event", actor="rock", verb="speaks", target="alice", note="impossible")
    scenarios.append(BenchmarkScenario(
        id="physics_001", name="Rock speaks",
        description="An inanimate rock cannot speak",
        category=ScenarioCategory.PHYSICS,
        state=state3, update=update3,
        expected_coherent=False, expected_violations=["physics"],
        difficulty="easy"
    ))
    
    # Scenario 4: Invalid physics - person flies without vehicle
    state4 = InternalWorldState()
    state4.entities["bob"] = Entity(id="e1", name="bob", type="person", created_at=1)
    state4.entities["city"] = Entity(id="e2", name="city", type="place", created_at=2)
    state4.time_step = 5
    update4 = ProposedUpdate(kind="event", actor="bob", verb="flies", target="city", note="to get there faster")
    scenarios.append(BenchmarkScenario(
        id="physics_002", name="Person flies",
        description="A person cannot fly without a vehicle",
        category=ScenarioCategory.PHYSICS,
        state=state4, update=update4,
        expected_coherent=False, expected_violations=["physics"],
        difficulty="medium"
    ))
    
    # Scenario 5: Valid physics - bird flies
    state5 = InternalWorldState()
    state5.entities["bird"] = Entity(id="e1", name="bird", type="bird", created_at=1)
    state5.entities["tree"] = Entity(id="e2", name="tree", type="object", created_at=2)
    state5.time_step = 5
    update5 = ProposedUpdate(kind="event", actor="bird", verb="flies", target="tree", note="to nest")
    scenarios.append(BenchmarkScenario(
        id="physics_003", name="Bird flies",
        description="A bird can fly to a tree",
        category=ScenarioCategory.PHYSICS,
        state=state5, update=update5,
        expected_coherent=True, expected_violations=[],
        difficulty="easy"
    ))
    
    # === IDENTITY SCENARIOS ===
    
    # Scenario 6: Invalid identity - type flip without transform
    state6 = InternalWorldState()
    state6.entities["cat"] = Entity(id="e1", name="cat", type="animal", created_at=1)
    state6.time_step = 5
    update6 = ProposedUpdate(kind="state", name="cat", new_type="person", note="sudden change",
                             allow_type_change=False)
    scenarios.append(BenchmarkScenario(
        id="identity_001", name="Type flip without transform",
        description="Cat becomes person without transformation context",
        category=ScenarioCategory.IDENTITY,
        state=state6, update=update6,
        expected_coherent=False, expected_violations=["identity"],
        difficulty="medium"
    ))
    
    # Scenario 7: Valid identity - type change with transform
    state7 = InternalWorldState()
    state7.entities["caterpillar"] = Entity(id="e1", name="caterpillar", type="insect", created_at=1)
    state7.time_step = 5
    update7 = ProposedUpdate(kind="state", name="caterpillar", new_type="butterfly", 
                             note="transformed through metamorphosis", allow_type_change=True)
    scenarios.append(BenchmarkScenario(
        id="identity_002", name="Valid transformation",
        description="Caterpillar transforms into butterfly with context",
        category=ScenarioCategory.IDENTITY,
        state=state7, update=update7,
        expected_coherent=True, expected_violations=[],
        difficulty="medium"
    ))
    
    # === CAUSALITY SCENARIOS ===
    
    # Scenario 8: Invalid causality - actor missing
    state8 = InternalWorldState()
    state8.entities["mouse"] = Entity(id="e1", name="mouse", type="animal", created_at=1)
    state8.time_step = 5
    update8 = ProposedUpdate(kind="event", actor="ghost", verb="scares", target="mouse", note="spooky")
    scenarios.append(BenchmarkScenario(
        id="causality_001", name="Missing actor",
        description="Event references non-existent actor",
        category=ScenarioCategory.CAUSALITY,
        state=state8, update=update8,
        expected_coherent=False, expected_violations=["causality"],
        difficulty="easy"
    ))
    
    # Scenario 9: Valid causality - creation verb
    state9 = InternalWorldState()
    state9.entities["bob"] = Entity(id="e1", name="bob", type="person", created_at=1)
    state9.time_step = 5
    update9 = ProposedUpdate(kind="event", actor="bob", verb="builds", target="house", note="new home")
    scenarios.append(BenchmarkScenario(
        id="causality_002", name="Valid creation",
        description="Bob builds a new house (creation verb)",
        category=ScenarioCategory.CAUSALITY,
        state=state9, update=update9,
        expected_coherent=True, expected_violations=[],
        difficulty="easy"
    ))
    
    # === CONTRADICTION SCENARIOS ===
    
    # Scenario 10: Invalid contradiction - age decrease
    state10 = InternalWorldState()
    state10.entities["alice"] = Entity(id="e1", name="alice", type="person", 
                                        attributes={"age": "30"}, created_at=1)
    state10.time_step = 5
    update10 = ProposedUpdate(kind="state", name="alice", attributes={"age": "25"},
                              note="time passed", allow_overwrite=False)
    scenarios.append(BenchmarkScenario(
        id="contradiction_001", name="Age decrease",
        description="Alice's age decreases without explanation",
        category=ScenarioCategory.CONTRADICTION,
        state=state10, update=update10,
        expected_coherent=False, expected_violations=["contradiction"],
        difficulty="medium"
    ))
    
    # Scenario 11: Valid age increase with birthday
    state11 = InternalWorldState()
    state11.entities["bob"] = Entity(id="e1", name="bob", type="person",
                                      attributes={"age": "30"}, created_at=1)
    state11.time_step = 5
    update11 = ProposedUpdate(kind="state", name="bob", attributes={"age": "31"},
                              note="one year later, birthday celebration", allow_overwrite=True)
    scenarios.append(BenchmarkScenario(
        id="contradiction_002", name="Valid birthday",
        description="Bob's age increases on birthday",
        category=ScenarioCategory.CONTRADICTION,
        state=state11, update=update11,
        expected_coherent=True, expected_violations=[],
        difficulty="easy"
    ))
    
    # === SEMANTIC SCENARIOS ===
    
    # Scenario 12: Invalid semantic - cluster jump
    state12 = InternalWorldState()
    state12.entities["cat"] = Entity(id="e1", name="cat", type="animal", created_at=1)
    state12.time_step = 5
    update12 = ProposedUpdate(kind="state", name="cat", new_type="number", 
                              note="regular update", allow_type_change=True)
    scenarios.append(BenchmarkScenario(
        id="semantic_001", name="Cluster jump",
        description="Cat becomes a number - incompatible semantic clusters",
        category=ScenarioCategory.SEMANTIC,
        state=state12, update=update12,
        expected_coherent=False, expected_violations=["semantic"],
        difficulty="hard"
    ))
    
    # === RELATIONAL SCENARIOS ===
    
    # Scenario 13: Invalid relational - self-relation
    state13 = InternalWorldState()
    state13.entities["cat"] = Entity(id="e1", name="cat", type="animal", created_at=1)
    state13.time_step = 5
    update13 = ProposedUpdate(kind="event", actor="cat", verb="chases", target="cat", note="confused")
    scenarios.append(BenchmarkScenario(
        id="relational_001", name="Self-relation",
        description="Cat chases itself",
        category=ScenarioCategory.RELATIONAL,
        state=state13, update=update13,
        expected_coherent=False, expected_violations=["relational"],
        difficulty="easy"
    ))
    
    # === COMPLEX SCENARIOS ===
    
    # Scenario 14: Multiple violations
    state14 = InternalWorldState()
    state14.time_step = 5
    update14 = ProposedUpdate(kind="event", actor="ghost", verb="speaks", target="ghost", 
                              note="before it existed")
    scenarios.append(BenchmarkScenario(
        id="complex_001", name="Multiple violations",
        description="Ghost speaks to itself before existing",
        category=ScenarioCategory.COMPLEX,
        state=state14, update=update14,
        expected_coherent=False, expected_violations=["causality", "temporal", "relational"],
        difficulty="hard"
    ))
    
    # Scenario 15: Valid complex scenario
    state15 = InternalWorldState()
    state15.entities["alice"] = Entity(id="e1", name="alice", type="person", 
                                        attributes={"mood": "happy"}, created_at=1)
    state15.entities["bob"] = Entity(id="e2", name="bob", type="person",
                                      attributes={"mood": "calm"}, created_at=2)
    state15.entities["garden"] = Entity(id="e3", name="garden", type="place", created_at=3)
    state15.time_step = 10
    update15 = ProposedUpdate(kind="event", actor="alice", verb="walks", target="garden",
                              note="to meet bob for lunch")
    scenarios.append(BenchmarkScenario(
        id="complex_002", name="Valid complex event",
        description="Alice walks to garden - all entities exist, valid action",
        category=ScenarioCategory.COMPLEX,
        state=state15, update=update15,
        expected_coherent=True, expected_violations=[],
        difficulty="medium"
    ))
    
    return scenarios


class CoherenceBenchmark:
    """Benchmark runner for coherence evaluation."""
    
    def __init__(self, rule_engine: Optional[RuleEngine] = None):
        self.rule_engine = rule_engine or RuleEngine()
        self.scenarios = create_benchmark_scenarios()
    
    def run_rule_engine_baseline(self) -> BenchmarkReport:
        """Run benchmark using rule engine as baseline."""
        results = []
        
        for scenario in self.scenarios:
            ok, reasons = self.rule_engine.aggregate_checks(scenario.state, scenario.update)
            
            result = BenchmarkResult(
                scenario_id=scenario.id,
                predicted_coherent=ok,
                expected_coherent=scenario.expected_coherent,
                is_correct=(ok == scenario.expected_coherent),
                predicted_violations=[r.split(":")[0] for r in reasons if r],
                expected_violations=scenario.expected_violations,
            )
            results.append(result)
        
        return self._create_report(results)
    
    def run_model_benchmark(
        self,
        tokenizer,
        model,
        threshold: float = 0.5,
    ) -> BenchmarkReport:
        """Run benchmark using a coherence model."""
        from coherence.generation import predict_coherence
        
        results = []
        
        for scenario in self.scenarios:
            # Build prompt
            prompt = f"World: {scenario.state.summary()}"
            raw_line = self._update_to_text(scenario.update)
            
            # Get model prediction
            coherence_score = predict_coherence(tokenizer, model, prompt, raw_line)
            
            if coherence_score is not None:
                predicted_coherent = coherence_score >= threshold
            else:
                # Fallback to rule engine
                ok, _ = self.rule_engine.aggregate_checks(scenario.state, scenario.update)
                predicted_coherent = ok
                coherence_score = 1.0 if ok else 0.0
            
            result = BenchmarkResult(
                scenario_id=scenario.id,
                predicted_coherent=predicted_coherent,
                expected_coherent=scenario.expected_coherent,
                is_correct=(predicted_coherent == scenario.expected_coherent),
                predicted_violations=[],  # Model doesn't provide diagnostics
                expected_violations=scenario.expected_violations,
                coherence_score=coherence_score,
            )
            results.append(result)
        
        return self._create_report(results)
    
    def _create_report(self, results: List[BenchmarkResult]) -> BenchmarkReport:
        """Create a benchmark report from results."""
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / len(results) if results else 0.0
        
        # Results by category
        results_by_category = {}
        for scenario in self.scenarios:
            cat = scenario.category.value
            if cat not in results_by_category:
                results_by_category[cat] = {"total": 0, "correct": 0}
            results_by_category[cat]["total"] += 1
            
            result = next((r for r in results if r.scenario_id == scenario.id), None)
            if result and result.is_correct:
                results_by_category[cat]["correct"] += 1
        
        # Calculate category accuracies
        for cat in results_by_category:
            total = results_by_category[cat]["total"]
            cor = results_by_category[cat]["correct"]
            results_by_category[cat]["accuracy"] = cor / total if total > 0 else 0.0
        
        return BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            total_scenarios=len(results),
            correct=correct,
            accuracy=accuracy,
            results_by_category=results_by_category,
            results=results,
        )
    
    def _update_to_text(self, update: ProposedUpdate) -> str:
        """Convert update to text."""
        if update.kind == "state":
            attrs = ", ".join(f"{k}:{v}" for k, v in update.attributes.items()) or "none"
            return f"state: entity={update.name}; type={update.new_type}; attr={attrs}; note={update.note}"
        else:
            return f"event: actor={update.actor}; verb={update.verb}; target={update.target}; note={update.note}"
    
    def print_report(self, report: BenchmarkReport) -> None:
        """Print a formatted benchmark report."""
        print("\n" + "=" * 70)
        print("COHERENCE BENCHMARK REPORT")
        print("=" * 70)
        print(f"Timestamp: {report.timestamp}")
        print(f"Total Scenarios: {report.total_scenarios}")
        print(f"Correct: {report.correct}")
        print(f"Accuracy: {report.accuracy:.2%}")
        
        print("\nResults by Category:")
        print("-" * 40)
        for cat, stats in sorted(report.results_by_category.items()):
            print(f"  {cat}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")
        
        print("\nFailed Scenarios:")
        print("-" * 40)
        for result in report.results:
            if not result.is_correct:
                scenario = next((s for s in self.scenarios if s.id == result.scenario_id), None)
                if scenario:
                    print(f"  [{scenario.id}] {scenario.name}")
                    print(f"    Expected: {'coherent' if result.expected_coherent else 'incoherent'}")
                    print(f"    Predicted: {'coherent' if result.predicted_coherent else 'incoherent'}")
                    if result.coherence_score is not None:
                        print(f"    Score: {result.coherence_score:.3f}")
        
        print("=" * 70 + "\n")
    
    def save_report(self, report: BenchmarkReport, path: str = "benchmark_report.json") -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report saved to {path}")


def run_benchmark(model_path: Optional[str] = None) -> BenchmarkReport:
    """Run the coherence benchmark."""
    benchmark = CoherenceBenchmark()
    
    if model_path:
        # Load model and run model benchmark
        from coherence import load_dream_model
        tokenizer, model = load_dream_model()
        report = benchmark.run_model_benchmark(tokenizer, model)
    else:
        # Run rule engine baseline
        report = benchmark.run_rule_engine_baseline()
    
    benchmark.print_report(report)
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run coherence benchmark")
    parser.add_argument("--model", type=str, help="Path to model weights")
    parser.add_argument("--output", type=str, default="benchmark_report.json",
                       help="Output path for report")
    args = parser.parse_args()
    
    report = run_benchmark(model_path=args.model)
    
    benchmark = CoherenceBenchmark()
    benchmark.save_report(report, args.output)
