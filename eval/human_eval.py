"""
Human Evaluation Tools for Coherence Model

Tools for:
- Generating samples for human evaluation
- Collecting and storing human ratings
- Comparing model predictions with human judgments
- Calculating inter-annotator agreement
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherence.state import InternalWorldState, ProposedUpdate, Entity
from coherence.rules import RuleEngine
from coherence.data_augmentation import DataAugmenter, AugmentedSample


@dataclass
class HumanEvalSample:
    """A sample prepared for human evaluation."""
    id: str
    world_state: str
    proposed_update: str
    context: str
    
    # Hidden from annotator
    rule_engine_verdict: Optional[bool] = None
    model_prediction: Optional[float] = None
    corruption_type: Optional[str] = None
    
    # Human annotations
    human_ratings: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class HumanRating:
    """A single human rating for a sample."""
    sample_id: str
    annotator_id: str
    is_coherent: bool
    confidence: int  # 1-5 scale
    reasoning: str
    timestamp: str
    violations_identified: List[str] = field(default_factory=list)


@dataclass
class EvaluationSession:
    """A human evaluation session."""
    session_id: str
    created_at: str
    samples: List[HumanEvalSample]
    ratings: List[HumanRating] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HumanEvaluator:
    """Manager for human evaluation tasks."""
    
    def __init__(
        self,
        output_dir: str = "eval_sessions",
        rule_engine: Optional[RuleEngine] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rule_engine = rule_engine or RuleEngine()
        self.augmenter = DataAugmenter(rule_engine=self.rule_engine)
    
    def generate_eval_samples(
        self,
        num_samples: int = 20,
        include_model_predictions: bool = False,
        tokenizer=None,
        model=None,
    ) -> List[HumanEvalSample]:
        """Generate samples for human evaluation."""
        samples = []
        
        # Generate a mix of positive and negative samples
        augmented = self.augmenter.generate_batch(
            batch_size=num_samples,
            positive_ratio=0.5,
        )
        
        for i, aug_sample in enumerate(augmented):
            sample = HumanEvalSample(
                id=f"eval_{i:04d}",
                world_state=self._extract_world_state(aug_sample.prompt),
                proposed_update=self._extract_update(aug_sample.prompt),
                context=self._generate_context(aug_sample),
                rule_engine_verdict=aug_sample.is_coherent,
                corruption_type=aug_sample.corruption_type.value if aug_sample.corruption_type else None,
            )
            
            # Add model predictions if requested
            if include_model_predictions and tokenizer and model:
                from coherence.generation import predict_coherence
                score = predict_coherence(
                    tokenizer, model,
                    sample.world_state,
                    sample.proposed_update,
                )
                sample.model_prediction = score
            
            samples.append(sample)
        
        return samples
    
    def create_session(
        self,
        num_samples: int = 20,
        session_name: Optional[str] = None,
        **kwargs,
    ) -> EvaluationSession:
        """Create a new evaluation session."""
        session_id = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        samples = self.generate_eval_samples(num_samples, **kwargs)
        
        session = EvaluationSession(
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            samples=samples,
            metadata={
                "num_samples": num_samples,
                "generation_params": kwargs,
            },
        )
        
        self.save_session(session)
        return session
    
    def save_session(self, session: EvaluationSession) -> str:
        """Save session to disk."""
        path = self.output_dir / f"{session.session_id}.json"
        
        # Convert to serializable format
        data = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "samples": [asdict(s) for s in session.samples],
            "ratings": [asdict(r) for r in session.ratings],
            "metadata": session.metadata,
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        return str(path)
    
    def load_session(self, session_id: str) -> EvaluationSession:
        """Load session from disk."""
        path = self.output_dir / f"{session_id}.json"
        
        with open(path, "r") as f:
            data = json.load(f)
        
        samples = [HumanEvalSample(**s) for s in data["samples"]]
        ratings = [HumanRating(**r) for r in data.get("ratings", [])]
        
        return EvaluationSession(
            session_id=data["session_id"],
            created_at=data["created_at"],
            samples=samples,
            ratings=ratings,
            metadata=data.get("metadata", {}),
        )
    
    def add_rating(
        self,
        session: EvaluationSession,
        sample_id: str,
        annotator_id: str,
        is_coherent: bool,
        confidence: int,
        reasoning: str,
        violations: Optional[List[str]] = None,
    ) -> HumanRating:
        """Add a human rating to a session."""
        rating = HumanRating(
            sample_id=sample_id,
            annotator_id=annotator_id,
            is_coherent=is_coherent,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now().isoformat(),
            violations_identified=violations or [],
        )
        
        session.ratings.append(rating)
        
        # Also add to the sample's ratings
        for sample in session.samples:
            if sample.id == sample_id:
                sample.human_ratings.append(asdict(rating))
                break
        
        self.save_session(session)
        return rating
    
    def export_for_annotation(
        self,
        session: EvaluationSession,
        output_path: Optional[str] = None,
    ) -> str:
        """Export session in a format suitable for annotation (hides ground truth)."""
        output_path = output_path or str(self.output_dir / f"{session.session_id}_annotation.json")
        
        annotation_data = {
            "session_id": session.session_id,
            "instructions": self._get_annotation_instructions(),
            "samples": [],
        }
        
        for sample in session.samples:
            annotation_data["samples"].append({
                "id": sample.id,
                "world_state": sample.world_state,
                "proposed_update": sample.proposed_update,
                "context": sample.context,
                # Explicitly exclude ground truth
            })
        
        with open(output_path, "w") as f:
            json.dump(annotation_data, f, indent=2)
        
        return output_path
    
    def compute_agreement(self, session: EvaluationSession) -> Dict[str, float]:
        """Compute inter-annotator agreement metrics."""
        # Group ratings by sample
        ratings_by_sample: Dict[str, List[HumanRating]] = {}
        for rating in session.ratings:
            if rating.sample_id not in ratings_by_sample:
                ratings_by_sample[rating.sample_id] = []
            ratings_by_sample[rating.sample_id].append(rating)
        
        # Compute pairwise agreement
        agreements = []
        for sample_id, ratings in ratings_by_sample.items():
            if len(ratings) >= 2:
                # Check if all annotators agree
                verdicts = [r.is_coherent for r in ratings]
                agreement = 1.0 if len(set(verdicts)) == 1 else 0.0
                agreements.append(agreement)
        
        # Simple percent agreement
        percent_agreement = sum(agreements) / len(agreements) if agreements else 0.0
        
        return {
            "percent_agreement": percent_agreement,
            "samples_with_multiple_ratings": len([r for r in ratings_by_sample.values() if len(r) >= 2]),
            "total_ratings": len(session.ratings),
        }
    
    def compare_with_model(self, session: EvaluationSession, threshold: float = 0.5) -> Dict[str, Any]:
        """Compare human ratings with model predictions."""
        comparisons = []
        
        for sample in session.samples:
            if not sample.human_ratings or sample.model_prediction is None:
                continue
            
            # Aggregate human ratings (majority vote)
            human_verdicts = [r["is_coherent"] for r in sample.human_ratings]
            human_majority = sum(human_verdicts) > len(human_verdicts) / 2
            
            model_verdict = sample.model_prediction >= threshold
            
            comparisons.append({
                "sample_id": sample.id,
                "human_verdict": human_majority,
                "model_verdict": model_verdict,
                "model_score": sample.model_prediction,
                "agree": human_majority == model_verdict,
            })
        
        if not comparisons:
            return {"error": "No samples with both human ratings and model predictions"}
        
        agreement_rate = sum(1 for c in comparisons if c["agree"]) / len(comparisons)
        
        return {
            "total_compared": len(comparisons),
            "agreement_rate": agreement_rate,
            "comparisons": comparisons,
        }
    
    def generate_report(self, session: EvaluationSession) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        report = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "report_generated_at": datetime.now().isoformat(),
            "summary": {
                "total_samples": len(session.samples),
                "total_ratings": len(session.ratings),
                "unique_annotators": len(set(r.annotator_id for r in session.ratings)),
            },
        }
        
        # Human agreement
        if session.ratings:
            report["inter_annotator_agreement"] = self.compute_agreement(session)
        
        # Human vs Rule Engine
        rule_correct = 0
        rule_total = 0
        for sample in session.samples:
            if sample.human_ratings and sample.rule_engine_verdict is not None:
                human_verdicts = [r["is_coherent"] for r in sample.human_ratings]
                human_majority = sum(human_verdicts) > len(human_verdicts) / 2
                if human_majority == sample.rule_engine_verdict:
                    rule_correct += 1
                rule_total += 1
        
        if rule_total > 0:
            report["human_vs_rule_engine"] = {
                "agreement_rate": rule_correct / rule_total,
                "samples_compared": rule_total,
            }
        
        # Human vs Model
        model_comparison = self.compare_with_model(session)
        if "error" not in model_comparison:
            report["human_vs_model"] = model_comparison
        
        # Confidence distribution
        if session.ratings:
            confidences = [r.confidence for r in session.ratings]
            report["confidence_distribution"] = {
                "mean": sum(confidences) / len(confidences),
                "min": min(confidences),
                "max": max(confidences),
            }
        
        return report
    
    def _extract_world_state(self, prompt: str) -> str:
        """Extract world state from prompt."""
        if "World:" in prompt:
            parts = prompt.split("World:")
            if len(parts) > 1:
                state_part = parts[1].split("Proposed:")[0]
                return state_part.strip()
        return prompt
    
    def _extract_update(self, prompt: str) -> str:
        """Extract proposed update from prompt."""
        if "Proposed:" in prompt:
            parts = prompt.split("Proposed:")
            if len(parts) > 1:
                return parts[1].strip()
        return ""
    
    def _generate_context(self, sample: AugmentedSample) -> str:
        """Generate context for human annotators."""
        return (
            "Evaluate whether the proposed update is coherent with the current world state. "
            "Consider: Does the update make logical sense? Are there any contradictions? "
            "Is the timing/causality valid? Are the entities behaving plausibly?"
        )
    
    def _get_annotation_instructions(self) -> str:
        """Get instructions for human annotators."""
        return """
COHERENCE ANNOTATION INSTRUCTIONS

For each sample, you will see:
1. A description of the current world state (entities, attributes, events)
2. A proposed update to the world

Your task is to judge whether the proposed update is COHERENT with the world state.

An update is COHERENT if:
- It follows logically from the current state
- It doesn't contradict existing facts
- Entities exist before they are referenced in events
- Actions are physically plausible (rocks can't speak, fish can't walk on land)
- Entity types are consistent (a cat doesn't suddenly become a number)

An update is INCOHERENT if:
- It references non-existent entities
- It contradicts established facts
- It violates physical laws
- It creates impossible temporal situations
- It makes illogical type changes

For each sample, provide:
1. Your verdict: Coherent or Incoherent
2. Your confidence: 1 (very unsure) to 5 (very confident)
3. Brief reasoning explaining your judgment
4. Any specific violations you identified
"""


def create_evaluation_session(
    num_samples: int = 20,
    output_dir: str = "eval_sessions",
    session_name: Optional[str] = None,
) -> str:
    """Create a new evaluation session and export for annotation."""
    evaluator = HumanEvaluator(output_dir=output_dir)
    session = evaluator.create_session(num_samples=num_samples, session_name=session_name)
    annotation_path = evaluator.export_for_annotation(session)
    
    print(f"Created evaluation session: {session.session_id}")
    print(f"Session saved to: {evaluator.output_dir / f'{session.session_id}.json'}")
    print(f"Annotation file: {annotation_path}")
    
    return session.session_id


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Human evaluation tools")
    parser.add_argument("--action", choices=["create", "report", "export"], default="create")
    parser.add_argument("--session", type=str, help="Session ID for report/export")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default="eval_sessions")
    args = parser.parse_args()
    
    evaluator = HumanEvaluator(output_dir=args.output_dir)
    
    if args.action == "create":
        session_id = create_evaluation_session(
            num_samples=args.samples,
            output_dir=args.output_dir,
        )
    elif args.action == "report" and args.session:
        session = evaluator.load_session(args.session)
        report = evaluator.generate_report(session)
        print(json.dumps(report, indent=2))
    elif args.action == "export" and args.session:
        session = evaluator.load_session(args.session)
        path = evaluator.export_for_annotation(session)
        print(f"Exported to: {path}")
