"""Opt-in semantic candidate detector for the existing MASW verifier interface."""
from memory_semantics import review_memory
from .d3_risk_filter import RuleBasedDetector,DetectionResult


class LLMCandidateDetector:
    name="rule_and_llm"

    def __init__(self,client,model):
        self.client,self.model=client,model
        self.rule=RuleBasedDetector()

    def detect(self,candidate):
        base=self.rule.detect(candidate)
        review=review_memory(self.client,self.model,candidate.as_text(),candidate.evidence_span)
        return DetectionResult(detector=self.name,risk=max(base.risk,1.0 if review.risky else 0.0),
                               reasons=base.reasons+((review.reason,) if review.risky else ()))
