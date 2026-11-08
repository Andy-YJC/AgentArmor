"""Opt-in semantic D1 extension; deterministic checks and source evidence remain."""
from memory_semantics import review_memory
from .d1_reflection_intent import D1ReflectionIntentDetector
from .types import DefenseVerdict


class LLMReflectionIntentDetector(D1ReflectionIntentDetector):
    def __init__(self,config,client,model):
        super().__init__(config)
        self.client,self.model=client,model

    def check(self,ctx):
        base=super().check(ctx)
        if not self.cfg.enabled or base.action=="BLOCK":
            return base
        review=review_memory(self.client,self.model,
                             "\n".join(ctx.candidate.fact_texts),ctx.joined_turn_text())
        if not review.risky:
            return base
        return DefenseVerdict("D1",passed=False,score=1.0,action="BLOCK",
                              reason=review.reason,metadata={"semantic_review":True,
                                                            "evidence":review.evidence})
