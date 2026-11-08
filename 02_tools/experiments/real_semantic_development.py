"""Fixed original real candidates: semantic admission development regression."""
import json,sys
from pathlib import Path
from dataclasses import asdict
from experiments.real_route_support import execute
from Reflection.pipeline import ReflectionDefensePipeline
from Reflection.config import PipelineConfig
from Reflection.llm_intent import LLMReflectionIntentDetector
from Reflection.types import ConversationTurn,SourceType,ReflectionCandidate
from Reflection.memory_store import MemoryStore
from Reflection.evaluation import load_dataset
from MASW.pipeline import build_default_pipeline
from MASW.llm_detector import LLMCandidateDetector
from MASW.tests.build_dataset import build_samples
from MASW.tests.eval_masw import _agents
from MASW.types import CandidateFact,Task,TaskContext,TrustLevel
BASE=Path("D:/Codes/LaTex_project/AgentArmorReport/07_runtime")
OLD={}
for route in ["reflection","masw"]:
 OLD[route]={r["id"]:r for r in map(json.loads,(BASE/(route+"-real-full-20261107")/"results.jsonl").read_text(encoding="utf-8").splitlines()) if r["status"]=="ok"}
RAWS={s.sample_id:s for s in load_dataset(Path(__file__).resolve().parents[2]/"03_defense/Reflection/datasets/reflection_type3_competition.csv")}
MASW={s["id"]:s for s in build_samples()}


def load():
 rows=[]
 for route,old in OLD.items():
  bad=[r for r in old.values() if r.get("unsafe_protected",{}).get("unsafe")]
  normal=[r for r in sorted(old.values(),key=lambda r:r["id"]) if r["label"]=="benign" and r["accepted"]][:8]
  for r in bad+normal:
   rows.append({"id":route+"-"+r["id"],"original_id":r["id"],"label":r["label"],
                "path":route,"old":r,"development_group":"previously_flagged" if r in bad else "normal_control"})
 return rows


def evaluate(s,api,embed,unused):
 old=s["old"];sid=s["original_id"]
 if s["path"]=="reflection":
  raw=RAWS[sid];facts=old["generated_facts"]
  cfg=PipelineConfig(audit_log_path=None)
  p=ReflectionDefensePipeline(cfg,d1=LLMReflectionIntentDetector(cfg.d1,api,api.model))
  r=p.evaluate([ConversationTurn(sid,SourceType(raw.source),raw.raw_text)],
               ReflectionCandidate(" ".join(facts),facts),MemoryStore())
  kept=[x.fact_text for x in r.accepted_records]
  details={"verdicts":[asdict(v) for v in r.verdicts]}
 else:
  raw=MASW[sid]
  events=[json.loads(x) for x in (BASE/"masw-real-full-20261107"/(sid+"-api.jsonl")).read_text(encoding="utf-8").splitlines()]
  facts=json.loads(events[0]["response"])["facts"]
  p=build_default_pipeline();a,v,b=_agents()
  class OriginalRealExtraction:
   def extract(self,agent,external_input,task):
    return [CandidateFact(**x,confidence=0.75,source=external_input.source_uri,writer=agent.id,
                          trust=TrustLevel.UNTRUSTED,taint=True,parent_ids=(external_input.id,)) for x in facts]
  p.extractor=OriginalRealExtraction()
  p.verifier.detector=LLMCandidateDetector(api,api.model)
  task=Task(sid,raw["user_query"],raw["source_uri"],raw["source_type"],TaskContext())
  written=p.path_external_to_shared_memory(a,v,raw["content"],task)
  kept=[r.as_text() for r in written]
  details={"quarantine":[q.reason for q in p.quarantine.items]}
 return {"accepted":bool(kept),"new_memory":kept,"original_protected_memory":old.get("protected_facts",old.get("protected_memory")),
         "development_group":s["development_group"],"original_path":s["path"],"fixed_original_real_candidates":True,**details}

if __name__=="__main__":execute("semantic_development",load,evaluate)
