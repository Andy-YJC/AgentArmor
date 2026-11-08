"""Known-case development regression for D3 prompt; not independent evaluation."""
from dataclasses import asdict
from experiments.real_route_support import execute
from experiments.real_minja_pilot import pipeline
from MINJA.tests.build_dataset import ALL_SAMPLES
from MINJA.types import CandidateEntry,WriteContext,SourceLabel,IntegrityLabel
IDS={"B-001","B-010","B-014","B-029","MI1-001","MI2-001","MI3-001","MI4-001"}

def load():
 return [{**asdict(s),"label":s.expected_label} for s in ALL_SAMPLES if s.id in IDS]

def evaluate(s,api,embed,unused):
 p=pipeline(embed,api,api.model)
 r=p.on_write_request(CandidateEntry(content=s["candidate_content"],embedding=embed.one(s["candidate_content"])),
     WriteContext(s["user_goal"],"",s["indication_prompt"],s["candidate_content"],s["triggering_query"]),
     [SourceLabel("external_input",IntegrityLabel.UNTRUSTED)])
 return {"accepted":r.accepted,"blocked_by":r.blocked_by,"verdicts":[asdict(v) for v in r.verdicts],
         "scope":"known development regression after prompt revision"}

if __name__=="__main__":execute("minja_development",load,evaluate)
