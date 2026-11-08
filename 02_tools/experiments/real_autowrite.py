"""Actual AutoWrite write pipeline, with real MINJA and local embeddings."""
from dataclasses import asdict
import os
import secrets
import numpy as np

from experiments.real_route_support import execute
from experiments.real_minja_pilot import pipeline
from AutoWrite.config import AutoWriteConfig
from AutoWrite.db_selective_write import MemoryIndex
from AutoWrite.pipeline import AutoWriteDefensePipeline
from AutoWrite.tests.build_dataset import ALL_SAMPLES
from AutoWrite.types import ChainedCandidateEntry, ChainedRetrievedEntry
from MINJA.types import IntegrityLabel, SourceLabel, WriteContext


class EmptyIndex(MemoryIndex):
    def get_all_embeddings(self): return np.empty((0,384))
    def count(self): return 0


def load():
    return [{**asdict(s),"id":s.id,"label":s.expected_label} for s in ALL_SAMPLES]


def evaluate(s,api,embed,unused):
    minja=pipeline(embed,api,api.model)
    cfg=AutoWriteConfig(audit_log_path=None)
    cfg.de.judge_model=api.model
    p=AutoWriteDefensePipeline.from_config(cfg,minja,embed,EmptyIndex(),api)
    e=ChainedCandidateEntry(entry_id=s["id"],content=s["candidate_content"])
    ctx=WriteContext(s["user_goal"],s["candidate_content"],s["indication_prompt"],
                     s["candidate_content"],s["triggering_query"])
    result=p.on_write_request(e,ctx,[SourceLabel("framework_context",IntegrityLabel.UNTRUSTED)])
    row={"accepted":result.accepted,"blocked_by":result.blocked_by,
         "verdicts":[asdict(v) for v in result.verdicts],"candidate":e.content,
         "baseline_memory":s["candidate_content"],"scope":"isolated-session write"}
    if result.accepted:
        r=ChainedRetrievedEntry(e.entry_id,e.content,embed.one(e.content),e.provenance,
                               chain_hash=e.chain_hash,prev_chain_hash=e.prev_chain_hash)
        got=p.on_retrieval([r],s["target_query"] or s["user_goal"])
        row.update(retrieved=len(got.entries),tampered_count=got.tampered_count,
                   retrieval_verdicts=[asdict(v) for v in got.verdicts])
    # D-E is also probed on supplied content, without treating FLAG as a write BLOCK.
    from MINJA.types import RetrievedEntry
    r=RetrievedEntry(s["id"],s["candidate_content"],embed.one(s["candidate_content"]))
    with api.phase("diagnostic"):
        aligned,vs=p.de.verify([r],s["target_query"] or s["user_goal"])
    row.update(alignment_flagged=aligned[0].flagged,alignment_verdicts=[asdict(v) for v in vs])
    return row


if __name__=="__main__":
    os.environ.setdefault("AUTOWRITE_CHAIN_KEY",secrets.token_hex(32))
    execute("autowrite",load,evaluate)
