"""A newly written chain entry must verify without changing any fields."""
import numpy as np
from AutoWrite.config import AutoWriteConfig
from AutoWrite.pipeline import AutoWriteDefensePipeline
from AutoWrite.db_selective_write import MockMemoryIndex
from AutoWrite.types import ChainedCandidateEntry
from experiments.real_minja_pilot import pipeline
from MINJA.types import SourceLabel, IntegrityLabel, WriteContext


def test_pipeline_records_previous_chain_head(monkeypatch):
    monkeypatch.setenv("AUTOWRITE_CHAIN_KEY", "ab"*32)
    embed=lambda texts:np.ones((len(texts),4),dtype=np.float32)
    minja=pipeline(embed,None,"unused")
    minja.cfg.d1.enabled=False
    minja.cfg.d2.enabled=False
    minja.cfg.d3.enabled=False
    cfg=AutoWriteConfig(audit_log_path=None)
    cfg.da.enabled=False
    cfg.db.enabled=False
    p=AutoWriteDefensePipeline.from_config(cfg,minja,embed,MockMemoryIndex(np.empty((0,4))))
    e=ChainedCandidateEntry(content="Meeting scheduled for Friday.")
    result=p.on_write_request(e,WriteContext("record meeting","", "",e.content,"meeting"),
                              [SourceLabel("user_input",IntegrityLabel.CANDIDATE)])
    assert result.accepted
    assert e.prev_chain_hash==cfg.dc.chain_anchor
    assert p.dc.verify_entry(e,e.chain_hash,e.prev_chain_hash).passed
