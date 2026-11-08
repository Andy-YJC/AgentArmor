from types import SimpleNamespace as NS
from unittest.mock import Mock
import json
import numpy as np
import pytest
from AutoWrite.config import DBConfig
from AutoWrite.db_selective_write import DBSelectiveWritePolicy, MemoryIndex
from AutoWrite.db_novelty_review import NoveltyReview, review_novelty
from MINJA.types import CandidateEntry, WriteContext

class Index(MemoryIndex):
    def __init__(self,matrix):self.matrix=matrix
    def count(self):return len(self.matrix)
    def get_all_embeddings(self):return self.matrix
    def get_all_contents(self):return ['old record '+str(i) for i in range(len(self.matrix))]

def call(policy):
    entry=CandidateEntry(content='New quantity is 27',embedding=[1.,0.])
    return policy.check(entry,WriteContext('record progress','','',entry.content,'progress'))

def test_low_vector_value_can_contain_a_real_update():
    review=Mock(return_value=NoveltyReview('novel','27',None,'updated quantity'))
    index=Index(np.array([[.6,.8]]*5))
    p=DBSelectiveWritePolicy(DBConfig(),lambda t:None,index,novelty_check_fn=review)
    r=call(p)
    assert r.action=='PASS' and r.metadata['value']<.3
    review.assert_called_once()

def test_single_duplicate_cannot_hide_in_large_diverse_index():
    index=Index(np.array([[1.,0.]]+[[0.,1.]]*9))
    review=Mock(return_value=NoveltyReview('duplicate','27',0,'already present'))
    p=DBSelectiveWritePolicy(DBConfig(),lambda t:None,index,novelty_check_fn=review)
    r=call(p)
    assert r.action=='BLOCK' and r.metadata['value']>.3

@pytest.mark.parametrize('cfg,matrix',[ (DBConfig(max_session_writes=0),[[0.,1.]]), (DBConfig(),[[1.,0.]]*5)])
def test_semantic_review_cannot_override_rate_or_density(cfg,matrix):
    review=Mock()
    r=call(DBSelectiveWritePolicy(cfg,lambda t:None,Index(np.array(matrix)),novelty_check_fn=review))
    assert r.action=='BLOCK'
    review.assert_not_called()

def test_review_error_is_not_a_pass():
    p=DBSelectiveWritePolicy(DBConfig(),lambda t:None,Index(np.array([[.6,.8]]*5)),
        novelty_check_fn=Mock(side_effect=ValueError('bad evidence')))
    with pytest.raises(ValueError):call(p)

@pytest.mark.parametrize('data',[
    {'decision':'novel','evidence':'absent','matched_index':None,'reason':'x'},
    {'decision':'duplicate','evidence':'27','matched_index':True,'reason':'x'},
    {'decision':'duplicate','evidence':'27','matched_index':5,'reason':'x'},
    {'decision':'other','evidence':'27','matched_index':None,'reason':'x'},
])
def test_model_review_requires_grounded_evidence_and_match(data):
    resp=NS(choices=[NS(message=NS(content=json.dumps(data)),finish_reason='stop')])
    api=NS(chat=NS(completions=NS(create=lambda **kw:resp)))
    with pytest.raises(ValueError):review_novelty(api,'test','New quantity is 27',['old record'])

def test_db_pass_does_not_bypass_downstream_safety(monkeypatch):
    from AutoWrite.pipeline import AutoWriteDefensePipeline
    from AutoWrite.config import AutoWriteConfig
    from AutoWrite.types import ChainedCandidateEntry
    from MINJA.pipeline import WriteResult
    monkeypatch.setenv('AUTOWRITE_CHAIN_KEY','ab'*32)
    minja=Mock()
    minja.on_write_request.return_value=WriteResult(False,None,[],blocked_by='D3')
    cfg=AutoWriteConfig(audit_log_path=None);cfg.da.enabled=False
    p=AutoWriteDefensePipeline.from_config(cfg,minja,lambda t:np.array([[1.,0.]]),Index(np.array([[.6,.8]]*5)))
    p.db._novelty_check=lambda *a:NoveltyReview('novel','27',None,'update')
    entry=ChainedCandidateEntry(content='New quantity is 27',embedding=[1.,0.])
    r=p.on_write_request(entry,WriteContext('record','','',entry.content,'record'),[])
    assert not r.accepted and r.blocked_by=='D3'
    minja.on_write_request.assert_called_once()


def test_paraphrase_must_not_bypass_review_when_vector_score_is_high():
    index=Index(np.array([[.3,.954]]*5))
    review=Mock(return_value=NoveltyReview('duplicate','27',0,'paraphrase of existing fact'))
    r=call(DBSelectiveWritePolicy(DBConfig(),lambda t:None,index,novelty_check_fn=review))
    assert r.action=='BLOCK'
    review.assert_called_once()
