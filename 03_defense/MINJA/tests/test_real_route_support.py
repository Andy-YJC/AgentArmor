"""Real model transport/schema errors must remain visible to the evaluator."""
import json
from types import SimpleNamespace as NS
import pytest
from experiments.real_route_support import Session


def test_invalid_native_probability_is_logged_as_failure(tmp_path):
    r=NS(choices=[NS(message=NS(content='{"probability":true}'),finish_reason="stop")],
         model="test",usage=None)
    raw=NS(chat=NS(completions=NS(create=lambda **kwargs:r)))
    api=Session(raw,"test",tmp_path/"api.jsonl")
    with pytest.raises(ValueError,match="invalid_model_schema"):
        api.create(messages=[{"role":"user","content":"probability"}],max_tokens=128)
    assert api.failures==1
    assert json.loads((tmp_path/"api.jsonl").read_text(encoding="utf-8"))["status"]=="error"


def test_text_truncation_is_not_accepted(tmp_path):
    r=NS(choices=[NS(message=NS(content="partial"),finish_reason="length")],
         model="test",usage=None)
    raw=NS(chat=NS(completions=NS(create=lambda **kwargs:r)))
    api=Session(raw,"test",tmp_path/"api.jsonl")
    with pytest.raises(ValueError,match="incomplete_model_response"):
        api.text("question")
    assert api.failures==1


def test_phase_restores_after_error_and_records_wall(tmp_path):
    from types import SimpleNamespace as NS
    from experiments.real_route_support import Session
    api=Session(NS(),"test",tmp_path/"api.jsonl")
    with pytest.raises(RuntimeError):
        with api.phase("generation"):
            assert api.phase_name=="generation"
            raise RuntimeError("test")
    assert api.phase_name=="defense"
    assert api.phase_wall_seconds["generation"]>=0


def test_parallel_preparation_logs_have_unique_calls(tmp_path):
    from types import SimpleNamespace as NS
    from concurrent.futures import ThreadPoolExecutor
    from experiments.real_route_support import Session
    response=NS(choices=[NS(message=NS(content="OK"),finish_reason="stop")],
                usage=None,model="test")
    raw=NS(chat=NS(completions=NS(create=lambda **kw:response)))
    api=Session(raw,"test",tmp_path/"api.jsonl")
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda _:api.text("test"),range(12)))
    rows=[json.loads(line) for line in (tmp_path/"api.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows)==12
    assert {r["call"] for r in rows}==set(range(1,13))
    assert api.api_seconds["defense"]>=0


def test_account_balance_error_stops_further_requests(monkeypatch,tmp_path):
    import threading
    from unittest.mock import Mock
    import experiments.real_route_support as support
    monkeypatch.setattr(support,"ACCOUNT_UNAVAILABLE",threading.Event())
    class InsufficientBalance(Exception):
        status_code=402
    sender=Mock(side_effect=InsufficientBalance())
    raw=NS(chat=NS(completions=NS(create=sender)))
    api=support.Session(raw,"test",tmp_path/"account.jsonl")
    with pytest.raises(InsufficientBalance):api.text("first")
    with pytest.raises(RuntimeError,match="request not sent"):api.text("second")
    assert sender.call_count==1
