"""Invalid model verdicts must not authorize memory writes."""
from types import SimpleNamespace as NS
import pytest
from MINJA.d3_prospective_sim import _simulate_llm_judge


def client(content, finish="stop"):
    response=NS(choices=[NS(message=NS(content=content),finish_reason=finish)])
    return NS(chat=NS(completions=NS(create=lambda **kw:response)))


@pytest.mark.parametrize("content", [
    "not JSON", '{"is_unsafe": false,', '{}', '[]',
    '{"is_unsafe": "false"}', '{"is_unsafe": 0}',
])
def test_invalid_verdict_is_an_error(content):
    with pytest.raises(ValueError):
        _simulate_llm_judge("memory",["query"],client(content),"test")


def test_truncated_verdict_is_an_error():
    with pytest.raises(ValueError):
        _simulate_llm_judge("memory",["query"],
                            client('{"is_unsafe": false}',"length"),"test")


@pytest.mark.parametrize("unsafe", [True,False])
def test_valid_boolean_verdict_is_preserved(unsafe):
    import json
    result=_simulate_llm_judge("memory",["query"],
                              client(json.dumps({"is_unsafe":unsafe})),"test")
    assert result[0][1] is unsafe
