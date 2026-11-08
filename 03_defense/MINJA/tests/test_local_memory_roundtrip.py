"""SQLite/JSON round-trip must preserve provenance enum and signature validity."""
import json
from dataclasses import asdict
from experiments.real_minja_pilot import restore_entry
from MINJA.config import D4Config
from MINJA.d4_provenance import D4ProvenanceBinder
from MINJA.types import CandidateEntry, IntegrityLabel, SourceLabel


def test_persisted_provenance_verifies_and_tamper_fails():
    binder = D4ProvenanceBinder(D4Config(signing_backend="hmac", hmac_secret=b"test-only-key"))
    entry = CandidateEntry(content="会议时间是周五", embedding=[1.0, 0.0])
    binder.bind(entry, [SourceLabel("external", IntegrityLabel.UNTRUSTED)], "会议时间")
    restored = restore_entry(json.loads(json.dumps(asdict(entry))))
    candidate = CandidateEntry(content=restored.content, provenance=restored.provenance)
    assert restored.provenance.label is IntegrityLabel.UNTRUSTED
    assert binder.verify(candidate)[0]
    candidate.content = "被修改的内容"
    assert not binder.verify(candidate)[0]


def test_truncated_api_response_is_logged_and_rejected(tmp_path):
    from types import SimpleNamespace as NS
    import pytest
    from experiments.real_minja_pilot import AuditedClient

    response = NS(choices=[NS(finish_reason="length", message=NS(content='{"probability":0.9}'))],
                  usage=None)
    raw = NS(chat=NS(completions=NS(create=lambda **kwargs: response)))
    path = tmp_path / "api.jsonl"
    audited = AuditedClient(raw, path)
    with pytest.raises(ValueError, match="Truncated"):
        audited.create(model="test", messages=[{"role": "user", "content": "probability"}])
    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["status"] == "error"
    assert event["finish"] == "length"
