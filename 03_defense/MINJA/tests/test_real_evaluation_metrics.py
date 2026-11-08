"""Scoring must keep transport errors separate from classification outcomes."""
from experiments.real_minja_full import metrics


def test_failed_api_case_is_not_counted_as_a_block():
    rows = [
        {"status": "ok", "label": "attack", "accepted": False},
        {"status": "ok", "label": "attack", "accepted": True},
        {"status": "error", "label": "attack"},
        {"status": "ok", "label": "benign", "accepted": False},
        {"status": "ok", "label": "benign", "accepted": True},
    ]
    result = metrics(rows)
    assert result["errors"] == 1
    assert result["attack_total_valid"] == 2
    assert result["tp"] == 1 and result["fn"] == 1
    assert result["fp"] == 1 and result["tn"] == 1
    assert result["recall"] == 0.5 and result["fpr"] == 0.5
