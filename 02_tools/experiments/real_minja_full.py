"""Frozen 120-case real MINJA write evaluation; no actor or tool execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from dotenv import dotenv_values
from openai import OpenAI

from experiments.real_minja_pilot import append, pipeline
from experiments.real_route_support import Session, timing_stats
from MINJA.local_embedding import LocalEmbedding
from MINJA.tests.build_dataset import ALL_SAMPLES
from MINJA.types import CandidateEntry, IntegrityLabel, SourceLabel, WriteContext

SEEN = {"MI1-001", "MI2-001", "MI3-001", "MI4-001", "B-001", "B-021"}


def metrics(rows):
    valid = [r for r in rows if r["status"] == "ok"]
    attacks = [r for r in valid if r["label"] == "attack"]
    benign = [r for r in valid if r["label"] == "benign"]
    tp = sum(not r["accepted"] for r in attacks)
    fp = sum(not r["accepted"] for r in benign)
    return {"total": len(rows), "valid": len(valid), "errors": len(rows)-len(valid),
            "attack_total_valid": len(attacks), "benign_total_valid": len(benign),
            "tp": tp, "fn": len(attacks)-tp, "fp": fp, "tn": len(benign)-fp,
            "recall": tp/len(attacks) if attacks else None,
            "fpr": fp/len(benign) if benign else None,
            "baseline": "deterministic allow-all write; not an LLM behavior baseline"}


def main():
    run_started=time.perf_counter()
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--embedding-model", required=True)
    a.add_argument("--output-dir", type=Path, required=True)
    a.add_argument("--workers", type=int, default=3, choices=[1,2,3])
    args = a.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    env = {**dotenv_values(root/".env"), **os.environ}
    model = env["DEEPSEEK_MODEL"]
    preparation_started=time.perf_counter()
    embed = LocalEmbedding(args.embedding_model)
    vectors = embed([s.candidate_content for s in ALL_SAMPLES])
    preparation_s=time.perf_counter()-preparation_started
    client = OpenAI(api_key=env["DEEPSEEK_API_KEY"], base_url=env["DEEPSEEK_BASE_URL"],
                    timeout=30, max_retries=0)
    frozen = pipeline(embed, None, model)
    cfg = asdict(frozen.cfg)
    cfg["d4"].pop("private_key_b64")
    cfg["d4"].pop("hmac_secret")
    manifest = {"model": model, "config": cfg, "thinking": "disabled", "workers": args.workers,
                "scope": "write admission only; supplied candidates; no actor/retrieval metrics",
                "previously_seen_ids": sorted(SEEN), "samples": len(ALL_SAMPLES),
                "embedding_sha256": hashlib.file_digest(
                    open(Path(args.embedding_model)/"model.safetensors", "rb"), "sha256").hexdigest()}
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for rel in ["02_tools/experiments/real_minja_full.py",
                "02_tools/experiments/real_minja_pilot.py", "03_defense/MINJA/local_embedding.py",
                "03_defense/MINJA/tests/build_dataset.py"]:
        dest = args.output_dir/"source"/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/rel, dest)

    def evaluate(index, sample):
        audit = Session(client, model, args.output_dir/(sample.id+"-api.jsonl"))
        row = {"id": sample.id, "subtype": sample.subtype, "label": sample.expected_label,
               "previously_seen": sample.id in SEEN}
        started = time.perf_counter()
        try:
            p = pipeline(embed, audit, model)
            result = p.on_write_request(
                CandidateEntry(content=sample.candidate_content, embedding=vectors[index].tolist()),
                WriteContext(sample.user_goal, "", sample.indication_prompt,
                             sample.candidate_content, sample.triggering_query),
                [SourceLabel("external_input", IntegrityLabel.UNTRUSTED)])
            row.update(status="ok", accepted=result.accepted, blocked_by=result.blocked_by,
                       verdicts=[asdict(v) for v in result.verdicts])
        except Exception as exc:
            row.update(status="error", error=type(exc).__name__,
                       http_status=getattr(exc, "status_code", None))
        row.update(elapsed_s=time.perf_counter()-started, api_calls=audit.calls, api_seconds_by_phase=dict(audit.api_seconds))
        return row

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, i, s) for i,s in enumerate(ALL_SAMPLES)]
        for f in as_completed(futures):
            row = f.result()
            results.append(row)
            append(args.output_dir/"results.jsonl", row)
            print(len(results), row["id"], row["status"], row.get("accepted"), flush=True)
    summary = {"all": metrics(results),
               "previously_seen_6": metrics([r for r in results if r["previously_seen"]]),
               "new_114": metrics([r for r in results if not r["previously_seen"]]),
               "by_subtype": {s: metrics([r for r in results if r["subtype"] == s])
                              for s in sorted({r["subtype"] for r in results})},
               "api_calls": sum(r["api_calls"] for r in results)}
    (args.output_dir/"timing.json").write_text(json.dumps({
        "script_wall_s_excluding_imports":time.perf_counter()-run_started,
        "preparation_s":preparation_s,"case_wall":timing_stats([r["elapsed_s"] for r in results]),
        "note":"write admission only; no scoring API; process wall measured by orchestrator"},indent=2),encoding="utf-8")
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    if summary["all"]["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
