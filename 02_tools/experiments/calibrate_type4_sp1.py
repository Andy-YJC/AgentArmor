"""Normal-only calibration: old200 is a regression set, not a holdout."""
import argparse,hashlib,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from experiments.real_route_support import Embeddings
from experiments.real_type4 import load
from External_Developer_Write.types import DocumentEmbedding,Document,PipelineConfig
from External_Developer_Write.pipeline import DefensePipeline
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--normal-corpus",type=Path,required=True)
parser.add_argument("--embedding-model",type=Path,required=True)
parser.add_argument("--output-dir",type=Path,required=True)
args=parser.parse_args()
out=args.output_dir
out.mkdir(parents=True,exist_ok=False)
rows=sorted([json.loads(x) for x in args.normal_corpus.read_text(encoding="utf-8").splitlines()],key=lambda r:r["id"])
assert len(rows)==320 and len({r["text"] for r in rows})==320
groups={k:[r for r in rows if r["split"]==k] for k in ["train","validation","normal_check"]}
assert [len(v) for v in groups.values()]==[160,80,80]
embed=Embeddings(str(args.embedding_model))
train=embed([r["text"] for r in groups["train"]])
mu=train.mean(axis=0)
_,_,vt=np.linalg.svd(train-mu,full_matrices=False)
components=vt[:16]
def vector(text):
 v=(embed([text])[0]-mu)@components.T
 return (v/max(np.linalg.norm(v),1e-12)).tolist()
docs=lambda key:[DocumentEmbedding(r["id"],vector(r["text"]),"MiniLM-PCA16") for r in groups[key]]
p=DefensePipeline(PipelineConfig.CONFIG_5_MAX,embed_model=vector)
p._active_sps={"sp1"}
p.initialize()
p.train_sp1(docs("train"))
detector=p._detectors["sp1"]
def check():
 return [dict(id=r["id"],**asdict(detector.detect(vector(r["text"]),p._sp1_clean_stats))) for r in groups["normal_check"]]
before=check()
threshold=p.calibrate_sp1(docs("validation"),false_alarm_rate=0.05)
after=check()
regression=[]
for s in load():
 r=p.upload_document(Document(s["id"],s["content"]),vector(s["content"]))
 regression.append({"id":s["id"],"label":s["label"],"allowed":r["allowed"],"SP1":asdict(r["details"]["sp1"])})
summary={"scope":"SP1 only; no test labels used for calibration","train":160,"validation":80,"new_normal_check":80,
"default_threshold":0.5,"calibrated_threshold":threshold,
"new_normal_flags_before":sum(r["is_anomaly"] for r in before),
"new_normal_flags_after":sum(r["is_anomaly"] for r in after),
"old200_regression":{"attack_blocked":sum(not r["allowed"] for r in regression if r["label"]=="attack"),
"benign_passed":sum(r["allowed"] for r in regression if r["label"]=="benign")},
"warning":"Synthetic development checks, no external malicious holdout; not full Type4"}
profile={"clean_stats":asdict(p._sp1_clean_stats),"threshold":threshold,
"mean":mu.tolist(),"components":components.tolist(),"training_texts":[r["text"] for r in groups["train"]],
"embedding_sha256":hashlib.file_digest(open(args.embedding_model/"model.safetensors","rb"),"sha256").hexdigest(),
"scope":"SP1 normal-only empirical calibration; no malicious labels"}
for name,value in [("profile.json",profile),("summary.json",summary),("normal-before.json",before),("normal-after.json",after),("old200-regression.json",regression),("projection.json",{"mean":mu.tolist(),"components":components.tolist()})]:
 (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False))
