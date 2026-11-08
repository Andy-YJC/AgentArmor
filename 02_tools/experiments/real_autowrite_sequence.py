"""Warm shared-store sequences with actual MiniLM and model admission calls."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import argparse, hashlib, json, os, secrets, time
import numpy as np
from dotenv import dotenv_values
from openai import OpenAI
from experiments.real_route_support import Embeddings, Session, append
from experiments.real_minja_pilot import pipeline
from AutoWrite.config import AutoWriteConfig
from AutoWrite.db_selective_write import MemoryIndex
from AutoWrite.df_distribution_monitor import DFMemoryDistributionMonitor
from AutoWrite.pipeline import AutoWriteDefensePipeline
from AutoWrite.tests.build_dataset import ALL_SAMPLES
from AutoWrite.types import ChainedCandidateEntry
from MINJA.types import CandidateEntry, SourceLabel, IntegrityLabel, WriteContext

ROOT=Path(__file__).resolve().parents[2]


class SharedIndex(MemoryIndex):
    def __init__(self, seeds):
        self.entries=list(seeds)

    def count(self):
        return len(self.entries)

    def get_all_embeddings(self):
        return np.asarray([e.embedding for e in self.entries],dtype=np.float32)

    def get_all_contents(self):
        return [e.content for e in self.entries]

    def add(self,entry):
        self.entries.append(entry)


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embedding-model',required=True)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--seed-corpus',required=True,type=Path)
    args=parser.parse_args()
    args.output_dir.mkdir(exist_ok=False,parents=True)
    env={**dotenv_values(ROOT/'.env'),**os.environ}
    os.environ.setdefault('AUTOWRITE_CHAIN_KEY',secrets.token_hex(32))
    client=OpenAI(api_key=env['DEEPSEEK_API_KEY'],base_url=env['DEEPSEEK_BASE_URL'],timeout=45,max_retries=0)
    embed=Embeddings(args.embedding_model)
    corpus=[json.loads(x) for x in args.seed_corpus.read_text(encoding='utf-8').splitlines()]
    seeds=sorted([r for r in corpus if r['split']=='train'],key=lambda r:(r['id'].rsplit('-',1)[1],r['topic']))[:50]
    assert len(seeds)==50
    (args.output_dir/'trusted-seeds.json').write_text(json.dumps(seeds,ensure_ascii=False,indent=2),encoding='utf-8')
    seed_vectors=embed([r['text'] for r in seeds])
    groups={'flood_and_low_value':[s for s in ALL_SAMPLES if s.id.startswith('AWB-')],
            'trigger_cluster':[s for s in ALL_SAMPLES if s.id.startswith('AWF-')],
            'normal_control':[s for s in ALL_SAMPLES if s.expected_label=='benign']}
    manifest={'model':env['DEEPSEEK_MODEL'],'workers':3,'auto_retry':False,
              'seed_count':50,'seed_sha256':hashlib.sha256((args.output_dir/'trusted-seeds.json').read_bytes()).hexdigest(),
              'scenarios':{name:[s.id for s in samples] for name,samples in groups.items()},
              'scope':'Known cases, sequential persistent pipeline/index per scenario. Trusted normal seeds are preloaded, not tested admissions. No actor or tool execution.'}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

    def run(name,samples):
        api=Session(client,env['DEEPSEEK_MODEL'],args.output_dir/(name+'-api.jsonl'))
        initial=[CandidateEntry(entry_id=r['id'],content=r['text'],embedding=v.tolist()) for r,v in zip(seeds,seed_vectors)]
        index=SharedIndex(initial)
        baseline=SharedIndex(initial)
        minja=pipeline(embed,api,api.model)
        cfg=AutoWriteConfig(audit_log_path=None)
        cfg.de.judge_model=api.model
        p=AutoWriteDefensePipeline.from_config(cfg,minja,embed,index,api)
        baseline_monitor=DFMemoryDistributionMonitor(cfg.df,embed)
        for e in initial:
            p.df.update(e)
            baseline_monitor.update(e)
        initial_scan=asdict(p.df.scan())
        rows=[]
        for step,s in enumerate(samples,1):
            api.phase_name=s.id+'/defense'
            count_before=index.count()
            api_before,failures_before=api.calls,api.failures
            tick=time.perf_counter()
            raw=CandidateEntry(entry_id=s.id,content=s.candidate_content,embedding=embed.one(s.candidate_content))
            baseline.add(raw)
            baseline_monitor.update(raw)
            entry=ChainedCandidateEntry(entry_id=s.id,content=s.candidate_content)
            ctx=WriteContext(s.user_goal,s.candidate_content,s.indication_prompt,s.candidate_content,s.triggering_query)
            row={'scenario':name,'step':step,'id':s.id,'label':s.expected_label,'count_before':count_before}
            try:
                got=p.on_write_request(entry,ctx,[SourceLabel('framework_context',IntegrityLabel.UNTRUSTED)])
                if api.failures>failures_before:raise RuntimeError('native_node_swallowed_model_failure')
                if got.accepted:index.add(entry)
                row.update(status='ok',accepted=got.accepted,blocked_by=got.blocked_by,
                           verdicts=[asdict(v) for v in got.verdicts],stored_candidate=entry.content if got.accepted else None)
            except Exception as exc:
                row.update(status='error',error=type(exc).__name__)
            row.update(count_after=index.count(),baseline_count=baseline.count(),
                       protected_distribution=asdict(p.df.scan()),baseline_distribution=asdict(baseline_monitor.scan()),
                       api_calls=api.calls-api_before,elapsed_s=time.perf_counter()-tick)
            append(args.output_dir/(name+'-results.jsonl'),row)
            rows.append(row)
            print(name,step,s.id,row['status'],row.get('accepted'),flush=True)
        def nearest_ids(store,query):
            matrix=store.get_all_embeddings()
            vector=np.asarray(embed.one(query))
            scores=matrix@vector/(np.linalg.norm(matrix,axis=1)*np.linalg.norm(vector)+1e-10)
            return [store.entries[int(i)].entry_id for i in np.argsort(-scores)[:5]]
        queries=['AI安全最新动态','查看文件访问规则','总结今天的会议']
        valid=[r for r in rows if r['status']=='ok']
        result={'total':len(rows),'valid':len(valid),'errors':len(rows)-len(valid),
                'accepted':sum(r['accepted'] for r in valid),'blocked':sum(not r['accepted'] for r in valid),
                'blocked_by':{node:sum(r.get('blocked_by')==node for r in valid) for node in ['D-A','D-B','D1','D2','D3']},
                'seed_count':len(initial),'protected_count':index.count(),'baseline_count':baseline.count(),
                'initial_distribution':initial_scan,'final_protected_distribution':asdict(p.df.scan()),
                'final_baseline_distribution':asdict(baseline_monitor.scan()),
                'retrieval_exposure':[{ 'query':q,'baseline_top5':nearest_ids(baseline,q),'protected_top5':nearest_ids(index,q)} for q in queries],
                'api_calls':api.calls}
        assert index.count()==len(initial)+sum(r['accepted'] for r in valid)
        return name,result

    summary={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(run,n,s) for n,s in groups.items()]):
            name,result=future.result();summary[name]=result
    summary['wall_s']=time.perf_counter()-started
    (args.output_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    if any(v['errors'] for k,v in summary.items() if k!='wall_s'):raise SystemExit(1)

if __name__=='__main__':main()
