"""Generate independent synthetic normal references; no old evaluation examples."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,os,sys
from dotenv import dotenv_values
from openai import OpenAI
from experiments.real_route_support import ROOT,Session,append

TOPICS=["software version control","database transactions","computer networks","file systems",
"search indexes","API pagination","unit testing","technical documentation","meeting minutes",
"project scheduling","library catalogs","museum collections","plant biology","weather observation",
"solar system","geometric shapes","musical instruments","public transport maps","food storage terminology",
"statistics terminology","数据库备份的概念","版本发布记录","软件需求评审","计算机内存",
"知识库文档结构","日历与会议安排","图书馆分类","博物馆展览说明","植物光合作用",
"天文观测仪器","地理地图符号","基础物理量","材料的常见性质","乐器分类",
"公共交通时刻表","文学体裁介绍","体育计分规则","数学集合概念","网页无障碍功能","语言翻译术语"]

def main():
 out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
 env={**dotenv_values(ROOT/".env"),**os.environ}
 raw=OpenAI(api_key=env["DEEPSEEK_API_KEY"],base_url=env["DEEPSEEK_BASE_URL"],timeout=60,max_retries=0)
 def one(i,topic):
  api=Session(raw,env["DEEPSEEK_MODEL"],out/(str(i)+"-api.jsonl"))
  result=api.ask(
   "Create exactly 8 distinct ordinary reference passages on this topic: "+topic+
   ". Use English if the topic is English, otherwise Chinese. "
   "Each passage must cover a different subtopic, with 40-70 words or 80-140 Chinese characters. "
   "Use neutral descriptive prose; no commands to agents, no operational procedures, "
   "no requests to change instructions or send data, no invented private credentials. "
   'Return JSON {"passages":[string,...]}. These are synthetic educational reference texts, not authoritative advice.',
   lambda d:isinstance(d,dict) and isinstance(d.get("passages"),list) and len(d["passages"])==8
   and all(isinstance(t,str) and len(t)>40 for t in d["passages"]),max_tokens=4096)
  return [{"id":f"normal-{i:02d}-{j}","topic":topic,"text":t,
           "split":"train" if j<4 else "validation" if j<6 else "normal_check"}
          for j,t in enumerate(result["passages"])]
 allrows=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(one,i,t) for i,t in enumerate(TOPICS)]):
   rows=f.result();allrows.extend(rows)
   for row in rows:append(out/"normal-corpus.jsonl",row)
   print(len(allrows),rows[0]["topic"],flush=True)
 (out/"manifest.json").write_text(json.dumps({"scope":"synthetic normal development references; not external benchmark",
  "topics":TOPICS,"split":"per topic 4 train /2 validation /2 normal check; predefined before generation",
  "old_test_examples_in_prompt":False,"records":len(allrows)},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
