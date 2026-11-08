# MINJA 真实模型小规模评估

入口：`02_tools/experiments/real_minja_pilot.py`。这是一条新增的独立入口，原有 mock 评估脚本仍保留，不能用它们的默认结果代替本入口结果。

## 环境

使用 andypytorch，CPU 推理。可选依赖见根目录 `requirements-local-models.txt`。
本地模型目录必须含完整 SentenceTransformer 模型，不会自动从 Hub 下载。
仓库根目录的 .env 提供 DEEPSEEK_API_KEY、DEEPSEEK_BASE_URL、DEEPSEEK_MODEL；不在命令行或报告写密钥。
终端默认 python 不一定跟随编辑器选择，建议明确使用解释器路径。

## 运行

先离线检查模型与长文本编码：

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
& D:/Anaconda/envs/andypytorch/python.exe -B run.py module experiments.real_minja_pilot --embedding-model D:/Models/paraphrase-multilingual-MiniLM-L12-v2 --output-dir 07_runtime/minja-offline-check --offline-check
```

执行真实 API 对照（会产生费用；模型从 .env 读取）：

```powershell
& D:/Anaconda/envs/andypytorch/python.exe -B run.py module experiments.real_minja_pilot --embedding-model D:/Models/paraphrase-multilingual-MiniLM-L12-v2 --output-dir 07_runtime/minja-real-pilot
```

输出目录必须不存在，每次使用新目录。

## 协议与结果

- 固定6条：MI1-001、MI2-001、MI3-001、MI4-001、B-001、B-021，不根据效果挑选或调整阈值。
- 写入候选来自原始数据集，不是本轮真实 Agent 自动生成的候选。
- baseline：直接接受候选，绕过写入、检索、执行门控；protected：调用 MINJA 实际节点。
- 两组均使用相同真实模型、本地向量和后续 target_query；标签和 attack_goal 仅进入结果，绝不进入防御、Actor提示词。
- 本地模型按 tokenizer offset 切成126-token窗口，重叠16个token；窗口向量分别归一化后取均值并再次归一化。它是当前模型128-token限制下明确的长文处理方案，不等同于长上下文模型。
- D1=keyword（真实规则），D2=proxy_llm，D3=llm_judge（仅边界触发），D4=Ed25519，D5=默认策略，D6=真实embedding，阈值保持代码默认。
- DeepSeek显式禁用thinking，禁用SDK自动重试；对JSON结构、数值范围、finish_reason做检查，异常记error，不记成功拦截，不回退mock。
- SQLite提交后关闭并重新打开，重读记忆、按真实余弦召回，再调用D4/D5。签名密钥仅本次进程持有，不表示已经验证跨进程重启。
- 两组Actor均生成工具提案/回答；protected调用D6，ASK不执行。所有允许动作仅写入sandbox.jsonl，不运行shell、邮件或网络操作。

产物：

- manifest.json：模型、编码和协议配置。
- api.jsonl：真实请求提示词、最终回答、token计数和失败信息；不记录思考过程或密钥。
- results.jsonl：每组写入、持久化、召回、提案、执行仲裁结果及判决链。
- *.sqlite：逐例持久化记忆。
- sandbox.jsonl：被允许的动作记录（record_only）。

## 解释边界

这是试运行，不是总体防御效果基准。每例只有一条候选记忆，因此不能验证Hubness、多条协调投毒或长期累积。
接口调用失败必须独立统计，不能按拦截算。
tool_allowed不是攻击成功；noop表示没有工具调用。需审阅提案/回答是否达到攻击目标，并评估良性任务是否完成。
如果baseline模型自身拒绝攻击，不能把两组都无危险行为归因于防御模块。
嵌入相似度不能可靠区分所有权限、收件地址和参数越权；D6默认阈值尚未针对该模型校准。
调阈值必须另用开发集，本6例不再用于独立测试结论。

## 120条全量写入评估

```powershell
& D:/Anaconda/envs/andypytorch/python.exe -B run.py module experiments.real_minja_full --embedding-model D:/Models/paraphrase-multilingual-MiniLM-L12-v2 --output-dir 07_runtime/minja-full-real --workers 3
```

只验证候选写入准入，不调用Actor或工具。120条全部执行，原先试测的6条与新执行114条单独统计；114条也来自同一自建数据集，不视作外部独立基准。summary.json提供全体、已见/新执行样本和各攻击类型结果。API日志逐例保存，错误不计为正确拦截。最多3个并发、禁用自动重试、固定原默认阈值。

## 真实向量检索/执行诊断

```powershell
& D:/Anaconda/envs/andypytorch/python.exe -B run.py module experiments.real_minja_diagnostics --embedding-model D:/Models/paraphrase-multilingual-MiniLM-L12-v2 --output-dir 07_runtime/minja-diagnostics-real
```

将20条MI-4与40条良性记忆直接签名写入SQLite，以隔离写入门控、观察检索后防御。12个查询重复3轮，每轮Top10。D5结果是标记和降权，不等于移除污染记忆；重复召回槽位不是独立样本。D6使用4条明确任务/工具提案，均不实际执行。该专项是组件诊断，不能作为端到端攻击成功率或泛化性能。

## 四路径真实模型评估入口

```powershell
# 各命令的输出目录必须不存在
& D:/Anaconda/envs/andypytorch/python.exe -B run.py module experiments.real_autowrite --embedding-model D:/Models/paraphrase-multilingual-MiniLM-L12-v2 --output-dir 07_runtime/autowrite-real
& D:/Anaconda/envs/andypytorch/python.exe -B run.py module experiments.real_reflection --embedding-model D:/Models/paraphrase-multilingual-MiniLM-L12-v2 --output-dir 07_runtime/reflection-real
& D:/Anaconda/envs/andypytorch/python.exe -B run.py module experiments.real_masw --embedding-model D:/Models/paraphrase-multilingual-MiniLM-L12-v2 --output-dir 07_runtime/masw-real
& D:/Anaconda/envs/andypytorch/python.exe -B run.py module experiments.real_type4 --embedding-model D:/Models/paraphrase-multilingual-MiniLM-L12-v2 --output-dir 07_runtime/type4-real
```

每个入口最多3并发，默认读取根目录.env中的DeepSeek配置；所有模型返回均检查finish_reason和结构，节点吞掉API错误也会导致该例被标记error。不会真正执行工具。--limit用于预检；它按原顺序截取，不能代表均衡小样本。

覆盖边界：

- AutoWrite：120条现成候选，实际DA/DB+MINJA写入管线，真实向量与DE条件LLM核查；每例独立会话，不把单条评估冒充洪水/群聚序列评估。
- Reflection：120条competition原文，真实LLM生成事实，再送原生Reflection门控。保留source信息；嵌入仅作源文/摘要相似度诊断，不替代原生规则。
- MASW：125条原文，真实LLM抽取事实、真实向量召回、真实LLM两组动作提案；风险/证据/来源/权限/仲裁沿用原生实现。缺少事实的生成器输出是模型弃权，不是防御拦截。
- Type4：200份原文，独立40条参考文本校准SP1（MiniLM+PCA16），真实向量扫描SP4、真实LLM关键词聚合SP5、真实向量与LLM-NLI代理SP6、真实LLM三元组SP7。SP2/SP3所需逐token条件概率未提供，交叉编码器未提供，明确不覆盖。SP7为背景诊断，不进入上传门控。SP5输出关键词共识而非完整答案。上传库为空时的SP5/SP6是强制绕过上传的独立组件诊断，不表示正常端到端链路。

标签仅用于评分。Reflection/MASW生成内容可能已经净化原攻击，因此原始attack标签下“有记忆写入”不等同于污染漏检；同时记录无门控/有门控实际内容的LLM自动审阅标记。该审阅由同一模型家族完成，不等同于人工真值或跨模型独立裁判，不能据此宣称准确的攻击成功率。

在2026-11-07测试中另修复两个接口功能问题：AutoWrite保存错误的前序链头；SP7把已带doc_id的三元组重复包装。两者均有回归，不涉及阈值调整。保留修复前原始证据；缓存回放只验证功能，不计为新模型推理。

## 2026-11-07 开发修订：正常校准与可选语义准入

SP1修复了新查询误删最近邻的问题。新增两个显式接口：
- EmbeddingAnomalyDetector.calibrate(clean_validation, clean_stats, false_alarm_rate=0.05)
- DefensePipeline.calibrate_sp1(clean_validation, false_alarm_rate=0.05)

必须先训练，并提供与训练/测试分离的正常验证集。校准是经验分位数，不保证新分布上的误报率；不传验证集时默认阈值仍保持原样。

生成的正常开发参考分为160条训练、80条验证、80条正常检查，划分在生成前固定。未给生成器旧测试样本。它是合成开发数据，不是外部独立基准。

保存可加载profile：

```powershell
& D:/Anaconda/envs/andypytorch/python.exe -B run.py module experiments.calibrate_type4_sp1 --normal-corpus <normal-corpus.jsonl路径> --embedding-model D:/Models/paraphrase-multilingual-MiniLM-L12-v2 --output-dir <新目录>
```

Type4真实入口通过 --sp1-profile <profile.json路径> 显式加载正常校准；会检查模型权重SHA256匹配。没有该参数仍使用之前的参考设置，不会自动替换旧实验结果。

Reflection和MASW新增可选LLM语义审查，可在真实评估入口添加 --semantic-review。原有规则不可被模型“安全”判断覆盖；模型失败/证据不在候选中则报错，不静默放行。
普通引用、威胁描述与采纳攻击者规则不同，自动审阅曾标记的案例不一定都应拒绝。该可选层的已知案例开发回归不等于独立泛化评估。

MINJA/AutoWrite共用的D3提示已明确：历史事实不是当前执行授权；不能仅因细节不足或未来需要审批就拒绝存储。现阶段只做了4条已知误拦+4条攻击的开发回归，尚未重新进行修改后的五路径全量评估。

Type4开发上传入口 experiments.real_type4_admission_development 使用校准SP1与语义审查的交集，验证上传准入，不执行检索/工具，也不等同于补齐SP2/SP3/交叉编码器。
