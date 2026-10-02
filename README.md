# 链甲 · AgentArmor

**基于写入路径分类的 LLM-Agent 全链记忆防御系统**

AgentArmor 面向具有长期记忆、RAG 知识库和共享工作区的 LLM Agent，研究恶意内容如何经由写入、检索和工具调用持续影响 Agent 行为，并为不同写入路径提供防御流水线。

本仓库包含五类防御模块、本地交互演示、实验脚本与研究材料，可用于阅读实现、验证攻击场景和开展消融评估。

[作品报告](05_docs/report/agentarmor-report.pdf) · [架构说明](05_docs/architecture.md) · [实验指南](05_docs/experiments.md) · [演示说明](01_demo-web/README.md)

## 防御思路

记忆污染可能从一次外部输入开始，经持久化存储跨会话保留，再通过检索影响后续决策。AgentArmor 围绕这一过程设置多层防御：

```text
外部输入 → 候选记忆 → 写入审查 → 记忆存储 → 检索复核 → 执行仲裁
                        │                      │             │
                        └────────── 审计与溯源 ──────────────┘
                                         │
                                   污染定位与级联撤销
```

各模块按自身场景组合输入净化、风险检测、来源验证、完整性保护和信任约束，保留独立配置与评估入口。

## 五类防御模块

| 类型 | 写入路径 | 模块 | 主要防御能力 |
| --- | --- | --- | --- |
| Type 1 | 框架自动写入 | [AutoWrite](03_defense/AutoWrite/) | Token 清洗、选择性写入、完整性链、时序衰减与分布监控 |
| Type 2 | Agent 自主写入 | [MINJA](03_defense/MINJA/) | 写入意图检测、因果审查、行为仿真、溯源签名与执行对齐 |
| Type 3 | 反思归纳写入 | [Reflection](03_defense/Reflection/) | 反思注入检测、证据锚定、事实一致性、策略审查与写入门控 |
| Type 4 | 外部开发者写入 | [External Developer Write](03_defense/External_Developer_Write/) | 文档异常检测、跨块连贯性、鲁棒检索与检索后验证 |
| Type 5 | 多 Agent 共享写入 | [MASW](03_defense/MASW/) | 信任标记、候选事实验证、共享写入网关、工具仲裁与级联撤销 |

当前主线实现位于 `03_defense/`。AutoWrite 复用 MINJA 的部分能力；MASW 使用完整版实现。各模块的防御节点与报告章节对应关系见 [架构说明](05_docs/architecture.md)。

## 快速开始

需要 **Python 3.10 或更新版本**。以下命令均在仓库根目录执行，建议使用独立的 Python 虚拟环境。

安装运行依赖并启动演示：

```bash
python -m pip install -r requirements.txt
python run.py demo
```

浏览器打开 **http://127.0.0.1:4173**。

本地演示无需配置 API Key。浏览器通过 Python 服务调用项目模块，支持单场景逐步执行、无防护与受保护路径对比，以及批量结果查看。工具执行使用本地沙箱，运行证据保存在 `07_runtime/demo/`。

若需更换端口，在 PowerShell 中执行：

```powershell
$env:AGENTARMOR_DEMO_PORT = "4183"
python run.py demo
```

更多操作见 [Web 演示说明](01_demo-web/README.md)。

## 仓库结构

目录按“运行入口 → 防御模块 → 结果与说明”连续编号：

```text
AgentArmor/
├── 01_demo-web/                   # 本地演示服务与前端
├── 02_tools/                      # 开发与实验工具
│   ├── scripts/                   # 全仓库测试入口
│   └── experiments/               # 实验、外部接入与旧报告工具
├── 03_defense/                    # 五类防御实现
│   ├── AutoWrite/
│   ├── MINJA/
│   ├── Reflection/
│   ├── External_Developer_Write/
│   └── MASW/
├── 04_benchmarks/                 # 实验数据、归档结果和图表
├── 05_docs/                       # 作品报告与使用文档
├── 06_archive/                    # 历史原型
├── 07_runtime/                    # 新运行产物，不纳入版本管理
├── run.py                         # 统一命令入口
├── pyproject.toml                 # 测试与静态检查配置
├── requirements.txt               # 运行依赖
├── requirements-dev.txt           # 开发与测试依赖
├── .env.example                   # 可选远程基线配置示例
└── LICENSE
```

`run.py` 负责配置编号目录下的模块搜索路径。运行独立模块时使用 `python run.py module <模块名>`，包名仍为 `AutoWrite`、`MINJA` 等原名称。

## 测试与代码检查

安装开发依赖后执行：

```bash
python -m pip install -r requirements-dev.txt
python run.py check
python -m ruff check .
```

`check` 检查五个主线模块，并在独立进程中测试两套历史原型。仅检查主线模块可执行：

```bash
python -m pytest
```

## 实验运行

运行端到端链路、Type 4/5、撤销、消融、良性语料和性能实验：

```bash
python run.py benchmark --no-figures
```

结果默认写入 `07_runtime/experiments/<时间戳>/`，包含数据、指标、日志及运行记录。

生成图表并指定输出目录：

```bash
python -m pip install pillow
python run.py benchmark --output-dir 07_runtime/my-run
```

指定目录必须不存在或为空。已有研究材料保存在 `04_benchmarks/`。

独立模块评估示例：

```bash
python run.py module AutoWrite.tests.eval_autowrite --mode mock
python run.py module MINJA.tests.eval_minja --mode mock
python run.py module Reflection.tests.eval_reflection
python run.py module MASW.tests.eval_masw
```

独立评估保留各自的数据协议与输出位置，部分结果写入模块的 `tests/results/`。完整命令、外部框架接入和产物说明见 [实验指南](05_docs/experiments.md)。

## 配置与开发

- **防御配置**：查看各模块的配置类及流水线构造参数，按场景启用防御节点。
- **可选远程基线**：参考 [.env.example](.env.example) 配置凭据与运行开关；本地演示和默认实验可独立运行。
- **历史代码**：`06_archive/` 保留早期原型，主程序使用 `03_defense/` 中的实现。
- **运行产物**：`07_runtime/`、审计日志和缓存已配置 Git 忽略规则；实验数据与归档结果按原有文件保留。

目录迁移、命名与依赖约定见 [仓库维护说明](05_docs/repository-guide.md)。

## 文档与许可

| 文档 | 内容 |
| --- | --- |
| [作品报告](05_docs/report/agentarmor-report.pdf) | 系统设计、实现思路与实验分析 |
| [架构说明](05_docs/architecture.md) | 五类路径、源码位置与模块依赖 |
| [实验指南](05_docs/experiments.md) | 评估命令、依赖和结果目录 |
| [演示说明](01_demo-web/README.md) | 交互场景、批量测试和运行证据 |
| [仓库维护说明](05_docs/repository-guide.md) | 目录、命名和开发约定 |

本项目采用 [MIT License](LICENSE)。
