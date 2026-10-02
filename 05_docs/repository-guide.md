# 仓库维护说明

## 目录职责

- `03_defense/` 的五个算法包：当前主线实现。
- `02_tools/experiments/`：跨模块实验和工具入口。
- `04_benchmarks/`：已有数据、结果、图表、协议和旧报告快照。
- `01_demo-web/`：本地展示与交互服务。
- `05_docs/`：作品报告、架构和使用说明。
- `06_archive/`：早期原型，仅用于追溯与隔离回归。
- `07_runtime/`：新生成的实验产物，不提交。

## 目录顺序与迁移

数字前缀按 01–07 连续排列：01–02 为前置入口，03 为防御模块，04–07 为材料、说明、归档和输出。

| 原路径 | 当前路径 |
| --- | --- |
| `demo-web/` | `01_demo-web/` |
| `scripts/` | `02_tools/scripts/` |
| `experiments/` | `02_tools/experiments/` |
| 五个算法包 | `03_defense/` 下的同名包 |
| `benchmarks/` | `04_benchmarks/` |
| `docs/` | `05_docs/` |
| `archive/` | `06_archive/` |
| `runtime/` | `07_runtime/` |

根目录 `run.py` 统一配置模块搜索路径；包本身保留原名称。用 `python run.py demo` 启动演示，用 `python run.py check` 测试，用 `python run.py module 包名.模块名` 运行独立评估。

隐藏的 `.git`、`.tools` 是版本管理和本地开发工具目录，不参与业务目录编号。历史实验文件里的旧路径作为原始记录保留。

## 命名与依赖

保留现有算法包名和对外接口。Python 文件采用 `snake_case.py`；Web 脚本采用 `kebab-case.js`。报告 R/E 编号与实现 D/SP 编号的关系见 [架构说明](architecture.md)。

AutoWrite 统一导入 MINJA 共享类型，不依赖仓库检出目录的名字。Type 4 的 pytest fixtures、数据构造和指标计算分别放在 `conftest.py`、`dataset_helpers.py`、`metrics.py`。

主程序只导入`03_defense/MASW/`，禁止将归档原型加入主程序搜索路径。保留归档测试时使用独立进程。

## 产物约定

论文、数据集和既有实验输出保留原内容。缓存、游离日志、临时数据库、旧预览截图不纳入版本管理。统一实验只写入新的运行目录；旧的独立模块评估和部分测试仍可能重写自身结果文件，运行后检查 Git 差异。

新增功能先定位所属防御路径；可共享逻辑须确认语义一致后再抽取，避免把不同信任模型或实验协议强行合并。

## 回归命令

```bash
python run.py check
python -m ruff check .
node --check 01_demo-web/scenarios.js
node --check 01_demo-web/batch-results.js
```
