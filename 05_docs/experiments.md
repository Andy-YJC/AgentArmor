# 实验与复现入口

所有主线命令从仓库根目录执行。

## 安装与测试

```bash
python -m pip install -r requirements-dev.txt
python run.py check
python -m ruff check .
```

`python -m pytest` 检查五个主包；`02_tools/scripts/check.py` 额外在独立进程中检查两套历史原型。

## 跨类型统一实验

```bash
python run.py benchmark --no-figures
```

覆盖端到端链路、Type 4/5、撤销、消融、良性语料和性能统计。默认创建 `07_runtime/experiments/<时间戳>/`，包含 `data/`、`results/`、`artifacts/`。

需要生成图表时：

```bash
python -m pip install pillow
python run.py benchmark --output-dir 07_runtime/my-run
```

指定目录必须不存在或为空，防止覆盖已有运行。`04_benchmarks/` 保存原有实验材料。Windows 的 CPU 时间使用可用的系统接口；无法测量的峰值内存留空。

## 独立模块评估

```bash
python run.py module AutoWrite.tests.eval_autowrite --mode mock
python run.py module MINJA.tests.eval_minja --mode mock
python run.py module Reflection.tests.eval_reflection
python run.py module MASW.tests.eval_masw
python run.py module MASW.tests.ablation_eval
python run.py module MASW.tests.e2e_attack_validation
```

这些入口保留各自的实验协议与输出方式，部分会写入模块的 `tests/results/` 或审计日志；运行后检查文件变化。跨类型实验不自动运行这些独立评估。

## 外部接入与旧报告工具

```bash
python run.py integrations
```

外部框架接入检查与既有 Type 1–3 文件汇总写入 `07_runtime/external-rerun/results/`。通过 `AGENTARMOR_EXTERNAL_PYTHON` 可指定另一个已安装外部依赖的解释器。API 基线需显式配置相应 SDK、凭据和运行开关。

`python run.py legacy-report` 只更新 `04_benchmarks/legacy-report.tex` 的旧模板。当前作品报告保存在 `05_docs/report/agentarmor-report.pdf`，不由旧模板生成。

## 结果目录

| 子目录 | 内容 |
| --- | --- |
| `e2e/` | SQLite 记忆链路、工具沙箱和逐案例日志 |
| `type4/` | 文档投毒主集、困难集与基线 |
| `type5/` | 共享记忆困难集、协作和撤销 |
| `ablation/` | 组件消融 |
| `benign/` | 良性语料误伤分析 |
| `performance/` | 延迟、吞吐量和资源统计 |
| `type123/` | 历史独立模块结果汇总 |

Web 演示的运行证据写入 `07_runtime/demo/`；实验输出写入根目录 `07_runtime/`。两者均被 Git 忽略。
