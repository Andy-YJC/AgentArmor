# 历史原型

保留不同阶段的独立实现，便于追溯和回归；当前主线使用根目录 [`03_defense/MASW/`](../03_defense/MASW)。

| 目录 | 来源 | 独立测试 |
| --- | --- | --- |
| `masw_minimal/MASW/` | 原根目录简化版 | 进入 `06_archive/masw_minimal/`，执行 `python -m unittest discover -s 03_defense/MASW/tests` |
| `masw_prototype/` | 最早的独立共享写入原型 | 进入该目录，执行 `python -m unittest discover -s tests` |

不要将归档目录添加到主程序 `sys.path`。`02_tools/scripts/check.py` 会在独立工作目录和进程中验证这些代码。
