"""Stable command entry point for the numerically ordered repository."""

from pathlib import Path
import argparse
import os
import runpy
import sys


ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "demo": ROOT / "01_demo-web/server.py",
    "check": ROOT / "02_tools/scripts/check.py",
    "benchmark": "experiments.run_benchmarks",
    "integrations": "experiments.check_integrations",
    "legacy-report": "experiments.update_legacy_report",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[*COMMANDS, "module"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "03_defense"), str(ROOT / "02_tools")]
    os.chdir(ROOT)
    if args.command == "module":
        if not args.arguments:
            parser.error("module requires a Python module name")
        target, *arguments = args.arguments
    else:
        target, arguments = COMMANDS[args.command], args.arguments
    sys.argv = [str(target), *arguments]
    if isinstance(target, Path):
        runpy.run_path(str(target), run_name="__main__")
    else:
        runpy.run_module(target, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
