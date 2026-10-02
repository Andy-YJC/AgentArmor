"""Run the five active modules and isolated historical prototype tests."""

from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    suites = [
        (ROOT, ["-m", "pytest", "-q"]),
        (ROOT / "06_archive" / "masw_minimal", ["-m", "unittest", "discover", "-s", "MASW/tests"]),
        (ROOT / "06_archive" / "masw_prototype", ["-m", "unittest", "discover", "-s", "tests"]),
    ]
    failed = False
    for cwd, args in suites:
        print(f"\n== {cwd.name} ==", flush=True)
        # The two MASW packages intentionally have independent implementations.
        # A fresh process prevents one version from leaking through sys.modules.
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "03_defense"), str(ROOT / "02_tools"), env.get("PYTHONPATH", "")])
        result = subprocess.run([sys.executable, *args], cwd=cwd, env=env)
        failed |= result.returncode != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
