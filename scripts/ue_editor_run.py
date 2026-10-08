"""Run one UE-side Python script in the full editor (headless), the way scripts/import_assets.py and build_level.sh do.

    env -u PYTHONPATH .venv/bin/python scripts/ue_editor_run.py autofly_ue5/scenes/ue/foliage_materials.py \\
        --out runs/assets/foliage_materials.json [--arg key=value ...]

The script reads env AUTOFLY_UE_OUT (where to write its JSON report) and AUTOFLY_UE_ARGS (a JSON object of the --arg
pairs), and must end by asking the editor to quit. Exit status follows the report's "pass", not the editor's exit code
(the editor's is unreliable after quit_editor). The editor log goes next to the report (<out>.log, .stdout).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

from autofly_ue5.paths import ENGINE_DIR, UE_CACHE_ENV, UE_PROJECT_DIR, UPROJECT, ZEN_DATA_DIR  # noqa: E402


def run(script: Path, out: Path, args: dict) -> int:
    editor = ENGINE_DIR / "Engine" / "Binaries" / "Linux" / "UnrealEditor"
    out = out.resolve()  # the editor runs in the UE project directory: a relative path would land there
    ZEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(DISPLAY=":1", SDL_VIDEODRIVER="x11", AUTOFLY_UE_OUT=str(out), AUTOFLY_UE_ARGS=json.dumps(args), **UE_CACHE_ENV)
    log = out.with_suffix(".log")
    cmd = [str(editor), str(UPROJECT), f"-ExecCmds=py {script.resolve()}", "-unattended", "-nop4", "-nosplash", "-notraceserver",
           "-RenderOffscreen", f"-ZenDataPath={ZEN_DATA_DIR}", "-stdout", "-FullStdOutLogOutput", f"-abslog={log}"]
    with open(str(log) + ".stdout", "w") as handle:
        subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, cwd=str(UE_PROJECT_DIR))
    if not out.exists():
        print(f"no report at {out}: see {log}", file=sys.stderr)
        return 2
    report = json.loads(out.read_text())
    print(json.dumps({"pass": report.get("pass"), "error": report.get("error"), "report": str(out)}))
    return 0 if report.get("pass") else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("script", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arg", action="append", default=[], help="key=value, passed to the script as JSON")
    a = p.parse_args(argv)
    args = dict(kv.split("=", 1) for kv in a.arg)
    return run(a.script, a.out, args)


if __name__ == "__main__":
    sys.exit(main())
