# autofly_ue5

Recreation of the AutoFly (arXiv 2602.09657) simulation side on Unreal Engine 5.7.4 + Project AirSim,
used to generate an AutoFly-format UAV navigation dataset.

- Design: `docs/superpowers/specs/2026-09-15-autofly-ue5-dataset-design.md`
- Plans: `docs/superpowers/plans/`
- Gate reports: `docs/gates/`; decision records: `docs/decisions/`
- Working notes for agents and people: `CLAUDE.md` (rules) and `MEMORY.md` (learnings)

## Rules

- Run every Python process with `env -u PYTHONPATH` (the host leaks a ROS Jazzy path).
- Run every Unreal process with `DISPLAY=:1 SDL_VIDEODRIVER=x11`.
- Launch and stop simulators only with `scripts/launch_sim.py` / `scripts/stop_sim.py` (PID files in `runs/sim/`).
- Run builds, cooks, launches and gate runs through `scripts/run_job.sh start|wait|stop` (PID files in `runs/jobs/`).
- Unreal processes get `UE_ZenDataPath` / `UE_LocalDataCachePath` (underscores: the engine rewrites `-` to `_` before
  its lookup) pointing into `ue_project/DerivedDataCache/`.

## Setup

Prerequisites:

- Linux x86_64 and Python 3.12. If the host has no `python3.12`, get one from uv, conda or deadsnakes and pass it as
  `PYTHON=/path/to/python3.12`.
- Offline work (tests, scene generation) needs nothing else; WSL2 is fine for it.
- Live work (building, packaging, simulating, training) needs the GPU host with the git-ignored `engine/` (Epic prebuilt
  5.7.4) and `platform/` (Project AirSim at 4d878bf), a running X display `:1`, Vulkan, `jq` (only
  `scripts/build_level.sh` uses it), and a user that can read the kernel journal (`adm` or `systemd-journal` group) --
  the GPU-fault audit fails closed without it.

```bash
bash scripts/setup_venv.sh                                   # .venv from requirements.lock (never rewrites it)
env -u PYTHONPATH .venv/bin/python scripts/build_scenes.py scenes/s01_white_pillars.json   # runs/levels/s01.*.json
env -u PYTHONPATH .venv/bin/python -m pytest                 # offline tests; under a minute on the GPU host
bash scripts/fetch_plugin.sh                                 # Project AirSim 1.0.1 Linux plugin, verified
bash scripts/create_ue_project.sh restore                    # Blocks content + plugin into ue_project/ (after a fresh clone)
bash scripts/build_editor.sh                                 # BlocksEditor Linux Development
```

`bash scripts/setup_venv.sh --relock` rebuilds the lock from `requirements.txt`; `--recreate` rebuilds `.venv` from
scratch. The test suite builds the s01 layout itself (`tests/conftest.py`) if `runs/levels/` is empty.
