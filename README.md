# autofly_ue5

Recreation of the AutoFly (arXiv 2602.09657) simulation side on Unreal Engine 5.7.4 + Project AirSim,
used to generate an AutoFly-format UAV navigation dataset.

- Design: `docs/superpowers/specs/2026-09-15-autofly-ue5-dataset-design.md`
- Plans: `docs/superpowers/plans/`
- Gate reports: `docs/gates/`

## Rules

- Run every Python process with `env -u PYTHONPATH` (the host leaks a ROS Jazzy path).
- Run every Unreal process with `DISPLAY=:1 SDL_VIDEODRIVER=x11`.
- Launch and stop simulators only with `scripts/launch_sim.py` / `scripts/stop_sim.py` (PID files in `runs/sim/`).
- Run builds, cooks, launches and gate runs through `scripts/run_job.sh start|wait|stop` (PID files in `runs/jobs/`).
- Unreal processes get `UE-ZenDataPath` / `UE-LocalDataCachePath` pointing into `ue_project/DerivedDataCache/`.

## Setup

```bash
bash scripts/setup_venv.sh                   # .venv with pinned requirements
env -u PYTHONPATH .venv/bin/python -m pytest # offline tests
bash scripts/fetch_plugin.sh                 # Project AirSim 1.0.1 Linux plugin, verified
bash scripts/create_ue_project.sh restore    # Blocks content + plugin into ue_project/ (after a fresh clone)
bash scripts/build_editor.sh                 # BlocksEditor Linux Development
```

`engine/` (Epic prebuilt 5.7.4) and `platform/` (Project AirSim at 4d878bf) are git-ignored and must be present.
