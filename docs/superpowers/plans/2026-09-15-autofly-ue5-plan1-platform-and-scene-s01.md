# AutoFly UE5 Dataset — Plan 1: Platform and Scene s01 (M0–M1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install and prove UE 5.7.4 + Project AirSim 1.0.1 on this machine (M0 gate), then deliver the `autofly_ue5.sim` interface with a Project AirSim backend and a fake, and scene s01 (white pillars) generated from JSON, built into a UE map, packaged, and validated live (M1 gate).

**Architecture:** Epic's prebuilt engine and Project AirSim's prebuilt Linux plugin are combined in `ue_project/` (a copy of the Blocks sample), built once with `Build.sh`, and run headless as processes whose PIDs this project records. All Python talks to the simulator through `autofly_ue5/sim/` only: a lock-step backend (steppable clock, 0.2 s per record, image topics filtered by timestamp) behind a `Simulator` Protocol that a fake also implements for offline tests. Scenes are JSON files expanded by a seeded generator, checked for reachability on a 2D occupancy grid, converted to a UE-centimetre level spec in the venv, and turned into a saved map by a stdlib-only UE Python commandlet script.

**Tech Stack:** Unreal Engine 5.7.4 (Epic prebuilt Linux, bundled clang 20.1.8), Project AirSim plugin 1.0.1 (prebuilt SimLibs) + `projectairsim==1.0.2` client, UE PythonScriptPlugin (embedded Python 3.11.8), system Python 3.12 venv, numpy 1.26.4, pillow 11.2.1, jsonschema 4.23.0, pytest 8.3.5, bash, `ss`, `nvidia-smi`, `journalctl`.

**Spec:** /home/nvidiasims/research_uav/autofly_ue5/docs/superpowers/specs/2026-09-15-autofly-ue5-dataset-design.md

## Global Constraints

- ROOT is `/home/nvidiasims/research_uav/autofly_ue5`; everything this plan creates lives under ROOT (spec §5). Known exception, not redirectable in the installed engine: UE itself writes under `~/.config/Epic/` (`UnrealEngine/5.7/` user config and saved data, `UnrealEngine/Common/Zen/Install/` zenserver binary copy, `UnrealBuildTool/` logs and config; `Paths.cpp:204-221`, `UnixPlatformProcess.cpp:340-355`, `ZenServerInterface.cpp:155-158`). `~/.config/Epic/AirVLN` and `UnrealEngine/4.27` belong to another project and are never touched. `engine_check` (Task 2) and `m0_gate faults` (Task 8) record `du -sk ~/.config/Epic` before and after M0. Other out-of-ROOT locations and how this plan keeps them unused: every non-Shipping UE process forks `engine/Engine/Binaries/Linux/UnrealTraceServer` (a daemon outside the recorded process group that writes `~/UnrealEngine/UnrealTrace/` and `/tmp/UnrealTraceServer.pid` and listens on TCP 1981/1989; `TraceAuxiliary.cpp:1980-1986,2553-2590`) unless `-notraceserver` is passed, so every editor, commandlet and game command line carries `-notraceserver` and the RunUAT cook gets `-AdditionalCookerOptions=-notraceserver`; RunUAT on an installed engine writes and clears its logs in `~/Documents/Unreal Engine/LocalBuildLogs/` (`CommandEnvironment.cs:104-122`, `LinuxHostPlatform.cs:64-67`) unless `uebp_LogFolder` is set, so `scripts/package_sim.sh` sets `uebp_LogFolder=ROOT/runs/package/uat_logs`; the Task 2 editor probe (no project) can create an empty `~/Documents/Unreal Projects` (`SProjectDialog.cpp:1748-1752`), which `engine_check` removes only when the probe created it and it is empty. `engine_check` records these paths before M0 and `m0_gate faults` fails when any of them was created during M0.
- While a simulator runs, its topics and services ports (8989/8990, 9001/9002) listen on all interfaces without authentication (`simserver.cpp:42-44`), so any host on the LAN can reach them. The executor never changes firewall rules (that needs sudo); the M0 milestone report asks the user whether a host firewall rule is wanted.
- Tool-call timeouts: the Bash tool defaults to 120 s. Every Bash tool call that runs `run_job.sh wait … 540`, `scripts/setup_venv.sh`, or `scripts/stop_sim.py` followed by `sleep` sets the tool's `timeout` parameter to 600000 ms, and a single call never contains more than one `run_job.sh wait … 540`.
- Every Unreal process (editor, commandlet, packaged game, RunUAT cook) runs with `UE-ZenDataPath=/home/nvidiasims/research_uav/autofly_ue5/ue_project/DerivedDataCache/Zen` and `UE-LocalDataCachePath=/home/nvidiasims/research_uav/autofly_ue5/ue_project/DerivedDataCache` (`autofly_ue5.paths.UE_CACHE_ENV`; read at `ZenServerInterface.cpp:861` and `BaseEngine.ini:2760`), so the derived-data cache stays inside ROOT instead of `~/.config/Epic/UnrealEngine/Common/Zen/Data`; the M0 gate fails if that default Zen data folder is created during M0.
- Never modify other projects under `/home/nvidiasims/research_uav` (phi_aerovla, AirVLN, TravelUAV, …) or `~/Documents/AirSim`.
- Engine: `ROOT/engine`, Epic prebuilt `Linux_Unreal_Engine_5.7.4.zip`, `Build.version` 5.7.4 CL 51494982, bundled toolchain `v26_clang-20.1.8-rockylinux8` (spec §4).
- Platform checkout: `ROOT/platform` at `4d878bf`; tag `v1.0.1` = `0975545`; C++ dirs must show no diff between them (spec §4).
- Plugin: `ProjectAirSim-Plugin-Linux-UE5_7-1.0.1.zip`, 669,317,542 bytes, sha256 `11f016ac7aa292a1a353dfde9ff7c4a1b5cebd660cf9e8f400f2ed4c030dcb49`; SimLibs are Release-only, so build and package **Development** only (never DebugGame).
- Python client: `projectairsim==1.0.2` in `ROOT/.venv`, created from `/usr/bin/python3.12`.
- Every Python process runs with `env -u PYTHONPATH` (host exports `/opt/ros/jazzy/lib/python3.12/site-packages`).
- Every UnrealEditor / UnrealEditor-Cmd / packaged-game process runs with `DISPLAY=:1 SDL_VIDEODRIVER=x11`.
- Skip `setup_linux_dev_tools.sh` and `setup_linux_unreal_prereqs.sh`; `SetupToolchain.sh` does not exist in the prebuilt engine and is not needed (UBT reads the in-tree SDK).
- Before launching a simulator: GPU check — `memory.used` ≤ 2000 MiB when none of our instances runs, and ≥ 6000 MiB free in every case (`autofly_ue5/gpu.py`).
- Only stop processes this project launched: each simulator PID/PGID/cmdline is recorded in `ROOT/runs/sim/inst<N>/pid.json`; never use `pkill -f` or kill a PID that is not recorded there.
- Ports: instance N uses topics `8989 + 12·N` and services `8990 + 12·N` (inst0 8989/8990, inst1 9001/9002).
- `autofly_ue5/sim/` is the only package code that imports `projectairsim` (spec §5, §7).
- Units: Project AirSim world is NED metres (+z down), yaw in radians; UE is X forward, Y right, Z up, centimetres; `(X, Y, Z)_cm = (100·x, 100·y, −100·z)` with no origin offset; UE yaw (deg) = NED yaw (deg); AutoFly `z_up = −z_ned` (spec §3.2).
- Clock: `"type": "steppable"`, `"step-ns": 5000000`, `"real-time-update-rate": 3000000`, `"pause-on-start": true`; one record = one `world.step(200000000)` (5 Hz); velocity command `duration = dt − 2·step-ns = 0.19` s sent before the step, a 2 ms client pause, the step, then the command reply awaited with a timeout (`CommandTimeoutError`). The server starts the duration at the sim time its request job reads (`core_sim/src/service_manager.cpp:440`, `vehicle_apis/include/common/function_caller.hpp` `IsTimeout`), which can be one 5 ms tick after the Step job has started the clock; with `dt − 1e-3` the reply would then never arrive inside the step. The 10 ms without a new goal stays far below simple flight's 60 ms hover fallback (`OffboardApi.hpp:56-66`, `Params.hpp:479`).
- Long-running commands (anything that can exceed 5 minutes: the engine check, downloads, builds, commandlets, packaging, simulator launches, smoke and gate runs) never run as a foreground tool call. Start them with `bash /home/nvidiasims/research_uav/autofly_ue5/scripts/run_job.sh start <name> -- <cmd…>` (new session; PID, PGID and command recorded in `runs/jobs/<name>.pid.json`; output in `runs/jobs/<name>.log`) and poll with `bash /home/nvidiasims/research_uav/autofly_ue5/scripts/run_job.sh wait <name> 540` (returns within 9 minutes: 0 = finished with exit 0, 1 = finished with another exit code, 124 = still running so call `wait` again, 3 = stopped or died). Each step states a total wait budget; when it is used up, run `bash scripts/run_job.sh stop <name>`, then `scripts/stop_sim.py --instance N` for any simulator the job launched, and report. Never re-launch after a timeout.
- UE commandlets (`UnrealEditor-Cmd … -run=pythonscript`) return exit code 1 whenever any error was logged, even when the script succeeded (`LaunchEngineLoop.cpp:4160-4166`; PythonScriptPlugin never sets `UseCommandletResultAsExitCode`). Gate only on the log line `Python script executed successfully` plus the script's JSON report; record the exit code and `grep -c "Error:"` of the log as information.
- Camera: `FrontCamera`, 256×256, `fov-degrees` 90, image type 0 (RGB, encoding `BGR`) and 1 (DepthPlanar, encoding `16FC1`, metres, `+inf` = no hit), `compress: false`, `capture-interval: 0.001`, origin `0.40 0.0 0.0`.
- Scenes: bounds 70 × 70 m centred on the origin; start band 2–6 m inside the boundary; target band 0–3 m inside; altitude band 1.0–3.0 m; reachability clearance 1.0 m (spec §6).
- Failure rule: when a live step fails or an UNCONFIRMED fact is contradicted, stop and report (command, exit code, log/report path, last 50 log lines); never switch approach silently. Before reporting, stop every simulator instance the current task launched (`env -u PYTHONPATH .venv/bin/python scripts/stop_sim.py --instance N`) and every job of the task still running (`bash scripts/run_job.sh stop <name>`), unless the step explicitly says to keep one, and include their outputs (`terminated`/`killed`/`not_running`) plus `nvidia-smi --query-gpu=memory.used --format=csv,noheader` in the report. The Cosys-AirSim `5.8-v3.4.1` fallback is decided only with the user (spec §4).
- Milestone rule: after Task 8 (M0 gate) stop for the user's go-ahead before Task 9 (spec §12).
- Git: no pushes, no remote. Commit with `git -C /home/nvidiasims/research_uav/autofly_ue5 add <paths> && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "<message>" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"`.
- Evidence-backed deviations from the spec text adopted here: images come from topic subscriptions filtered by `time_stamp == T`, not from `get_images` (it is only probed at M0); runtime `set_object_material` is exercised with the opaque base `UMaterial` `/Game/Geometry/Materials/M_Orange` because the server loads with a `UMaterial` class filter (`WorldSimApi.cpp:1030-1032`), and the spec's material-instance route is attempted once with the `MaterialInstanceConstant` `/Game/Geometry/Materials/M_Blue` and recorded (`material_instance_ok`, predicted `false`), while s01 pillar colour is baked into the level with a material instance by the editor script; the reachability check (Task 13) keeps the spec §6.2 rule and adds a crossing rule on top of it (every free start cell must also reach the target band of the opposite edge through the obstacle field, not around its perimeter), because the spec rule alone accepts every layout whose outer 6 m ring is free, which makes it vacuous for s01. The crossing rule only rejects more layouts; it is put to the user for confirmation in the M0 milestone report (before any M1 work) and listed again in the M1 milestone report.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `.gitignore` | Modify | Ignore engine, platform, venv, runs, downloads, UE build outputs, plugin copy, UE content copy, client logs |
| `README.md` | Create | How to set up, build, launch and test; pointers to spec and plans |
| `pyproject.toml` | Create | Package metadata, editable install, pytest configuration |
| `requirements.txt` | Create | Direct pins (projectairsim 1.0.2 and the libraries below) |
| `requirements.lock` | Create (generated) | Full `pip freeze` of the working venv |
| `docs/superpowers/plans/2026-09-15-autofly-ue5-plan1-platform-and-scene-s01.md` | Create (this file, committed in Task 1) | This plan |
| `scripts/setup_venv.sh` | Create | Create `.venv` from python3.12 without PYTHONPATH, install pins, editable install, write lock |
| `scripts/run_job.sh` | Create | Start, wait for and stop long-running commands owned by this project (PID/PGID files in `runs/jobs/`) |
| `autofly_ue5/__init__.py` | Create | Package marker and version |
| `autofly_ue5/paths.py` | Create | Absolute project paths and the UE derived-data-cache environment used everywhere |
| `autofly_ue5/gpu.py` | Create | nvidia-smi memory parsing and the pre-launch GPU rule |
| `autofly_ue5/frames.py` | Create | NED ↔ UE-cm conversions, yaw/quaternion helpers, body-to-NED rotation, angle wrapping |
| `autofly_ue5/validate/__init__.py` | Create | Package marker |
| `autofly_ue5/validate/engine_check.py` | Create | M0 engine readiness checks and report; Xid (across reboots), boot id, `VK_ERROR_DEVICE_LOST`, `~/.config/Epic` and other out-of-ROOT location helpers |
| `autofly_ue5/validate/plugin_manifest.py` | Create | Verify the unzipped plugin against its `build-manifest.json` |
| `autofly_ue5/validate/m0_gate.py` | Create | VRAM samples, GPU-fault, reboot, `~/.config/Epic` and out-of-ROOT records, two-instance overlap check, M0 gate report assembly |
| `autofly_ue5/validate/geometry.py` | Create | Ray/cylinder depth expectation, depth-probe selection, depth agreement, tracking-error metrics |
| `autofly_ue5/validate/live_m1.py` | Create | M1 gate live checks through the `Simulator` interface |
| `autofly_ue5/sim/__init__.py` | Create | Package marker |
| `autofly_ue5/sim/process.py` | Create | Command lines, launch with PID file, port readiness, owned-only stop, handshake, client-log routing |
| `autofly_ue5/sim/decode.py` | Create | Decode `BGR` RGB and `16FC1` depth image messages |
| `autofly_ue5/sim/smoke_m0.py` | Create | M0 live smoke test against a running simulator, JSON report, fixtures |
| `autofly_ue5/sim/types.py` | Create | `Pose`, `CollisionEvent`, `Observation`, `ObjectNotFoundError`, `dt_to_ns` |
| `autofly_ue5/sim/protocol.py` | Create | `Simulator` Protocol (spec §7) |
| `autofly_ue5/sim/fake.py` | Create | `FakeSimulator` for offline tests |
| `autofly_ue5/sim/sync.py` | Create | `FrameCollector`: timestamp-exact frame wait |
| `autofly_ue5/sim/events.py` | Create | Collision event parsing, de-duplication, episode filtering |
| `autofly_ue5/sim/airsim_backend.py` | Create | `ProjectAirSimSimulator` implementing the Protocol |
| `autofly_ue5/sim/check_map.py` | Create | Confirm a launched simulator has the s01 map loaded |
| `autofly_ue5/scenes/__init__.py` | Create | Package marker |
| `autofly_ue5/scenes/scene.schema.json` | Create | JSON Schema of scene files (spec §6.1) |
| `autofly_ue5/scenes/model.py` | Create | Dataclasses; load/validate scene files and the asset registry |
| `autofly_ue5/scenes/generate.py` | Create | Seeded layout generator with `jittered_grid` placement |
| `autofly_ue5/scenes/reachability.py` | Create | 2D occupancy grid and start→target reachability check (spec §6.2) |
| `autofly_ue5/scenes/level_spec.py` | Create | Layout → UE-centimetre level spec |
| `autofly_ue5/scenes/ue/build_level.py` | Create | UE-side (embedded Python) map builder |
| `autofly_ue5/scenes/ue/verify_level.py` | Create | UE-side reload-and-verify of the saved map |
| `autofly_ue5/scenes/ue/probe_python.py` | Create | UE-side probe proving the editor Python APIs in a commandlet (Task 4) |
| `assets/registry.json` | Create (Task 12), Modify (Task 14) | Logical assets/materials used by s01 (engine primitives only): UE path, size, pivot, category, seen/unseen, bounds measured live in Task 14 |
| `scenes/s01_white_pillars.json` | Create | Scene s01 file |
| `configs/robot_autofly_quadrotor.jsonc` | Create | Fast-physics quadrotor with FrontCamera RGB + DepthPlanar |
| `configs/scene_autofly_m0.jsonc` | Create | M0 scene on BlocksMap, 3 ms real-time update |
| `configs/scene_autofly_m0_fast.jsonc` | Create | Same with 1 ms real-time update (throughput measurement) |
| `configs/scene_autofly_s01.jsonc` | Create | s01 scene config (drone origin in the south start band, x = −33 m) |
| `scripts/fetch_plugin.sh` | Create | Download, size/sha256 check, unzip, manifest verification |
| `scripts/create_ue_project.sh` | Create | `init`/`restore` of `ue_project` from Blocks + plugin |
| `scripts/build_editor.sh` | Create | `BlocksEditor Linux Development` build with checks |
| `scripts/launch_sim.py` | Create | CLI: GPU check, launch editor-game or packaged instance, readiness, handshake |
| `scripts/stop_sim.py` | Create | CLI: stop an owned instance |
| `scripts/build_scenes.py` | Create | CLI: scene file → layout + reachability + level spec JSON |
| `scripts/build_level.sh` | Create | Run the UE build and verify commandlets |
| `scripts/package_sim.sh` | Create | `RunUAT BuildCookRun` Development package with S01 and BlocksMap |
| `ue_project/Blocks.uproject` | Create | Project descriptor (5.7, Python plugins, Linux target) |
| `ue_project/Config/DefaultGame.ini` | Create | Packaging settings (cook dirs and maps) |
| `ue_project/Config/*` (other ini), `ue_project/Source/*` | Create (copied) | Blocks sample config and module sources |
| `tests/test_paths.py` | Create | Paths and UE cache environment |
| `tests/test_run_job.py` | Create | `scripts/run_job.sh` start/wait/stop |
| `tests/test_gpu.py` | Create | GPU rule |
| `tests/test_engine_check.py` | Create | Engine-check parsers |
| `tests/test_plugin_manifest.py` | Create | Manifest verification |
| `tests/test_process.py` | Create | Launch/readiness/stop with a fake server |
| `tests/test_frames.py` | Create | Unit conversions |
| `tests/test_decode.py` | Create | Synthetic image decoding |
| `tests/test_pas_configs.py` | Create | Configs validate against the Project AirSim schema |
| `tests/test_m0_gate.py` | Create | M0 gate assembly |
| `tests/test_sim_types_fake.py` | Create | Types and FakeSimulator |
| `tests/test_sync_events.py` | Create | FrameCollector and CollisionLog |
| `tests/test_decode_fixtures.py` | Create | Decoding real M0 messages |
| `tests/test_airsim_backend.py` | Create | Backend orchestration with fake client/world/drone |
| `tests/test_scene_model.py` | Create | Schema, scene and registry loading |
| `tests/test_generate.py` | Create | jittered_grid and layout generation |
| `tests/test_reachability.py` | Create | Occupancy and reachability |
| `tests/test_level_spec.py` | Create | Level spec conversion |
| `tests/test_check_map.py` | Create | Bounding-box comparison used to confirm the cooked map |
| `tests/test_geometry.py` | Create | Depth expectation, probes, depth agreement, tracking metrics |
| `tests/test_live_m1_fake.py` | Create | M1 gate check functions run against `FakeSimulator` and the generated s01 layout |
| `tests/fixtures/pas/{rgb_msg,depth_msg}.{json,bin}` | Create (captured) | Real image messages captured at M0 |
| `docs/gates/*.json` | Create (captured) | Committed copies of the M0/M1 gate reports and the M1 package manifest |

---

## Milestone M0 — platform installed, sample environment runs

### Task 1: Project skeleton and pinned Python environment

**Files:**
- Modify: `.gitignore`
- Create: `README.md`, `pyproject.toml`, `requirements.txt`, `requirements.lock` (generated), `scripts/setup_venv.sh`, `scripts/run_job.sh`, `autofly_ue5/__init__.py`, `autofly_ue5/paths.py`
- Commit (already on disk, untracked): `docs/superpowers/plans/2026-09-15-autofly-ue5-plan1-platform-and-scene-s01.md`
- Test: `tests/test_paths.py`, `tests/test_run_job.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `autofly_ue5.paths` constants (all `pathlib.Path`): `ROOT`, `ENGINE_DIR`, `PLATFORM_DIR`, `UE_PROJECT_DIR`, `UPROJECT`, `CONFIGS_DIR`, `SCENES_DIR`, `ASSET_REGISTRY`, `RUNS_DIR`, `JOBS_DIR`, `FIXTURES_DIR`, `UNREAL_EDITOR`, `UNREAL_EDITOR_CMD`, `PACKAGED_BINARY`, `DDC_DIR`, `ZEN_DATA_DIR`; plus `UE_CACHE_ENV: dict[str, str]` (`{"UE-ZenDataPath": str(ZEN_DATA_DIR), "UE-LocalDataCachePath": str(DDC_DIR)}`).
  - `bash scripts/run_job.sh start <name> -- <cmd…>` (prints `job <name> started: pid <PID>, log <path>`; exit 1 if a job of that name is still running), `bash scripts/run_job.sh wait <name> [timeout_s=540]` (prints the log tail and `job <name> finished: exit=<code>`; returns 0 / 1 / 124 still running / 3 stopped or died), `bash scripts/run_job.sh stop <name>` (prints `terminated`, `killed`, `not_running` or `no_job`; refuses a PID whose argv marker or process group does not match). Files: `$AUTOFLY_JOBS_DIR` (default `ROOT/runs/jobs`)`/<name>.{pid.json,log,exit,stopped}`.
  - The interpreter `ROOT/.venv/bin/python` with `projectairsim==1.0.2` and `autofly_ue5` (editable) installed.

Dependency choice: `jsonschema` validates scene files declaratively (it is already a dependency of `projectairsim`, so it adds nothing, and the schema file doubles as format documentation with path-accurate error messages); standard-library `dataclasses` hold the typed in-memory objects. numpy 1.26.4, pillow 11.2.1, pytest 8.3.5, opencv-python 4.11.0.86, matplotlib 3.10.3 and msgpack 1.1.0 are the versions already used on this machine by another project; pinning opencv/matplotlib stops pip from pulling numpy-2-only releases.

- [ ] **Step 1: Confirm the starting state**

Run:
```bash
git -C /home/nvidiasims/research_uav/autofly_ue5 status --short && git -C /home/nvidiasims/research_uav/autofly_ue5 log --oneline | head -3 && /usr/bin/python3.12 --version && ls /home/nvidiasims/research_uav/autofly_ue5
```
Expected: status prints exactly one line, `?? docs/superpowers/plans/` (this plan, committed in Step 12); top commit `a9e4098 Pin UE 5.7.4, …`; `Python 3.12.3`; listing shows `docs engine platform`. If status shows anything else, stop and report. Also run `pgrep -a zenserver || echo none` and `du -sk ~/.config/Epic` and note both outputs (expected `none`; if a zenserver is already running, stop and report it, because UE may attach to a running instance on port 8558 whose data path is not ours).

- [ ] **Step 2: Replace `.gitignore`**

Write `/home/nvidiasims/research_uav/autofly_ue5/.gitignore`:
```gitignore
# large or generated content: never commit
/engine/
/platform/
/.venv/
/runs/
/checkpoints/
/data/
/downloads/
ue_project/Binaries/
ue_project/Intermediate/
ue_project/Saved/
ue_project/Saved_*/
ue_project/DerivedDataCache/
ue_project/Packaged/
ue_project/Plugins/
ue_project/Content/
ue_project/projectairsim_server.log
ue_project/.vscode/
ue_project/*.code-workspace
projectairsim_client.log
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
```
(`ue_project/Content/` is regenerated: Blocks content is copied by `scripts/create_ue_project.sh restore`, AutoFly maps and materials by `scripts/build_level.sh`.)

- [ ] **Step 3: Write `pyproject.toml`, `requirements.txt`, the package marker and `README.md`**

`/home/nvidiasims/research_uav/autofly_ue5/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "autofly_ue5"
version = "0.1.0"
description = "AutoFly-format UAV navigation dataset generation on UE 5.7.4 + Project AirSim"
requires-python = ">=3.12,<3.13"

[tool.setuptools.packages.find]
include = ["autofly_ue5*"]

[tool.setuptools.package-data]
"autofly_ue5.scenes" = ["*.json"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--import-mode=importlib -q"
```

`/home/nvidiasims/research_uav/autofly_ue5/requirements.txt`:
```text
projectairsim==1.0.2
numpy==1.26.4
pillow==11.2.1
pytest==8.3.5
jsonschema==4.23.0
opencv-python==4.11.0.86
matplotlib==3.10.3
msgpack==1.1.0
```

`/home/nvidiasims/research_uav/autofly_ue5/autofly_ue5/__init__.py`:
```python
"""AutoFly-format dataset generation on Unreal Engine 5.7.4 + Project AirSim."""

__version__ = "0.1.0"
```

`/home/nvidiasims/research_uav/autofly_ue5/README.md`:
````markdown
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
````

- [ ] **Step 4: Write `scripts/setup_venv.sh`**

`/home/nvidiasims/research_uav/autofly_ue5/scripts/setup_venv.sh`:
```bash
#!/usr/bin/env bash
# Create ROOT/.venv from system python3.12 with the host PYTHONPATH removed,
# install the pinned requirements and the package (editable), and record the lock.
set -euo pipefail
ROOT=/home/nvidiasims/research_uav/autofly_ue5
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  env -u PYTHONPATH /usr/bin/python3.12 -m venv .venv
fi
if [ -f requirements.lock ]; then REQ=requirements.lock; else REQ=requirements.txt; fi
env -u PYTHONPATH .venv/bin/python -m pip install -r "$REQ"
env -u PYTHONPATH .venv/bin/python -m pip install --no-deps -e .
env -u PYTHONPATH .venv/bin/python -m pip freeze --exclude-editable > requirements.lock
env -u PYTHONPATH .venv/bin/python -c "import projectairsim, numpy, PIL, jsonschema; print('projectairsim', projectairsim.__version__, 'numpy', numpy.__version__)"
```

- [ ] **Step 5: Create the venv**

Run (Bash tool `timeout` 600000 ms: pip downloads opencv/matplotlib and builds two source packages):
```bash
bash /home/nvidiasims/research_uav/autofly_ue5/scripts/setup_venv.sh 2>&1 | tail -5
```
Expected last line: `projectairsim 1.0.2 numpy 1.26.4`. This also builds the source-only `commentjson` and `lark-parser` on Python 3.12 (UNCONFIRMED until now). If pip fails on them or on any pin, stop and report the pip error; do not change pins without the user.

- [ ] **Step 6: Confirm the ROS path cannot leak and the source-only deps built**

Run:
```bash
cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -c "import sys; assert not any('ros' in p for p in sys.path), sys.path; import commentjson, lark; print('ok')" && env -u PYTHONPATH .venv/bin/python -m pip show commentjson lark-parser | grep -E '^(Name|Version)'
```
Expected: `ok`, then `Name: commentjson` / `Version: 0.9.0` and `Name: lark-parser` / `Version: 0.7.8`.

- [ ] **Step 7: Write the failing tests**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_paths.py`:
```python
from pathlib import Path

from autofly_ue5 import paths


def test_root_is_project_root():
    assert paths.ROOT == Path("/home/nvidiasims/research_uav/autofly_ue5")


def test_engine_binaries_exist():
    assert paths.UNREAL_EDITOR.is_file()
    assert paths.UNREAL_EDITOR_CMD.is_file()


def test_derived_paths():
    assert paths.UPROJECT == paths.ROOT / "ue_project" / "Blocks.uproject"
    assert paths.CONFIGS_DIR == paths.ROOT / "configs"
    assert paths.ASSET_REGISTRY == paths.ROOT / "assets" / "registry.json"
    assert paths.JOBS_DIR == paths.ROOT / "runs" / "jobs"
    assert paths.PACKAGED_BINARY.parts[-6:] == ("Development", "Linux", "Blocks", "Binaries", "Linux", "Blocks")


def test_ue_cache_env_stays_inside_root():
    assert paths.UE_CACHE_ENV == {
        "UE-ZenDataPath": "/home/nvidiasims/research_uav/autofly_ue5/ue_project/DerivedDataCache/Zen",
        "UE-LocalDataCachePath": "/home/nvidiasims/research_uav/autofly_ue5/ue_project/DerivedDataCache",
    }
```

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_run_job.py`:
```python
import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_job.sh"


def run_job(jobs_dir: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["AUTOFLY_JOBS_DIR"] = str(jobs_dir)
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=90)


def test_start_records_the_job_and_wait_reports_success(tmp_path):
    started = run_job(tmp_path, "start", "ok", "--", "bash", "-c", "echo hello; sleep 1")
    assert started.returncode == 0, started.stdout + started.stderr
    record = json.loads((tmp_path / "ok.pid.json").read_text())
    assert record["pid"] == record["pgid"]
    assert record["cmd"] == ["bash", "-c", "echo hello; sleep 1"]
    waited = run_job(tmp_path, "wait", "ok", "30")
    assert waited.returncode == 0
    assert "hello" in waited.stdout and "job ok finished: exit=0" in waited.stdout


def test_wait_reports_a_failing_exit_code(tmp_path):
    run_job(tmp_path, "start", "bad", "--", "bash", "-c", "exit 3")
    waited = run_job(tmp_path, "wait", "bad", "30")
    assert waited.returncode == 1 and "job bad finished: exit=3" in waited.stdout


def test_wait_times_out_duplicate_start_is_refused_and_stop_ends_the_group(tmp_path):
    run_job(tmp_path, "start", "slow", "--", "bash", "-c", "sleep 300 & wait")
    waited = run_job(tmp_path, "wait", "slow", "1")
    assert waited.returncode == 124 and "still running" in waited.stdout
    again = run_job(tmp_path, "start", "slow", "--", "true")
    assert again.returncode == 1 and "still running" in again.stdout
    pgid = json.loads((tmp_path / "slow.pid.json").read_text())["pgid"]
    stopped = run_job(tmp_path, "stop", "slow")
    assert stopped.returncode == 0 and stopped.stdout.strip() == "terminated"
    assert subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True).returncode == 1
    assert run_job(tmp_path, "wait", "slow", "5").returncode == 3
```

- [ ] **Step 8: Run them to verify they fail**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_paths.py tests/test_run_job.py`
Expected: FAIL: `ModuleNotFoundError: No module named 'autofly_ue5.paths'` for `test_paths.py`, and the three `test_run_job.py` tests fail on `returncode` (bash reports `scripts/run_job.sh: No such file or directory`).

- [ ] **Step 9: Write `autofly_ue5/paths.py`**

```python
"""Absolute project paths; every other module takes its paths from here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engine"
PLATFORM_DIR = ROOT / "platform"
UE_PROJECT_DIR = ROOT / "ue_project"
UPROJECT = UE_PROJECT_DIR / "Blocks.uproject"
CONFIGS_DIR = ROOT / "configs"
SCENES_DIR = ROOT / "scenes"
ASSET_REGISTRY = ROOT / "assets" / "registry.json"
RUNS_DIR = ROOT / "runs"
JOBS_DIR = RUNS_DIR / "jobs"
FIXTURES_DIR = ROOT / "tests" / "fixtures"
UNREAL_EDITOR = ENGINE_DIR / "Engine" / "Binaries" / "Linux" / "UnrealEditor"
UNREAL_EDITOR_CMD = ENGINE_DIR / "Engine" / "Binaries" / "Linux" / "UnrealEditor-Cmd"
PACKAGED_BINARY = (
    UE_PROJECT_DIR / "Packaged" / "Development" / "Linux" / "Blocks" / "Binaries" / "Linux" / "Blocks"
)
# Derived-data cache inside ROOT (git-ignored) instead of ~/.config/Epic/UnrealEngine/Common/Zen/Data.
DDC_DIR = UE_PROJECT_DIR / "DerivedDataCache"
ZEN_DATA_DIR = DDC_DIR / "Zen"
UE_CACHE_ENV = {"UE-ZenDataPath": str(ZEN_DATA_DIR), "UE-LocalDataCachePath": str(DDC_DIR)}
```

- [ ] **Step 10: Write `scripts/run_job.sh`**

`/home/nvidiasims/research_uav/autofly_ue5/scripts/run_job.sh`:
```bash
#!/usr/bin/env bash
# Long-running commands owned by this project, detached from the caller's shell and tool timeout.
#   run_job.sh start <name> -- <cmd> [args...]  new session; writes <name>.pid.json, output to <name>.log
#   run_job.sh wait <name> [timeout_s]          polls (default 540 s); prints the log tail and "job <name> finished: exit=<code>"
#                                               returns 0 (exit 0), 1 (other exit code), 124 (still running), 3 (stopped or died)
#   run_job.sh stop <name>                      SIGTERM, then SIGKILL after 30 s, to the recorded process group, only if it is ours
# Files live in $AUTOFLY_JOBS_DIR (default ROOT/runs/jobs).
set -euo pipefail
ROOT=/home/nvidiasims/research_uav/autofly_ue5
JOBS=${AUTOFLY_JOBS_DIR:-$ROOT/runs/jobs}
USAGE="usage: run_job.sh start <name> -- <cmd...> | wait <name> [timeout_s] | stop <name>"
MODE="${1:?$USAGE}"
NAME="${2:?$USAGE}"
shift 2
mkdir -p "$JOBS"
PIDFILE=$JOBS/$NAME.pid.json
EXITFILE=$JOBS/$NAME.exit
STOPPED=$JOBS/$NAME.stopped
LOG=$JOBS/$NAME.log
MARKER="autofly_job:$NAME"

recorded() { jq -r ".$1" "$PIDFILE"; }

is_ours() {  # recorded PID alive, argv[3] is our marker, process group unchanged
  local pid pgid
  pid=$(recorded pid)
  pgid=$(recorded pgid)
  kill -0 "$pid" 2>/dev/null || return 1
  [ "$(tr '\0' '\n' < "/proc/$pid/cmdline" | sed -n 4p)" = "$MARKER" ] || return 1
  [ "$(ps -o pgid= -p "$pid" | tr -d ' ')" = "$pgid" ]
}

case "$MODE" in
  start)
    { [ "${1:-}" = "--" ] && [ $# -ge 2 ]; } || { echo "$USAGE"; exit 2; }
    shift
    if [ -f "$PIDFILE" ] && is_ours; then
      echo "job $NAME is still running with pid $(recorded pid)"
      exit 1
    fi
    rm -f "$PIDFILE" "$EXITFILE" "$STOPPED"
    setsid bash -c 'exit_file=$1; shift; code=0; "$@" || code=$?; echo "$code" > "$exit_file"' \
      "$MARKER" "$EXITFILE" "$@" < /dev/null > "$LOG" 2>&1 &
    PID=$!
    PGID=$PID
    for _ in $(seq 50); do  # wait until setsid has made the job its own process group
      SEEN=$(ps -o pgid= -p "$PID" | tr -d ' ' || true)
      [ -z "$SEEN" ] && break
      PGID=$SEEN
      [ "$PGID" = "$PID" ] && break
      sleep 0.1
    done
    env -u PYTHONPATH /usr/bin/python3.12 -c 'import json, sys, time; pid, pgid, log = sys.argv[1:4]; print(json.dumps({"pid": int(pid), "pgid": int(pgid), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "log": log, "cmd": sys.argv[4:]}))' \
      "$PID" "$PGID" "$LOG" "$@" > "$PIDFILE"
    echo "job $NAME started: pid $PID, log $LOG"
    ;;
  wait)
    TIMEOUT=${1:-540}
    [ -f "$PIDFILE" ] || { echo "no job $NAME"; exit 2; }
    START=$(date +%s)
    while [ ! -f "$EXITFILE" ]; do
      if [ -f "$STOPPED" ]; then echo "job $NAME was stopped"; exit 3; fi
      if ! kill -0 "$(recorded pid)" 2>/dev/null; then
        sleep 1
        if [ ! -f "$EXITFILE" ]; then tail -n 20 "$LOG"; echo "job $NAME died without an exit code"; exit 3; fi
        break
      fi
      if [ $(( $(date +%s) - START )) -ge "$TIMEOUT" ]; then
        tail -n 5 "$LOG"
        echo "job $NAME still running after ${TIMEOUT} s (pid $(recorded pid)); call wait again"
        exit 124
      fi
      sleep 2
    done
    CODE=$(cat "$EXITFILE")
    tail -n 20 "$LOG"
    echo "job $NAME finished: exit=$CODE"
    [ "$CODE" = "0" ] || exit 1
    ;;
  stop)
    [ -f "$PIDFILE" ] || { echo "no_job"; exit 0; }
    PID=$(recorded pid)
    PGID=$(recorded pgid)
    if ! kill -0 "$PID" 2>/dev/null; then echo "not_running"; exit 0; fi
    is_ours || { echo "pid $PID does not match job $NAME (marker or process group); not stopping it"; exit 1; }
    touch "$STOPPED"
    kill -TERM -- "-$PGID"
    RESULT=terminated
    for _ in $(seq 150); do kill -0 -- "-$PGID" 2>/dev/null || break; sleep 0.2; done
    if kill -0 -- "-$PGID" 2>/dev/null; then kill -KILL -- "-$PGID"; RESULT=killed; sleep 1; fi
    echo "$RESULT"
    ;;
  *)
    echo "$USAGE"
    exit 2
    ;;
esac
```
(The job's command runs in its own session, so a tool timeout that kills the caller never orphans an unrecorded build, cook or simulator. Unreal starts its helpers (ShaderCompileWorker, zenserver, CrashReportClient) in their own process groups (`UnixPlatformProcess.cpp:1048-1049`), so `stop` cannot reach them; the steps that stop Unreal processes check for leftovers with `pgrep` and report them instead of killing them.)

- [ ] **Step 11: Run the tests to verify they pass**

Run: `chmod +x /home/nvidiasims/research_uav/autofly_ue5/scripts/run_job.sh && cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_paths.py tests/test_run_job.py`
Expected: `7 passed`.

- [ ] **Step 12: Commit (including this plan)**

```bash
chmod +x /home/nvidiasims/research_uav/autofly_ue5/scripts/setup_venv.sh
git -C /home/nvidiasims/research_uav/autofly_ue5 add .gitignore README.md pyproject.toml requirements.txt requirements.lock scripts/setup_venv.sh scripts/run_job.sh autofly_ue5/__init__.py autofly_ue5/paths.py tests/test_paths.py tests/test_run_job.py docs/superpowers/plans/2026-09-15-autofly-ue5-plan1-platform-and-scene-s01.md && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Add the autofly_ue5 package skeleton, a pinned Python 3.12 venv with projectairsim 1.0.2, the job runner and Plan 1" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Engine readiness check

**Files:**
- Create: `autofly_ue5/gpu.py`, `autofly_ue5/validate/__init__.py`, `autofly_ue5/validate/engine_check.py`
- Create (captured): `docs/gates/m0_engine_check.json`
- Test: `tests/test_gpu.py`, `tests/test_engine_check.py`

**Interfaces:**
- Consumes: `autofly_ue5.paths` (`ENGINE_DIR`, `PLATFORM_DIR`, `RUNS_DIR`, `UNREAL_EDITOR`, `UNREAL_EDITOR_CMD`, `UE_CACHE_ENV`); `scripts/run_job.sh` (Task 1).
- Produces:
  - `autofly_ue5.gpu.GpuBusyError(RuntimeError)`
  - `autofly_ue5.gpu.parse_gpu_memory(text: str) -> tuple[int, int]` (used, total MiB)
  - `autofly_ue5.gpu.gpu_memory_mib() -> tuple[int, int]`
  - `autofly_ue5.gpu.check_gpu_for_launch(used_mib: int, total_mib: int, own_running: int, idle_max_mib: int = 2000, min_free_mib: int = 6000) -> None`
  - `autofly_ue5.validate.engine_check`: `ZEN_DEFAULT_DATA = Path.home() / ".config/Epic/UnrealEngine/Common/Zen/Data"`; `OUT_OF_ROOT_PATHS: dict[str, Path]` (keys `unreal_trace_store` = `~/UnrealEngine`, `unreal_trace_server_pid` = `/tmp/UnrealTraceServer.pid`, `documents_unreal_projects` = `~/Documents/Unreal Projects`, `documents_unreal_engine_logs` = `~/Documents/Unreal Engine`); `parse_build_version(text: str) -> str` (e.g. `"5.7.4-51494982"`), `count_xid(journal_text: str) -> int`, `xid_journal_command(since: str | None) -> list[str]` (`since` = local time `"YYYY-MM-DD HH:MM:SS"` → `journalctl _TRANSPORT=kernel --since <since> --no-pager`, which spans reboots because `-k` would imply `-b`; `None` → `journalctl -k -b --no-pager`, current boot), `xid_count(since: str | None = None) -> int`, `boot_id() -> str` (`/proc/sys/kernel/random/boot_id`), `clang_version(text: str) -> str | None`, `find_fatal_lines(log_text: str) -> list[str]`, `count_device_lost(log_paths: list[Path]) -> dict[str, int]` (`VK_ERROR_DEVICE_LOST` lines per existing file), `epic_config_usage() -> dict` (`{"kib": int | None, "zen_default_data_exists": bool}`), `out_of_root_state() -> dict[str, bool]` (existence of each `OUT_OF_ROOT_PATHS` entry), `run_checks(editor_wait_s: float = 120.0) -> dict` (report keys `started_at`, `boot_id`, `checks`, `epic_config_before`, `out_of_root_before`, `pass`; `checks.nvidia_xid` holds `since`, `before`/`after` counts since `started_at` and `current_boot_total_at_start`, and passes when no new Xid appeared during the probe), CLI `python -m autofly_ue5.validate.engine_check --out <json>` (exit 0 iff `report["pass"]`). While the editor probe runs, `runs/m0/editor_probe.pid.json` records its `pid`, `pgid` and `cmd`.

- [ ] **Step 1: Write the failing tests**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_gpu.py`:
```python
import pytest

from autofly_ue5.gpu import GpuBusyError, check_gpu_for_launch, parse_gpu_memory


def test_parse_gpu_memory():
    assert parse_gpu_memory("622, 24564\n") == (622, 24564)


def test_idle_gpu_allows_first_instance():
    check_gpu_for_launch(622, 24564, own_running=0)


def test_foreign_load_blocks_first_instance():
    with pytest.raises(GpuBusyError, match="no simulator of ours"):
        check_gpu_for_launch(3000, 24564, own_running=0)


def test_second_instance_needs_headroom_only():
    check_gpu_for_launch(9000, 24564, own_running=1)
    with pytest.raises(GpuBusyError, match="free"):
        check_gpu_for_launch(20000, 24564, own_running=1)
```

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_engine_check.py`:
```python
from autofly_ue5.validate.engine_check import (
    clang_version,
    count_device_lost,
    count_xid,
    find_fatal_lines,
    parse_build_version,
    xid_journal_command,
)

BUILD_VERSION = """{
	"MajorVersion": 5,
	"MinorVersion": 7,
	"PatchVersion": 4,
	"Changelist": 51494982,
	"CompatibleChangelist": 47537391,
	"IsLicenseeVersion": 0,
	"IsPromotedBuild": 1,
	"BranchName": "++UE5+Release-5.7"
}"""


def test_parse_build_version():
    assert parse_build_version(BUILD_VERSION) == "5.7.4-51494982"


def test_count_xid_ignores_network_driver_xid():
    journal = (
        "Sep 15 11:13:27 host kernel: r8169 0000:0a:00.0 eth0: RTL8125B, XID 641, IRQ 86\n"
        "Sep 15 12:00:00 host kernel: NVRM: Xid (PCI:0000:01:00): 31, pid=1234, name=UnrealEditor\n"
    )
    assert count_xid(journal) == 1


def test_xid_journal_command_spans_reboots_when_given_a_start_time():
    cmd = xid_journal_command("2026-09-15 12:00:00")
    assert "_TRANSPORT=kernel" in cmd and "-k" not in cmd and "-b" not in cmd  # -k implies -b (current boot only)
    assert cmd[cmd.index("--since") + 1] == "2026-09-15 12:00:00"
    assert xid_journal_command(None) == ["journalctl", "-k", "-b", "--no-pager"]


def test_clang_version():
    assert clang_version("clang version 20.1.8 (https://github.com/llvm/llvm-project 87f0227)\nTarget: x86_64") == "20.1.8"
    assert clang_version("no compiler here") is None


def test_find_fatal_lines():
    log = "LogInit: Display: ok\nLogVulkanRHI: Error: VK_ERROR_DEVICE_LOST\nLogCore: Fatal error: boom\n"
    assert find_fatal_lines(log) == ["LogVulkanRHI: Error: VK_ERROR_DEVICE_LOST", "LogCore: Fatal error: boom"]


def test_count_device_lost_per_file(tmp_path):
    good, bad = tmp_path / "sim.log", tmp_path / "sim-backup-2026.09.15.log"
    good.write_text("LogInit: Display: ok\n")
    bad.write_text("LogVulkanRHI: Error: VK_ERROR_DEVICE_LOST\nLogVulkanRHI: Error: VK_ERROR_DEVICE_LOST again\n")
    counts = count_device_lost([good, bad, tmp_path / "missing.log"])
    assert counts == {str(good): 0, str(bad): 2}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_gpu.py tests/test_engine_check.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'autofly_ue5.gpu'` (and `autofly_ue5.validate`).

- [ ] **Step 3: Write `autofly_ue5/gpu.py`**

```python
"""GPU memory reading and the rule applied before launching a simulator."""

import subprocess


class GpuBusyError(RuntimeError):
    """The GPU is in use by someone else or lacks headroom for another instance."""


def parse_gpu_memory(text: str) -> tuple[int, int]:
    first = text.strip().splitlines()[0]
    used, total = (int(v.strip()) for v in first.split(","))
    return used, total


def gpu_memory_mib() -> tuple[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return parse_gpu_memory(out)


def check_gpu_for_launch(
    used_mib: int, total_mib: int, own_running: int, idle_max_mib: int = 2000, min_free_mib: int = 6000
) -> None:
    if own_running == 0 and used_mib > idle_max_mib:
        raise GpuBusyError(
            f"GPU has {used_mib} MiB in use and no simulator of ours is running; another job is using it"
        )
    free = total_mib - used_mib
    if free < min_free_mib:
        raise GpuBusyError(f"only {free} MiB free on the GPU, need {min_free_mib} MiB")
```

- [ ] **Step 4: Write `autofly_ue5/validate/__init__.py` and `autofly_ue5/validate/engine_check.py`**

`autofly_ue5/validate/__init__.py`:
```python
"""Platform, simulator and dataset checks."""
```

`autofly_ue5/validate/engine_check.py`:
```python
"""M0 engine readiness: versions, toolchain, plugins, platform checkout, GPU, Xid, editor opens on X11, out-of-ROOT paths.

Usage: env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.engine_check --out runs/m0/engine_check.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from autofly_ue5.gpu import gpu_memory_mib
from autofly_ue5.paths import ENGINE_DIR, PLATFORM_DIR, RUNS_DIR, UE_CACHE_ENV, UNREAL_EDITOR, UNREAL_EDITOR_CMD

EPIC_CONFIG_DIR = Path.home() / ".config" / "Epic"
ZEN_DEFAULT_DATA = EPIC_CONFIG_DIR / "UnrealEngine" / "Common" / "Zen" / "Data"
# Written by UE unless prevented (-notraceserver, uebp_LogFolder) or cleaned up (the probe's empty project folder).
OUT_OF_ROOT_PATHS = {
    "unreal_trace_store": Path.home() / "UnrealEngine",
    "unreal_trace_server_pid": Path("/tmp/UnrealTraceServer.pid"),
    "documents_unreal_projects": Path.home() / "Documents" / "Unreal Projects",
    "documents_unreal_engine_logs": Path.home() / "Documents" / "Unreal Engine",
}
EXPECTED_BUILD = "5.7.4-51494982"
TOOLCHAIN = "v26_clang-20.1.8-rockylinux8"
FATAL_PATTERNS = ("Fatal error", "VK_ERROR_DEVICE_LOST", "Unhandled Exception", "Segmentation fault", "Critical error")
ENGINE_PLUGINS = {
    "PythonScriptPlugin": "Experimental/PythonScriptPlugin/PythonScriptPlugin.uplugin",
    "EditorScriptingUtilities": "Editor/EditorScriptingUtilities/EditorScriptingUtilities.uplugin",
    "SunPosition": "Runtime/SunPosition/SunPosition.uplugin",
    "ChaosVehiclesPlugin": "Experimental/ChaosVehiclesPlugin/ChaosVehiclesPlugin.uplugin",
}


def parse_build_version(text: str) -> str:
    data = json.loads(text)
    return f"{data['MajorVersion']}.{data['MinorVersion']}.{data['PatchVersion']}-{data['Changelist']}"


def count_xid(journal_text: str) -> int:
    return sum(1 for line in journal_text.splitlines() if "NVRM: Xid" in line)


def clang_version(text: str) -> str | None:
    match = re.search(r"clang version (\d+\.\d+\.\d+)", text)
    return match.group(1) if match else None


def find_fatal_lines(log_text: str) -> list[str]:
    return [line for line in log_text.splitlines() if any(p in line for p in FATAL_PATTERNS)]


def count_device_lost(log_paths: list[Path]) -> dict[str, int]:
    return {
        str(p): sum(1 for line in p.read_text(errors="replace").splitlines() if "VK_ERROR_DEVICE_LOST" in line)
        for p in log_paths
        if p.is_file()
    }


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def xid_journal_command(since: str | None) -> list[str]:
    if since is None:
        return ["journalctl", "-k", "-b", "--no-pager"]
    # No -k here: journalctl's -k implies -b, which would hide Xid lines logged before a reboot.
    return ["journalctl", "_TRANSPORT=kernel", "--since", since, "--no-pager"]


def xid_count(since: str | None = None) -> int:
    return count_xid(_run(xid_journal_command(since)))


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def epic_config_usage() -> dict:
    out = _run(["du", "-sk", str(EPIC_CONFIG_DIR)]).split()
    return {"kib": int(out[0]) if out else None, "zen_default_data_exists": ZEN_DEFAULT_DATA.exists()}


def out_of_root_state() -> dict[str, bool]:
    return {name: path.exists() for name, path in OUT_OF_ROOT_PATHS.items()}


def open_editor_probe(wait_s: float) -> dict:
    """Open the editor (no project) on :1. It stays in this process's group, so stopping the job that runs
    engine_check (scripts/run_job.sh stop) also stops the editor; engine_check itself signals only the editor PID.
    The Project Browser may create an empty ~/Documents/Unreal Projects (SProjectDialog.cpp:1748-1752); it is removed
    afterwards only if this probe created it and it is still empty."""
    log = RUNS_DIR / "m0" / "editor_open.log"
    pid_file = RUNS_DIR / "m0" / "editor_probe.pid.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    projects_dir = OUT_OF_ROOT_PATHS["documents_unreal_projects"]
    projects_existed = projects_dir.exists()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update({"DISPLAY": ":1", "SDL_VIDEODRIVER": "x11", **UE_CACHE_ENV})
    cmd = [str(UNREAL_EDITOR), "-nosplash", "-notraceserver", f"-abslog={log}"]
    with open(log.with_suffix(".stdout"), "wb") as out:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
    pid_file.write_text(json.dumps({"pid": proc.pid, "pgid": os.getpgid(proc.pid), "cmd": cmd}, indent=2))
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(2.0)
    alive = proc.poll() is None
    exit_code = None if alive else proc.returncode
    stop_result = "not_running"
    if alive:
        proc.terminate()
        stop_result = "terminated"
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
            stop_result = "killed"
    pid_file.unlink()
    projects_created = projects_dir.exists() and not projects_existed
    projects_removed = False
    if projects_created:
        try:
            projects_dir.rmdir()  # fails (and is reported) if the folder is not empty
            projects_removed = True
        except OSError:
            projects_removed = False
    text = log.read_text(errors="replace") if log.exists() else ""
    fatal = find_fatal_lines(text)
    return {
        "pid": proc.pid,
        "alive_after_wait": alive,
        "exit_code_before_stop": exit_code,
        "stop": stop_result,
        "fatal_lines": fatal[:20],
        "gpu_named_in_log": "RTX 4090" in text,
        "documents_unreal_projects": {"existed_before": projects_existed, "created_by_probe": projects_created,
                                      "removed_empty": projects_removed},
        "log": str(log),
        "pass": alive and not fatal and (not projects_created or projects_removed),
    }


def run_checks(editor_wait_s: float = 120.0) -> dict:
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    epic_before = epic_config_usage()
    out_of_root_before = out_of_root_state()
    xid_boot_total = xid_count(None)
    checks: dict[str, dict] = {}
    build = parse_build_version((ENGINE_DIR / "Engine/Build/Build.version").read_text())
    checks["engine_version"] = {"value": build, "pass": build == EXPECTED_BUILD}
    checks["installed_build"] = {"pass": (ENGINE_DIR / "Engine/Build/InstalledBuild.txt").is_file()}
    clang = ENGINE_DIR / f"Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/{TOOLCHAIN}/x86_64-unknown-linux-gnu/bin/clang++"
    version = clang_version(_run([str(clang), "--version"])) if clang.is_file() else None
    checks["toolchain"] = {"clang": version, "pass": version == "20.1.8"}
    sdk = json.loads((ENGINE_DIR / "Engine/Config/Linux/Linux_SDK.json").read_text())
    checks["linux_sdk_json"] = {"main": sdk["MainVersion"], "pass": sdk["MainVersion"] == TOOLCHAIN}
    checks["editor_binaries"] = {
        "pass": os.access(UNREAL_EDITOR, os.X_OK) and os.access(UNREAL_EDITOR_CMD, os.X_OK)
    }
    present = {name: (ENGINE_DIR / "Engine/Plugins" / rel).is_file() for name, rel in ENGINE_PLUGINS.items()}
    checks["engine_plugins"] = {"present": present, "pass": all(present.values())}
    missing = [line.strip() for line in _run(["ldd", str(UNREAL_EDITOR)]).splitlines() if "not found" in line]
    checks["editor_shared_libs"] = {"missing": missing, "pass": not missing}
    head = _run(["git", "-C", str(PLATFORM_DIR), "rev-parse", "HEAD"]).strip()
    diff = subprocess.run(
        ["git", "-C", str(PLATFORM_DIR), "diff", "--quiet", "0975545", "4d878bf", "--",
         "unreal", "core_sim", "physics", "vehicle_apis", "simserver"]
    ).returncode
    checks["platform_checkout"] = {
        "head": head,
        "cpp_identical_to_v1_0_1": diff == 0,
        "pass": head.startswith("4d878bf") and diff == 0,
    }
    used, total = gpu_memory_mib()
    checks["gpu_idle"] = {"used_mib": used, "total_mib": total, "pass": used <= 2000}
    xid_before = xid_count(started_at)
    if checks["gpu_idle"]["pass"]:
        checks["editor_opens_x11"] = open_editor_probe(editor_wait_s)
    else:
        checks["editor_opens_x11"] = {"pass": False, "skipped": "GPU busy"}
    xid_after = xid_count(started_at)
    # Xid lines from before this check (e.g. another program) are recorded, not failed on; counts from started_at on
    # span reboots, and M0 gates on the delta from this baseline plus an unchanged boot id.
    checks["nvidia_xid"] = {"since": started_at, "before": xid_before, "after": xid_after,
                            "current_boot_total_at_start": xid_boot_total, "pass": xid_after == xid_before}
    return {"started_at": started_at, "boot_id": boot_id(), "checks": checks, "epic_config_before": epic_before,
            "out_of_root_before": out_of_root_before, "pass": all(c["pass"] for c in checks.values())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--editor-wait", type=float, default=120.0)
    args = parser.parse_args(argv)
    report = run_checks(args.editor_wait)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({name: c["pass"] for name, c in report["checks"].items()}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_gpu.py tests/test_engine_check.py`
Expected: `10 passed`.

- [ ] **Step 6: Run the live engine check as a job (opens the editor on display :1 for 120 s, then stops it; total wait budget 10 minutes)**

This puts a visible Unreal Project Browser window on the user's X display :1 for about two minutes: say so in the progress message before starting the job. Run (Bash tool `timeout` 600000 ms):
```bash
cd /home/nvidiasims/research_uav/autofly_ue5 && nvidia-smi --query-gpu=memory.used --format=csv,noheader && bash scripts/run_job.sh start engine_check -- env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.engine_check --out runs/m0/engine_check.json && bash scripts/run_job.sh wait engine_check 540; echo "wait=$?"
```
Expected: the log tail shows every key `true`, then `job engine_check finished: exit=0` and `wait=0` (`wait=124`: run `bash scripts/run_job.sh wait engine_check 60` once more; after that, apply the timeout rule of the Global Constraints, which here means `bash scripts/run_job.sh stop engine_check`, which also stops the editor because it shares the job's process group). The editor started without a project shows the Project Browser; "opens" is judged as alive after 120 s with no fatal log line and no Xid added during the probe (the exact UE log wording is UNCONFIRMED, so `gpu_named_in_log` is informational only). If any check is `false`: stop and report `runs/m0/engine_check.json` plus `tail -50 runs/m0/editor_open.log`. Afterwards confirm nothing is left running and the cache went into ROOT:
```bash
cd /home/nvidiasims/research_uav/autofly_ue5 && sleep 30; pgrep -a -f "$(pwd)/engine/Engine/Binaries/Linux/" || echo none; pgrep -a zenserver || echo no-zenserver; ls runs/m0/editor_probe.pid.json 2>/dev/null || echo no-pid-file; jq '{started_at, boot_id, epic_config_before, out_of_root_before, probe_projects_dir: .checks.editor_opens_x11.documents_unreal_projects}' runs/m0/engine_check.json; ls -d ~/.config/Epic/UnrealEngine/Common/Zen/Data 2>/dev/null || echo zen-default-absent; ls -d ~/UnrealEngine /tmp/UnrealTraceServer.pid ~/"Documents/Unreal Projects" 2>/dev/null || echo out-of-root-absent; ls ue_project/DerivedDataCache 2>/dev/null || echo ddc-not-created-yet
```
Expected: `none`; `no-zenserver` (a zenserver still listed after 30 s is reported, not killed: it was started by UE with `--owner-pid` and exits on its own); `no-pid-file`; the recorded JSON (`started_at`, `boot_id`, usage, `out_of_root_before`, and the probe's `documents_unreal_projects` record); `zen-default-absent` (if the folder is listed while `epic_config_before.zen_default_data_exists` was `false`, the cache override did not work: stop and report); `out-of-root-absent` (any of those paths listed while `out_of_root_before` had it `false` means `-notraceserver` or the empty-folder cleanup did not work: stop and report). `ddc-not-created-yet` is acceptable: the editor may not have touched the cache without a project, and later tasks check again.

- [ ] **Step 7: Commit (with the captured report)**

```bash
mkdir -p /home/nvidiasims/research_uav/autofly_ue5/docs/gates && cp /home/nvidiasims/research_uav/autofly_ue5/runs/m0/engine_check.json /home/nvidiasims/research_uav/autofly_ue5/docs/gates/m0_engine_check.json
git -C /home/nvidiasims/research_uav/autofly_ue5 add autofly_ue5/gpu.py autofly_ue5/validate/__init__.py autofly_ue5/validate/engine_check.py tests/test_gpu.py tests/test_engine_check.py docs/gates/m0_engine_check.json && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Add the M0 engine readiness check (toolchain, plugins, checkout, GPU, Xid, editor on X11) and its report" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Prebuilt plugin download and verification

**Files:**
- Create: `autofly_ue5/validate/plugin_manifest.py`, `scripts/fetch_plugin.sh`
- Test: `tests/test_plugin_manifest.py`

**Interfaces:**
- Consumes: nothing from earlier tasks besides the venv.
- Produces:
  - `autofly_ue5.validate.plugin_manifest.verify_manifest(root: pathlib.Path) -> dict` with keys `files` (int), `missing` (list[str]), `mismatched` (list[str]), `missing_dirs` (list[str]), `binaries_dirs` (list[str]), `source_sha`, `unreal_version`, `pass` (bool); CLI `python -m autofly_ue5.validate.plugin_manifest <root>` (exit 0 iff pass).
  - On disk: `ROOT/downloads/ProjectAirSim-Plugin-Linux-UE5_7-1.0.1.zip` (+ `.sha256`) and `ROOT/downloads/plugin_ue57_1.0.1/Plugins/{ProjectAirSim,Drone,Rover}`.

- [ ] **Step 1: Write the failing test**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_plugin_manifest.py`:
```python
import hashlib
import json

from autofly_ue5.validate.plugin_manifest import verify_manifest


def _make_plugin_tree(root):
    files = {
        "Plugins/ProjectAirSim/Source/ProjectAirSim/ProjectAirSim.Build.cs": b"build rules",
        "Plugins/ProjectAirSim/SimLibs/core_sim/Release/libcore_sim.a": b"archive",
        "Plugins/Drone/Drone.uplugin": b"{}",
        "Plugins/Rover/Rover.uplugin": b"{}",
    }
    entries = []
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"source_sha": "09755454b8d8", "unreal_version": "5.7", "files": entries}
    (root / "build-manifest.json").write_text(json.dumps(manifest))
    return files


def test_intact_tree_passes(tmp_path):
    _make_plugin_tree(tmp_path)
    report = verify_manifest(tmp_path)
    assert report["pass"] is True
    assert report["files"] == 4
    assert report["mismatched"] == [] and report["missing"] == [] and report["missing_dirs"] == []


def test_corrupted_file_fails(tmp_path):
    _make_plugin_tree(tmp_path)
    (tmp_path / "Plugins/Drone/Drone.uplugin").write_bytes(b"tampered")
    report = verify_manifest(tmp_path)
    assert report["pass"] is False
    assert report["mismatched"] == ["Plugins/Drone/Drone.uplugin"]


def test_wrong_engine_version_fails(tmp_path):
    _make_plugin_tree(tmp_path)
    manifest = json.loads((tmp_path / "build-manifest.json").read_text())
    manifest["unreal_version"] = "5.2"
    (tmp_path / "build-manifest.json").write_text(json.dumps(manifest))
    assert verify_manifest(tmp_path)["pass"] is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_plugin_manifest.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'autofly_ue5.validate.plugin_manifest'`.

- [ ] **Step 3: Write `autofly_ue5/validate/plugin_manifest.py`**

```python
"""Verify an unzipped Project AirSim plugin release against its build-manifest.json.

Usage: env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.plugin_manifest downloads/plugin_ue57_1.0.1
"""

import hashlib
import json
import sys
from pathlib import Path

REQUIRED_DIRS = (
    "Plugins/ProjectAirSim/Source",
    "Plugins/ProjectAirSim/SimLibs",
    "Plugins/Drone",
    "Plugins/Rover",
)
EXPECTED_SOURCE_PREFIX = "0975545"
EXPECTED_UNREAL_VERSION = "5.7"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> dict:
    manifest = json.loads((root / "build-manifest.json").read_text())
    missing: list[str] = []
    mismatched: list[str] = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            missing.append(entry["path"])
        elif _sha256(path) != entry["sha256"]:
            mismatched.append(entry["path"])
    missing_dirs = [d for d in REQUIRED_DIRS if not (root / d).is_dir()]
    plugins = root / "Plugins"
    binaries = sorted(str(p.relative_to(root)) for p in plugins.rglob("Binaries") if p.is_dir()) if plugins.is_dir() else []
    source_sha = str(manifest.get("source_sha", ""))
    unreal_version = str(manifest.get("unreal_version", ""))
    return {
        "files": len(manifest["files"]),
        "missing": missing[:50],
        "mismatched": mismatched[:50],
        "missing_dirs": missing_dirs,
        "binaries_dirs": binaries,
        "source_sha": source_sha,
        "unreal_version": unreal_version,
        "pass": not missing
        and not mismatched
        and not missing_dirs
        and source_sha.startswith(EXPECTED_SOURCE_PREFIX)
        and unreal_version == EXPECTED_UNREAL_VERSION,
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    report = verify_manifest(Path(args[0]))
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_plugin_manifest.py`
Expected: `3 passed`.

- [ ] **Step 5: Write `scripts/fetch_plugin.sh`**

```bash
#!/usr/bin/env bash
# Download the Project AirSim 1.0.1 Linux UE5.7 plugin, check size and sha256, unzip, verify the manifest.
set -euo pipefail
ROOT=/home/nvidiasims/research_uav/autofly_ue5
NAME=ProjectAirSim-Plugin-Linux-UE5_7-1.0.1.zip
URL=https://github.com/iamaisim/ProjectAirSim/releases/download/v1.0.1/$NAME
SHA=11f016ac7aa292a1a353dfde9ff7c4a1b5cebd660cf9e8f400f2ed4c030dcb49
SIZE=669317542
DL=$ROOT/downloads
mkdir -p "$DL"
cd "$DL"
if [ ! -f "$NAME" ] || [ "$(stat -c %s "$NAME")" != "$SIZE" ]; then
  curl -fL --retry 5 -C - -o "$NAME" "$URL"
fi
test "$(stat -c %s "$NAME")" = "$SIZE"
echo "$SHA  $NAME" | sha256sum -c -
sha256sum "$NAME" > "$NAME.sha256"
unzip -tq "$NAME"
rm -rf plugin_ue57_1.0.1
unzip -q "$NAME" -d plugin_ue57_1.0.1
env -u PYTHONPATH "$ROOT/.venv/bin/python" -m autofly_ue5.validate.plugin_manifest "$DL/plugin_ue57_1.0.1" > "$DL/plugin_manifest_report.json"
cat "$DL/plugin_manifest_report.json"
```

- [ ] **Step 6: Cross-check the published digest (read-only API call)**

Run:
```bash
curl -s https://api.github.com/repos/iamaisim/ProjectAirSim/releases/tags/v1.0.1 | jq -r '.assets[] | select(.name=="ProjectAirSim-Plugin-Linux-UE5_7-1.0.1.zip") | "\(.size) \(.digest)"'
```
Expected: `669317542 sha256:11f016ac7aa292a1a353dfde9ff7c4a1b5cebd660cf9e8f400f2ed4c030dcb49`. Any difference: stop and report (the asset changed since the design was pinned).

- [ ] **Step 7: Download and verify as a job (about 670 MB; total wait budget 60 minutes)**

Run: `chmod +x /home/nvidiasims/research_uav/autofly_ue5/scripts/fetch_plugin.sh && cd /home/nvidiasims/research_uav/autofly_ue5 && bash scripts/run_job.sh start fetch_plugin -- bash scripts/fetch_plugin.sh && bash scripts/run_job.sh wait fetch_plugin 540; echo "wait=$?"` (repeat `bash scripts/run_job.sh wait fetch_plugin 540` while it returns 124, within the budget; `curl -C -` resumes a partial file if the job must be restarted after a network failure).
Expected: `wait=0` and the job log (`runs/jobs/fetch_plugin.log`) shows `ProjectAirSim-Plugin-Linux-UE5_7-1.0.1.zip: OK`, then a JSON report with `"files": 2395`, `"missing": []`, `"mismatched": []`, `"missing_dirs": []`, `"source_sha"` starting `0975545`, `"unreal_version": "5.7"`, `"pass": true`, and `job fetch_plugin finished: exit=0`. The manifest key names (`files[].path`, `files[].sha256`, `source_sha`, `unreal_version`) are UNCONFIRMED until this run: if the script fails with a `KeyError`, stop and report the first 40 lines of `downloads/plugin_ue57_1.0.1/build-manifest.json`.

- [ ] **Step 8: Confirm the SimLibs are Release-only**

Run: `find /home/nvidiasims/research_uav/autofly_ue5/downloads/plugin_ue57_1.0.1/Plugins/ProjectAirSim/SimLibs -mindepth 2 -maxdepth 2 -type d -printf '%f\n' | sort | uniq -c`
Expected: a `Release` entry (header folders such as `include` may also appear) and no `Debug` entry. A `Debug` entry does not block anything; note it in the task report.

- [ ] **Step 9: Commit**

```bash
git -C /home/nvidiasims/research_uav/autofly_ue5 add autofly_ue5/validate/plugin_manifest.py scripts/fetch_plugin.sh tests/test_plugin_manifest.py && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Fetch the Project AirSim 1.0.1 Linux UE5.7 plugin and verify size, sha256 and build manifest" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `ue_project` from Blocks and the BlocksEditor build

**Files:**
- Create: `scripts/create_ue_project.sh`, `scripts/build_editor.sh`, `ue_project/Blocks.uproject`, `ue_project/Config/DefaultGame.ini`, `autofly_ue5/scenes/ue/probe_python.py`
- Create (copied from `platform/unreal/Blocks`): `ue_project/Config/{DefaultEditor.ini,DefaultEditorPerProjectUserSettings.ini,DefaultEngine.ini,DefaultGameUserSettings.ini,DefaultInput.ini,HoloLens/}`, `ue_project/Source/{Blocks.Target.cs,BlocksEditor.Target.cs,Blocks/}`
- Test: live build and commandlet probe (no pytest)

**Interfaces:**
- Consumes: `ROOT/downloads/plugin_ue57_1.0.1/Plugins` (Task 3); `scripts/run_job.sh` (Task 1).
- Produces: `ROOT/ue_project/Blocks.uproject` with modules built: `ue_project/Binaries/Linux/libUnrealEditor-Blocks.so`, `ue_project/Plugins/ProjectAirSim/Binaries/Linux/libUnrealEditor-ProjectAirSim.so`; `bash scripts/create_ue_project.sh init|restore`; `bash scripts/build_editor.sh` (writes `runs/build/build_result.json` = `{"build_sh_exit": 0, "outputs_ok": true, "pass": true}` only when every check passed); a proven `UnrealEditor-Cmd … -run=pythonscript` path with report `runs/build/python_probe.json` (read by the M0 gate in Task 8).

- [ ] **Step 1: Write `scripts/create_ue_project.sh`**

```bash
#!/usr/bin/env bash
# init:    first creation of ue_project from platform/unreal/Blocks (Config, Source) plus the ignored parts.
# restore: copy only the git-ignored parts (Blocks content, prebuilt plugin) into an existing ue_project.
set -euo pipefail
ROOT=/home/nvidiasims/research_uav/autofly_ue5
BLOCKS=$ROOT/platform/unreal/Blocks
PLUGINS=$ROOT/downloads/plugin_ue57_1.0.1/Plugins
DST=$ROOT/ue_project
MODE="${1:?usage: create_ue_project.sh init|restore}"

copy_ignored() {
  test -d "$PLUGINS/ProjectAirSim/SimLibs" || { echo "missing $PLUGINS (run scripts/fetch_plugin.sh)"; exit 1; }
  mkdir -p "$DST/Content"
  rm -rf "$DST/Plugins" "$DST/Content/Geometry" "$DST/Content/BlocksMap.umap"
  cp -a "$PLUGINS" "$DST/Plugins"
  cp -a "$BLOCKS/Content/BlocksMap.umap" "$BLOCKS/Content/Geometry" "$DST/Content/"
}

case "$MODE" in
  init)
    if [ -e "$DST/Blocks.uproject" ]; then echo "ue_project already initialised; use restore"; exit 1; fi
    mkdir -p "$DST"
    cp -a "$BLOCKS/Config" "$BLOCKS/Source" "$DST/"
    cp -a "$BLOCKS/Blocks.uproject" "$DST/"
    copy_ignored
    ;;
  restore)
    test -f "$DST/Blocks.uproject" || { echo "ue_project not initialised; use init"; exit 1; }
    copy_ignored
    ;;
  *)
    echo "unknown mode $MODE"
    exit 1
    ;;
esac
echo "ue_project $MODE done"
```
(The Blocks plugin folder from git is deliberately not copied: it has no SimLibs. `GISMap`, `glTF_tiles_to_pak`, the `.bat` files and the broken VS Code helper are not copied.)

- [ ] **Step 2: Run `init`**

Run: `chmod +x /home/nvidiasims/research_uav/autofly_ue5/scripts/create_ue_project.sh && bash /home/nvidiasims/research_uav/autofly_ue5/scripts/create_ue_project.sh init && ls /home/nvidiasims/research_uav/autofly_ue5/ue_project /home/nvidiasims/research_uav/autofly_ue5/ue_project/Plugins /home/nvidiasims/research_uav/autofly_ue5/ue_project/Content`
Expected: `ue_project init done`; `Blocks.uproject Config Content Plugins Source` (plus `DerivedDataCache` if Task 2's editor probe already created `DerivedDataCache/Zen` through `UE-ZenDataPath`); `Drone ProjectAirSim Rover`; `BlocksMap.umap Geometry`.

- [ ] **Step 3: Overwrite the descriptor `ue_project/Blocks.uproject`**

```json
{
	"FileVersion": 3,
	"EngineAssociation": "5.7",
	"Category": "",
	"Description": "AutoFly UE5 dataset simulator, derived from the Project AirSim Blocks sample",
	"Modules": [
		{
			"Name": "Blocks",
			"Type": "Runtime",
			"LoadingPhase": "Default"
		}
	],
	"Plugins": [
		{ "Name": "ProjectAirSim", "Enabled": true },
		{ "Name": "USDImporter", "Enabled": false },
		{ "Name": "GLTFExporter", "Enabled": false },
		{ "Name": "SunPosition", "Enabled": true },
		{ "Name": "StaticMeshEditorModeling", "Enabled": true },
		{ "Name": "ChaosVehiclesPlugin", "Enabled": true },
		{ "Name": "Drone", "Enabled": true },
		{ "Name": "Rover", "Enabled": true },
		{ "Name": "PythonScriptPlugin", "Enabled": true, "TargetAllowList": [ "Editor" ] },
		{ "Name": "EditorScriptingUtilities", "Enabled": true, "TargetAllowList": [ "Editor" ] }
	],
	"TargetPlatforms": [ "Linux" ]
}
```
(SteamVR and OculusVR are removed because they do not exist in 5.7; both Python plugins are enabled before the first build.)

- [ ] **Step 4: Overwrite `ue_project/Config/DefaultGame.ini`**

```ini
[/Script/EngineSettings.GeneralProjectSettings]
ProjectID=367FFC384956CDC4377673B3217F380D
ProjectName="Blocks"
CompanyName=Microsoft
Homepage="https://microsoft.com/"
SupportContact="https://microsoft.com"
LicensingTerms=Licence
ProjectDisplayedTitle=NSLOCTEXT("[/Script/EngineSettings]", "8F8B6B2A472F9FDFB69E2B8CFAE8C4E0", "Blocks Environment")
ProjectDebugTitleInfo=NSLOCTEXT("[/Script/EngineSettings]", "F31D7C524A9E9BC66DD2AA922D309408", "Blocks Environment")

[/Script/UnrealEd.ProjectPackagingSettings]
Build=IfProjectHasCode
BuildConfiguration=PPBC_Development
FullRebuild=True
ForDistribution=False
IncludeDebugFiles=False
BlueprintNativizationMethod=Disabled
bIncludeNativizedAssetsInProjectGeneration=False
UsePakFile=True
bGenerateChunks=False
bGenerateNoChunks=False
bChunkHardReferencesOnly=False
bBuildHttpChunkInstallData=False
HttpChunkInstallDataDirectory=(Path="")
HttpChunkInstallDataVersion=
IncludePrerequisites=True
IncludeAppLocalPrerequisites=False
bShareMaterialShaderCode=True
bSharedMaterialNativeLibraries=True
ApplocalPrerequisitesDirectory=(Path="")
IncludeCrashReporter=False
InternationalizationPreset=English
-CulturesToStage=en
+CulturesToStage=en
bCookAll=False
bCookMapsOnly=False
+DirectoriesToAlwaysCook=(Path="/Game/Geometry")
+DirectoriesToAlwaysCook=(Path="/Game/AutoFly/Materials")
+DirectoriesToAlwaysCook=(Path="/ProjectAirSim")
+DirectoriesToAlwaysCook=(Path="/Drone")
+DirectoriesToAlwaysCook=(Path="/Rover")
bCompressed=True
bEncryptIniFiles=False
bEncryptPakIndex=False
bSkipEditorContent=False
+MapsToCook=(FilePath="/Game/BlocksMap")
+MapsToCook=(FilePath="/Game/AutoFly/Maps/S01")
bNativizeBlueprintAssets=False
bNativizeOnlySelectedBlueprints=False
```
(Removed from the Blocks original: `StagingDirectory C:/temp`, cooking all of `/Game`, `/InternalDevUseContent`, the glTF UFS directory and `GISMap`.)

- [ ] **Step 5: Write `scripts/build_editor.sh`**

```bash
#!/usr/bin/env bash
# Build BlocksEditor (Linux, Development) with the engine's bundled toolchain and check the outputs.
set -euo pipefail
ROOT=/home/nvidiasims/research_uav/autofly_ue5
LOG=$ROOT/runs/build/build_BlocksEditor_Development.log
RESULT=$ROOT/runs/build/build_result.json
mkdir -p "$(dirname "$LOG")"
rm -f "$RESULT"
set +e
env -u PYTHONPATH "$ROOT/engine/Engine/Build/BatchFiles/Linux/Build.sh" BlocksEditor Linux Development \
  -project="$ROOT/ue_project/Blocks.uproject" -waitmutex > "$LOG" 2>&1
CODE=$?
set -e
echo "Build.sh exit code: $CODE"
tail -3 "$LOG"
test "$CODE" -eq 0
for f in "$ROOT/ue_project/Binaries/Linux/libUnrealEditor-Blocks.so" \
         "$ROOT/ue_project/Plugins/ProjectAirSim/Binaries/Linux/libUnrealEditor-ProjectAirSim.so"; do
  test -f "$f" || { echo "missing $f"; exit 1; }
done
ldd "$ROOT/ue_project/Plugins/ProjectAirSim/Binaries/Linux/libUnrealEditor-ProjectAirSim.so" > "$ROOT/runs/build/ldd_ProjectAirSim.txt"
MISSING=$(grep "not found" "$ROOT/runs/build/ldd_ProjectAirSim.txt" | grep -v "libUnrealEditor-" || true)
test -z "$MISSING" || { echo "unresolved third-party libraries: $MISSING"; exit 1; }
printf '{"build_sh_exit": %d, "outputs_ok": true, "pass": true}\n' "$CODE" > "$RESULT"
echo "BlocksEditor build OK"
```
(Engine module libraries `libUnrealEditor-*.so` are resolved by the editor process at load time, so only third-party libraries such as onnxruntime/JSBSim are required to resolve under plain `ldd`.)

- [ ] **Step 6: Build as a job (3–15 minutes; total wait budget 45 minutes)**

Run: `chmod +x /home/nvidiasims/research_uav/autofly_ue5/scripts/build_editor.sh && cd /home/nvidiasims/research_uav/autofly_ue5 && bash scripts/run_job.sh start build_editor -- bash scripts/build_editor.sh && bash scripts/run_job.sh wait build_editor 540; echo "wait=$?"` (repeat `bash scripts/run_job.sh wait build_editor 540` while it returns 124, within the budget).
Expected: the job log tail shows `Build.sh exit code: 0`, a tail containing `Result: Succeeded` (wording UNCONFIRMED; the exit code is authoritative), `BlocksEditor build OK`, then `job build_editor finished: exit=0` and `wait=0`; `runs/build/build_result.json` exists with `"pass": true`. Expected harmless warning: obsolete `bEnableUndefinedIdentifierWarnings`. If the build fails: stop and report the first `error:` lines of `runs/build/build_BlocksEditor_Development.log` and `~/.config/Epic/UnrealBuildTool/Log.txt` tail. Do not run `setup_linux_dev_tools.sh`, do not build SimLibs from source, do not try a 22.04 container without the user (spec §4 fallback trigger).

- [ ] **Step 7: Write the commandlet probe `autofly_ue5/scenes/ue/probe_python.py`**

```python
"""UE-side probe (UnrealEditor-Cmd -run=pythonscript): proves the editor Python APIs the level builder needs.

Writes JSON to the path in env AUTOFLY_PROBE_OUT. Runs on UE's embedded Python; stdlib + unreal only.
"""
import json
import os

import unreal

les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
ass = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)
game_mode = unreal.load_class(None, "/Script/ProjectAirSim.ProjectAirSimGameMode")
sunsky = unreal.load_class(None, "/SunPosition/SunSky.SunSky_C")
cylinder = unreal.load_asset("/Engine/BasicShapes/Cylinder")
cube = unreal.load_asset("/Engine/BasicShapes/Cube")
basic_material = unreal.load_asset("/Engine/BasicShapes/BasicShapeMaterial")
grid_material = unreal.load_asset("/Engine/EngineMaterials/WorldGridMaterial")
params = []
mic_readback = None
mic_parent = None
mic_set_return = None
if basic_material is not None:
    params = [str(n) for n in unreal.MaterialEditingLibrary.get_vector_parameter_names(basic_material)]
    # Throwaway MaterialInstanceConstant, never saved (AssetTools CreateAsset only marks the package dirty), so nothing
    # is written to ue_project/Content. Same calls as the level builder in Task 14.
    probe_mic = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        "MI_ProbeColor", "/Game/AutoFly/Probe", unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    if probe_mic is not None:
        unreal.MaterialEditingLibrary.set_material_instance_parent(probe_mic, basic_material)
        parent = probe_mic.get_editor_property("parent")
        mic_parent = parent.get_path_name() if parent is not None else None
        # UE 5.7.4 returns False here even when the value was set (MaterialEditingLibrary.cpp:1253-1262): read it back.
        mic_set_return = bool(unreal.MaterialEditingLibrary.set_material_instance_vector_parameter_value(
            probe_mic, "Color", unreal.LinearColor(0.1, 0.2, 0.3, 1.0)))
        got = unreal.MaterialEditingLibrary.get_material_instance_vector_parameter_value(probe_mic, "Color")
        mic_readback = [got.r, got.g, got.b, got.a]
result = {
    "engine_version": unreal.SystemLibrary.get_engine_version(),
    "level_editor_subsystem": les is not None,
    "editor_actor_subsystem": eas is not None,
    "editor_asset_subsystem": ass is not None,
    "game_mode_class": game_mode.get_path_name() if game_mode is not None else None,
    "sunsky_class": sunsky.get_path_name() if sunsky is not None else None,
    "cylinder_mesh": cylinder is not None,
    "cube_mesh": cube is not None,
    "world_grid_material": grid_material is not None,
    "basic_shape_material_vector_params": params,
    "mic_parent": mic_parent,
    "mic_set_vector_return_value": mic_set_return,
    "mic_color_readback": mic_readback,
    "mic_color_readback_ok": mic_readback is not None
    and all(abs(a - b) < 1e-4 for a, b in zip(mic_readback, (0.1, 0.2, 0.3, 1.0))),
}
with open(os.environ["AUTOFLY_PROBE_OUT"], "w") as handle:
    json.dump(result, handle, indent=2)
```

- [ ] **Step 8: Run the probe headless as a job (no GPU use; first start can take several minutes; total wait budget 30 minutes)**

Run:
```bash
cd /home/nvidiasims/research_uav/autofly_ue5 && mkdir -p runs/build && rm -f runs/build/python_probe.json && bash scripts/run_job.sh start python_probe -- env -u PYTHONPATH DISPLAY=:1 SDL_VIDEODRIVER=x11 "UE-ZenDataPath=$PWD/ue_project/DerivedDataCache/Zen" "UE-LocalDataCachePath=$PWD/ue_project/DerivedDataCache" AUTOFLY_PROBE_OUT=$PWD/runs/build/python_probe.json engine/Engine/Binaries/Linux/UnrealEditor-Cmd $PWD/ue_project/Blocks.uproject -run=pythonscript -script=$PWD/autofly_ue5/scenes/ue/probe_python.py -unattended -nop4 -nosplash -nullrhi -notraceserver -stdout -FullStdOutLogOutput -abslog=$PWD/runs/build/python_probe.log && bash scripts/run_job.sh wait python_probe 540; echo "wait=$?"
```
Repeat `bash scripts/run_job.sh wait python_probe 540` while it returns 124, within the budget. Then:
```bash
cd /home/nvidiasims/research_uav/autofly_ue5 && cat runs/jobs/python_probe.exit; grep -c "Error:" runs/build/python_probe.log; grep -E "Python script executed (successfully|with errors)" runs/build/python_probe.log; cat runs/build/python_probe.json; ls -d ue_project/DerivedDataCache/Zen ~/.config/Epic/UnrealEngine/Common/Zen/Data 2>&1; ls -d ue_project/Content/AutoFly/Probe 2>/dev/null || echo probe-asset-not-saved; ls -d ~/UnrealEngine /tmp/UnrealTraceServer.pid 2>/dev/null || echo no-trace-server-files
```
Expected: `Python script executed successfully` and JSON with `"engine_version"` starting `5.7.4`, all three subsystems `true`, `"game_mode_class": "/Script/ProjectAirSim.ProjectAirSimGameMode"`, `"sunsky_class": "/SunPosition/SunSky.SunSky_C"`, `cylinder_mesh`/`cube_mesh`/`world_grid_material` `true`, `"Color"` in `basic_shape_material_vector_params`, `"mic_parent": "/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"`, `"mic_color_readback_ok": true` (`mic_set_vector_return_value` is information only; the engine source predicts `false`); `probe-asset-not-saved`; `no-trace-server-files`; `~/.config/Epic/UnrealEngine/Common/Zen/Data` reported as missing (if it exists and did not exist in Task 2, the cache override failed: stop and report); `ue_project/DerivedDataCache/Zen` is listed once UE has started its cache, which a commandlet may not do. The exit code (`0` or `1`) and the `Error:` count are information only: exit code 1 with the success line and a correct JSON is not a failure (Global Constraints); report the first 10 `Error:` lines in that case. These facts are UNCONFIRMED in the research notes (commandlet editor subsystems, SunSky path, `Color` parameter, setting a material-instance colour from Python). If the success line is missing or any JSON value differs, stop and report the JSON and `grep -iE "error|warning: .*python" runs/build/python_probe.log | head -40`; do not switch the builder to other APIs without the user.

- [ ] **Step 9: Commit**

```bash
git -C /home/nvidiasims/research_uav/autofly_ue5 add scripts/create_ue_project.sh scripts/build_editor.sh ue_project/Blocks.uproject ue_project/Config ue_project/Source autofly_ue5/scenes/ue/probe_python.py && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Create ue_project from Blocks with the prebuilt plugin, build BlocksEditor and prove the editor Python commandlet" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Owned simulator processes and a headless BlocksMap run

**Files:**
- Create: `autofly_ue5/sim/__init__.py`, `autofly_ue5/sim/process.py`, `scripts/launch_sim.py`, `scripts/stop_sim.py`
- Test: `tests/test_process.py`

**Interfaces:**
- Consumes: `autofly_ue5.paths` (`RUNS_DIR`, `UNREAL_EDITOR`, `UPROJECT`, `PACKAGED_BINARY`, `UE_CACHE_ENV`), `autofly_ue5.gpu` (`gpu_memory_mib`, `check_gpu_for_launch`), `scripts/run_job.sh` (Task 1).
- Produces (`autofly_ue5.sim.process`):
  - `SIM_RUN_DIR: Path` (= `RUNS_DIR / "sim"`), exceptions `NotOwnedError`, `SimExitedError` (both `RuntimeError`)
  - `@dataclass(frozen=True) SimPorts(topics: int, services: int)`; `ports_for_instance(instance: int) -> SimPorts`
  - `instance_dir(instance: int, run_root: Path = SIM_RUN_DIR) -> Path`
  - `editor_game_command(map_path: str, ports: SimPorts, log_path: Path, instance: int, uproject: Path = UPROJECT, editor: Path = UNREAL_EDITOR) -> list[str]`
  - `packaged_command(map_path: str, ports: SimPorts, log_path: Path, instance: int, binary: Path = PACKAGED_BINARY) -> list[str]`
  - `sim_environment(editor_mode: bool) -> dict[str, str]` (no `PYTHONPATH`; `DISPLAY=:1`, `SDL_VIDEODRIVER=x11`, the `UE_CACHE_ENV` entries; `PROJECTAIRSIM_CI=1` only in editor mode)
  - `route_client_log(path: Path) -> None` (sends the `projectairsim` client logger to `path`, so concurrent clients never share `./projectairsim_client.log`, which `projectairsim.utils.projectairsim_log()` otherwise opens with `mode="w"` in the working directory, `utils.py:464-486`)
  - `@dataclass SimProcess(pid: int, pgid: int, instance: int, topics_port: int, services_port: int, cmd: list[str], log_path: str, started_unix: float)`
  - `launch_process(cmd: list[str], instance: int, ports: SimPorts, env: dict[str, str], run_root: Path = SIM_RUN_DIR) -> SimProcess`
  - `listening_pids(port: int) -> set[int]`; `is_alive(pid: int) -> bool`; `is_owned(sp: SimProcess) -> bool`; `read_pid_file(path: Path) -> SimProcess`
  - `wait_ready(sp: SimProcess, timeout_s: float, poll_s: float = 1.0) -> float`
  - `stop(instance: int, grace_s: float = 30.0, run_root: Path = SIM_RUN_DIR) -> str` (`"terminated"`, `"killed"`, `"not_running"`, `"no_pid_file"`)
  - `own_running_instances(run_root: Path = SIM_RUN_DIR) -> list[SimProcess]`
  - `handshake(ports: SimPorts, timeout_s: float = 120.0) -> float`
- CLIs: `scripts/launch_sim.py --mode {editor,packaged} --map <path> --instance N [--timeout S]` (prints one JSON line), `scripts/stop_sim.py --instance N`.

- [ ] **Step 1: Write the failing tests**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_process.py`:
```python
import json
import logging
import os
import socket
import sys
import time

import pytest

from autofly_ue5.sim.process import (
    NotOwnedError,
    SimExitedError,
    SimPorts,
    editor_game_command,
    instance_dir,
    launch_process,
    listening_pids,
    own_running_instances,
    packaged_command,
    ports_for_instance,
    route_client_log,
    sim_environment,
    stop,
    wait_ready,
)

FAKE_SERVER = (
    "import socket, sys, time\n"
    "held = []\n"
    "for port in sys.argv[1:]:\n"
    "    s = socket.socket()\n"
    "    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    "    s.bind(('127.0.0.1', int(port)))\n"
    "    s.listen()\n"
    "    held.append(s)\n"
    "time.sleep(600)\n"
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_ports_for_instance():
    assert ports_for_instance(0) == SimPorts(8989, 8990)
    assert ports_for_instance(1) == SimPorts(9001, 9002)
    with pytest.raises(ValueError):
        ports_for_instance(-1)


def test_editor_game_command(tmp_path):
    log = tmp_path / "sim.log"
    cmd = editor_game_command("/Game/BlocksMap", SimPorts(9001, 9002), log, 1)
    assert cmd[0].endswith("engine/Engine/Binaries/Linux/UnrealEditor")
    assert cmd[1].endswith("ue_project/Blocks.uproject")
    assert cmd[2] == "/Game/BlocksMap"
    for flag in ("-game", "-RenderOffScreen", "-vulkan", "-nosound", "-unattended", "-notraceserver", "-topicsport=9001",
                 "-servicesport=9002", f"-abslog={log}", "-saveddirsuffix=inst1"):
        assert flag in cmd


def test_packaged_command(tmp_path):
    cmd = packaged_command("/Game/AutoFly/Maps/S01", SimPorts(8989, 8990), tmp_path / "sim.log", 0)
    assert cmd[0].endswith("Packaged/Development/Linux/Blocks/Binaries/Linux/Blocks")
    assert cmd[1:3] == ["Blocks", "/Game/AutoFly/Maps/S01"]
    assert "-game" not in cmd and "-notraceserver" in cmd
    assert not any(c.startswith("-saveddirsuffix") for c in cmd)


def test_sim_environment(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/opt/ros/jazzy/lib/python3.12/site-packages")
    editor_env = sim_environment(editor_mode=True)
    assert "PYTHONPATH" not in editor_env
    assert editor_env["DISPLAY"] == ":1" and editor_env["SDL_VIDEODRIVER"] == "x11"
    assert editor_env["PROJECTAIRSIM_CI"] == "1"
    assert editor_env["UE-ZenDataPath"] == "/home/nvidiasims/research_uav/autofly_ue5/ue_project/DerivedDataCache/Zen"
    packaged_env = sim_environment(editor_mode=False)
    assert "PROJECTAIRSIM_CI" not in packaged_env
    assert packaged_env["UE-LocalDataCachePath"] == "/home/nvidiasims/research_uav/autofly_ue5/ue_project/DerivedDataCache"


def test_route_client_log_replaces_the_working_directory_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = logging.getLogger("projectairsim")
    saved = (list(logger.handlers), logger.level)
    try:
        route_client_log(tmp_path / "client" / "inst1.client.log")
        logger.info("hello from the client")
        for handler in logger.handlers:
            handler.flush()
        assert "hello from the client" in (tmp_path / "client" / "inst1.client.log").read_text()
        assert logger.hasHandlers()  # projectairsim_log() will not add ./projectairsim_client.log
        assert not (tmp_path / "projectairsim_client.log").exists()
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers[:] = saved[0]
        logger.setLevel(saved[1])


@pytest.fixture
def fake_sim(tmp_path):
    ports = SimPorts(free_port(), free_port())
    cmd = [sys.executable, "-c", FAKE_SERVER, str(ports.topics), str(ports.services)]
    sp = launch_process(cmd, 3, ports, dict(os.environ), run_root=tmp_path)
    yield sp, ports, tmp_path
    try:
        stop(3, grace_s=5.0, run_root=tmp_path)
    except Exception:
        pass


def test_launch_records_pid_and_becomes_ready(fake_sim):
    sp, ports, root = fake_sim
    data = json.loads((instance_dir(3, root) / "pid.json").read_text())
    assert data["pid"] == sp.pid and data["pgid"] == sp.pid
    assert wait_ready(sp, timeout_s=20.0, poll_s=0.2) < 20.0
    assert sp.pid in listening_pids(ports.topics)
    assert [p.pid for p in own_running_instances(root)] == [sp.pid]


def test_stop_terminates_owned_process(fake_sim):
    sp, ports, root = fake_sim
    wait_ready(sp, timeout_s=20.0, poll_s=0.2)
    assert stop(3, grace_s=10.0, run_root=root) == "terminated"
    assert not (instance_dir(3, root) / "pid.json").exists()
    deadline = time.monotonic() + 5.0
    while listening_pids(ports.topics) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert listening_pids(ports.topics) == set()


def test_stop_refuses_a_process_it_did_not_launch(tmp_path):
    d = instance_dir(5, tmp_path)
    d.mkdir(parents=True)
    foreign = {"pid": os.getpid(), "pgid": os.getpgid(os.getpid()), "instance": 5, "topics_port": 1,
               "services_port": 2, "cmd": ["/not/our/UnrealEditor"], "log_path": "x", "started_unix": 0.0}
    (d / "pid.json").write_text(json.dumps(foreign))
    with pytest.raises(NotOwnedError):
        stop(5, run_root=tmp_path)
    assert (d / "pid.json").exists()


def test_wait_ready_reports_early_exit(tmp_path):
    ports = SimPorts(free_port(), free_port())
    sp = launch_process([sys.executable, "-c", "import sys; sys.exit(3)"], 4, ports, dict(os.environ), run_root=tmp_path)
    with pytest.raises(SimExitedError):
        wait_ready(sp, timeout_s=10.0, poll_s=0.1)
    assert stop(4, run_root=tmp_path) == "not_running"


def test_launch_refuses_a_busy_port(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        with pytest.raises(RuntimeError, match="already in use"):
            launch_process([sys.executable, "-c", "pass"], 6, SimPorts(busy, free_port()), dict(os.environ), run_root=tmp_path)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_process.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'autofly_ue5.sim'`.

- [ ] **Step 3: Write `autofly_ue5/sim/__init__.py` and `autofly_ue5/sim/process.py`**

`autofly_ue5/sim/__init__.py`:
```python
"""Simulator interface; the only package that imports the Project AirSim client."""
```

`autofly_ue5/sim/process.py`:
```python
"""Launch, readiness and stop of simulator processes owned by this project.

Each launched process runs in its own session (PGID == PID) and is recorded in
runs/sim/inst<N>/pid.json. stop() only signals a process whose recorded PID is
alive, whose /proc cmdline[0] equals the recorded cmd[0], and whose process
group equals the recorded PGID. Unreal starts its helpers (ShaderCompileWorker,
zenserver, CrashReportClient) in their own process groups (UnixPlatformProcess.cpp:1048-1049),
so they are not signalled here; callers check for survivors with pgrep and report them.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from autofly_ue5.paths import PACKAGED_BINARY, RUNS_DIR, UE_CACHE_ENV, UNREAL_EDITOR, UPROJECT

SIM_RUN_DIR = RUNS_DIR / "sim"
BASE_TOPICS_PORT = 8989
BASE_SERVICES_PORT = 8990
PORT_STRIDE = 12
# -notraceserver: otherwise every non-Shipping UE process forks UnrealTraceServer, a daemon outside our process group
# that writes ~/UnrealEngine/UnrealTrace and /tmp/UnrealTraceServer.pid (TraceAuxiliary.cpp:1980-1986).
COMMON_FLAGS = ["-RenderOffScreen", "-nosound", "-unattended", "-nopause", "-nosplash", "-notraceserver", "-log",
                "-ResX=640", "-ResY=480"]


class NotOwnedError(RuntimeError):
    """The pid file points at a process this project did not launch."""


class SimExitedError(RuntimeError):
    """The simulator process exited before becoming ready."""


@dataclass(frozen=True)
class SimPorts:
    topics: int
    services: int


@dataclass
class SimProcess:
    pid: int
    pgid: int
    instance: int
    topics_port: int
    services_port: int
    cmd: list[str]
    log_path: str
    started_unix: float


def ports_for_instance(instance: int) -> SimPorts:
    if instance < 0:
        raise ValueError(f"instance must be >= 0, got {instance}")
    return SimPorts(BASE_TOPICS_PORT + PORT_STRIDE * instance, BASE_SERVICES_PORT + PORT_STRIDE * instance)


def instance_dir(instance: int, run_root: Path = SIM_RUN_DIR) -> Path:
    return Path(run_root) / f"inst{instance}"


def _port_flags(ports: SimPorts, log_path: Path, instance: int) -> list[str]:
    flags = [f"-topicsport={ports.topics}", f"-servicesport={ports.services}", f"-abslog={log_path}"]
    if instance > 0:
        flags.append(f"-saveddirsuffix=inst{instance}")
    return flags


def editor_game_command(
    map_path: str, ports: SimPorts, log_path: Path, instance: int, uproject: Path = UPROJECT, editor: Path = UNREAL_EDITOR
) -> list[str]:
    return [str(editor), str(uproject), map_path, "-game", "-vulkan", *COMMON_FLAGS, *_port_flags(ports, log_path, instance)]


def packaged_command(
    map_path: str, ports: SimPorts, log_path: Path, instance: int, binary: Path = PACKAGED_BINARY
) -> list[str]:
    return [str(binary), "Blocks", map_path, *COMMON_FLAGS, *_port_flags(ports, log_path, instance)]


def sim_environment(editor_mode: bool) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["DISPLAY"] = ":1"
    env["SDL_VIDEODRIVER"] = "x11"
    env.update(UE_CACHE_ENV)
    if editor_mode:
        env["PROJECTAIRSIM_CI"] = "1"
    else:
        env.pop("PROJECTAIRSIM_CI", None)
    return env


def route_client_log(path: Path) -> None:
    """Point the Project AirSim client logger at `path`; projectairsim_log() then skips its ./projectairsim_client.log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger = logging.getLogger("projectairsim")
    logger.handlers[:] = [handler]
    logger.setLevel(logging.DEBUG)


def read_pid_file(path: Path) -> SimProcess:
    return SimProcess(**json.loads(Path(path).read_text()))


def _proc_state(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return stat.rsplit(")", 1)[1].split()[0]


def is_alive(pid: int) -> bool:
    state = _proc_state(pid)
    return state is not None and state != "Z"


def _proc_cmdline(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def is_owned(sp: SimProcess) -> bool:
    cmdline = _proc_cmdline(sp.pid)
    if not cmdline:
        return False
    try:
        pgid = os.getpgid(sp.pid)
    except ProcessLookupError:
        return False
    return cmdline[0] == sp.cmd[0] and pgid == sp.pgid


def _reap(pid: int) -> None:
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass


def listening_pids(port: int) -> set[int]:
    out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"], capture_output=True, text=True, check=True).stdout
    pids: set[int] = set()
    for line in out.splitlines():
        if not line.strip():
            continue
        found = [int(p) for p in re.findall(r"pid=(\d+)", line)]
        pids.update(found if found else [-1])
    return pids


def launch_process(
    cmd: list[str], instance: int, ports: SimPorts, env: dict[str, str], run_root: Path = SIM_RUN_DIR
) -> SimProcess:
    directory = instance_dir(instance, run_root)
    directory.mkdir(parents=True, exist_ok=True)
    pid_file = directory / "pid.json"
    if pid_file.exists():
        existing = read_pid_file(pid_file)
        if is_alive(existing.pid) and is_owned(existing):
            raise RuntimeError(f"instance {instance} is already running with pid {existing.pid}")
        pid_file.unlink()
    for port in (ports.topics, ports.services):
        if listening_pids(port):
            raise RuntimeError(f"port {port} is already in use")
    with open(directory / "stdout.log", "wb") as out:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env,
            start_new_session=True, cwd=directory,
        )
    log_path = next((c.split("=", 1)[1] for c in cmd if c.startswith("-abslog=")), str(directory / "stdout.log"))
    sp = SimProcess(
        pid=proc.pid, pgid=os.getpgid(proc.pid), instance=instance, topics_port=ports.topics,
        services_port=ports.services, cmd=list(cmd), log_path=log_path, started_unix=time.time(),
    )
    pid_file.write_text(json.dumps(asdict(sp), indent=2))
    return sp


def wait_ready(sp: SimProcess, timeout_s: float, poll_s: float = 1.0) -> float:
    start = time.monotonic()
    while True:
        if not is_alive(sp.pid):
            _reap(sp.pid)
            raise SimExitedError(f"simulator pid {sp.pid} exited before its ports opened; see {sp.log_path}")
        if sp.pid in listening_pids(sp.topics_port) and sp.pid in listening_pids(sp.services_port):
            return time.monotonic() - start
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"ports {sp.topics_port}/{sp.services_port} not open after {timeout_s} s; see {sp.log_path}")
        time.sleep(poll_s)


def stop(instance: int, grace_s: float = 30.0, run_root: Path = SIM_RUN_DIR) -> str:
    pid_file = instance_dir(instance, run_root) / "pid.json"
    if not pid_file.exists():
        return "no_pid_file"
    sp = read_pid_file(pid_file)
    if not is_alive(sp.pid):
        _reap(sp.pid)
        pid_file.unlink()
        return "not_running"
    if not is_owned(sp):
        raise NotOwnedError(
            f"pid {sp.pid} in {pid_file} does not match the recorded command/process group; not stopping it"
        )
    os.killpg(sp.pgid, signal.SIGTERM)
    result = "terminated"
    deadline = time.monotonic() + grace_s
    while is_alive(sp.pid) and time.monotonic() < deadline:
        _reap(sp.pid)
        time.sleep(0.2)
    if is_alive(sp.pid):
        os.killpg(sp.pgid, signal.SIGKILL)
        result = "killed"
        deadline = time.monotonic() + 10.0
        while is_alive(sp.pid) and time.monotonic() < deadline:
            _reap(sp.pid)
            time.sleep(0.2)
    pid_file.unlink()
    return result


def own_running_instances(run_root: Path = SIM_RUN_DIR) -> list[SimProcess]:
    running = []
    for pid_file in sorted(Path(run_root).glob("inst*/pid.json")):
        sp = read_pid_file(pid_file)
        if is_alive(sp.pid) and is_owned(sp):
            running.append(sp)
    return running


def handshake(ports: SimPorts, timeout_s: float = 120.0) -> float:
    """Connect a Project AirSim client, fetch the topic list and disconnect; retried until timeout."""
    from projectairsim import ProjectAirSimClient

    start = time.monotonic()
    last_error: Exception | None = None
    while time.monotonic() - start < timeout_s:
        client = ProjectAirSimClient(port_topics=ports.topics, port_services=ports.services)
        try:
            client.connect()
            client.get_topic_info()
            return time.monotonic() - start
        except Exception as err:  # the server may still be starting
            last_error = err
            time.sleep(2.0)
        finally:
            client.disconnect()
    raise TimeoutError(f"handshake on {ports} failed for {timeout_s} s: {last_error}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_process.py`
Expected: `10 passed`.

- [ ] **Step 5: Write the CLIs**

`/home/nvidiasims/research_uav/autofly_ue5/scripts/launch_sim.py`:
```python
"""Launch one owned simulator instance, wait for its ports and a client handshake, print a JSON line.

env -u PYTHONPATH .venv/bin/python scripts/launch_sim.py --mode editor --map /Game/BlocksMap --instance 0
"""

import argparse
import json
import sys

from autofly_ue5.gpu import check_gpu_for_launch, gpu_memory_mib
from autofly_ue5.sim.process import (
    editor_game_command,
    handshake,
    instance_dir,
    launch_process,
    own_running_instances,
    packaged_command,
    ports_for_instance,
    route_client_log,
    sim_environment,
    stop,
    wait_ready,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["editor", "packaged"], required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    ports = ports_for_instance(args.instance)
    used, total = gpu_memory_mib()
    check_gpu_for_launch(used, total, own_running=len(own_running_instances()))
    directory = instance_dir(args.instance)
    directory.mkdir(parents=True, exist_ok=True)
    route_client_log(directory / "launch.client.log")
    log = directory / "sim.log"
    if args.mode == "editor":
        cmd = editor_game_command(args.map, ports, log, args.instance)
    else:
        cmd = packaged_command(args.map, ports, log, args.instance)
    sp = launch_process(cmd, args.instance, ports, sim_environment(editor_mode=args.mode == "editor"))
    try:
        ready_s = wait_ready(sp, args.timeout)
        handshake_s = handshake(ports, timeout_s=120.0)
    except Exception:
        print(f"launch failed; stopping pid {sp.pid}: {stop(args.instance)}", file=sys.stderr)
        raise
    print(json.dumps({
        "pid": sp.pid, "instance": args.instance, "ports": [ports.topics, ports.services],
        "ports_ready_s": round(ready_s, 1), "handshake_s": round(handshake_s, 1),
        "gpu_used_mib_before": used, "log": str(log),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`/home/nvidiasims/research_uav/autofly_ue5/scripts/stop_sim.py`:
```python
"""Stop one simulator instance launched by this project (refuses processes it did not launch).

env -u PYTHONPATH .venv/bin/python scripts/stop_sim.py --instance 0
"""

import argparse
import sys

from autofly_ue5.sim.process import stop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, required=True)
    parser.add_argument("--grace", type=float, default=30.0)
    args = parser.parse_args()
    print(stop(args.instance, grace_s=args.grace))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Launch BlocksMap headless as a job (first launch compiles shaders; total wait budget 25 minutes)**

Run:
```bash
cd /home/nvidiasims/research_uav/autofly_ue5 && nvidia-smi --query-gpu=memory.used --format=csv,noheader && ss -ltnH '( sport = :8989 or sport = :8990 )' && bash scripts/run_job.sh start launch_inst0 -- env -u PYTHONPATH .venv/bin/python scripts/launch_sim.py --mode editor --map /Game/BlocksMap --instance 0 --timeout 900 && bash scripts/run_job.sh wait launch_inst0 540; echo "wait=$?"
```
Repeat `bash scripts/run_job.sh wait launch_inst0 540` while it returns 124, within the budget. If the budget is used up: `bash scripts/run_job.sh stop launch_inst0`, then `env -u PYTHONPATH .venv/bin/python scripts/stop_sim.py --instance 0` (the simulator runs in its own recorded session), and report; do not re-launch.
Expected: GPU memory below 2000 MiB, no listener on 8989/8990, then in the job log a JSON line such as `{"pid": 123456, "instance": 0, "ports": [8989, 8990], "ports_ready_s": …, "handshake_s": …}`, `job launch_inst0 finished: exit=0` and `wait=0`. Whether the positional map argument `/Game/BlocksMap` is honoured by `UnrealEditor <uproject> <map> -game` is UNCONFIRMED: confirm with `grep -m3 -E "LoadMap|Browse" runs/sim/inst0/sim.log` (expected: a line naming `/Game/BlocksMap`). If launch fails, the CLI stops the process; stop and report `tail -80 runs/sim/inst0/sim.log` and `grep -iE "Xid|VK_ERROR|Fatal" runs/sim/inst0/sim.log`.

- [ ] **Step 7: Check the process, its session, the server log and the cache location**

Run:
```bash
cd /home/nvidiasims/research_uav/autofly_ue5 && PID=$(jq .pid runs/sim/inst0/pid.json) && ps -o pid,pgid,etimes,cmd -p "$PID" | cut -c1-160 && ss -ltnpH '( sport = :8989 or sport = :8990 )' && grep -m2 "active at" ue_project/projectairsim_server.log; nvidia-smi --query-gpu=memory.used --format=csv,noheader; echo "xid now $(journalctl _TRANSPORT=kernel --since "$(jq -r .started_at runs/m0/engine_check.json)" --no-pager | grep -c 'NVRM: Xid'), baseline $(jq .checks.nvidia_xid.after runs/m0/engine_check.json)"; echo "boot $(cat /proc/sys/kernel/random/boot_id), m0 start $(jq -r .boot_id runs/m0/engine_check.json)"; du -sh ue_project/DerivedDataCache; ls -d ~/.config/Epic/UnrealEngine/Common/Zen/Data 2>/dev/null || echo zen-default-absent; ls -d ~/UnrealEngine /tmp/UnrealTraceServer.pid 2>/dev/null || echo no-trace-server-files
```
Expected: PID equals PGID; both ports listed with `pid=<PID>`; `Topics active at: 'tcp://*:8989'` and `Services active at: 'tcp://*:8990'` (log wording UNCONFIRMED; missing lines alone are not a failure when the handshake succeeded); GPU memory printed (note it); Xid `now` equal to `baseline` (counted from the M0 start across reboots); the two boot ids equal; the size of `ue_project/DerivedDataCache` (information: expected to grow as shaders are compiled); `zen-default-absent`; `no-trace-server-files`. A higher Xid count, a different boot id, the default Zen folder or a trace-server file appearing: stop and report (after stopping instance 0, per the failure rule).

- [ ] **Step 8: Stop it and confirm cleanup**

Run (Bash tool `timeout` 600000 ms): `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python scripts/stop_sim.py --instance 0 && sleep 30; ss -ltnH '( sport = :8989 or sport = :8990 )' | wc -l; ls runs/sim/inst0/; pgrep -a -f "$(pwd)/engine/Engine/Binaries/Linux/" || echo none; pgrep -a zenserver || echo no-zenserver`
Expected: `terminated` (or `killed` after 30 s, which is acceptable but note it), `0` listeners, no `pid.json` in the listing, `none`, `no-zenserver`. Unreal helpers run in their own process groups (`UnixPlatformProcess.cpp:1048-1049`), so `stop_sim.py` cannot signal them: if any `ShaderCompileWorker`, `CrashReportClient` or `zenserver` is still listed, wait 60 s and check again, then report the survivors without killing them.

- [ ] **Step 9: Commit**

```bash
git -C /home/nvidiasims/research_uav/autofly_ue5 add autofly_ue5/sim/__init__.py autofly_ue5/sim/process.py scripts/launch_sim.py scripts/stop_sim.py tests/test_process.py && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Launch and stop simulator processes with recorded PIDs, port readiness and a client handshake" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Frame conversions and image decoding

**Files:**
- Create: `autofly_ue5/frames.py`, `autofly_ue5/sim/decode.py`
- Test: `tests/test_frames.py`, `tests/test_decode.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `autofly_ue5.frames`: `CM_PER_M = 100.0`; `ned_to_ue_cm(x: float, y: float, z: float) -> tuple[float, float, float]`; `ue_cm_to_ned(x_cm: float, y_cm: float, z_cm: float) -> tuple[float, float, float]`; `ned_yaw_deg_to_ue_yaw_deg(yaw_deg: float) -> float`; `wrap_pi(angle: float) -> float`; `yaw_to_quat(yaw: float) -> tuple[float, float, float, float]` (w, x, y, z); `quat_to_yaw(w: float, x: float, y: float, z: float) -> float`; `z_up_from_ned(z_ned: float) -> float`; `heading_unit(yaw: float) -> tuple[float, float]` (north, east components); `body_to_ned(w: float, x: float, y: float, z: float, v: tuple[float, float, float]) -> tuple[float, float, float]` (rotates a body-frame FRD vector into NED with the body-to-world orientation quaternion, as used for the camera mount offset).
  - `autofly_ue5.sim.decode`: `ImageFormatError(ValueError)`; `decode_rgb(msg: dict) -> np.ndarray` (H×W×3 uint8, RGB order); `decode_depth(msg: dict) -> np.ndarray` (H×W float32 metres, `+inf` = no hit). Both accept `data` as `bytes` (topic callbacks) or a list of ints (`get_images`).

- [ ] **Step 1: Write the failing tests**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_frames.py`:
```python
import math

import pytest

from autofly_ue5.frames import (
    body_to_ned,
    heading_unit,
    ned_to_ue_cm,
    ned_yaw_deg_to_ue_yaw_deg,
    quat_to_yaw,
    ue_cm_to_ned,
    wrap_pi,
    yaw_to_quat,
    z_up_from_ned,
)


def test_ned_to_ue_cm_flips_z_and_scales():
    assert ned_to_ue_cm(1.0, -2.0, -3.0) == (100.0, -200.0, 300.0)


def test_ue_cm_round_trip():
    assert ue_cm_to_ned(*ned_to_ue_cm(12.5, -7.25, -1.75)) == pytest.approx((12.5, -7.25, -1.75))


def test_ground_top_is_ue_zero():
    assert ned_to_ue_cm(0.0, 0.0, 0.0) == (0.0, 0.0, -0.0)


def test_yaw_is_identical_in_ue():
    assert ned_yaw_deg_to_ue_yaw_deg(37.5) == 37.5


def test_yaw_quaternion_round_trip():
    w, x, y, z = yaw_to_quat(math.pi / 2)
    assert (w, x, y, z) == pytest.approx((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))
    assert quat_to_yaw(w, x, y, z) == pytest.approx(math.pi / 2)
    assert quat_to_yaw(*yaw_to_quat(-2.5)) == pytest.approx(-2.5)


def test_wrap_pi():
    assert wrap_pi(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert wrap_pi(-3 * math.pi / 2) == pytest.approx(math.pi / 2)
    assert wrap_pi(0.3) == pytest.approx(0.3)


def test_autofly_z_up():
    assert z_up_from_ned(-2.0) == 2.0


def test_heading_unit_east():
    assert heading_unit(math.pi / 2) == pytest.approx((0.0, 1.0))


def test_body_to_ned_yaw_and_nose_up_pitch():
    assert body_to_ned(*yaw_to_quat(math.pi / 2), (0.4, 0.0, 0.0)) == pytest.approx((0.0, 0.4, 0.0), abs=1e-12)
    half = math.radians(30.0) / 2  # 30 deg nose-up pitch about body y: a forward point moves up (negative z in NED)
    forward = body_to_ned(math.cos(half), 0.0, math.sin(half), 0.0, (0.4, 0.0, 0.0))
    assert forward == pytest.approx((0.4 * math.cos(math.radians(30.0)), 0.0, -0.4 * math.sin(math.radians(30.0))))
```

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_decode.py`:
```python
import numpy as np
import pytest

from autofly_ue5.sim.decode import ImageFormatError, decode_depth, decode_rgb


def _rgb_msg(rgb: np.ndarray, as_list: bool = False) -> dict:
    data = np.ascontiguousarray(rgb[:, :, ::-1]).tobytes()
    return {"encoding": "BGR", "height": rgb.shape[0], "width": rgb.shape[1], "data": list(data) if as_list else data}


def test_decode_rgb_reverses_bgr():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[10, 20, 30], [40, 50, 60], [70, 80, 90]]], dtype=np.uint8)
    out = decode_rgb(_rgb_msg(rgb))
    assert out.dtype == np.uint8 and out.shape == (2, 3, 3)
    assert np.array_equal(out, rgb)


def test_decode_rgb_accepts_int_list():
    rgb = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    assert np.array_equal(decode_rgb(_rgb_msg(rgb, as_list=True)), rgb)


def test_decode_depth_half_float_metres_with_inf():
    depth = np.array([[1.5, np.inf], [0.25, 30.0]], dtype="<f2")
    msg = {"encoding": "16FC1", "height": 2, "width": 2, "data": depth.tobytes()}
    out = decode_depth(msg)
    assert out.dtype == np.float32
    assert out[0, 0] == pytest.approx(1.5) and np.isinf(out[0, 1])
    assert out[1, 0] == pytest.approx(0.25) and out[1, 1] == pytest.approx(30.0)


def test_wrong_encoding_is_rejected():
    with pytest.raises(ImageFormatError, match="PNG"):
        decode_rgb({"encoding": "PNG", "height": 1, "width": 1, "data": b"\x00\x00\x00"})
    with pytest.raises(ImageFormatError, match="BGR"):
        decode_depth({"encoding": "BGR", "height": 1, "width": 1, "data": b"\x00\x00"})


def test_size_mismatch_is_rejected():
    with pytest.raises(ImageFormatError, match="bytes"):
        decode_rgb({"encoding": "BGR", "height": 2, "width": 2, "data": b"\x00" * 11})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_frames.py tests/test_decode.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'autofly_ue5.frames'` and `No module named 'autofly_ue5.sim.decode'`.

- [ ] **Step 3: Write `autofly_ue5/frames.py`**

```python
"""Coordinate conventions.

Project AirSim world: NED, metres (x north, y east, z down); yaw rotates north toward east, radians.
Unreal Engine world: X forward (= north), Y right (= east), Z up, centimetres; there is no origin
offset between the UE world origin and the NED origin, and a pure yaw is the same angle in both.
AutoFly state: z is up, metres, so z_up = -z_ned.
"""

import math

CM_PER_M = 100.0


def ned_to_ue_cm(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (x * CM_PER_M, y * CM_PER_M, -z * CM_PER_M)


def ue_cm_to_ned(x_cm: float, y_cm: float, z_cm: float) -> tuple[float, float, float]:
    return (x_cm / CM_PER_M, y_cm / CM_PER_M, -z_cm / CM_PER_M)


def ned_yaw_deg_to_ue_yaw_deg(yaw_deg: float) -> float:
    return yaw_deg


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def z_up_from_ned(z_ned: float) -> float:
    return -z_ned


def heading_unit(yaw: float) -> tuple[float, float]:
    return (math.cos(yaw), math.sin(yaw))


def body_to_ned(w: float, x: float, y: float, z: float, v: tuple[float, float, float]) -> tuple[float, float, float]:
    vx, vy, vz = v
    return (
        (1.0 - 2.0 * (y * y + z * z)) * vx + 2.0 * (x * y - w * z) * vy + 2.0 * (x * z + w * y) * vz,
        2.0 * (x * y + w * z) * vx + (1.0 - 2.0 * (x * x + z * z)) * vy + 2.0 * (y * z - w * x) * vz,
        2.0 * (x * z - w * y) * vx + 2.0 * (y * z + w * x) * vy + (1.0 - 2.0 * (x * x + y * y)) * vz,
    )
```

- [ ] **Step 4: Write `autofly_ue5/sim/decode.py`**

```python
"""Decode Project AirSim image messages (camera compress=false).

RGB (image type 0): encoding "BGR", 3 uint8 per pixel in B, G, R order, rows top to bottom.
Depth (image types 1/2): encoding "16FC1", one little-endian IEEE half float per pixel, metres, +inf = no hit.
"""

import numpy as np


class ImageFormatError(ValueError):
    """The message encoding or payload size is not what the decoder expects."""


def _as_uint8(data) -> np.ndarray:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(data, dtype=np.uint8)
    return np.asarray(data, dtype=np.uint8)


def decode_rgb(msg: dict) -> np.ndarray:
    if msg["encoding"] != "BGR":
        raise ImageFormatError(f"expected encoding BGR, got {msg['encoding']}")
    height, width = int(msg["height"]), int(msg["width"])
    raw = _as_uint8(msg["data"])
    if raw.size != height * width * 3:
        raise ImageFormatError(f"expected {height * width * 3} bytes for {height}x{width} BGR, got {raw.size}")
    return np.ascontiguousarray(raw.reshape(height, width, 3)[:, :, ::-1])


def decode_depth(msg: dict) -> np.ndarray:
    if msg["encoding"] != "16FC1":
        raise ImageFormatError(f"expected encoding 16FC1, got {msg['encoding']}")
    height, width = int(msg["height"]), int(msg["width"])
    raw = _as_uint8(msg["data"])
    if raw.size != height * width * 2:
        raise ImageFormatError(f"expected {height * width * 2} bytes for {height}x{width} 16FC1, got {raw.size}")
    return raw.view("<f2").reshape(height, width).astype(np.float32)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_frames.py tests/test_decode.py`
Expected: `14 passed`.

- [ ] **Step 6: Commit**

```bash
git -C /home/nvidiasims/research_uav/autofly_ue5 add autofly_ue5/frames.py autofly_ue5/sim/decode.py tests/test_frames.py tests/test_decode.py && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Add NED/UE-centimetre frame conversions and BGR/16FC1 image decoding" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Project AirSim configs and the M0 smoke test (single instance)

**Files:**
- Create: `configs/robot_autofly_quadrotor.jsonc`, `configs/scene_autofly_m0.jsonc`, `configs/scene_autofly_m0_fast.jsonc`, `autofly_ue5/sim/smoke_m0.py`
- Create (captured): `tests/fixtures/pas/rgb_msg.json`, `tests/fixtures/pas/rgb_msg.bin`, `tests/fixtures/pas/depth_msg.json`, `tests/fixtures/pas/depth_msg.bin`, `docs/gates/m0_smoke_inst0.json`
- Test: `tests/test_pas_configs.py` (offline schema validation) + live smoke run

**Interfaces:**
- Consumes: `autofly_ue5.paths.CONFIGS_DIR`; `autofly_ue5.frames.{body_to_ned, quat_to_yaw, wrap_pi, yaw_to_quat, heading_unit}`; `autofly_ue5.sim.decode.{decode_rgb, decode_depth}`; `autofly_ue5.sim.process.{ports_for_instance, route_client_log}`; `autofly_ue5.gpu.gpu_memory_mib`; `scripts/launch_sim.py`, `scripts/stop_sim.py` (Task 5); `scripts/run_job.sh` (Task 1).
- Produces:
  - Config files: robot `Drone1` config with sensor `FrontCamera` (topics `drone.sensors["FrontCamera"]["scene_camera"]` and `["depth_planar_camera"]`), scene ids `SceneAutoFlyM0` / `SceneAutoFlyM0Fast`.
  - CLI `python -m autofly_ue5.sim.smoke_m0 --instance N --scene <file> --phases lockstep,velocity,reset,spawn,collision,teleport --steps 50 --out <json> [--warmup-steps 0] [--start-at <unix seconds>] [--fixtures-dir <dir>] [--command-timeout 10]`; exit 0 iff report `pass`; the client log goes to `<out>.client.log`. Report keys: `instance`, `ports`, `scene`, `load_scene_s`, `clock_type`, `paused_on_load`, `camera_topics`, `phases.{lockstep,velocity,reset,spawn,collision,teleport}` (each with `pass`), `pass`. `phases.lockstep` runs `--warmup-steps` untimed records, sleeps until `--start-at` if given, then times `--steps` records; it carries `steps_per_s` (timed records only), `warmup_steps`, `timed_start_unix`, `timed_end_unix`, `start_late_s`, `vram_used_mib` and `vram_sample_unix` (one sample taken after timed record `steps // 2`, while the camera is capturing), `max_camera_pose_error_m`, `get_images_probe`; `phases.reset` carries `camera_pose_vs_state_m`; `phases.spawn` carries `center_face_width_px`, `expected_face_width_px`, `hfov_consistent`, `material_ok`, `material_patch_mean_abs_diff`, `material_center_rgb` and `material_instance_ok` (informational); `phases.teleport` records what `set_pose` through an obstacle does to the Unreal actor (research gotcha 6).
  - `CommandTimeoutError(RuntimeError)`, `camera_pose_error(msg: dict | None, rec: dict) -> float` (`inf` when the message is missing), `contiguous_width(mask_row: np.ndarray, center: int = 128) -> int` and `flat_face_width_px(half_width_m: float, depth_m: float, image_width: int = 256, hfov_deg: float = 90.0) -> float` in `autofly_ue5.sim.smoke_m0`.
  - Fixture files: `<stem>.bin` (raw `data` bytes) and `<stem>.json` (all other message fields plus `data_len`; RGB adds `center_pixel_bgr`; depth adds `inf_count` and `expected_center_depth_m`).

- [ ] **Step 1: Write the failing config test**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_pas_configs.py`:
```python
import pytest
from projectairsim.utils import load_scene_config_as_dict

from autofly_ue5.paths import CONFIGS_DIR


@pytest.mark.parametrize("scene", ["scene_autofly_m0.jsonc", "scene_autofly_m0_fast.jsonc"])
def test_scene_validates_against_project_airsim_schema(scene):
    config, _paths = load_scene_config_as_dict(scene, str(CONFIGS_DIR))
    assert config["clock"]["type"] == "steppable"
    assert config["clock"]["step-ns"] == 5_000_000
    assert config["clock"]["pause-on-start"] is True
    robot = config["actors"][0]["robot-config"]
    camera = next(s for s in robot["sensors"] if s["id"] == "FrontCamera")
    assert camera["capture-interval"] == 0.001
    settings = {c["image-type"]: c for c in camera["capture-settings"]}
    assert set(settings) == {0, 1}
    for c in settings.values():
        assert (c["width"], c["height"], c["fov-degrees"]) == (256, 256, 90)
        assert c["capture-enabled"] is True and c["compress"] is False
    assert camera["origin"]["xyz"] == "0.40 0.0 0.0"
```
(This test imports `projectairsim` only to reuse the upstream schema validation; package code outside `sim/` still never imports it.)

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_pas_configs.py`
Expected: FAIL (2 failed) with `FileNotFoundError: [Errno 2] No such file or directory: '/home/nvidiasims/research_uav/autofly_ue5/configs/scene_autofly_m0.jsonc'` (and the same for `scene_autofly_m0_fast.jsonc`).

- [ ] **Step 3: Write `configs/robot_autofly_quadrotor.jsonc`**

(Airframe, controller and actuators are copied from `platform/client/python/example_user_scripts/sim_config/robot_quadrotor_manual_step.jsonc`; the chase and down cameras are replaced by `FrontCamera`.)
```jsonc
{
  "physics-type": "fast-physics",
  "links": [
    {
      "name": "Frame",
      "inertial": {
        "mass": 1.0,
        "inertia": { "type": "geometry", "geometry": { "box": { "size": "0.180 0.110 0.040" } } },
        "aerodynamics": { "drag-coefficient": 0.325, "type": "geometry", "geometry": { "box": { "size": "0.180 0.110 0.040" } } }
      },
      "collision": { "restitution": 0.1, "friction": 0.5 },
      "visual": { "geometry": { "type": "unreal_mesh", "name": "/Drone/Quadrotor1" } }
    },
    {
      "name": "Prop_FL",
      "inertial": {
        "origin": { "xyz": "0.253 -0.253 -0.01", "rpy-deg": "0 0 0" },
        "mass": 0.055,
        "inertia": { "type": "point-mass" },
        "aerodynamics": { "drag-coefficient": 0.325, "type": "geometry", "geometry": { "cylinder": { "radius": 0.1143, "length": 0.01 } } }
      },
      "visual": { "origin": { "xyz": "0.253 -0.253 -0.01", "rpy-deg": "0 0 0" }, "geometry": { "type": "unreal_mesh", "name": "/Drone/PropellerRed" } }
    },
    {
      "name": "Prop_FR",
      "inertial": {
        "origin": { "xyz": "0.253 0.253 -0.01", "rpy-deg": "0 0 0" },
        "mass": 0.055,
        "inertia": { "type": "point-mass" },
        "aerodynamics": { "drag-coefficient": 0.325, "type": "geometry", "geometry": { "cylinder": { "radius": 0.1143, "length": 0.01 } } }
      },
      "visual": { "origin": { "xyz": "0.253 0.253 -0.01", "rpy-deg": "0 0 0" }, "geometry": { "type": "unreal_mesh", "name": "/Drone/PropellerRed" } }
    },
    {
      "name": "Prop_RL",
      "inertial": {
        "origin": { "xyz": "-0.253 -0.253 -0.01", "rpy-deg": "0 0 0" },
        "mass": 0.055,
        "inertia": { "type": "point-mass" },
        "aerodynamics": { "drag-coefficient": 0.325, "type": "geometry", "geometry": { "cylinder": { "radius": 0.1143, "length": 0.01 } } }
      },
      "visual": { "origin": { "xyz": "-0.253 -0.253 -0.01", "rpy-deg": "0 0 0" }, "geometry": { "type": "unreal_mesh", "name": "/Drone/PropellerWhite" } }
    },
    {
      "name": "Prop_RR",
      "inertial": {
        "origin": { "xyz": "-0.253 0.253 -0.01", "rpy-deg": "0 0 0" },
        "mass": 0.055,
        "inertia": { "type": "point-mass" },
        "aerodynamics": { "drag-coefficient": 0.325, "type": "geometry", "geometry": { "cylinder": { "radius": 0.1143, "length": 0.01 } } }
      },
      "visual": { "origin": { "xyz": "-0.253 0.253 -0.01", "rpy-deg": "0 0 0" }, "geometry": { "type": "unreal_mesh", "name": "/Drone/PropellerWhite" } }
    }
  ],
  "joints": [
    { "id": "Frame_Prop_FL", "type": "fixed", "parent-link": "Frame", "child-link": "Prop_FL", "axis": "0 0 1" },
    { "id": "Frame_Prop_FR", "type": "fixed", "parent-link": "Frame", "child-link": "Prop_FR", "axis": "0 0 1" },
    { "id": "Frame_Prop_RL", "type": "fixed", "parent-link": "Frame", "child-link": "Prop_RL", "axis": "0 0 1" },
    { "id": "Frame_Prop_RR", "type": "fixed", "parent-link": "Frame", "child-link": "Prop_RR", "axis": "0 0 1" }
  ],
  "controller": {
    "id": "Simple_Flight_Controller",
    "airframe-setup": "quadrotor-x",
    "type": "simple-flight-api",
    "simple-flight-api-settings": {
      "actuator-order": [ { "id": "Prop_FR_actuator" }, { "id": "Prop_RL_actuator" }, { "id": "Prop_FL_actuator" }, { "id": "Prop_RR_actuator" } ]
    }
  },
  "actuators": [
    {
      "name": "Prop_FL_actuator", "type": "rotor", "enabled": true, "parent-link": "Frame", "child-link": "Prop_FL",
      "origin": { "xyz": "0.253 -0.253 -0.01", "rpy-deg": "0 0 0" },
      "rotor-settings": { "turning-direction": "clock-wise", "normal-vector": "0.0 0.0 -1.0", "coeff-of-thrust": 0.109919, "coeff-of-torque": 0.040164, "max-rpm": 6396.667, "propeller-diameter": 0.2286, "smoothing-tc": 0.005 }
    },
    {
      "name": "Prop_FR_actuator", "type": "rotor", "enabled": true, "parent-link": "Frame", "child-link": "Prop_FR",
      "origin": { "xyz": "0.253 0.253 -0.01", "rpy-deg": "0 0 0" },
      "rotor-settings": { "turning-direction": "counter-clock-wise", "normal-vector": "0.0 0.0 -1.0", "coeff-of-thrust": 0.109919, "coeff-of-torque": 0.040164, "max-rpm": 6396.667, "propeller-diameter": 0.2286, "smoothing-tc": 0.005 }
    },
    {
      "name": "Prop_RL_actuator", "type": "rotor", "enabled": true, "parent-link": "Frame", "child-link": "Prop_RL",
      "origin": { "xyz": "-0.253 -0.253 -0.01", "rpy-deg": "0 0 0" },
      "rotor-settings": { "turning-direction": "counter-clock-wise", "normal-vector": "0.0 0.0 -1.0", "coeff-of-thrust": 0.109919, "coeff-of-torque": 0.040164, "max-rpm": 6396.667, "propeller-diameter": 0.2286, "smoothing-tc": 0.005 }
    },
    {
      "name": "Prop_RR_actuator", "type": "rotor", "enabled": true, "parent-link": "Frame", "child-link": "Prop_RR",
      "origin": { "xyz": "-0.253 0.253 -0.01", "rpy-deg": "0 0 0" },
      "rotor-settings": { "turning-direction": "clock-wise", "normal-vector": "0.0 0.0 -1.0", "coeff-of-thrust": 0.109919, "coeff-of-torque": 0.040164, "max-rpm": 6396.667, "propeller-diameter": 0.2286, "smoothing-tc": 0.005 }
    }
  ],
  "sensors": [
    {
      "id": "FrontCamera",
      "type": "camera",
      "enabled": true,
      "parent-link": "Frame",
      "capture-interval": 0.001,
      "capture-settings": [
        { "image-type": 0, "width": 256, "height": 256, "fov-degrees": 90, "capture-enabled": true, "streaming-enabled": false,
          "pixels-as-float": false, "compress": false, "target-gamma": 2.5, "motion-blur-amount": 0.0 },
        { "image-type": 1, "width": 256, "height": 256, "fov-degrees": 90, "capture-enabled": true, "streaming-enabled": false,
          "pixels-as-float": true, "compress": false }
      ],
      "origin": { "xyz": "0.40 0.0 0.0", "rpy-deg": "0 0 0" }
    },
    {
      "id": "IMU1",
      "type": "imu",
      "enabled": true,
      "parent-link": "Frame",
      "accelerometer": { "velocity-random-walk": 2.353e-3, "tau": 800, "bias-stability": 3.53e-4, "turn-on-bias": "0 0 0" },
      "gyroscope": { "angle-random-walk": 8.72644e-5, "tau": 500, "bias-stability": 2.23014e-5, "turn-on-bias": "0 0 0" }
    },
    { "id": "GPS", "type": "gps", "enabled": false, "parent-link": "Frame" },
    { "id": "Barometer", "type": "barometer", "enabled": false, "parent-link": "Frame" },
    { "id": "Magnetometer", "type": "magnetometer", "enabled": false, "parent-link": "Frame" }
  ]
}
```

- [ ] **Step 4: Write the two M0 scene configs**

`configs/scene_autofly_m0.jsonc`:
```jsonc
{
  "id": "SceneAutoFlyM0",
  "actors": [
    {
      "type": "robot",
      "name": "Drone1",
      "origin": { "xyz": "0.0 0.0 -4.0", "rpy-deg": "0 0 0" },
      "robot-config": "robot_autofly_quadrotor.jsonc"
    }
  ],
  "clock": { "type": "steppable", "step-ns": 5000000, "real-time-update-rate": 3000000, "pause-on-start": true },
  "home-geo-point": { "latitude": 47.641468, "longitude": -122.140165, "altitude": 122.0 },
  "scene-type": "UnrealNative"
}
```

`configs/scene_autofly_m0_fast.jsonc`:
```jsonc
{
  "id": "SceneAutoFlyM0Fast",
  "actors": [
    {
      "type": "robot",
      "name": "Drone1",
      "origin": { "xyz": "0.0 0.0 -4.0", "rpy-deg": "0 0 0" },
      "robot-config": "robot_autofly_quadrotor.jsonc"
    }
  ],
  "clock": { "type": "steppable", "step-ns": 5000000, "real-time-update-rate": 1000000, "pause-on-start": true },
  "home-geo-point": { "latitude": 47.641468, "longitude": -122.140165, "altitude": 122.0 },
  "scene-type": "UnrealNative"
}
```

- [ ] **Step 5: Run the config test to verify it passes**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_pas_configs.py`
Expected: `2 passed`. A `jsonschema.exceptions.ValidationError` means a config typo: fix the config text to match Steps 3–4 exactly.

- [ ] **Step 6: Write `autofly_ue5/sim/smoke_m0.py` (part 1: lock-step driver)**

Create the file with this content (Step 7 appends the phases and the CLI):
```python
"""M0 gate smoke test against a running Project AirSim simulator (launched with scripts/launch_sim.py).

env -u PYTHONPATH .venv/bin/python -m autofly_ue5.sim.smoke_m0 --instance 0 --scene scene_autofly_m0.jsonc \
    --out runs/m0/smoke_inst0.json --fixtures-dir runs/m0/fixtures
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.drone import YawControlMode
from projectairsim.types import BoxAlignment, Pose

from autofly_ue5.frames import body_to_ned, heading_unit, quat_to_yaw, wrap_pi, yaw_to_quat
from autofly_ue5.gpu import gpu_memory_mib
from autofly_ue5.paths import CONFIGS_DIR
from autofly_ue5.sim.decode import decode_depth, decode_rgb
from autofly_ue5.sim.process import ports_for_instance, route_client_log

DT_NS = 200_000_000
DT_S = 0.2
STEP_NS = 5_000_000
# The server starts a command's duration when its request job runs, possibly one clock tick after Step started
# (service_manager.cpp:440); two ticks of margin keep the reply inside the step.
COMMAND_DURATION_S = (DT_NS - 2 * STEP_NS) / 1e9
ROBOT = "Drone1"
CAMERA = "FrontCamera"
CAMERA_X_OFFSET_M = 0.40
IMAGE_W = 256
HFOV_DEG = 90.0
CUBE_ASSET = "1M_Cube"
PILLAR_SCALE = [1.0, 3.0, 12.0]  # 1 m deep along the view, 3 m wide face, 12 m tall
# Opaque base UMaterial (BLEND_Opaque) cooked with /Game/Geometry; the server loads materials with a UMaterial class
# filter (WorldSimApi.cpp:1030-1032). M_Blue is a MaterialInstanceConstant: attempted once, predicted to fail.
BASE_MATERIAL = "/Game/Geometry/Materials/M_Orange"
INSTANCE_MATERIAL = "/Game/Geometry/Materials/M_Blue"
PILLAR_NAME = "AF_M0_Pillar"
TMP_NAME = "AF_M0_Tmp"
SAFE_Z = -40.0
PHASE_ORDER = ["lockstep", "velocity", "reset", "spawn", "collision", "teleport"]


class CommandTimeoutError(RuntimeError):
    """A velocity command's reply did not arrive after its step."""


def make_pose(x: float, y: float, z: float, yaw: float) -> Pose:
    w, qx, qy, qz = yaw_to_quat(yaw)
    return Pose({"frame_id": "DEFAULT_FRAME", "translation": {"x": x, "y": y, "z": z},
                 "rotation": {"w": w, "x": qx, "y": qy, "z": qz}})


def center_depth(depth: np.ndarray) -> float:
    return float(np.median(depth[124:132, 124:132]))


def contiguous_width(mask_row: np.ndarray, center: int = 128) -> int:
    """Length of the run of True values in mask_row that contains index center (0 when center is False)."""
    if not mask_row[center]:
        return 0
    left = center
    while left > 0 and mask_row[left - 1]:
        left -= 1
    right = center
    while right < len(mask_row) - 1 and mask_row[right + 1]:
        right += 1
    return right - left + 1


def flat_face_width_px(half_width_m: float, depth_m: float, image_width: int = IMAGE_W, hfov_deg: float = HFOV_DEG) -> float:
    """Pinhole image width of a flat face perpendicular to the optical axis and centred on it."""
    return image_width * half_width_m / (depth_m * math.tan(math.radians(hfov_deg) / 2.0))


def camera_pose_error(msg: dict | None, rec: dict) -> float:
    """Distance between the camera position stamped in an image message and kinematics + the rotated mount offset."""
    if msg is None:
        return math.inf
    off = body_to_ned(*rec["quat"], (CAMERA_X_OFFSET_M, 0.0, 0.0))
    expected = (rec["x"] + off[0], rec["y"] + off[1], rec["z"] + off[2])
    return math.dist((float(msg["pos_x"]), float(msg["pos_y"]), float(msg["pos_z"])), expected)


class FrameSlots:
    """Newest RGB and depth message per stream, filled from the client's receive thread."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._latest: dict[str, dict | None] = {"rgb": None, "depth": None}

    def callback(self, key: str):
        def _on_message(_topic, msg):
            with self._cond:
                self._latest[key] = msg
                self._cond.notify_all()
        return _on_message

    def wait(self, t_ns: int, timeout_s: float) -> dict[str, dict | None]:
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while True:
                msgs = dict(self._latest)
                if all(m is not None and m["time_stamp"] >= t_ns for m in msgs.values()):
                    return msgs
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return msgs
                self._cond.wait(remaining)


class Smoke:
    def __init__(self, client: ProjectAirSimClient, world: World, drone: Drone, frame_timeout_s: float,
                 first_frame_timeout_s: float, command_timeout_s: float) -> None:
        self.client, self.world, self.drone = client, world, drone
        self.frame_timeout_s = frame_timeout_s
        self.first_frame_timeout_s = first_frame_timeout_s
        self.command_timeout_s = command_timeout_s
        self.frames = FrameSlots()
        self.collision_msgs: list[dict] = []
        self.loop = asyncio.new_event_loop()
        self.t_ns = int(world.get_sim_time())
        self.records = 0
        self.origin: dict | None = None
        self.pillar_name: str | None = None
        self.pillar_center: tuple[float, float, float] | None = None
        self.pillar_yaw = 0.0
        cams = drone.sensors[CAMERA]
        client.subscribe(cams["scene_camera"], self.frames.callback("rgb"))
        client.subscribe(cams["depth_planar_camera"], self.frames.callback("depth"))
        client.subscribe(drone.robot_info["collision_info"], lambda _t, m: self.collision_msgs.append(m))

    def record(self, v_forward: float = 0.0, yaw_rate: float = 0.0, v_up: float = 0.0) -> dict:
        target = self.t_ns + DT_NS
        wall_start = time.monotonic()
        task = self.loop.run_until_complete(self.drone.move_by_velocity_body_frame_async(
            v_forward, 0.0, -v_up, duration=COMMAND_DURATION_S,
            yaw_control_mode=YawControlMode.MaxDegreeOfFreedom, yaw_is_rate=True, yaw=yaw_rate))
        self.loop.run_until_complete(asyncio.sleep(0.002))  # let the server job read the command start time
        result = self.world.step(DT_NS)
        try:
            self.loop.run_until_complete(asyncio.wait_for(task, timeout=self.command_timeout_s))
        except asyncio.TimeoutError as err:
            raise CommandTimeoutError(
                f"move command reply missing {self.command_timeout_s} s after the step to {target} ns") from err
        timeout = self.first_frame_timeout_s if self.records == 0 else self.frame_timeout_s
        msgs = self.frames.wait(target, timeout)
        kin = self.drone.get_ground_truth_kinematics()
        for _ in range(4):
            if int(kin["time_stamp"]) == target:
                break
            time.sleep(0.01)
            kin = self.drone.get_ground_truth_kinematics()
        wall_s = time.monotonic() - wall_start
        self.t_ns = int(result["sim_time_ns"])
        self.records += 1
        pos, ori = kin["pose"]["position"], kin["pose"]["orientation"]
        rec = {
            "target_ns": target,
            "sim_time_ns": int(result["sim_time_ns"]),
            "rgb_ts": msgs["rgb"]["time_stamp"] if msgs["rgb"] else None,
            "depth_ts": msgs["depth"]["time_stamp"] if msgs["depth"] else None,
            "kin_ts": int(kin["time_stamp"]),
            "x": float(pos["x"]), "y": float(pos["y"]), "z": float(pos["z"]),
            "yaw": quat_to_yaw(ori["w"], ori["x"], ori["y"], ori["z"]),
            "quat": [float(ori[k]) for k in ("w", "x", "y", "z")],
            "vel": [float(kin["twist"]["linear"][k]) for k in ("x", "y", "z")],
            "events": [e for e in result["robots"][ROBOT]["events"] if e.get("type") == "collision"],
            "rgb_msg": msgs["rgb"],
            "depth_msg": msgs["depth"],
            "wall_s": wall_s,
        }
        rec["camera_pose_error_m"] = camera_pose_error(msgs["rgb"], rec)
        if self.origin is None:
            self.origin = {"x": rec["x"], "y": rec["y"], "z": rec["z"], "yaw": rec["yaw"]}
        return rec
```

- [ ] **Step 7: Append the phases and CLI to `autofly_ue5/sim/smoke_m0.py` (part 2)**

Append:
```python


def frames_at_target(rec: dict) -> bool:
    return rec["rgb_ts"] == rec["target_ns"] and rec["depth_ts"] == rec["target_ns"]


def probe_get_images(smoke: Smoke) -> dict:
    start = time.monotonic()
    try:
        images = smoke.drone.get_images(CAMERA, [0, 1])
    except RuntimeError as err:
        return {"ok": False, "error": str(err), "elapsed_s": round(time.monotonic() - start, 3)}
    stamps = {str(k): int(v["time_stamp"]) for k, v in images.items()}
    ok = set(images) == {0, 1} and all(s >= smoke.t_ns for s in stamps.values())
    return {"ok": ok, "time_stamps": stamps, "sim_time_ns": smoke.t_ns, "elapsed_s": round(time.monotonic() - start, 3)}


def phase_lockstep(smoke: Smoke, args: argparse.Namespace) -> dict:
    warm = [smoke.record() for _ in range(args.warmup_steps)]  # first-frame wait and shader warm-up stay untimed
    start_late_s = None
    if args.start_at is not None:  # lets two instances time overlapping windows
        start_late_s = time.time() - args.start_at
        if start_late_s < 0:
            time.sleep(-start_late_s)
    n = args.steps
    used = total = vram_sample_unix = None
    recs = []
    timed_start_unix = time.time()
    wall_start = time.monotonic()
    for i in range(n):
        recs.append(smoke.record())
        if i == n // 2:  # mid-window sample, while this camera (and any concurrent instance) is capturing
            used, total = gpu_memory_mib()
            vram_sample_unix = time.time()
    wall = time.monotonic() - wall_start
    timed_end_unix = time.time()
    checked = warm + recs
    m = len(checked)
    last = recs[-1]
    rgb = decode_rgb(last["rgb_msg"]) if last["rgb_msg"] else None
    depth = decode_depth(last["depth_msg"]) if last["depth_msg"] else None
    z0 = checked[0]["z"]
    out = {
        "steps": n,
        "warmup_steps": len(warm),
        "sim_time_exact": sum(r["sim_time_ns"] == r["target_ns"] for r in checked),
        "frames_eq_target": sum(frames_at_target(r) for r in checked),
        "kin_ts_eq_target": sum(r["kin_ts"] == r["target_ns"] for r in checked),
        "max_camera_pose_error_m": max(r["camera_pose_error_m"] for r in checked),
        "z_drift_m": max(abs(r["z"] - z0) for r in checked),
        "steps_per_s": n / wall,
        "timed_start_unix": timed_start_unix,
        "timed_end_unix": timed_end_unix,
        "start_late_s": start_late_s,
        "median_record_wall_s": float(np.median([r["wall_s"] for r in recs])),
        "rgb_shape": list(rgb.shape) if rgb is not None else None,
        "depth_shape": list(depth.shape) if depth is not None else None,
        "rgb_mean": float(rgb.mean()) if rgb is not None else None,
        "depth_min_m": float(np.min(depth)) if depth is not None else None,
        "vram_used_mib": used,
        "vram_total_mib": total,
        "vram_sample_unix": vram_sample_unix,
        "get_images_probe": probe_get_images(smoke),
    }
    out["pass"] = (
        out["sim_time_exact"] == m and out["frames_eq_target"] == m and out["kin_ts_eq_target"] == m
        and out["max_camera_pose_error_m"] < 0.1
        and out["z_drift_m"] < 0.5 and out["rgb_shape"] == [256, 256, 3] and out["depth_shape"] == [256, 256]
        and out["rgb_mean"] is not None and out["rgb_mean"] > 10.0
        and out["depth_min_m"] is not None and out["depth_min_m"] >= 0.5
    )
    return out


def phase_velocity(smoke: Smoke, args: argparse.Namespace) -> dict:
    start = smoke.record()
    ahead = center_depth(decode_depth(start["depth_msg"]))
    if not ahead > 15.0:
        return {"pass": False, "reason": f"BlocksMap geometry {ahead:.2f} m ahead of the start pose; not flying"}
    north, east = heading_unit(start["yaw"])
    fwd = [smoke.record(v_forward=2.0) for _ in range(10)]
    dx, dy = fwd[-1]["x"] - start["x"], fwd[-1]["y"] - start["y"]
    along, lateral = dx * north + dy * east, -dx * east + dy * north
    hold = [smoke.record() for _ in range(5)]
    up = [smoke.record(v_up=1.0) for _ in range(5)]
    climb = -(up[-1]["z"] - hold[-1]["z"])
    hold2 = [smoke.record() for _ in range(5)]
    yaws = [hold2[-1]["yaw"]] + [smoke.record(yaw_rate=0.5)["yaw"] for _ in range(10)]
    dyaw = sum(wrap_pi(b - a) for a, b in zip(yaws, yaws[1:]))
    for _ in range(5):
        smoke.record()
    out = {
        "forward_along_m": along, "forward_lateral_m": lateral, "climb_m": climb,
        "yaw_change_rad": dyaw, "yaw_rate_measured_rad_s": dyaw / 2.0, "yaw_rate_ratio": (dyaw / 2.0) / 0.5,
    }
    out["pass"] = along >= 2.0 and abs(lateral) < 1.0 and climb >= 0.5 and abs(dyaw) >= 0.3
    return out


def phase_reset(smoke: Smoke, args: argparse.Namespace) -> dict:
    o = smoke.origin
    ok = bool(smoke.drone.set_pose(make_pose(o["x"], o["y"], o["z"], 0.0), reset_kinematics=True))
    recs = [smoke.record() for _ in range(5)]
    last = recs[-1]
    pos_err = math.dist((last["x"], last["y"], last["z"]), (o["x"], o["y"], o["z"]))
    speed = math.hypot(*last["vel"])
    yaw_err = abs(wrap_pi(last["yaw"]))
    out = {"set_pose_ok": ok, "position_error_m": pos_err, "speed_m_s": speed, "yaw_error_rad": yaw_err,
           "camera_pose_vs_state_m": last["camera_pose_error_m"]}
    out["pass"] = ok and pos_err < 0.5 and speed < 0.5 and yaw_err < 0.1 and last["camera_pose_error_m"] < 0.1
    return out


def save_fixture(msg: dict, stem: str, directory: Path, extra: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    data = bytes(msg["data"])
    (directory / f"{stem}.bin").write_bytes(data)
    meta = {k: v for k, v in msg.items() if k != "data"}
    meta.update(extra)
    meta["data_len"] = len(data)
    (directory / f"{stem}.json").write_text(json.dumps(meta, indent=2, default=str))


def phase_spawn(smoke: Smoke, args: argparse.Namespace) -> dict:
    before = smoke.record()
    assets = smoke.world.list_assets(f"^{CUBE_ASSET}$")
    north, east = heading_unit(before["yaw"])
    px, py, pz = before["x"] + 8.0 * north, before["y"] + 8.0 * east, before["z"]
    name = smoke.world.spawn_object(PILLAR_NAME, CUBE_ASSET, make_pose(px, py, pz, before["yaw"]), PILLAR_SCALE, False)
    smoke.pillar_name, smoke.pillar_center, smoke.pillar_yaw = name, (px, py, pz), before["yaw"]
    bbox = smoke.world.get_3d_bounding_box(name, BoxAlignment.OBJECT_ORIENTED)
    # Depth, face width and the fixtures come from the cube's own opaque material, before any material change.
    after = smoke.record()
    rgb_before, rgb_after = decode_rgb(before["rgb_msg"]), decode_rgb(after["rgb_msg"])
    depth_after = decode_depth(after["depth_msg"])
    along = (px - after["x"]) * north + (py - after["y"]) * east
    expected = along - PILLAR_SCALE[0] / 2.0 - CAMERA_X_OFFSET_M
    measured = center_depth(depth_after)
    # Field of view: the 3 m face is a flat plane facing the camera, so its planar depth is constant across row 128.
    row = depth_after[128]
    face_width = contiguous_width(np.isfinite(row) & (np.abs(row - measured) < 0.2))
    expected_face_width = flat_face_width_px(PILLAR_SCALE[1] / 2.0, measured) if 0.0 < measured < math.inf else None
    hfov_consistent = expected_face_width is not None and abs(face_width - expected_face_width) <= 3.0
    material_ok = bool(smoke.world.set_object_material(name, BASE_MATERIAL))
    after_material = smoke.record()
    rgb_material = decode_rgb(after_material["rgb_msg"])
    patch_diff = float(np.mean(np.abs(rgb_material[112:144, 112:144].astype(np.int16) - rgb_after[112:144, 112:144].astype(np.int16))))
    material_center_rgb = [float(v) for v in rgb_material[120:136, 120:136].reshape(-1, 3).mean(axis=0)]
    center_err = math.dist((bbox["center"]["x"], bbox["center"]["y"], bbox["center"]["z"]), (px, py, pz)) if bbox else None
    size = [bbox["size"][k] for k in ("x", "y", "z")] if bbox else None
    tx, ty = px - 10.0 * east, py + 10.0 * north
    tmp = smoke.world.spawn_object(TMP_NAME, CUBE_ASSET, make_pose(tx, ty, pz, 0.0), [1.0, 1.0, 1.0], False)
    listed = smoke.world.list_objects(f"{tmp}.*")
    moved_ok = bool(smoke.world.set_object_pose(tmp, make_pose(tx + 2.0, ty, pz - 1.0, 0.0), True))
    scaled_ok = bool(smoke.world.set_object_scale(tmp, [2.0, 2.0, 2.0]))
    tmp_translation = smoke.world.get_object_pose(tmp)["translation"]
    tmp_move_err = math.dist((tmp_translation["x"], tmp_translation["y"], tmp_translation["z"]), (tx + 2.0, ty, pz - 1.0))
    tmp_bbox = smoke.world.get_3d_bounding_box(tmp, BoxAlignment.WORLD_AXIS)
    tmp_size = [tmp_bbox["size"][k] for k in ("x", "y", "z")] if tmp_bbox else None
    # Spec §4.1's material-instance route, informational: the UMaterial class filter predicts False.
    material_instance_ok = bool(smoke.world.set_object_material(tmp, INSTANCE_MATERIAL))
    destroyed = bool(smoke.world.destroy_object(tmp))
    listed_after = smoke.world.list_objects(f"{tmp}.*")
    if args.fixtures_dir is not None:
        rgb_data = bytes(after["rgb_msg"]["data"])
        center_index = (128 * 256 + 128) * 3
        save_fixture(after["rgb_msg"], "rgb_msg", args.fixtures_dir,
                     {"center_pixel_bgr": list(rgb_data[center_index:center_index + 3])})
        depth_data = bytes(after["depth_msg"]["data"])
        inf_count = sum(1 for (v,) in struct.iter_unpack("<e", depth_data) if math.isinf(v))
        save_fixture(after["depth_msg"], "depth_msg", args.fixtures_dir,
                     {"inf_count": inf_count, "expected_center_depth_m": expected})
        Image.fromarray(rgb_after).save(args.fixtures_dir / "rgb_after_spawn.png")
        Image.fromarray(rgb_material).save(args.fixtures_dir / "rgb_after_material.png")
    out = {
        "assets_found": assets, "spawned_name": name, "bbox_center_error_m": center_err, "bbox_size_m": size,
        "rgb_mean_abs_diff": float(np.mean(np.abs(rgb_after.astype(np.int16) - rgb_before.astype(np.int16)))),
        "center_depth_m": measured, "expected_center_depth_m": expected, "depth_error_m": abs(measured - expected),
        "center_face_width_px": face_width, "expected_face_width_px": expected_face_width, "hfov_consistent": hfov_consistent,
        "material": BASE_MATERIAL, "material_ok": material_ok, "material_patch_mean_abs_diff": patch_diff,
        "material_center_rgb": material_center_rgb,
        "material_instance": INSTANCE_MATERIAL, "material_instance_ok": material_instance_ok,
        "tmp_name": tmp, "tmp_listed": listed, "tmp_set_pose_ok": moved_ok, "tmp_move_error_m": tmp_move_err,
        "tmp_set_scale_ok": scaled_ok, "tmp_bbox_size_after_scale_m": tmp_size,
        "tmp_destroyed": destroyed, "tmp_listed_after": listed_after,
    }
    out["pass"] = (
        len(assets) > 0 and name.startswith(PILLAR_NAME) and center_err is not None and center_err < 0.1
        and size is not None and all(abs(a - b) < 0.1 for a, b in zip(size, PILLAR_SCALE))
        and out["rgb_mean_abs_diff"] > 5.0 and out["depth_error_m"] < 0.3 and hfov_consistent
        and material_ok and patch_diff > 10.0
        and len(listed) > 0 and moved_ok and tmp_move_err < 0.05
        and scaled_ok and tmp_size is not None and all(abs(v - 2.0) < 0.1 for v in tmp_size)
        and destroyed and len(listed_after) == 0
    )
    return out


def phase_collision(smoke: Smoke, args: argparse.Namespace) -> dict:
    if smoke.pillar_name is None:
        return {"pass": False, "reason": "spawn phase did not run"}
    smoke.collision_msgs.clear()
    events: list[dict] = []
    hit_step = None
    for i in range(30):
        rec = smoke.record(v_forward=2.0)
        events.extend(rec["events"])
        if rec["events"]:
            hit_step = i
            break
    tail = smoke.record()
    events.extend(tail["events"])
    names = sorted({e["object_name"] for e in events})
    topic_names = sorted({m["object_name"] for m in smoke.collision_msgs})
    out = {"hit_step": hit_step, "step_event_names": names, "topic_event_names": topic_names,
           "first_event": events[0] if events else None, "drone_at_end": [tail["x"], tail["y"], tail["z"]]}
    out["pass"] = hit_step is not None and any(smoke.pillar_name in n for n in names)
    return out


def teleport_and_record(smoke: Smoke, x: float, y: float, z: float, yaw: float, records: int = 1) -> dict:
    ok = bool(smoke.drone.set_pose(make_pose(x, y, z, yaw), reset_kinematics=True))
    recs = [smoke.record() for _ in range(records)]
    last = dict(recs[-1])
    last["set_pose_ok"] = ok
    last["all_events"] = [e for r in recs for e in r["events"]]
    return last


def phase_teleport(smoke: Smoke, args: argparse.Namespace) -> dict:
    """Research gotcha 6 live: set_pose sweeps the Unreal actor, so a teleport through an obstacle can leave the camera
    behind while kinematics jump. Gated: the backend's reset path (straight up to a safe altitude, across, down) must
    resynchronise camera and kinematics after the collision phase left the drone touching the pillar.
    Informational: camera/kinematics mismatch and events after a teleport straight through the pillar."""
    if smoke.pillar_center is None:
        return {"pass": False, "reason": "spawn phase did not run"}
    px, py, pz = smoke.pillar_center
    yaw = smoke.pillar_yaw
    north, east = heading_unit(yaw)
    now = smoke.record()
    up = teleport_and_record(smoke, now["x"], now["y"], SAFE_Z, yaw)
    across = teleport_and_record(smoke, px - 5.0 * north, py - 5.0 * east, SAFE_Z, yaw)
    front = teleport_and_record(smoke, px - 5.0 * north, py - 5.0 * east, pz, yaw, records=2)
    through = teleport_and_record(smoke, px + 5.0 * north, py + 5.0 * east, pz, yaw, records=2)
    msg = through["rgb_msg"]
    out = {
        "recovery_camera_errors_m": [up["camera_pose_error_m"], across["camera_pose_error_m"], front["camera_pose_error_m"]],
        "recovery_event_names": sorted({e["object_name"] for r in (up, across, front) for e in r["all_events"]}),
        "through_camera_error_m": through["camera_pose_error_m"],
        "through_event_names": sorted({e["object_name"] for e in through["all_events"]}),
        "through_camera_position": [msg["pos_x"], msg["pos_y"], msg["pos_z"]] if msg else None,
        "through_kinematics_position": [through["x"], through["y"], through["z"]],
        "set_pose_ok": all(r["set_pose_ok"] for r in (up, across, front, through)),
    }
    out["pass"] = out["set_pose_ok"] and all(e < 0.1 for e in out["recovery_camera_errors_m"])
    return out


PHASES = {"lockstep": phase_lockstep, "velocity": phase_velocity, "reset": phase_reset,
          "spawn": phase_spawn, "collision": phase_collision, "teleport": phase_teleport}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--scene", default="scene_autofly_m0.jsonc")
    parser.add_argument("--phases", default=",".join(PHASE_ORDER))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--start-at", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixtures-dir", type=Path, default=None)
    parser.add_argument("--frame-timeout", type=float, default=5.0)
    parser.add_argument("--first-frame-timeout", type=float, default=120.0)
    parser.add_argument("--command-timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    phases = args.phases.split(",")
    unknown = set(phases) - set(PHASE_ORDER)
    if unknown:
        parser.error(f"unknown phases {sorted(unknown)}")
    ports = ports_for_instance(args.instance)
    report: dict = {"instance": args.instance, "ports": [ports.topics, ports.services], "scene": args.scene, "phases": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    route_client_log(args.out.with_suffix(".client.log"))
    client = ProjectAirSimClient(port_topics=ports.topics, port_services=ports.services)
    try:
        client.connect()
        load_start = time.monotonic()
        world = World(client, args.scene, delay_after_load_sec=0, sim_config_path=str(CONFIGS_DIR))
        report["load_scene_s"] = time.monotonic() - load_start
        report["clock_type"] = world.get_sim_clock_type()
        report["paused_on_load"] = world.is_paused()
        drone = Drone(client, world, ROBOT)
        report["camera_topics"] = sorted(drone.sensors.get(CAMERA, {}))
        drone.enable_api_control()
        drone.arm()
        smoke = Smoke(client, world, drone, args.frame_timeout, args.first_frame_timeout, args.command_timeout)
        for name in PHASE_ORDER:
            if name in phases:
                report["phases"][name] = PHASES[name](smoke, args)
                args.out.write_text(json.dumps(report, indent=2, default=str))
    except Exception as err:
        report["error"] = f"{type(err).__name__}: {err}"
    finally:
        client.disconnect()
    report["pass"] = (
        "error" not in report and report.get("clock_type") == "steppable"
        and set(report["phases"]) == set(phases) and all(p["pass"] for p in report["phases"].values())
    )
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"pass": report["pass"], "error": report.get("error"),
                      "phases": {k: v["pass"] for k, v in report["phases"].items()}}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8: Check the module imports and the offline suite still passes**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -c "import autofly_ue5.sim.smoke_m0 as s; print(s.PHASE_ORDER, s.COMMAND_DURATION_S)" && env -u PYTHONPATH .venv/bin/python -m pytest`
Expected: `['lockstep', 'velocity', 'reset', 'spawn', 'collision', 'teleport'] 0.19` and all tests passed (`46 passed`).

- [ ] **Step 9: Launch instance 0 on BlocksMap as a job (total wait budget 25 minutes)**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && bash scripts/run_job.sh start launch_inst0 -- env -u PYTHONPATH .venv/bin/python scripts/launch_sim.py --mode editor --map /Game/BlocksMap --instance 0 --timeout 900 && bash scripts/run_job.sh wait launch_inst0 540; echo "wait=$?"` (repeat `wait` while it returns 124, within the budget).
Expected: the launch JSON line in the job log, `job launch_inst0 finished: exit=0`, `wait=0` (as in Task 5 Step 6). On failure or an exhausted budget: `bash scripts/run_job.sh stop launch_inst0`, `env -u PYTHONPATH .venv/bin/python scripts/stop_sim.py --instance 0`, then stop and report.

- [ ] **Step 10: Run the full smoke test as a job (total wait budget 20 minutes)**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && bash scripts/run_job.sh start smoke_inst0 -- env -u PYTHONPATH .venv/bin/python -m autofly_ue5.sim.smoke_m0 --instance 0 --scene scene_autofly_m0.jsonc --warmup-steps 5 --out runs/m0/smoke_inst0.json --fixtures-dir runs/m0/fixtures && bash scripts/run_job.sh wait smoke_inst0 540; echo "wait=$?"` (repeat `wait` while it returns 124, within the budget).
Expected: last log line `{"pass": true, "error": null, "phases": {"lockstep": true, "velocity": true, "reset": true, "spawn": true, "collision": true, "teleport": true}}`, `job smoke_inst0 finished: exit=0`, `wait=0`; the client log is `runs/m0/smoke_inst0.client.log`. This run verifies live the UNCONFIRMED items: every step (warm-up included) lands exactly on `T` and delivers RGB and depth with `time_stamp == T` (subscription pattern), kinematics timestamps equal `T`, every velocity command reply arrives after its step (no `CommandTimeoutError`), the camera position stamped in each image (`pos_x/pos_y/pos_z`) equals kinematics plus the rotated 0.40 m mount offset within 0.1 m, velocity commands move the drone from an armed in-air start without takeoff, `set_pose` resets pose, velocity and the camera, `1M_Cube` is centred and 1 m (bounding box), `set_object_pose` and `set_object_scale` move and scale a spawned object, planar depth of the spawned face matches geometry within 0.3 m (measured on the cube's own opaque material), the face's width in depth row 128 matches a 90° horizontal field of view within 3 px (`hfov_consistent`; spec §4.1 camera row), `set_object_material` with the opaque base `UMaterial` `M_Orange` returns true and changes the centre patch by more than 10 grey levels, a flight into a spawned object yields a `collision` event naming it, the up-across-down teleport path resynchronises the camera after that collision, and no drone part is closer than 0.5 m in depth. Also read `jq '.phases.velocity.yaw_rate_ratio, .phases.lockstep.get_images_probe, .phases.teleport, (.phases.spawn | {center_face_width_px, expected_face_width_px, material_center_rgb, material_instance_ok})' runs/m0/smoke_inst0.json`: `material_instance_ok` records whether the spec's material-instance route (`M_Blue`) works (predicted `false`; report either way); a ratio near 1.0 confirms rad/s, a ratio near 0.017 means deg/s (report either way); `teleport.through_camera_error_m` shows whether a teleport straight through an obstacle leaves the camera behind (report the number; the backend never teleports horizontally at flight altitude). If any phase is `false` or `error` is set: stop and report the whole `runs/m0/smoke_inst0.json`, `tail -80 runs/sim/inst0/sim.log` and `tail -40 runs/m0/smoke_inst0.client.log` (after stopping instance 0, per the failure rule). This is exactly the spec §4 M0 fallback evidence; do not change the backend design in response.

- [ ] **Step 11: Look at the captured frame**

Open `runs/m0/fixtures/rgb_after_spawn.png` and `runs/m0/fixtures/rgb_after_material.png` (Read tool). Expected: correctly coloured BlocksMap views (sky above, floor below) with the spawned pillar in the centre and no propeller or frame visible; in the second image the pillar is orange, and `phases.spawn.material_center_rgb` has R > G > B (Task 16's runtime-colour check relies on `M_Orange` rendering orange). If colours look swapped (sky orange, pillar blue), stop instance 0 (`scripts/stop_sim.py --instance 0`) and report: the BGR assumption is contradicted. If the pillar is not orange or R > G > B does not hold, stop instance 0 and report it with the image and `material_center_rgb` (the Task 16 colour check must then be revisited with the user).

- [ ] **Step 12: Stop instance 0**

Run (Bash tool `timeout` 600000 ms): `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python scripts/stop_sim.py --instance 0 && sleep 30; pgrep -a -f "$(pwd)/engine/Engine/Binaries/Linux/" || echo none`
Expected: `terminated`, then `none` (survivors after another 60 s are reported, not killed).

- [ ] **Step 13: Commit configs, smoke module, fixtures and report**

```bash
mkdir -p /home/nvidiasims/research_uav/autofly_ue5/tests/fixtures/pas && cp /home/nvidiasims/research_uav/autofly_ue5/runs/m0/fixtures/{rgb_msg.json,rgb_msg.bin,depth_msg.json,depth_msg.bin} /home/nvidiasims/research_uav/autofly_ue5/tests/fixtures/pas/ && cp /home/nvidiasims/research_uav/autofly_ue5/runs/m0/smoke_inst0.json /home/nvidiasims/research_uav/autofly_ue5/docs/gates/m0_smoke_inst0.json
git -C /home/nvidiasims/research_uav/autofly_ue5 add configs/robot_autofly_quadrotor.jsonc configs/scene_autofly_m0.jsonc configs/scene_autofly_m0_fast.jsonc autofly_ue5/sim/smoke_m0.py tests/test_pas_configs.py tests/fixtures/pas docs/gates/m0_smoke_inst0.json && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Add FrontCamera robot and steppable scene configs and the M0 smoke test; capture real RGB/depth fixtures" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: M0 gate — throughput, VRAM, second instance, gate report

**Files:**
- Create: `autofly_ue5/validate/m0_gate.py`
- Create (captured): `docs/gates/m0_gate.json`
- Test: `tests/test_m0_gate.py` + live runs

**Interfaces:**
- Consumes: `autofly_ue5.gpu.gpu_memory_mib`; `autofly_ue5.paths.{ROOT, RUNS_DIR}`; `autofly_ue5.validate.engine_check.{xid_count, boot_id, count_device_lost, epic_config_usage, out_of_root_state}` (Task 2); reports `runs/m0/engine_check.json` (Task 2), `downloads/plugin_manifest_report.json` (Task 3), `runs/build/build_result.json` and `runs/build/python_probe.json` (Task 4), `runs/m0/smoke_inst0.json` (Task 7); CLIs `scripts/launch_sim.py`, `scripts/stop_sim.py`, `python -m autofly_ue5.sim.smoke_m0`, `scripts/run_job.sh`.
- Produces:
  - `autofly_ue5.validate.m0_gate.VRAM_KEYS = ("baseline", "idle_instance")`
  - `record_vram(path: Path, key: str, used_mib: int) -> dict`
  - `probe_ok(probe: dict) -> bool` (the Task 4 editor-Python facts)
  - `fault_record(engine: dict, xid_now: int, device_lost: dict[str, int], epic_after: dict, boot_id_now: str, out_of_root_after: dict[str, bool]) -> dict` with keys `xid_since` (Task 2 `started_at`), `xid_baseline` (Task 2 `checks.nvidia_xid.after`), `xid_now` (counted from `xid_since` across reboots), `xid_delta`, `boot_id_m0_start`, `boot_id_now`, `boot_changed`, `device_lost`, `epic_config_before`, `epic_config_after`, `zen_default_data_created`, `out_of_root_before`, `out_of_root_after`, `out_of_root_created` (sorted keys that exist now but did not at the M0 start)
  - `two_instance_concurrency(smoke_a: dict, smoke_b: dict) -> dict` with `overlap_s` (length of the intersection of the two timed lockstep windows, `None` when a timing key is missing) and `vram_samples_inside_overlap` (both VRAM samples were taken inside that intersection)
  - `assemble_m0_gate(engine: dict, plugin: dict, build: dict, probe: dict, smoke: dict, smoke_fast: dict, smoke_inst1: dict, smoke_concurrent: dict, vram: dict, faults: dict) -> dict` with keys `engine_check`, `plugin_manifest`, `blocks_editor_build`, `editor_python_probe`, `smoke_single_instance`, `second_instance` (both concurrent runs passed and their VRAM samples lie inside the overlap of their timed windows), `two_instance_concurrency`, `gpu_faults`, `steps_per_s_rtur_3ms`, `steps_per_s_rtur_1ms`, `rtur_1ms_lockstep_pass`, `steps_per_s_two_instances`, `vram_mib`, `vram_one_instance_capturing_mib`, `vram_two_instances_capturing_mib`, `vram_per_instance_mib`, `yaw_rate_ratio`, `get_images_probe`, `teleport_through_camera_error_m`, `hfov_face_width_px` (`[measured, expected]`), `set_object_material_instance_ok`, `faults`, `pass`. VRAM per instance is measured while cameras capture (the `lockstep` phase samples), never from an idle instance.
  - CLI: `python -m autofly_ue5.validate.m0_gate vram <baseline|idle_instance>`; `python -m autofly_ue5.validate.m0_gate faults` → `runs/m0/faults.json`, exit 0 iff no new Xid since the M0 start, the same boot, no `VK_ERROR_DEVICE_LOST`, no default Zen cache and no out-of-ROOT path created; `python -m autofly_ue5.validate.m0_gate assemble` → `runs/m0/m0_gate.json`, exit 0 iff pass.

- [ ] **Step 1: Write the failing test**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_m0_gate.py`:
```python
import json

import pytest

from autofly_ue5.validate.m0_gate import (
    assemble_m0_gate,
    fault_record,
    probe_ok,
    record_vram,
    two_instance_concurrency,
)

PROBE = {"engine_version": "5.7.4-51494982+++UE5+Release-5.7", "level_editor_subsystem": True,
         "editor_actor_subsystem": True, "editor_asset_subsystem": True,
         "game_mode_class": "/Script/ProjectAirSim.ProjectAirSimGameMode", "sunsky_class": "/SunPosition/SunSky.SunSky_C",
         "cylinder_mesh": True, "cube_mesh": True, "world_grid_material": True,
         "basic_shape_material_vector_params": ["Color"],
         "mic_parent": "/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial", "mic_set_vector_return_value": False,
         "mic_color_readback": [0.1, 0.2, 0.3, 1.0], "mic_color_readback_ok": True}
OUT_OF_ROOT_CLEAN = {"unreal_trace_store": False, "unreal_trace_server_pid": False,
                     "documents_unreal_projects": False, "documents_unreal_engine_logs": False}
ENGINE = {"pass": True, "started_at": "2026-09-15 12:00:00", "boot_id": "boot-a",
          "checks": {"nvidia_xid": {"before": 0, "after": 0}},
          "epic_config_before": {"kib": 1000, "zen_default_data_exists": False}, "out_of_root_before": OUT_OF_ROOT_CLEAN}
EPIC_AFTER = {"kib": 5000, "zen_default_data_exists": False}
FAULTS_OK = fault_record(ENGINE, 0, {"runs/sim/inst0/sim.log": 0, "runs/sim/inst1/sim.log": 0}, EPIC_AFTER, "boot-a",
                         OUT_OF_ROOT_CLEAN)
VRAM = {"baseline": 600, "idle_instance": 2100}


def _smoke(passed: bool, steps_per_s: float = 5.0, vram_used_mib: int = 3300,
           window: tuple[float, float, float] = (1000.0, 1020.0, 1010.0)) -> dict:
    start, end, sample = window
    return {"pass": passed,
            "phases": {"lockstep": {"pass": passed, "steps_per_s": steps_per_s, "vram_used_mib": vram_used_mib,
                                    "timed_start_unix": start, "timed_end_unix": end, "vram_sample_unix": sample,
                                    "get_images_probe": {"ok": True}},
                       "velocity": {"pass": passed, "yaw_rate_ratio": 0.97},
                       "spawn": {"pass": passed, "center_face_width_px": 54, "expected_face_width_px": 54.1,
                                 "material_instance_ok": False},
                       "teleport": {"pass": passed, "through_camera_error_m": 4.6}}}


def _gate(**overrides) -> dict:
    parts = dict(engine=ENGINE, plugin={"pass": True}, build={"pass": True}, probe=PROBE, smoke=_smoke(True),
                 smoke_fast=_smoke(False, 7.5, 3300), smoke_inst1=_smoke(True, 4.0, 5990, (1002.0, 1025.0, 1012.0)),
                 smoke_concurrent=_smoke(True, 4.2, 6000, (1001.0, 1024.0, 1011.0)), vram=VRAM, faults=FAULTS_OK)
    parts.update(overrides)
    return assemble_m0_gate(**parts)


def test_record_vram_creates_and_updates(tmp_path):
    path = tmp_path / "vram.json"
    record_vram(path, "baseline", 600)
    data = record_vram(path, "idle_instance", 2100)
    assert data == {"baseline": 600, "idle_instance": 2100}
    assert json.loads(path.read_text()) == data
    with pytest.raises(ValueError):
        record_vram(path, "one_instance", 1)


def test_gate_passes_and_measures_vram_while_capturing():
    gate = _gate()
    assert gate["pass"] is True
    assert gate["rtur_1ms_lockstep_pass"] is False  # informational only
    assert gate["steps_per_s_rtur_1ms"] == 7.5 and gate["steps_per_s_two_instances"] == [4.2, 4.0]
    assert gate["vram_one_instance_capturing_mib"] == 2700 and gate["vram_two_instances_capturing_mib"] == 5400
    assert gate["vram_per_instance_mib"] == 2700
    assert gate["yaw_rate_ratio"] == 0.97 and gate["teleport_through_camera_error_m"] == 4.6
    assert gate["hfov_face_width_px"] == [54, 54.1] and gate["set_object_material_instance_ok"] is False
    assert gate["two_instance_concurrency"] == {"overlap_s": 22.0, "vram_samples_inside_overlap": True}


def test_gate_fails_without_second_instance_or_vram():
    assert _gate(smoke_inst1=_smoke(False))["pass"] is False
    gate = _gate(vram={"idle_instance": 2100})
    assert gate["pass"] is False and gate["vram_per_instance_mib"] is None


def test_two_instance_runs_must_overlap_around_both_vram_samples():
    early = _smoke(True, 4.2, 6000, (1000.0, 1010.0, 1005.0))
    late = _smoke(True, 4.0, 5990, (1030.0, 1040.0, 1035.0))
    assert two_instance_concurrency(early, late) == {"overlap_s": -20.0, "vram_samples_inside_overlap": False}
    gate = _gate(smoke_concurrent=early, smoke_inst1=late)
    assert gate["second_instance"] is False and gate["pass"] is False
    partial = _smoke(True, 4.0, 5990, (1008.0, 1030.0, 1025.0))  # windows overlap on [1008, 1010] only
    assert two_instance_concurrency(early, partial)["vram_samples_inside_overlap"] is False
    assert two_instance_concurrency({}, late) == {"overlap_s": None, "vram_samples_inside_overlap": False}


def test_gate_requires_build_plugin_and_probe_evidence():
    assert _gate(build={})["pass"] is False
    assert _gate(plugin={"pass": False})["pass"] is False
    assert probe_ok(PROBE) is True
    assert probe_ok(dict(PROBE, basic_shape_material_vector_params=["Tint"])) is False
    assert probe_ok(dict(PROBE, mic_color_readback_ok=False)) is False
    assert _gate(probe={})["pass"] is False


def test_gate_fails_on_new_xid_reboot_device_lost_default_zen_cache_or_out_of_root_writes():
    new_xid = fault_record(ENGINE, 1, {"sim.log": 0}, EPIC_AFTER, "boot-a", OUT_OF_ROOT_CLEAN)
    assert new_xid["xid_delta"] == 1 and _gate(faults=new_xid)["gpu_faults"] is False
    rebooted = fault_record(ENGINE, 0, {"sim.log": 0}, EPIC_AFTER, "boot-b", OUT_OF_ROOT_CLEAN)
    assert rebooted["boot_changed"] is True and _gate(faults=rebooted)["pass"] is False
    lost = fault_record(ENGINE, 0, {"sim.log": 2}, EPIC_AFTER, "boot-a", OUT_OF_ROOT_CLEAN)
    assert _gate(faults=lost)["pass"] is False
    zen = fault_record(ENGINE, 0, {"sim.log": 0}, {"kib": 9000, "zen_default_data_exists": True}, "boot-a",
                       OUT_OF_ROOT_CLEAN)
    assert zen["zen_default_data_created"] is True and _gate(faults=zen)["pass"] is False
    trace = fault_record(ENGINE, 0, {"sim.log": 0}, EPIC_AFTER, "boot-a", dict(OUT_OF_ROOT_CLEAN, unreal_trace_store=True))
    assert trace["out_of_root_created"] == ["unreal_trace_store"] and _gate(faults=trace)["pass"] is False
    assert _gate(faults={})["pass"] is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_m0_gate.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'autofly_ue5.validate.m0_gate'`.

- [ ] **Step 3: Write `autofly_ue5/validate/m0_gate.py`**

```python
"""M0 gate: VRAM samples, GPU-fault and ~/.config/Epic records, and the assembled gate report.

env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.m0_gate vram baseline
env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.m0_gate faults
env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.m0_gate assemble
"""

import json
import sys
from pathlib import Path

from autofly_ue5.gpu import gpu_memory_mib
from autofly_ue5.paths import ROOT, RUNS_DIR
from autofly_ue5.validate.engine_check import boot_id, count_device_lost, epic_config_usage, out_of_root_state, xid_count

M0_DIR = RUNS_DIR / "m0"
PROBE_MIC_PARENT = "/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"
VRAM_KEYS = ("baseline", "idle_instance")
REQUIRED = ("engine_check", "plugin_manifest", "blocks_editor_build", "editor_python_probe", "smoke_single_instance",
            "second_instance", "gpu_faults")


def record_vram(path: Path, key: str, used_mib: int) -> dict:
    if key not in VRAM_KEYS:
        raise ValueError(f"unknown VRAM sample key {key!r}; expected one of {VRAM_KEYS}")
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = int(used_mib)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return data


def probe_ok(probe: dict) -> bool:
    flags = ("level_editor_subsystem", "editor_actor_subsystem", "editor_asset_subsystem", "cylinder_mesh", "cube_mesh",
             "world_grid_material")
    return (
        str(probe.get("engine_version", "")).startswith("5.7.4")
        and all(probe.get(k) is True for k in flags)
        and probe.get("game_mode_class") == "/Script/ProjectAirSim.ProjectAirSimGameMode"
        and probe.get("sunsky_class") == "/SunPosition/SunSky.SunSky_C"
        and "Color" in probe.get("basic_shape_material_vector_params", [])
        and probe.get("mic_parent") == PROBE_MIC_PARENT
        and probe.get("mic_color_readback_ok") is True
    )


def fault_record(engine: dict, xid_now: int, device_lost: dict[str, int], epic_after: dict, boot_id_now: str,
                 out_of_root_after: dict[str, bool]) -> dict:
    baseline = engine.get("checks", {}).get("nvidia_xid", {}).get("after")
    before = engine.get("epic_config_before", {})
    out_of_root_before = engine.get("out_of_root_before", {})
    return {
        "xid_since": engine.get("started_at"),
        "xid_baseline": baseline,
        "xid_now": xid_now,
        "xid_delta": None if baseline is None else xid_now - baseline,
        "boot_id_m0_start": engine.get("boot_id"),
        "boot_id_now": boot_id_now,
        "boot_changed": engine.get("boot_id") != boot_id_now,
        "device_lost": device_lost,
        "epic_config_before": before,
        "epic_config_after": epic_after,
        "zen_default_data_created": bool(epic_after.get("zen_default_data_exists"))
        and not before.get("zen_default_data_exists", False),
        "out_of_root_before": out_of_root_before,
        "out_of_root_after": out_of_root_after,
        "out_of_root_created": sorted(name for name, exists in out_of_root_after.items()
                                      if exists and not out_of_root_before.get(name, False)),
    }


def _faults_ok(faults: dict) -> bool:
    lost = faults.get("device_lost") or {}
    return (faults.get("xid_delta") == 0 and faults.get("boot_changed") is False
            and len(lost) > 0 and sum(lost.values()) == 0
            and faults.get("zen_default_data_created") is False and faults.get("out_of_root_created") == [])


def _lockstep(report: dict) -> dict:
    return report.get("phases", {}).get("lockstep", {})


def two_instance_concurrency(smoke_a: dict, smoke_b: dict) -> dict:
    keys = ("timed_start_unix", "timed_end_unix", "vram_sample_unix")
    a, b = _lockstep(smoke_a), _lockstep(smoke_b)
    if any(not isinstance(lock.get(k), (int, float)) for lock in (a, b) for k in keys):
        return {"overlap_s": None, "vram_samples_inside_overlap": False}
    low = max(a["timed_start_unix"], b["timed_start_unix"])
    high = min(a["timed_end_unix"], b["timed_end_unix"])
    inside = all(low <= lock["vram_sample_unix"] <= high for lock in (a, b))
    return {"overlap_s": high - low, "vram_samples_inside_overlap": inside}


def _minus(value, baseline):
    return value - baseline if isinstance(value, int) and isinstance(baseline, int) else None


def assemble_m0_gate(engine: dict, plugin: dict, build: dict, probe: dict, smoke: dict, smoke_fast: dict,
                     smoke_inst1: dict, smoke_concurrent: dict, vram: dict, faults: dict) -> dict:
    baseline = vram.get("baseline")
    one_used = _lockstep(smoke_fast).get("vram_used_mib")
    two_samples = [v for v in (_lockstep(smoke_concurrent).get("vram_used_mib"), _lockstep(smoke_inst1).get("vram_used_mib"))
                   if isinstance(v, int)]
    two_used = max(two_samples) if len(two_samples) == 2 else None
    one, two = _minus(one_used, baseline), _minus(two_used, baseline)
    concurrency = two_instance_concurrency(smoke_concurrent, smoke_inst1)
    spawn = smoke.get("phases", {}).get("spawn", {})
    gate = {
        "engine_check": bool(engine.get("pass")),
        "plugin_manifest": bool(plugin.get("pass")),
        "blocks_editor_build": bool(build.get("pass")),
        "editor_python_probe": probe_ok(probe),
        "smoke_single_instance": bool(smoke.get("pass")),
        "second_instance": bool(smoke_inst1.get("pass")) and bool(smoke_concurrent.get("pass"))
        and concurrency["vram_samples_inside_overlap"],
        "two_instance_concurrency": concurrency,
        "gpu_faults": _faults_ok(faults),
        "steps_per_s_rtur_3ms": _lockstep(smoke).get("steps_per_s"),
        "steps_per_s_rtur_1ms": _lockstep(smoke_fast).get("steps_per_s"),
        "rtur_1ms_lockstep_pass": bool(_lockstep(smoke_fast).get("pass")),
        "steps_per_s_two_instances": [_lockstep(smoke_concurrent).get("steps_per_s"), _lockstep(smoke_inst1).get("steps_per_s")],
        "vram_mib": {"baseline": baseline, "idle_instance": vram.get("idle_instance"),
                     "one_instance_capturing": one_used, "two_instances_capturing": two_used},
        "vram_one_instance_capturing_mib": one,
        "vram_two_instances_capturing_mib": two,
        "vram_per_instance_mib": max(one, two / 2) if one is not None and two is not None else None,
        "yaw_rate_ratio": smoke.get("phases", {}).get("velocity", {}).get("yaw_rate_ratio"),
        "get_images_probe": _lockstep(smoke).get("get_images_probe"),
        "teleport_through_camera_error_m": smoke.get("phases", {}).get("teleport", {}).get("through_camera_error_m"),
        "hfov_face_width_px": [spawn.get("center_face_width_px"), spawn.get("expected_face_width_px")],
        "set_object_material_instance_ok": spawn.get("material_instance_ok"),
        "faults": faults,
    }
    gate["pass"] = all(gate[k] for k in REQUIRED) and gate["vram_per_instance_mib"] is not None
    return gate


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) == 2 and args[0] == "vram":
        used, _total = gpu_memory_mib()
        print(json.dumps(record_vram(M0_DIR / "vram.json", args[1], used)))
        return 0
    if args == ["faults"]:
        logs = sorted((RUNS_DIR / "sim").glob("inst*/sim*.log")) + sorted(M0_DIR.glob("editor_open*.log"))
        engine = _load(M0_DIR / "engine_check.json")
        record = fault_record(engine, xid_count(engine.get("started_at")), count_device_lost(logs), epic_config_usage(),
                              boot_id(), out_of_root_state())
        (M0_DIR / "faults.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2))
        return 0 if _faults_ok(record) else 1
    if args == ["assemble"]:
        gate = assemble_m0_gate(
            _load(M0_DIR / "engine_check.json"), _load(ROOT / "downloads" / "plugin_manifest_report.json"),
            _load(RUNS_DIR / "build" / "build_result.json"), _load(RUNS_DIR / "build" / "python_probe.json"),
            _load(M0_DIR / "smoke_inst0.json"), _load(M0_DIR / "smoke_fast.json"), _load(M0_DIR / "smoke_inst1.json"),
            _load(M0_DIR / "smoke_inst0_concurrent.json"), _load(M0_DIR / "vram.json"), _load(M0_DIR / "faults.json"),
        )
        (M0_DIR / "m0_gate.json").write_text(json.dumps(gate, indent=2))
        print(json.dumps(gate, indent=2))
        return 0 if gate["pass"] else 1
    print("usage: m0_gate vram <baseline|idle_instance> | m0_gate faults | m0_gate assemble", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_m0_gate.py`
Expected: `6 passed`.

- [ ] **Step 5: Baseline VRAM and instance 0 (launch as a job; total wait budget 25 minutes)**

Run:
```bash
cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.m0_gate vram baseline && bash scripts/run_job.sh start launch_inst0 -- env -u PYTHONPATH .venv/bin/python scripts/launch_sim.py --mode editor --map /Game/BlocksMap --instance 0 --timeout 900 && bash scripts/run_job.sh wait launch_inst0 540; echo "wait=$?"
```
Repeat `bash scripts/run_job.sh wait launch_inst0 540` while it returns 124, within the budget; then run `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.m0_gate vram idle_instance`.
Expected: `{"baseline": <≤2000>}`, the launch JSON line with `job launch_inst0 finished: exit=0` and `wait=0`, then `{"baseline": …, "idle_instance": …}` (an idle instance with no scene loaded; information only, the gate uses capturing samples). On failure: stop and report.

- [ ] **Step 6: Throughput with a 1 ms real-time update rate and one-instance VRAM while capturing (job; total wait budget 15 minutes)**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && bash scripts/run_job.sh start smoke_fast -- env -u PYTHONPATH .venv/bin/python -m autofly_ue5.sim.smoke_m0 --instance 0 --scene scene_autofly_m0_fast.jsonc --phases lockstep --warmup-steps 5 --steps 50 --out runs/m0/smoke_fast.json && bash scripts/run_job.sh wait smoke_fast 540; echo "wait=$?"; jq '.error, (.phases.lockstep | {pass, steps_per_s, frames_eq_target, sim_time_exact, vram_used_mib, max_camera_pose_error_m})' runs/m0/smoke_fast.json`
Expected: `null` and a JSON summary with a numeric `vram_used_mib`. `pass` (and a job exit code of 1) may be `false` here without stopping the plan (the M1 scene keeps the verified 3 ms rate); if `error` is set (for example the scene reload failed) or `vram_used_mib` is missing, stop and report.

- [ ] **Step 7: Launch instance 1 on ports 9001/9002 while instance 0 runs (job; total wait budget 25 minutes)**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && bash scripts/run_job.sh start launch_inst1 -- env -u PYTHONPATH .venv/bin/python scripts/launch_sim.py --mode editor --map /Game/BlocksMap --instance 1 --timeout 900 && bash scripts/run_job.sh wait launch_inst1 540; echo "wait=$?"` (repeat `wait` while it returns 124, within the budget).
Expected: JSON line with `"ports": [9001, 9002]`, `job launch_inst1 finished: exit=0`, `wait=0`. `GpuBusyError` (less than 6000 MiB free) or any failure: `bash scripts/run_job.sh stop launch_inst1`, stop instance 1 if it started (`scripts/stop_sim.py --instance 1`), then stop instance 0 too (failure rule) and report. Both instances share `ue_project/projectairsim_server.log` (known; `-saveddirsuffix=inst1` separates `Saved_inst1/`).

- [ ] **Step 8: Step both instances concurrently (two jobs with a common timed window; total wait budget 15 minutes)**

Start both jobs in one tool call. Both scenes load and run 5 untimed warm-up records (instance 1's first frame can take up to 120 s while it loads shaders from the shared DDC), then each sleeps until the shared start time 240 s from now and times 100 records, sampling VRAM after record 50:
```bash
cd /home/nvidiasims/research_uav/autofly_ue5 && START_AT=$(( $(date +%s) + 240 )) && echo "start_at=$START_AT" && bash scripts/run_job.sh start smoke_inst0_concurrent -- env -u PYTHONPATH .venv/bin/python -m autofly_ue5.sim.smoke_m0 --instance 0 --scene scene_autofly_m0.jsonc --phases lockstep --warmup-steps 5 --start-at "$START_AT" --steps 100 --out runs/m0/smoke_inst0_concurrent.json && bash scripts/run_job.sh start smoke_inst1 -- env -u PYTHONPATH .venv/bin/python -m autofly_ue5.sim.smoke_m0 --instance 1 --scene scene_autofly_m0.jsonc --phases lockstep --warmup-steps 5 --start-at "$START_AT" --steps 100 --out runs/m0/smoke_inst1.json
```
Then poll each job in its own tool call (Bash tool `timeout` 600000 ms), repeating while it returns 124 within the budget:
```bash
cd /home/nvidiasims/research_uav/autofly_ue5 && bash scripts/run_job.sh wait smoke_inst1 540; echo "inst1 wait=$?"
```
```bash
cd /home/nvidiasims/research_uav/autofly_ue5 && bash scripts/run_job.sh wait smoke_inst0_concurrent 540; echo "inst0 wait=$?"; jq '.phases.lockstep | {pass, steps_per_s, start_late_s, timed_start_unix, timed_end_unix, vram_sample_unix, vram_used_mib}' runs/m0/smoke_inst0_concurrent.json runs/m0/smoke_inst1.json
```
Expected: last log line `{"pass": true, …}` for both jobs, `inst1 wait=0`, `inst0 wait=0`; both `start_late_s` ≤ 0 (a positive value means that instance finished its warm-up after the shared start time); the two timed windows overlap and each `vram_sample_unix` lies inside both windows (the gate checks this in Step 10); separate client logs `runs/m0/smoke_inst1.client.log` and `runs/m0/smoke_inst0_concurrent.client.log`. Any `false`, or a positive `start_late_s`: stop both instances and report both JSON files and both client logs.

- [ ] **Step 9: Stop both instances, record GPU faults, `~/.config/Epic` usage and out-of-ROOT paths**

Run (Bash tool `timeout` 600000 ms): `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python scripts/stop_sim.py --instance 1 && env -u PYTHONPATH .venv/bin/python scripts/stop_sim.py --instance 0 && sleep 30; pgrep -a -f "$(pwd)/engine/Engine/Binaries/Linux/" || echo none; pgrep -a zenserver || echo no-zenserver`
Expected: `terminated` twice, `none`, `no-zenserver` (survivors after another 60 s are reported, not killed).

Then, in a separate tool call: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.m0_gate faults; echo "faults=$?"; nvidia-smi --query-gpu=memory.used --format=csv,noheader`
Expected: a faults JSON with `"xid_delta": 0` (Xid lines counted from the Task 2 `started_at` across reboots), `"boot_changed": false`, all `device_lost` counts `0` over the `runs/sim/inst0`, `runs/sim/inst1` and editor-probe logs, `"zen_default_data_created": false`, `"out_of_root_created": []`, `epic_config_before`/`epic_config_after` sizes, `faults=0`, and GPU memory back near the baseline. `faults=1`: stop and report `runs/m0/faults.json` (spec §4 fallback trigger evidence for Xid or `VK_ERROR_DEVICE_LOST`; `boot_changed: true` means the machine rebooted during M0, which the user must explain before the gate can pass; a created default Zen folder means the cache override failed; an `out_of_root_created` entry names a location UE wrote despite `-notraceserver`, `uebp_LogFolder` or the probe cleanup).

- [ ] **Step 10: Assemble the gate**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.m0_gate assemble; echo "exit=$?"`
Expected: gate JSON with `"engine_check"`, `"plugin_manifest"`, `"blocks_editor_build"`, `"editor_python_probe"`, `"smoke_single_instance"`, `"second_instance"` and `"gpu_faults"` all `true`, `two_instance_concurrency.vram_samples_inside_overlap` `true` with `overlap_s` > 0, numeric `steps_per_s_*`, `vram_one_instance_capturing_mib`, `vram_two_instances_capturing_mib` and `vram_per_instance_mib`, `"pass": true`, `exit=0`. Any `false`: stop and report the gate JSON and the input file of that key.

- [ ] **Step 11: Commit**

```bash
cp /home/nvidiasims/research_uav/autofly_ue5/runs/m0/m0_gate.json /home/nvidiasims/research_uav/autofly_ue5/docs/gates/m0_gate.json
git -C /home/nvidiasims/research_uav/autofly_ue5 add autofly_ue5/validate/m0_gate.py tests/test_m0_gate.py docs/gates/m0_gate.json && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Pass the M0 gate: build and probe evidence, lock-step smoke, throughput, capturing VRAM, GPU faults and a concurrent second instance" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 12: Milestone stop (spec §12)**

Report to the user: `docs/gates/m0_gate.json` (steps/s at 3 ms and 1 ms and for two concurrent instances with their overlap, VRAM per instance while capturing, yaw-rate ratio, the `get_images` probe result, `teleport_through_camera_error_m`, `hfov_face_width_px`, `set_object_material_instance_ok` (whether spec §4.1's material-instance colouring route works at runtime; M3 target colouring depends on the answer), Xid/`VK_ERROR_DEVICE_LOST` counts and boot id, `~/.config/Epic` size before and after M0 and the out-of-ROOT locations listed in the Global Constraints). Also put two questions to the user: (1) confirm the reachability crossing rule that Task 13 adds on top of spec §6.2 (every free start-band cell must also reach the opposite edge's target band through the obstacle field; measured on the plan's code, the spec rule alone gives `unreachable_start_cells = 0` both for the s01 layout and for an empty layout with 15872 free start cells and 12864 free target cells, and a full-width wall across x = 0 passes the spec rule but leaves 3712 start cells on each of the `x_min` and `x_max` edges unable to cross, while s01 crosses from all four edges); (2) whether a host firewall rule is wanted for the simulator ports 8989/8990 and 9001/9002, which listen on all interfaces without authentication while a simulator runs (the executor does not change firewall rules). Wait for the go-ahead (and the crossing-rule answer) before Task 9.

---

## Milestone M1 — `sim/` module; scene s01 built from its JSON file into a packaged map

Start only after the user's go-ahead on the M0 gate.

### Task 9: Simulator interface types, Protocol and fake simulator

**Files:**
- Create: `autofly_ue5/sim/types.py`, `autofly_ue5/sim/protocol.py`, `autofly_ue5/sim/fake.py`
- Test: `tests/test_sim_types_fake.py`

**Interfaces:**
- Consumes: `autofly_ue5.frames.wrap_pi`.
- Produces:
  - `autofly_ue5.sim.types`: constants `STEP_NS = 5_000_000`, `CONTROL_DT_S = 0.2`; `ObjectNotFoundError(KeyError)`; `@dataclass(frozen=True) Pose(x: float, y: float, z: float, yaw: float)` (NED metres, yaw rad); `@dataclass(frozen=True) CollisionEvent(sim_time_ns: int, object_name: str, impact_point: tuple[float, float, float], normal: tuple[float, float, float])`; `@dataclass(frozen=True) Observation(rgb: np.ndarray, depth: np.ndarray, pose: Pose, velocity_ned: tuple[float, float, float], yaw_rate: float, sim_time_ns: int, collided: bool, step_collisions: tuple[CollisionEvent, ...], rgb_time_ns: int, depth_time_ns: int, kinematics_time_ns: int, camera_pose_error_m: float)` (the three `*_time_ns` are the timestamps carried by the RGB message, the depth message and the kinematics that built the observation; `camera_pose_error_m` is the distance between the camera position stamped in the RGB message and kinematics plus the mount offset); `dt_to_ns(dt: float, step_ns: int = STEP_NS) -> int`.
  - `autofly_ue5.sim.protocol.Simulator` (`typing.Protocol`, runtime-checkable): property `steps_taken -> int` (reset steps included); `launch(map_path: str, instance: int) -> None`; `close() -> None`; `reset(pose: Pose) -> Observation` (advances the clock by whole control steps); `spawn(name: str, asset: str, pose: Pose, scale: tuple[float, float, float], material: str | None = None) -> str`; `destroy(name: str) -> None` (raises `ObjectNotFoundError` for an unknown name); `command_velocity(v_forward: float, yaw_rate: float, v_z: float) -> None` (m/s, rad/s, m/s with **v_z positive up**); `step(dt: float = CONTROL_DT_S) -> int` (sim time after the step, ns; requires a `command_velocity` call since the previous step); `observe() -> Observation`.
  - `autofly_ue5.sim.fake.FakeSimulator(obstacles: list[tuple[float, float, float]] | None = None, image_size: int = 256, reset_steps: int = 4)` implementing `Simulator` (kinematic integration, circle obstacles `(x, y, radius)`, pose-seeded RGB, all-`inf` depth; `reset` advances `reset_steps` 0.2 s steps like the backend's two waypoint steps plus two settle steps).

- [ ] **Step 1: Write the failing test**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_sim_types_fake.py`:
```python
import math

import numpy as np
import pytest

from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.sim.protocol import Simulator
from autofly_ue5.sim.types import ObjectNotFoundError, Pose, dt_to_ns


def test_dt_to_ns():
    assert dt_to_ns(0.2) == 200_000_000
    with pytest.raises(ValueError):
        dt_to_ns(0.203)
    with pytest.raises(ValueError):
        dt_to_ns(0.0)


def test_fake_implements_protocol():
    assert isinstance(FakeSimulator(), Simulator)


def test_step_requires_launch_and_command():
    sim = FakeSimulator()
    with pytest.raises(RuntimeError, match="not launched"):
        sim.step()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    with pytest.raises(RuntimeError, match="command_velocity"):
        sim.step()


def test_reset_advances_the_clock_like_the_backend():
    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    obs = sim.reset(Pose(1.0, 2.0, -2.0, 0.5))
    assert sim.steps_taken == 4 and obs.sim_time_ns == 800_000_000
    assert obs.rgb_time_ns == obs.depth_time_ns == obs.kinematics_time_ns == 800_000_000
    assert obs.pose == Pose(1.0, 2.0, -2.0, 0.5) and obs.velocity_ned == (0.0, 0.0, 0.0) and obs.camera_pose_error_m == 0.0


def test_forward_yaw_and_vertical_integration():
    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    sim.command_velocity(2.0, 0.0, 0.0)
    assert sim.step(0.2) == 1_000_000_000
    obs = sim.observe()
    assert obs.pose.x == pytest.approx(0.4) and obs.sim_time_ns == 1_000_000_000 and sim.steps_taken == 5
    assert obs.rgb_time_ns == obs.depth_time_ns == obs.kinematics_time_ns == 1_000_000_000
    sim.reset(Pose(0.0, 0.0, -2.0, math.pi / 2))
    sim.command_velocity(1.0, 0.0, 0.5)
    sim.step()
    obs = sim.observe()
    assert obs.pose.y == pytest.approx(0.2) and obs.pose.x == pytest.approx(0.0, abs=1e-9)
    assert obs.pose.z == pytest.approx(-2.1)
    sim.command_velocity(0.0, 1.0, 0.0)
    sim.step()
    assert sim.observe().pose.yaw == pytest.approx(math.pi / 2 + 0.2)


def test_collision_sets_flag_until_reset():
    sim = FakeSimulator(obstacles=[(1.0, 0.0, 0.5)])
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    sim.reset(Pose(0.0, 0.0, -2.0, 0.0))
    sim.command_velocity(2.0, 0.0, 0.0)
    sim.step()
    assert sim.observe().collided is False
    sim.command_velocity(2.0, 0.0, 0.0)
    sim.step()
    obs = sim.observe()
    assert obs.collided is True and obs.step_collisions[0].object_name == "obstacle_0"
    assert sim.reset(Pose(0.0, 0.0, -2.0, 0.0)).collided is False


def test_rgb_depends_on_pose_only():
    sim = FakeSimulator(image_size=32)
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    a = sim.reset(Pose(0.0, 0.0, -2.0, 0.0)).rgb
    b = sim.reset(Pose(5.0, 0.0, -2.0, 0.0)).rgb
    a2 = sim.reset(Pose(0.0, 0.0, -2.0, 0.0)).rgb
    assert a.shape == (32, 32, 3) and a.dtype == np.uint8
    assert not np.array_equal(a, b) and np.array_equal(a, a2)
    assert np.isinf(sim.observe().depth).all()


def test_spawn_unique_names_and_destroy():
    sim = FakeSimulator()
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    first = sim.spawn("AF_Target", "cylinder", Pose(1.0, 2.0, -1.0, 0.0), (1.0, 1.0, 1.0), "white")
    second = sim.spawn("AF_Target", "cylinder", Pose(3.0, 2.0, -1.0, 0.0), (1.0, 1.0, 1.0))
    assert first == "AF_Target" and second == "AF_Target1"
    sim.destroy(first)
    with pytest.raises(ObjectNotFoundError):
        sim.destroy(first)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_sim_types_fake.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'autofly_ue5.sim.fake'`.

- [ ] **Step 3: Write `autofly_ue5/sim/types.py`**

```python
"""Value types of the simulator interface (spec §7). World frame: NED metres, yaw radians."""

from dataclasses import dataclass

import numpy as np

STEP_NS = 5_000_000
CONTROL_DT_S = 0.2


class ObjectNotFoundError(KeyError):
    """destroy() was given a name that no spawned object has."""


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class CollisionEvent:
    sim_time_ns: int
    object_name: str
    impact_point: tuple[float, float, float]
    normal: tuple[float, float, float]


@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray
    depth: np.ndarray
    pose: Pose
    velocity_ned: tuple[float, float, float]
    yaw_rate: float
    sim_time_ns: int
    collided: bool
    step_collisions: tuple[CollisionEvent, ...]
    rgb_time_ns: int
    depth_time_ns: int
    kinematics_time_ns: int
    camera_pose_error_m: float


def dt_to_ns(dt: float, step_ns: int = STEP_NS) -> int:
    dt_ns = round(dt * 1e9)
    if dt_ns <= 0 or dt_ns % step_ns != 0:
        raise ValueError(f"dt={dt} s is not a positive multiple of the {step_ns} ns clock step")
    return dt_ns
```

- [ ] **Step 4: Write `autofly_ue5/sim/protocol.py`**

```python
"""The simulator interface every backend implements (spec §7)."""

from typing import Protocol, runtime_checkable

from autofly_ue5.sim.types import CONTROL_DT_S, Observation, Pose


@runtime_checkable
class Simulator(Protocol):
    @property
    def steps_taken(self) -> int:
        """Number of simulator clock steps issued since construction (reset steps included)."""

    def launch(self, map_path: str, instance: int) -> None:
        """Start (or attach to) the simulator process for `map_path` as instance `instance`."""

    def close(self) -> None:
        """Disconnect and stop the process this object launched."""

    def reset(self, pose: Pose) -> Observation:
        """Teleport the vehicle to `pose` with zero velocity, advancing the clock by whole control steps (counted in
        steps_taken); returns the settled observation with collided=False."""

    def spawn(self, name: str, asset: str, pose: Pose, scale: tuple[float, float, float], material: str | None = None) -> str:
        """Spawn a static object; returns the actual (possibly uniquified) name.

        `asset` is the short AssetRegistry name of a cooked static mesh (e.g. "1M_Cube"), not an object path: Project
        AirSim keys its spawn table by short name and a later duplicate silently replaces an earlier one
        (WorldSimApi.cpp:464-475), so spawnable assets need globally unique names. `material` is a package path of a
        base UMaterial (e.g. "/Game/Geometry/Materials/M_Orange")."""

    def destroy(self, name: str) -> None:
        """Destroy a spawned object; raises ObjectNotFoundError if it does not exist."""

    def command_velocity(self, v_forward: float, yaw_rate: float, v_z: float) -> None:
        """Set the command held for the next step: body forward m/s, yaw rate rad/s, vertical m/s (positive up)."""

    def step(self, dt: float = CONTROL_DT_S) -> int:
        """Advance the simulator clock by exactly dt; returns the simulator time in ns after the step."""

    def observe(self) -> Observation:
        """Observation at the end of the last step or reset."""
```

- [ ] **Step 5: Write `autofly_ue5/sim/fake.py`**

```python
"""Deterministic in-memory simulator with the Simulator interface, for offline tests."""

import hashlib
import math

import numpy as np

from autofly_ue5.frames import wrap_pi
from autofly_ue5.sim.types import CONTROL_DT_S, CollisionEvent, ObjectNotFoundError, Observation, Pose, dt_to_ns


class FakeSimulator:
    def __init__(self, obstacles: list[tuple[float, float, float]] | None = None, image_size: int = 256,
                 reset_steps: int = 4) -> None:
        self._obstacles = list(obstacles or [])
        self._size = image_size
        self._reset_steps = reset_steps
        self._launched: tuple[str, int] | None = None
        self._pose = Pose(0.0, 0.0, -2.0, 0.0)
        self._velocity = (0.0, 0.0, 0.0)
        self._yaw_rate = 0.0
        self._t_ns = 0
        self._pending: tuple[float, float, float] | None = None
        self._steps = 0
        self._collided = False
        self._step_collisions: tuple[CollisionEvent, ...] = ()
        self._objects: dict[str, tuple[str, Pose, tuple[float, float, float], str | None]] = {}
        self._has_observation = False

    @property
    def steps_taken(self) -> int:
        return self._steps

    def launch(self, map_path: str, instance: int) -> None:
        if instance < 0:
            raise ValueError("instance must be >= 0")
        self._launched = (map_path, instance)

    def close(self) -> None:
        self._launched = None

    def _require_launched(self) -> None:
        if self._launched is None:
            raise RuntimeError("simulator is not launched")

    def reset(self, pose: Pose) -> Observation:
        self._require_launched()
        self._t_ns += self._reset_steps * dt_to_ns(CONTROL_DT_S)
        self._steps += self._reset_steps
        self._pose = pose
        self._velocity = (0.0, 0.0, 0.0)
        self._yaw_rate = 0.0
        self._pending = None
        self._collided = False
        self._step_collisions = ()
        self._has_observation = True
        return self.observe()

    def spawn(self, name: str, asset: str, pose: Pose, scale: tuple[float, float, float], material: str | None = None) -> str:
        self._require_launched()
        unique, k = name, 1
        while unique in self._objects:
            unique = f"{name}{k}"
            k += 1
        self._objects[unique] = (asset, pose, tuple(scale), material)
        return unique

    def destroy(self, name: str) -> None:
        if name not in self._objects:
            raise ObjectNotFoundError(name)
        del self._objects[name]

    def command_velocity(self, v_forward: float, yaw_rate: float, v_z: float) -> None:
        self._pending = (float(v_forward), float(yaw_rate), float(v_z))

    def step(self, dt: float = CONTROL_DT_S) -> int:
        self._require_launched()
        dt_ns = dt_to_ns(dt)
        if self._pending is None:
            raise RuntimeError("command_velocity must be called before every step")
        v_forward, yaw_rate, v_z = self._pending
        self._pending = None
        dt_s = dt_ns / 1e9
        p = self._pose
        vx, vy, vz_ned = v_forward * math.cos(p.yaw), v_forward * math.sin(p.yaw), -v_z
        new = Pose(p.x + vx * dt_s, p.y + vy * dt_s, p.z + vz_ned * dt_s, wrap_pi(p.yaw + yaw_rate * dt_s))
        self._t_ns += dt_ns
        self._steps += 1
        hits = tuple(
            CollisionEvent(self._t_ns, f"obstacle_{i}", (new.x, new.y, new.z), (0.0, 0.0, 0.0))
            for i, (ox, oy, radius) in enumerate(self._obstacles)
            if math.hypot(new.x - ox, new.y - oy) <= radius
        )
        self._pose, self._velocity, self._yaw_rate = new, (vx, vy, vz_ned), yaw_rate
        self._step_collisions = hits
        self._collided = self._collided or bool(hits)
        self._has_observation = True
        return self._t_ns

    def observe(self) -> Observation:
        if not self._has_observation:
            raise RuntimeError("no observation before reset() or step()")
        return Observation(
            rgb=self._render_rgb(),
            depth=np.full((self._size, self._size), np.inf, dtype=np.float32),
            pose=self._pose,
            velocity_ned=self._velocity,
            yaw_rate=self._yaw_rate,
            sim_time_ns=self._t_ns,
            collided=self._collided,
            step_collisions=self._step_collisions,
            rgb_time_ns=self._t_ns,
            depth_time_ns=self._t_ns,
            kinematics_time_ns=self._t_ns,
            camera_pose_error_m=0.0,
        )

    def _render_rgb(self) -> np.ndarray:
        p = self._pose
        digest = hashlib.sha256(f"{p.x:.3f},{p.y:.3f},{p.z:.3f},{p.yaw:.4f}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        return rng.integers(0, 256, size=(self._size, self._size, 3), dtype=np.uint8)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_sim_types_fake.py`
Expected: `8 passed`.

- [ ] **Step 7: Commit**

```bash
git -C /home/nvidiasims/research_uav/autofly_ue5 add autofly_ue5/sim/types.py autofly_ue5/sim/protocol.py autofly_ue5/sim/fake.py tests/test_sim_types_fake.py && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Add the Simulator protocol, its value types and a deterministic fake simulator" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Frame synchronisation, collision events and decoding of the real M0 fixtures

**Files:**
- Create: `autofly_ue5/sim/sync.py`, `autofly_ue5/sim/events.py`
- Test: `tests/test_sync_events.py`, `tests/test_decode_fixtures.py`

**Interfaces:**
- Consumes: `autofly_ue5.sim.types.CollisionEvent`; `autofly_ue5.sim.decode.{decode_rgb, decode_depth}`; `autofly_ue5.paths.FIXTURES_DIR`; fixtures `tests/fixtures/pas/{rgb_msg,depth_msg}.{json,bin}` (Task 7).
- Produces:
  - `autofly_ue5.sim.sync`: `FrameTimeoutError(RuntimeError)`, `FrameTimestampError(RuntimeError)`; `FrameCollector(keys: tuple[str, ...] = ("rgb", "depth"))` with `callback(key: str) -> Callable[[object, dict], None]`, `arm(target_ns: int) -> None`, `wait(timeout_s: float) -> dict[str, dict]` (every key has a message with `time_stamp == target_ns`), attribute `received: int`.
  - `autofly_ue5.sim.events`: `event_from_step(event: dict) -> CollisionEvent`; `event_from_topic(msg: dict) -> CollisionEvent`; `collisions_after(events: Iterable[CollisionEvent], t_ns: int) -> tuple[CollisionEvent, ...]`; `CollisionLog()` with `topic_callback(topic: object, msg: dict) -> None`, `collect(step_events: list[dict], up_to_ns: int) -> tuple[CollisionEvent, ...]` (merged, de-duplicated by `(sim_time_ns, object_name)`, time-sorted), `clear() -> None`.

- [ ] **Step 1: Write the failing tests**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_sync_events.py`:
```python
import threading
import time

import pytest

from autofly_ue5.sim.events import CollisionLog, collisions_after, event_from_step, event_from_topic
from autofly_ue5.sim.sync import FrameCollector, FrameTimeoutError, FrameTimestampError
from autofly_ue5.sim.types import CollisionEvent


def _msg(t_ns: int) -> dict:
    return {"time_stamp": t_ns, "encoding": "BGR", "height": 1, "width": 1, "data": b"\x00\x00\x00"}


def test_collector_returns_frames_at_target():
    fc = FrameCollector()
    fc.arm(200)
    fc.callback("rgb")(None, _msg(195))  # older frame is ignored
    fc.callback("rgb")(None, _msg(200))
    fc.callback("depth")(None, _msg(200))
    frames = fc.wait(timeout_s=0.5)
    assert frames["rgb"]["time_stamp"] == 200 and frames["depth"]["time_stamp"] == 200
    assert fc.received == 3


def test_collector_waits_for_late_frames_from_another_thread():
    fc = FrameCollector()
    fc.arm(400)

    def deliver():
        time.sleep(0.05)
        fc.callback("rgb")(None, _msg(400))
        fc.callback("depth")(None, _msg(400))

    threading.Thread(target=deliver).start()
    assert set(fc.wait(timeout_s=2.0)) == {"rgb", "depth"}


def test_collector_times_out_naming_missing_stream():
    fc = FrameCollector()
    fc.arm(600)
    fc.callback("rgb")(None, _msg(600))
    with pytest.raises(FrameTimeoutError, match="depth"):
        fc.wait(timeout_s=0.05)


def test_collector_rejects_frames_past_target():
    fc = FrameCollector()
    fc.arm(800)
    fc.callback("rgb")(None, _msg(805))
    fc.callback("depth")(None, _msg(800))
    with pytest.raises(FrameTimestampError):
        fc.wait(timeout_s=0.5)


def test_arm_clears_previous_frames():
    fc = FrameCollector()
    fc.arm(200)
    fc.callback("rgb")(None, _msg(200))
    fc.callback("depth")(None, _msg(200))
    fc.arm(400)
    with pytest.raises(FrameTimeoutError):
        fc.wait(timeout_s=0.05)


def test_unknown_stream_is_rejected():
    with pytest.raises(KeyError):
        FrameCollector().callback("segmentation")


STEP_EVENT = {"type": "collision", "sim_time_ns": 400, "object_name": "StaticMeshActor_12",
              "impact_point": {"x": 1.0, "y": 2.0, "z": -1.5}, "normal": {"x": -1.0, "y": 0.0, "z": 0.0}}
TOPIC_MSG = {"time_stamp": 400, "object_name": "StaticMeshActor_12", "segmentation_id": 3,
             "position": {"x": 0.9, "y": 2.0, "z": -1.5}, "impact_point": {"x": 1.0, "y": 2.0, "z": -1.5},
             "normal": {"x": -1.0, "y": 0.0, "z": 0.0}, "penetration_depth": 0.01}


def test_event_parsing():
    expected = CollisionEvent(400, "StaticMeshActor_12", (1.0, 2.0, -1.5), (-1.0, 0.0, 0.0))
    assert event_from_step(STEP_EVENT) == expected
    assert event_from_topic(TOPIC_MSG) == expected


def test_log_merges_and_deduplicates_step_and_topic_events():
    log = CollisionLog()
    log.topic_callback(None, TOPIC_MSG)
    log.topic_callback(None, dict(TOPIC_MSG, time_stamp=600))  # belongs to a later step
    first = log.collect([STEP_EVENT, {"type": "gate_pass", "sim_time_ns": 400}], up_to_ns=400)
    assert [e.sim_time_ns for e in first] == [400]
    second = log.collect([STEP_EVENT], up_to_ns=600)  # late duplicate of the 400 hit plus the 600 hit
    assert [e.sim_time_ns for e in second] == [600]
    log.clear()
    assert log.collect([STEP_EVENT], up_to_ns=600)[0].sim_time_ns == 400


def test_collisions_after():
    events = [CollisionEvent(t, "a", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) for t in (200, 400, 600)]
    assert [e.sim_time_ns for e in collisions_after(events, 400)] == [600]
```

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_decode_fixtures.py`:
```python
import json

import numpy as np

from autofly_ue5.paths import FIXTURES_DIR
from autofly_ue5.sim.decode import decode_depth, decode_rgb

PAS = FIXTURES_DIR / "pas"


def _load(stem: str) -> dict:
    msg = json.loads((PAS / f"{stem}.json").read_text())
    msg["data"] = (PAS / f"{stem}.bin").read_bytes()
    return msg


def test_real_rgb_message_decodes_to_256_rgb():
    msg = _load("rgb_msg")
    assert msg["encoding"] == "BGR" and msg["data_len"] == 256 * 256 * 3
    img = decode_rgb(msg)
    assert img.shape == (256, 256, 3) and img.dtype == np.uint8
    b, g, r = msg["center_pixel_bgr"]
    assert img[128, 128].tolist() == [r, g, b]


def test_real_rgb_message_as_int_list_matches_bytes():
    msg = _load("rgb_msg")
    as_list = dict(msg, data=list(msg["data"]))
    assert np.array_equal(decode_rgb(as_list), decode_rgb(msg))


def test_real_depth_message_is_metres_with_inf_sky():
    msg = _load("depth_msg")
    assert msg["encoding"] == "16FC1" and msg["data_len"] == 256 * 256 * 2
    depth = decode_depth(msg)
    assert depth.shape == (256, 256) and depth.dtype == np.float32
    assert int(np.isinf(depth).sum()) == msg["inf_count"]
    center = float(np.median(depth[124:132, 124:132]))
    assert abs(center - msg["expected_center_depth_m"]) < 0.3
    assert float(np.nanmin(depth)) >= 0.5
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_sync_events.py; env -u PYTHONPATH .venv/bin/python -m pytest tests/test_decode_fixtures.py`
Expected: the first run FAILS with `ModuleNotFoundError: No module named 'autofly_ue5.sim.events'`; the second run already PASSES (`3 passed`) because decoding exists since Task 6 — this confirms the synthetic decoder matches real server bytes. If a fixture test fails, stop and report: the Task 6 decoder contradicts the real message format.

- [ ] **Step 3: Write `autofly_ue5/sim/sync.py`**

```python
"""Wait for camera frames whose timestamp equals the simulator time a step ended at."""

import threading
import time
from typing import Callable


class FrameTimeoutError(RuntimeError):
    """A stream delivered no frame at or after the target time within the timeout."""


class FrameTimestampError(RuntimeError):
    """A frame arrived with a timestamp other than the target time."""


class FrameCollector:
    def __init__(self, keys: tuple[str, ...] = ("rgb", "depth")) -> None:
        self._keys = tuple(keys)
        self._cond = threading.Condition()
        self._target: int | None = None
        self._slots: dict[str, dict | None] = {k: None for k in self._keys}
        self.received = 0

    def callback(self, key: str) -> Callable[[object, dict], None]:
        if key not in self._keys:
            raise KeyError(key)

        def _on_message(_topic: object, msg: dict) -> None:
            with self._cond:
                self.received += 1
                if self._target is not None and int(msg["time_stamp"]) >= self._target:
                    self._slots[key] = msg
                    self._cond.notify_all()

        return _on_message

    def arm(self, target_ns: int) -> None:
        with self._cond:
            self._target = int(target_ns)
            self._slots = {k: None for k in self._keys}

    def wait(self, timeout_s: float) -> dict[str, dict]:
        deadline = time.monotonic() + timeout_s
        with self._cond:
            if self._target is None:
                raise RuntimeError("arm() must be called before wait()")
            while any(v is None for v in self._slots.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [k for k, v in self._slots.items() if v is None]
                    raise FrameTimeoutError(f"no {missing} frame with time_stamp >= {self._target} within {timeout_s} s")
                self._cond.wait(remaining)
            late = {k: int(v["time_stamp"]) for k, v in self._slots.items() if int(v["time_stamp"]) != self._target}
            if late:
                raise FrameTimestampError(f"frames {late} do not match the step time {self._target}")
            return dict(self._slots)
```

- [ ] **Step 4: Write `autofly_ue5/sim/events.py`**

```python
"""Collision events from Step() results and the collision_info topic.

A hit on a step's final pose can be reported in the next Step() result; the topic
message usually arrives first. Events are merged, de-duplicated by
(sim_time_ns, object_name) and assigned by their own timestamps.
"""

import threading
from typing import Iterable

from autofly_ue5.sim.types import CollisionEvent


def _vec(d: dict) -> tuple[float, float, float]:
    return (float(d["x"]), float(d["y"]), float(d["z"]))


def event_from_step(event: dict) -> CollisionEvent:
    return CollisionEvent(int(event["sim_time_ns"]), str(event["object_name"]), _vec(event["impact_point"]), _vec(event["normal"]))


def event_from_topic(msg: dict) -> CollisionEvent:
    return CollisionEvent(int(msg["time_stamp"]), str(msg["object_name"]), _vec(msg["impact_point"]), _vec(msg["normal"]))


def collisions_after(events: Iterable[CollisionEvent], t_ns: int) -> tuple[CollisionEvent, ...]:
    return tuple(e for e in events if e.sim_time_ns > t_ns)


class CollisionLog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[CollisionEvent] = []
        self._seen: set[tuple[int, str]] = set()

    def topic_callback(self, _topic: object, msg: dict) -> None:
        event = event_from_topic(msg)
        with self._lock:
            self._pending.append(event)

    def collect(self, step_events: list[dict], up_to_ns: int) -> tuple[CollisionEvent, ...]:
        with self._lock:
            candidates = [event_from_step(e) for e in step_events if e.get("type") == "collision"]
            candidates += [e for e in self._pending if e.sim_time_ns <= up_to_ns]
            self._pending = [e for e in self._pending if e.sim_time_ns > up_to_ns]
            fresh = []
            for event in sorted(candidates, key=lambda e: e.sim_time_ns):
                key = (event.sim_time_ns, event.object_name)
                if key not in self._seen:
                    self._seen.add(key)
                    fresh.append(event)
            return tuple(fresh)

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()
            self._seen.clear()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_sync_events.py tests/test_decode_fixtures.py`
Expected: `12 passed`.

- [ ] **Step 6: Commit**

```bash
git -C /home/nvidiasims/research_uav/autofly_ue5 add autofly_ue5/sim/sync.py autofly_ue5/sim/events.py tests/test_sync_events.py tests/test_decode_fixtures.py && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Add timestamp-exact frame collection, collision event merging and tests on the real M0 image messages" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Project AirSim backend (`ProjectAirSimSimulator`)

**Files:**
- Create: `autofly_ue5/sim/airsim_backend.py`
- Test: `tests/test_airsim_backend.py`

**Interfaces:**
- Consumes: `autofly_ue5.sim.types.{Pose, Observation, ObjectNotFoundError, CONTROL_DT_S, STEP_NS, dt_to_ns}`; `autofly_ue5.sim.sync.FrameCollector`; `autofly_ue5.sim.events.{CollisionLog, collisions_after}`; `autofly_ue5.sim.decode.{decode_rgb, decode_depth}`; `autofly_ue5.frames.{body_to_ned, quat_to_yaw, yaw_to_quat}`; `autofly_ue5.sim.process.{SimPorts, ports_for_instance, instance_dir, packaged_command, sim_environment, launch_process, wait_ready, stop, own_running_instances, SIM_RUN_DIR}`; `autofly_ue5.gpu.{gpu_memory_mib, check_gpu_for_launch}`; `autofly_ue5.paths.{CONFIGS_DIR, PACKAGED_BINARY}`.
- Produces:
  - `autofly_ue5.sim.airsim_backend.PasApi` (frozen dataclass: `client_cls`, `world_cls`, `drone_cls`, `pose_cls`, `yaw_mode_max_dof`) and `real_api() -> PasApi`
  - `StepTimingError(RuntimeError)`, `StaleStateError(RuntimeError)`, `CommandTimeoutError(RuntimeError)`, `CameraPoseError(RuntimeError)`
  - `ProjectAirSimSimulator(scene_config: str = "scene_autofly_s01.jsonc", config_dir: Path = CONFIGS_DIR, binary: Path = PACKAGED_BINARY, run_root: Path = SIM_RUN_DIR, api: PasApi | None = None, frame_timeout_s: float = 5.0, first_frame_timeout_s: float = 120.0, ready_timeout_s: float = 900.0, safe_altitude_m: float = 20.0, settle_steps: int = 2, collision_grace_s: float = 0.02, kinematics_retries: int = 5, command_timeout_s: float = 10.0, camera_offset_m: float = 0.40, camera_pose_tolerance_m: float = 0.10)` implementing `Simulator`, plus `connect(ports: SimPorts) -> None` (attach to an already running simulator; used by tests and by `launch`).
  - Constants `ROBOT = "Drone1"`, `CAMERA = "FrontCamera"`.

Behaviour fixed by M0 evidence: one `world.step(dt_ns)` per `step()`; the move command (duration `dt − 2·STEP_NS`, `v_down = −v_z`, `yaw_is_rate=True`) is sent before the step, followed by a 2 ms loop pause, and its reply is awaited after the step with `command_timeout_s` (`CommandTimeoutError` instead of blocking for the client's 300 s receive timeout); frames must have `time_stamp == T`; kinematics are re-read until `time_stamp == T`; the camera position stamped in the RGB message must equal kinematics plus the orientation-rotated `camera_offset_m` within `camera_pose_tolerance_m`, otherwise `CameraPoseError` (a `set_pose` sweep that left the Unreal actor stuck, research gotcha 6), except on a step that reports a new episode collision (collision messages are collected once more after `CAMERA_RECHECK_S = 0.2` s before raising, because a hit can arrive after the frame), where the error is only recorded in `Observation.camera_pose_error_m`; `reset` teleports via two waypoints at `−safe_altitude_m` (avoids sweeping through obstacles) and settles with zero commands, and a camera desync during reset always raises; collisions at or before the settle time never count for the new episode; `destroy` of an unknown name raises `ObjectNotFoundError`.

- [ ] **Step 1: Write the failing test**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_airsim_backend.py`:
```python
import asyncio
import math

import pytest

from autofly_ue5.frames import quat_to_yaw, yaw_to_quat
from autofly_ue5.sim.airsim_backend import (
    CameraPoseError,
    CommandTimeoutError,
    PasApi,
    ProjectAirSimSimulator,
    StaleStateError,
    StepTimingError,
)
from autofly_ue5.sim.process import SimPorts
from autofly_ue5.sim.protocol import Simulator
from autofly_ue5.sim.sync import FrameTimeoutError
from autofly_ue5.sim.types import ObjectNotFoundError, Pose


class FakeClient:
    def __init__(self, port_topics, port_services):
        self.ports = (port_topics, port_services)
        self.subs = {}
        self.connected = False

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def subscribe(self, topic, callback):
        self.subs.setdefault(topic, []).append(callback)

    def publish(self, topic, msg):
        for callback in self.subs.get(topic, []):
            callback(topic, msg)


class FakeWorld:
    def __init__(self, client, scene_config_name, delay_after_load_sec=0, sim_config_path=""):
        self.client, self.scene, self.config_path = client, scene_config_name, sim_config_path
        self.t = 0
        self.step_calls = 0
        self.drone = None
        self.spawned = {}
        self.pending_events = []
        self.stale_kinematics = 0
        self.drop_depth = False
        self.time_skew_ns = 0
        self.camera_lag_m = 0.0  # simulated stuck Unreal actor: the camera stays this far behind kinematics

    def get_sim_clock_type(self):
        return "steppable"

    def get_sim_time(self):
        return self.t

    def step(self, dt_ns):
        self.step_calls += 1
        self.t += dt_ns + self.time_skew_ns
        self.drone.integrate(dt_ns / 1e9)
        cams = self.drone.sensors["FrontCamera"]
        offset = 0.4 - self.camera_lag_m
        cam = {"pos_x": self.drone.x + offset * math.cos(self.drone.yaw), "pos_y": self.drone.y + offset * math.sin(self.drone.yaw),
               "pos_z": self.drone.z}
        self.client.publish(cams["scene_camera"], {"time_stamp": self.t, "encoding": "BGR", "height": 4, "width": 4, "data": bytes(48), **cam})
        if not self.drop_depth:
            self.client.publish(cams["depth_planar_camera"], {"time_stamp": self.t, "encoding": "16FC1", "height": 4, "width": 4, "data": bytes(32), **cam})
        events, self.pending_events = self.pending_events, []
        return {"sim_time_ns": self.t, "robots": {"Drone1": {"state": {}, "events": events}}}

    def spawn_object(self, object_name, asset_path, object_pose, object_scale, enable_physics):
        self.spawned[object_name] = (asset_path, object_pose, object_scale, enable_physics)
        return object_name

    def set_object_material(self, object_name, material_asset_path):
        return object_name in self.spawned

    def destroy_object(self, object_name):
        return self.spawned.pop(object_name, None) is not None


class FakeDrone:
    def __init__(self, client, world, name):
        self.client, self.world, self.name = client, world, name
        world.drone = self
        self.sensors = {"FrontCamera": {"scene_camera": "/rgb", "depth_planar_camera": "/depth"}}
        self.robot_info = {"collision_info": "/collision"}
        self.x, self.y, self.z, self.yaw = 0.0, 0.0, -2.0, 0.0
        self.cmd = (0.0, 0.0, 0.0)
        self.commands, self.poses = [], []
        self.api_enabled = self.armed = False
        self.hang_reply = False

    def enable_api_control(self):
        self.api_enabled = True
        return True

    def arm(self):
        self.armed = True
        return True

    async def move_by_velocity_body_frame_async(self, v_forward, v_right, v_down, duration, yaw_control_mode, yaw_is_rate, yaw):
        self.commands.append({"v_forward": v_forward, "v_right": v_right, "v_down": v_down, "duration": duration,
                              "yaw_is_rate": yaw_is_rate, "yaw": yaw, "step_calls_at_send": self.world.step_calls})
        self.cmd = (v_forward, v_down, yaw)
        hang = self.hang_reply

        async def _reply():
            if hang:  # the server never answers, e.g. the command's start was read one tick late
                await asyncio.get_running_loop().create_future()
            return True

        return asyncio.ensure_future(_reply())

    def integrate(self, dt):
        v_forward, v_down, yaw_rate = self.cmd
        self.x += v_forward * math.cos(self.yaw) * dt
        self.y += v_forward * math.sin(self.yaw) * dt
        self.z += v_down * dt
        self.yaw += yaw_rate * dt

    def set_pose(self, pose, reset_kinematics=True):
        self.poses.append(pose)
        t, r = pose["translation"], pose["rotation"]
        self.x, self.y, self.z = t["x"], t["y"], t["z"]
        self.yaw = quat_to_yaw(r["w"], r["x"], r["y"], r["z"])
        return True

    def get_ground_truth_kinematics(self):
        stale = self.world.stale_kinematics > 0
        if stale:
            self.world.stale_kinematics -= 1
        w, qx, qy, qz = yaw_to_quat(self.yaw)
        return {"time_stamp": self.world.t - (5_000_000 if stale else 0),
                "pose": {"position": {"x": self.x, "y": self.y, "z": self.z}, "orientation": {"w": w, "x": qx, "y": qy, "z": qz}},
                "twist": {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}}


API = PasApi(client_cls=FakeClient, world_cls=FakeWorld, drone_cls=FakeDrone, pose_cls=dict, yaw_mode_max_dof=0)


def make_sim(**kwargs):
    options = dict(api=API, frame_timeout_s=0.2, first_frame_timeout_s=0.2, collision_grace_s=0.0, command_timeout_s=0.2)
    options.update(kwargs)
    sim = ProjectAirSimSimulator(**options)
    sim.connect(SimPorts(8989, 8990))
    return sim


def test_connect_subscribes_arms_and_loads_config():
    sim = make_sim()
    assert isinstance(sim, Simulator)
    drone = sim._drone
    assert drone.api_enabled and drone.armed
    assert set(sim._client.subs) == {"/rgb", "/depth", "/collision"}
    assert sim._world.scene == "scene_autofly_s01.jsonc" and sim._world.config_path.endswith("/configs")


def test_step_sends_command_before_one_world_step():
    sim = make_sim()
    with pytest.raises(RuntimeError, match="command_velocity"):
        sim.step()
    sim.command_velocity(2.0, 0.5, 1.0)
    assert sim.step(0.2) == 200_000_000
    cmd = sim._drone.commands[-1]
    assert cmd["step_calls_at_send"] == 0 and sim._world.step_calls == 1 and sim.steps_taken == 1
    assert cmd["duration"] == pytest.approx(0.19) and cmd["v_down"] == -1.0 and cmd["v_right"] == 0.0
    assert cmd["yaw_is_rate"] is True and cmd["yaw"] == 0.5
    obs = sim.observe()
    assert obs.sim_time_ns == 200_000_000 and obs.rgb.shape == (4, 4, 3) and obs.depth.shape == (4, 4)
    assert obs.rgb_time_ns == obs.depth_time_ns == obs.kinematics_time_ns == 200_000_000
    assert obs.pose.x == pytest.approx(0.4) and obs.collided is False
    assert obs.camera_pose_error_m == pytest.approx(0.0, abs=1e-9)


def test_missing_command_reply_is_an_error():
    sim = make_sim()
    sim._drone.hang_reply = True
    sim.command_velocity(1.0, 0.0, 0.0)
    with pytest.raises(CommandTimeoutError):
        sim.step()


def test_camera_left_behind_is_an_error_outside_collisions():
    sim = make_sim()
    sim._world.camera_lag_m = 0.5
    sim.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(CameraPoseError):
        sim.step()


def test_camera_error_on_a_collision_step_is_recorded_not_raised():
    sim = make_sim()
    sim._world.camera_lag_m = 0.5
    sim._world.pending_events = [{"type": "collision", "sim_time_ns": 200_000_000, "object_name": "obs_0003",
                                  "impact_point": {"x": 0.4, "y": 0.0, "z": -2.0}, "normal": {"x": -1.0, "y": 0.0, "z": 0.0}}]
    sim.command_velocity(2.0, 0.0, 0.0)
    sim.step()
    obs = sim.observe()
    assert obs.collided is True and obs.camera_pose_error_m == pytest.approx(0.5)


def test_missing_depth_frame_times_out():
    sim = make_sim()
    sim._world.drop_depth = True
    sim.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(FrameTimeoutError):
        sim.step()


def test_step_time_mismatch_is_an_error():
    sim = make_sim()
    sim._world.time_skew_ns = 5_000_000
    sim.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(StepTimingError):
        sim.step()


def test_stale_kinematics_are_retried_then_rejected():
    sim = make_sim(kinematics_retries=3)
    sim._world.stale_kinematics = 1
    sim.command_velocity(0.0, 0.0, 0.0)
    sim.step()
    sim._world.stale_kinematics = 10
    sim.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(StaleStateError):
        sim.step()


def test_collision_sets_flag_and_reset_clears_it():
    sim = make_sim(settle_steps=2)
    sim._world.pending_events = [{"type": "collision", "sim_time_ns": 200_000_000, "object_name": "StaticMeshActor_7",
                                  "impact_point": {"x": 0.4, "y": 0.0, "z": -2.0}, "normal": {"x": -1.0, "y": 0.0, "z": 0.0}}]
    sim.command_velocity(2.0, 0.0, 0.0)
    sim.step()
    obs = sim.observe()
    assert obs.collided is True and obs.step_collisions[0].object_name == "StaticMeshActor_7"
    steps_before = sim.steps_taken
    obs = sim.reset(Pose(-31.0, -25.0, -2.0, math.pi / 2))
    assert obs.collided is False and obs.step_collisions == ()
    assert sim.steps_taken - steps_before == 2 + 2
    zs = [p["translation"]["z"] for p in sim._drone.poses]
    assert zs == [-20.0, -20.0, -2.0]
    assert obs.pose.x == pytest.approx(-31.0) and obs.pose.yaw == pytest.approx(math.pi / 2)


def test_spawn_and_destroy():
    sim = make_sim()
    name = sim.spawn("AF_Target", "SM_Target", Pose(1.0, 2.0, -0.5, 0.0), (1.0, 1.0, 1.0), "/Game/AutoFly/Materials/M_Red")
    asset, pose, scale, physics = sim._world.spawned[name]
    assert asset == "SM_Target" and pose["translation"] == {"x": 1.0, "y": 2.0, "z": -0.5}
    assert scale == [1.0, 1.0, 1.0] and physics is False
    sim.destroy(name)
    with pytest.raises(ObjectNotFoundError):
        sim.destroy(name)


def test_invalid_dt_is_rejected():
    sim = make_sim()
    sim.command_velocity(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        sim.step(0.203)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_airsim_backend.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'autofly_ue5.sim.airsim_backend'`.

- [ ] **Step 3: Write `autofly_ue5/sim/airsim_backend.py`**

```python
"""Project AirSim implementation of the Simulator interface (lock-step, 5 Hz records)."""

from __future__ import annotations

import asyncio
import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path

from autofly_ue5.frames import body_to_ned, quat_to_yaw, yaw_to_quat
from autofly_ue5.gpu import check_gpu_for_launch, gpu_memory_mib
from autofly_ue5.paths import CONFIGS_DIR, PACKAGED_BINARY
from autofly_ue5.sim.decode import decode_depth, decode_rgb
from autofly_ue5.sim.events import CollisionLog, collisions_after
from autofly_ue5.sim.process import (
    SIM_RUN_DIR,
    SimPorts,
    SimProcess,
    instance_dir,
    launch_process,
    own_running_instances,
    packaged_command,
    ports_for_instance,
    sim_environment,
    stop,
    wait_ready,
)
from autofly_ue5.sim.sync import FrameCollector
from autofly_ue5.sim.types import CONTROL_DT_S, STEP_NS, ObjectNotFoundError, Observation, Pose, dt_to_ns

ROBOT = "Drone1"
CAMERA = "FrontCamera"
NO_EPISODE_NS = 2**62
CAMERA_RECHECK_S = 0.2


@dataclass(frozen=True)
class PasApi:
    client_cls: type
    world_cls: type
    drone_cls: type
    pose_cls: object
    yaw_mode_max_dof: int


def real_api() -> PasApi:
    from projectairsim import Drone, ProjectAirSimClient, World
    from projectairsim.drone import YawControlMode
    from projectairsim.types import Pose as PasPose

    return PasApi(ProjectAirSimClient, World, Drone, PasPose, YawControlMode.MaxDegreeOfFreedom)


class StepTimingError(RuntimeError):
    """world.step() ended at a simulator time other than the requested one."""


class StaleStateError(RuntimeError):
    """Ground-truth kinematics never reached the step time."""


class CommandTimeoutError(RuntimeError):
    """A velocity command's reply did not arrive after its step (the server measured its duration from a later tick)."""


class CameraPoseError(RuntimeError):
    """The camera pose stamped in the image disagrees with kinematics (Unreal actor left behind by a set_pose sweep)."""


class ProjectAirSimSimulator:
    def __init__(
        self,
        scene_config: str = "scene_autofly_s01.jsonc",
        config_dir: Path = CONFIGS_DIR,
        binary: Path = PACKAGED_BINARY,
        run_root: Path = SIM_RUN_DIR,
        api: PasApi | None = None,
        frame_timeout_s: float = 5.0,
        first_frame_timeout_s: float = 120.0,
        ready_timeout_s: float = 900.0,
        safe_altitude_m: float = 20.0,
        settle_steps: int = 2,
        collision_grace_s: float = 0.02,
        kinematics_retries: int = 5,
        command_timeout_s: float = 10.0,
        camera_offset_m: float = 0.40,
        camera_pose_tolerance_m: float = 0.10,
    ) -> None:
        self._scene_config = scene_config
        self._config_dir = Path(config_dir)
        self._binary = Path(binary)
        self._run_root = Path(run_root)
        self._api = api
        self._frame_timeout_s = frame_timeout_s
        self._first_frame_timeout_s = first_frame_timeout_s
        self._ready_timeout_s = ready_timeout_s
        self._safe_altitude_m = safe_altitude_m
        self._settle_steps = settle_steps
        self._collision_grace_s = collision_grace_s
        self._kinematics_retries = kinematics_retries
        self._command_timeout_s = command_timeout_s
        self._camera_offset_m = camera_offset_m
        self._camera_pose_tolerance_m = camera_pose_tolerance_m
        self._frames = FrameCollector(("rgb", "depth"))
        self._collisions = CollisionLog()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client = self._world = self._drone = None
        self._proc: SimProcess | None = None
        self._t_ns = 0
        self._pending: tuple[float, float, float] | None = None
        self._steps = 0
        self._frame_steps = 0
        self._collided = False
        self._episode_start_ns = 0
        self._last_obs: Observation | None = None

    @property
    def steps_taken(self) -> int:
        return self._steps

    def launch(self, map_path: str, instance: int) -> None:
        ports = ports_for_instance(instance)
        used, total = gpu_memory_mib()
        check_gpu_for_launch(used, total, own_running=len(own_running_instances(self._run_root)))
        directory = instance_dir(instance, self._run_root)
        directory.mkdir(parents=True, exist_ok=True)
        cmd = packaged_command(map_path, ports, directory / "sim.log", instance, binary=self._binary)
        self._proc = launch_process(cmd, instance, ports, sim_environment(editor_mode=False), run_root=self._run_root)
        try:
            wait_ready(self._proc, self._ready_timeout_s)
            self.connect(ports)
        except Exception:
            self.close()
            raise

    def connect(self, ports: SimPorts) -> None:
        api = self._api if self._api is not None else real_api()
        self._api = api
        self._loop = asyncio.new_event_loop()
        self._client = api.client_cls(port_topics=ports.topics, port_services=ports.services)
        self._client.connect()
        self._world = api.world_cls(self._client, self._scene_config, delay_after_load_sec=0, sim_config_path=str(self._config_dir))
        clock = self._world.get_sim_clock_type()
        if clock != "steppable":
            raise RuntimeError(f"scene clock is {clock!r}; the backend needs 'steppable'")
        self._drone = api.drone_cls(self._client, self._world, ROBOT)
        cams = self._drone.sensors[CAMERA]
        self._client.subscribe(cams["scene_camera"], self._frames.callback("rgb"))
        self._client.subscribe(cams["depth_planar_camera"], self._frames.callback("depth"))
        self._client.subscribe(self._drone.robot_info["collision_info"], self._collisions.topic_callback)
        self._drone.enable_api_control()
        self._drone.arm()
        self._t_ns = int(self._world.get_sim_time())

    def close(self) -> None:
        if self._client is not None:
            self._client.disconnect()
            self._client = self._world = self._drone = None
        if self._loop is not None:
            self._loop.close()
            self._loop = None
        if self._proc is not None:
            stop(self._proc.instance, run_root=self._run_root)
            self._proc = None

    def _require_connected(self) -> None:
        if self._drone is None:
            raise RuntimeError("not connected; call launch() or connect() first")

    def _pas_pose(self, pose: Pose):
        w, qx, qy, qz = yaw_to_quat(pose.yaw)
        return self._api.pose_cls({"frame_id": "DEFAULT_FRAME",
                                   "translation": {"x": float(pose.x), "y": float(pose.y), "z": float(pose.z)},
                                   "rotation": {"w": w, "x": qx, "y": qy, "z": qz}})

    def _current_pose(self) -> Pose:
        if self._last_obs is not None:
            return self._last_obs.pose
        kin = self._drone.get_ground_truth_kinematics()
        pos, ori = kin["pose"]["position"], kin["pose"]["orientation"]
        return Pose(float(pos["x"]), float(pos["y"]), float(pos["z"]), quat_to_yaw(ori["w"], ori["x"], ori["y"], ori["z"]))

    def _teleport(self, pose: Pose) -> None:
        if not self._drone.set_pose(self._pas_pose(pose), reset_kinematics=True):
            raise RuntimeError(f"set_pose({pose}) failed")

    def reset(self, pose: Pose) -> Observation:
        self._require_connected()
        self._episode_start_ns = NO_EPISODE_NS
        current = self._current_pose()
        safe_z = -abs(self._safe_altitude_m)
        for waypoint in (Pose(current.x, current.y, safe_z, current.yaw), Pose(pose.x, pose.y, safe_z, pose.yaw)):
            self._teleport(waypoint)
            self.command_velocity(0.0, 0.0, 0.0)
            self.step()
        self._teleport(pose)
        for _ in range(self._settle_steps):
            self.command_velocity(0.0, 0.0, 0.0)
            self.step()
        self._episode_start_ns = self._t_ns
        self._collided = False
        self._last_obs = dataclasses.replace(self._last_obs, collided=False, step_collisions=())
        return self._last_obs

    def spawn(self, name: str, asset: str, pose: Pose, scale: tuple[float, float, float], material: str | None = None) -> str:
        self._require_connected()
        actual = self._world.spawn_object(name, asset, self._pas_pose(pose), [float(s) for s in scale], False)
        if material is not None and not self._world.set_object_material(actual, material):
            raise RuntimeError(f"set_object_material({actual!r}, {material!r}) failed")
        return actual

    def destroy(self, name: str) -> None:
        self._require_connected()
        if not self._world.destroy_object(name):
            raise ObjectNotFoundError(f"destroy_object({name!r}) found no such object")

    def command_velocity(self, v_forward: float, yaw_rate: float, v_z: float) -> None:
        self._pending = (float(v_forward), float(yaw_rate), float(v_z))

    def _kinematics_at(self, target_ns: int) -> dict:
        kin = self._drone.get_ground_truth_kinematics()
        for _ in range(self._kinematics_retries - 1):
            if int(kin["time_stamp"]) == target_ns:
                return kin
            time.sleep(0.01)
            kin = self._drone.get_ground_truth_kinematics()
        if int(kin["time_stamp"]) != target_ns:
            raise StaleStateError(f"kinematics time_stamp {kin['time_stamp']} != step time {target_ns}")
        return kin

    def step(self, dt: float = CONTROL_DT_S) -> int:
        self._require_connected()
        dt_ns = dt_to_ns(dt, STEP_NS)
        if self._pending is None:
            raise RuntimeError("command_velocity must be called before every step")
        v_forward, yaw_rate, v_z = self._pending
        self._pending = None
        target = self._t_ns + dt_ns
        self._frames.arm(target)
        # Duration two clock ticks short of dt: the server may read the command's start one tick after Step started.
        task = self._loop.run_until_complete(self._drone.move_by_velocity_body_frame_async(
            v_forward, 0.0, -v_z, duration=(dt_ns - 2 * STEP_NS) / 1e9,
            yaw_control_mode=self._api.yaw_mode_max_dof, yaw_is_rate=True, yaw=yaw_rate))
        self._loop.run_until_complete(asyncio.sleep(0.002))
        result = self._world.step(dt_ns)
        self._steps += 1
        self._t_ns = int(result["sim_time_ns"])
        if self._t_ns != target:
            raise StepTimingError(f"step ended at {self._t_ns} ns, expected {target} ns")
        try:
            self._loop.run_until_complete(asyncio.wait_for(task, timeout=self._command_timeout_s))
        except asyncio.TimeoutError as err:
            raise CommandTimeoutError(
                f"move command reply missing {self._command_timeout_s} s after the step to {target} ns") from err
        timeout = self._first_frame_timeout_s if self._frame_steps == 0 else self._frame_timeout_s
        frames = self._frames.wait(timeout)
        self._frame_steps += 1
        kin = self._kinematics_at(target)
        if self._collision_grace_s > 0:
            time.sleep(self._collision_grace_s)
        events = self._collisions.collect(result["robots"][ROBOT].get("events", []), target)
        new_events = collisions_after(events, self._episode_start_ns)
        self._collided = self._collided or bool(new_events)
        pos, ori = kin["pose"]["position"], kin["pose"]["orientation"]
        lin, ang = kin["twist"]["linear"], kin["twist"]["angular"]
        mount = body_to_ned(float(ori["w"]), float(ori["x"]), float(ori["y"]), float(ori["z"]), (self._camera_offset_m, 0.0, 0.0))
        rgb_msg = frames["rgb"]
        camera_error = math.dist(
            (float(rgb_msg["pos_x"]), float(rgb_msg["pos_y"]), float(rgb_msg["pos_z"])),
            (float(pos["x"]) + mount[0], float(pos["y"]) + mount[1], float(pos["z"]) + mount[2]),
        )
        if camera_error > self._camera_pose_tolerance_m and not new_events:
            # The sweep hit that stopped the actor may still be on its way as a collision_info message: look once more.
            time.sleep(CAMERA_RECHECK_S)
            new_events = collisions_after(self._collisions.collect([], target), self._episode_start_ns)
            self._collided = self._collided or bool(new_events)
        if camera_error > self._camera_pose_tolerance_m and not new_events:
            raise CameraPoseError(
                f"camera at step {target} ns is {camera_error:.3f} m from kinematics + mount offset "
                f"(tolerance {self._camera_pose_tolerance_m} m); the Unreal actor was probably stopped by a sweep")
        self._last_obs = Observation(
            rgb=decode_rgb(rgb_msg),
            depth=decode_depth(frames["depth"]),
            pose=Pose(float(pos["x"]), float(pos["y"]), float(pos["z"]), quat_to_yaw(ori["w"], ori["x"], ori["y"], ori["z"])),
            velocity_ned=(float(lin["x"]), float(lin["y"]), float(lin["z"])),
            yaw_rate=float(ang["z"]),
            sim_time_ns=target,
            collided=self._collided,
            step_collisions=new_events,
            rgb_time_ns=int(rgb_msg["time_stamp"]),
            depth_time_ns=int(frames["depth"]["time_stamp"]),
            kinematics_time_ns=int(kin["time_stamp"]),
            camera_pose_error_m=camera_error,
        )
        return target

    def observe(self) -> Observation:
        if self._last_obs is None:
            raise RuntimeError("no observation before reset() or step()")
        return self._last_obs
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_airsim_backend.py`
Expected: `11 passed`.

- [ ] **Step 5: Confirm the import boundary and the whole suite**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && grep -rln "projectairsim" autofly_ue5 | grep -v "^autofly_ue5/sim/" ; env -u PYTHONPATH .venv/bin/python -m pytest`
Expected: grep prints nothing (only `autofly_ue5/sim/` mentions `projectairsim`); all tests pass (`83 passed`).

- [ ] **Step 6: Commit**

```bash
git -C /home/nvidiasims/research_uav/autofly_ue5 add autofly_ue5/sim/airsim_backend.py tests/test_airsim_backend.py && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Add the lock-step Project AirSim backend of the Simulator interface" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Scene schema, asset registry, `jittered_grid` generator and the s01 scene file

**Files:**
- Create: `autofly_ue5/scenes/__init__.py`, `autofly_ue5/scenes/scene.schema.json`, `autofly_ue5/scenes/model.py`, `autofly_ue5/scenes/generate.py`, `assets/registry.json`, `scenes/s01_white_pillars.json`
- Test: `tests/test_scene_model.py`, `tests/test_generate.py`

**Interfaces:**
- Consumes: `autofly_ue5.paths.{ASSET_REGISTRY, SCENES_DIR}`.
- Produces:
  - `autofly_ue5.scenes.model`: `SceneFileError(ValueError)`; frozen dataclasses `Bounds(x_min, x_max, y_min, y_max)` (+ `width`, `height` properties), `ObstacleGroup(asset: str, count: int, scale_xy: tuple[float, float], scale_z: tuple[float, float], palette: tuple[str, ...], placement: dict)`, `SceneFile(id, split, seed, bounds: Bounds, ground: str, obstacle_groups: tuple[ObstacleGroup, ...], start_band: tuple[float, float], target_band: tuple[float, float], altitude_band: tuple[float, float], instruction_obstacle: str, sha256: str, path: str)`, `AssetEntry(name, ue_path, base_size_m: tuple[float, float, float], pivot, footprint, category, role, seen: bool | None, measured_extent_cm_at_unit_scale: tuple[float, float, float] | None)` (`seen` is `null` for obstacle and ground assets and `true`/`false` for targets from M4; the measured extent is UE's bounds half-extent divided by the actor scale, written by Task 14 Step 11), `MaterialEntry(name, kind, ue_path, parent: str | None, parameter: str | None, rgba: tuple[float, float, float, float] | None)`, `AssetRegistry(assets: dict[str, AssetEntry], materials: dict[str, MaterialEntry])`, `Instance(tag, asset, x, y, z_center, yaw, scale: tuple[float, float, float], material, radius_m, height_m)`, `Layout(scene_id, seed, bounds: Bounds, instances: tuple[Instance, ...])` with `to_json() -> dict`; functions `load_scene_file(path: Path) -> SceneFile` (schema plus: bounds must be 70 × 70 m centred on the origin, spec §6.1), `load_registry(path: Path = ASSET_REGISTRY) -> AssetRegistry`.
  - `autofly_ue5.scenes.generate`: `UnsupportedPlacementError(ValueError)`; `jittered_grid(rng: random.Random, bounds: Bounds, count: int, margin_m: float, jitter_m: float) -> list[tuple[float, float]]`; `generate_layout(scene: SceneFile, registry: AssetRegistry, seed: int | None = None) -> Layout`.
  - Instances are NED metres: `z_center = −height_m / 2` (standing on the ground at z = 0), tags `obs_0000`, `obs_0001`, …

- [ ] **Step 1: Write the data files**

`/home/nvidiasims/research_uav/autofly_ue5/assets/registry.json`:
```json
{
  "version": 1,
  "assets": {
    "cylinder": {
      "ue_path": "/Engine/BasicShapes/Cylinder",
      "base_size_m": [1.0, 1.0, 1.0],
      "pivot": "center",
      "footprint": "circle",
      "category": "geometry",
      "role": "obstacle",
      "seen": null,
      "measured_extent_cm_at_unit_scale": null,
      "source": "Unreal Engine 5.7.4 engine content (BasicShapes)",
      "licence": "Unreal Engine EULA",
      "size_and_pivot_verified_by": "scripts/build_level.sh bounds assertion (Task 14)"
    },
    "cube": {
      "ue_path": "/Engine/BasicShapes/Cube",
      "base_size_m": [1.0, 1.0, 1.0],
      "pivot": "center",
      "footprint": "none",
      "category": "geometry",
      "role": "ground",
      "seen": null,
      "measured_extent_cm_at_unit_scale": null,
      "source": "Unreal Engine 5.7.4 engine content (BasicShapes)",
      "licence": "Unreal Engine EULA",
      "size_and_pivot_verified_by": "scripts/build_level.sh bounds assertion (Task 14)"
    }
  },
  "materials": {
    "grid": { "kind": "engine", "ue_path": "/Engine/EngineMaterials/WorldGridMaterial" },
    "white": {
      "kind": "color_instance",
      "ue_path": "/Game/AutoFly/Materials/MI_White",
      "parent": "/Engine/BasicShapes/BasicShapeMaterial",
      "parameter": "Color",
      "rgba": [0.9, 0.9, 0.9, 1.0]
    }
  }
}
```

`/home/nvidiasims/research_uav/autofly_ue5/scenes/s01_white_pillars.json`:
```json
{
  "id": "s01",
  "split": "train",
  "seed": 1001,
  "bounds": { "x_min": -35.0, "x_max": 35.0, "y_min": -35.0, "y_max": 35.0 },
  "ground": "grid",
  "obstacle_groups": [
    {
      "asset": "cylinder",
      "count": 80,
      "scale_range": { "xy": [0.8, 1.2], "z": [8.0, 12.0] },
      "palette": ["white"],
      "placement": { "type": "jittered_grid", "margin_m": 8.0, "jitter_m": 1.0 }
    }
  ],
  "start_band": [2.0, 6.0],
  "target_band": [0.0, 3.0],
  "altitude_band": [1.0, 3.0],
  "instruction_obstacle": "white pillars"
}
```
(80 pillars of 0.8–1.2 m diameter and 8–12 m height on a 9 × 9 grid of 6 m cells inside an 8 m margin, so starts and targets near the boundary are clear and pillars are taller than the altitude band. The schema has no `$id`, so its internal `#/$defs/...` references resolve against the document itself with any jsonschema version.)

- [ ] **Step 2: Write the schema `autofly_ue5/scenes/scene.schema.json`**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AutoFly UE5 scene file (spec 6.1)",
  "type": "object",
  "additionalProperties": false,
  "required": ["id", "split", "seed", "bounds", "ground", "obstacle_groups", "start_band", "target_band", "altitude_band", "instruction_obstacle"],
  "properties": {
    "id": { "type": "string", "pattern": "^s[0-9]{2}r?$" },
    "split": { "enum": ["train", "test_seen", "test_unseen"] },
    "seed": { "type": "integer", "minimum": 0 },
    "bounds": {
      "type": "object",
      "additionalProperties": false,
      "required": ["x_min", "x_max", "y_min", "y_max"],
      "properties": {
        "x_min": { "type": "number" }, "x_max": { "type": "number" },
        "y_min": { "type": "number" }, "y_max": { "type": "number" }
      }
    },
    "ground": { "type": "string", "minLength": 1 },
    "obstacle_groups": { "type": "array", "minItems": 1, "items": { "$ref": "#/$defs/group" } },
    "start_band": { "$ref": "#/$defs/range" },
    "target_band": { "$ref": "#/$defs/range" },
    "altitude_band": { "$ref": "#/$defs/range" },
    "instruction_obstacle": { "type": "string", "minLength": 1 }
  },
  "$defs": {
    "range": { "type": "array", "items": { "type": "number", "minimum": 0 }, "minItems": 2, "maxItems": 2 },
    "int_range": { "type": "array", "items": { "type": "integer", "minimum": 1 }, "minItems": 2, "maxItems": 2 },
    "group": {
      "type": "object",
      "additionalProperties": false,
      "required": ["asset", "count", "scale_range", "palette", "placement"],
      "properties": {
        "asset": { "type": "string", "minLength": 1 },
        "count": { "type": "integer", "minimum": 1 },
        "scale_range": {
          "type": "object",
          "additionalProperties": false,
          "required": ["xy", "z"],
          "properties": { "xy": { "$ref": "#/$defs/range" }, "z": { "$ref": "#/$defs/range" } }
        },
        "palette": { "type": "array", "minItems": 1, "items": { "type": "string", "minLength": 1 } },
        "placement": {
          "oneOf": [
            { "$ref": "#/$defs/jittered_grid" },
            { "$ref": "#/$defs/poisson" },
            { "$ref": "#/$defs/clusters" },
            { "$ref": "#/$defs/stacks" }
          ]
        }
      }
    },
    "jittered_grid": {
      "type": "object", "additionalProperties": false, "required": ["type", "margin_m", "jitter_m"],
      "properties": { "type": { "const": "jittered_grid" }, "margin_m": { "type": "number", "minimum": 0 }, "jitter_m": { "type": "number", "minimum": 0 } }
    },
    "poisson": {
      "type": "object", "additionalProperties": false, "required": ["type", "margin_m", "min_distance_m"],
      "properties": { "type": { "const": "poisson" }, "margin_m": { "type": "number", "minimum": 0 }, "min_distance_m": { "type": "number", "exclusiveMinimum": 0 } }
    },
    "clusters": {
      "type": "object", "additionalProperties": false, "required": ["type", "margin_m", "cluster_count", "per_cluster", "radius_m"],
      "properties": { "type": { "const": "clusters" }, "margin_m": { "type": "number", "minimum": 0 }, "cluster_count": { "type": "integer", "minimum": 1 },
                      "per_cluster": { "$ref": "#/$defs/int_range" }, "radius_m": { "type": "number", "exclusiveMinimum": 0 } }
    },
    "stacks": {
      "type": "object", "additionalProperties": false, "required": ["type", "margin_m", "stack_count", "height_range"],
      "properties": { "type": { "const": "stacks" }, "margin_m": { "type": "number", "minimum": 0 }, "stack_count": { "type": "integer", "minimum": 1 },
                      "height_range": { "$ref": "#/$defs/int_range" } }
    }
  }
}
```

- [ ] **Step 3: Write the failing tests**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_scene_model.py`:
```python
import json

import pytest

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.model import SceneFileError, load_registry, load_scene_file

S01 = SCENES_DIR / "s01_white_pillars.json"


def test_load_s01():
    scene = load_scene_file(S01)
    assert scene.id == "s01" and scene.split == "train" and scene.seed == 1001
    assert (scene.bounds.width, scene.bounds.height) == (70.0, 70.0)
    assert scene.start_band == (2.0, 6.0) and scene.target_band == (0.0, 3.0) and scene.altitude_band == (1.0, 3.0)
    group = scene.obstacle_groups[0]
    assert group.asset == "cylinder" and group.count == 80 and group.palette == ("white",)
    assert group.scale_xy == (0.8, 1.2) and group.scale_z == (8.0, 12.0)
    assert group.placement == {"type": "jittered_grid", "margin_m": 8.0, "jitter_m": 1.0}
    assert len(scene.sha256) == 64 and scene.instruction_obstacle == "white pillars"


def _variant(tmp_path, mutate):
    data = json.loads(S01.read_text())
    mutate(data)
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(data))
    return path


def test_schema_rejects_missing_field(tmp_path):
    with pytest.raises(SceneFileError, match="instruction_obstacle"):
        load_scene_file(_variant(tmp_path, lambda d: d.pop("instruction_obstacle")))


def test_schema_rejects_unknown_placement(tmp_path):
    def mutate(d):
        d["obstacle_groups"][0]["placement"] = {"type": "spiral", "margin_m": 1.0}
    with pytest.raises(SceneFileError):
        load_scene_file(_variant(tmp_path, mutate))


def test_inverted_range_is_rejected(tmp_path):
    with pytest.raises(SceneFileError, match="start_band"):
        load_scene_file(_variant(tmp_path, lambda d: d.update(start_band=[6.0, 2.0])))


def test_inverted_bounds_are_rejected(tmp_path):
    with pytest.raises(SceneFileError, match="bounds"):
        load_scene_file(_variant(tmp_path, lambda d: d["bounds"].update(x_min=40.0)))


def test_off_centre_or_resized_bounds_are_rejected(tmp_path):
    with pytest.raises(SceneFileError, match="70"):
        load_scene_file(_variant(tmp_path, lambda d: d["bounds"].update(x_min=-30.0, x_max=40.0)))
    with pytest.raises(SceneFileError, match="70"):
        load_scene_file(_variant(tmp_path, lambda d: d.update(bounds={"x_min": -50.0, "x_max": 50.0, "y_min": -50.0, "y_max": 50.0})))


def test_load_registry():
    registry = load_registry()
    cyl = registry.assets["cylinder"]
    assert cyl.ue_path == "/Engine/BasicShapes/Cylinder" and cyl.base_size_m == (1.0, 1.0, 1.0)
    assert cyl.pivot == "center" and cyl.footprint == "circle" and cyl.seen is None
    measured = cyl.measured_extent_cm_at_unit_scale  # null until Task 14 Step 11 writes the live measurement
    assert measured is None or (len(measured) == 3 and all(abs(v - 50.0) <= 1.0 for v in measured))
    white = registry.materials["white"]
    assert white.kind == "color_instance" and white.parameter == "Color" and white.rgba == (0.9, 0.9, 0.9, 1.0)
    assert registry.materials["grid"].parent is None
```

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_generate.py`:
```python
import itertools
import json
import math
import random

import pytest

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.generate import UnsupportedPlacementError, generate_layout, jittered_grid
from autofly_ue5.scenes.model import Bounds, load_registry, load_scene_file

S01 = SCENES_DIR / "s01_white_pillars.json"
BOUNDS = Bounds(-35.0, 35.0, -35.0, 35.0)


def test_jittered_grid_count_margin_and_spacing():
    points = jittered_grid(random.Random(7), BOUNDS, 80, margin_m=8.0, jitter_m=1.0)
    assert len(points) == 80
    for x, y in points:
        assert -28.0 <= x <= 28.0 and -28.0 <= y <= 28.0
    closest = min(math.dist(a, b) for a, b in itertools.combinations(points, 2))
    assert closest >= 4.0 - 1e-9  # 6 m cells minus twice the 1 m jitter


def test_jittered_grid_is_deterministic():
    a = jittered_grid(random.Random(3), BOUNDS, 20, margin_m=8.0, jitter_m=1.0)
    b = jittered_grid(random.Random(3), BOUNDS, 20, margin_m=8.0, jitter_m=1.0)
    assert a == b


def test_jitter_larger_than_half_cell_is_rejected():
    with pytest.raises(ValueError, match="jitter"):
        jittered_grid(random.Random(1), BOUNDS, 80, margin_m=8.0, jitter_m=3.5)


def test_generate_s01_layout():
    scene, registry = load_scene_file(S01), load_registry()
    layout = generate_layout(scene, registry)
    assert layout.scene_id == "s01" and layout.seed == 1001 and len(layout.instances) == 80
    assert [i.tag for i in layout.instances[:2]] == ["obs_0000", "obs_0001"]
    assert len({i.tag for i in layout.instances}) == 80
    for inst in layout.instances:
        assert 0.4 <= inst.radius_m <= 0.6 and 8.0 <= inst.height_m <= 12.0
        assert inst.z_center == pytest.approx(-inst.height_m / 2, abs=1e-3)
        assert inst.scale[0] == inst.scale[1] and inst.material == "white" and inst.asset == "cylinder"
    assert generate_layout(scene, registry) == layout
    assert generate_layout(scene, registry, seed=2002) != layout
    assert json.loads(json.dumps(layout.to_json()))["instances"][0]["tag"] == "obs_0000"


def test_unimplemented_placement_is_explicit(tmp_path):
    data = json.loads(S01.read_text())
    data["obstacle_groups"][0]["placement"] = {"type": "poisson", "margin_m": 8.0, "min_distance_m": 3.0}
    path = tmp_path / "poisson.json"
    path.write_text(json.dumps(data))
    with pytest.raises(UnsupportedPlacementError, match="poisson"):
        generate_layout(load_scene_file(path), load_registry())
```

- [ ] **Step 4: Run them to verify they fail**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_scene_model.py tests/test_generate.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'autofly_ue5.scenes.model'`.

- [ ] **Step 5: Write `autofly_ue5/scenes/__init__.py` and `autofly_ue5/scenes/model.py`**

`autofly_ue5/scenes/__init__.py`:
```python
"""Scene files, layout generation, reachability and level specs (spec §6)."""
```

`autofly_ue5/scenes/model.py`:
```python
"""Scene file and asset registry loading; typed layout objects. Coordinates are NED metres."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jsonschema

from autofly_ue5.paths import ASSET_REGISTRY

SCHEMA_PATH = Path(__file__).with_name("scene.schema.json")


class SceneFileError(ValueError):
    """A scene file failed schema or consistency validation."""


@dataclass(frozen=True)
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class ObstacleGroup:
    asset: str
    count: int
    scale_xy: tuple[float, float]
    scale_z: tuple[float, float]
    palette: tuple[str, ...]
    placement: dict


@dataclass(frozen=True)
class SceneFile:
    id: str
    split: str
    seed: int
    bounds: Bounds
    ground: str
    obstacle_groups: tuple[ObstacleGroup, ...]
    start_band: tuple[float, float]
    target_band: tuple[float, float]
    altitude_band: tuple[float, float]
    instruction_obstacle: str
    sha256: str
    path: str


@dataclass(frozen=True)
class AssetEntry:
    name: str
    ue_path: str
    base_size_m: tuple[float, float, float]
    pivot: str
    footprint: str
    category: str
    role: str
    seen: bool | None
    measured_extent_cm_at_unit_scale: tuple[float, float, float] | None


@dataclass(frozen=True)
class MaterialEntry:
    name: str
    kind: str
    ue_path: str
    parent: str | None
    parameter: str | None
    rgba: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class AssetRegistry:
    assets: dict[str, AssetEntry]
    materials: dict[str, MaterialEntry]


@dataclass(frozen=True)
class Instance:
    tag: str
    asset: str
    x: float
    y: float
    z_center: float
    yaw: float
    scale: tuple[float, float, float]
    material: str
    radius_m: float
    height_m: float


@dataclass(frozen=True)
class Layout:
    scene_id: str
    seed: int
    bounds: Bounds
    instances: tuple[Instance, ...]

    def to_json(self) -> dict:
        return {"scene_id": self.scene_id, "seed": self.seed, "bounds": asdict(self.bounds),
                "instances": [asdict(i) for i in self.instances]}


def _range(values: list, name: str) -> tuple[float, float]:
    low, high = float(values[0]), float(values[1])
    if low > high:
        raise SceneFileError(f"{name}: minimum {low} is greater than maximum {high}")
    return (low, high)


def load_scene_file(path: Path) -> SceneFile:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    schema = json.loads(SCHEMA_PATH.read_text())
    try:
        jsonschema.validate(data, schema, cls=jsonschema.Draft202012Validator)
    except jsonschema.ValidationError as err:
        location = "/".join(str(p) for p in err.absolute_path) or "<root>"
        raise SceneFileError(f"{path}: {location}: {err.message}") from err
    b = data["bounds"]
    bounds = Bounds(float(b["x_min"]), float(b["x_max"]), float(b["y_min"]), float(b["y_max"]))
    if bounds.width <= 0 or bounds.height <= 0:
        raise SceneFileError(f"{path}: bounds must have x_min < x_max and y_min < y_max")
    if bounds.x_min != -bounds.x_max or bounds.y_min != -bounds.y_max or bounds.width != 70.0 or bounds.height != 70.0:
        raise SceneFileError(f"{path}: bounds must be 70 x 70 m centred on the origin (spec 6.1), got {b}")
    groups = tuple(
        ObstacleGroup(
            asset=g["asset"], count=int(g["count"]),
            scale_xy=_range(g["scale_range"]["xy"], f"obstacle_groups[{i}].scale_range.xy"),
            scale_z=_range(g["scale_range"]["z"], f"obstacle_groups[{i}].scale_range.z"),
            palette=tuple(g["palette"]), placement=dict(g["placement"]),
        )
        for i, g in enumerate(data["obstacle_groups"])
    )
    return SceneFile(
        id=data["id"], split=data["split"], seed=int(data["seed"]), bounds=bounds, ground=data["ground"],
        obstacle_groups=groups,
        start_band=_range(data["start_band"], "start_band"),
        target_band=_range(data["target_band"], "target_band"),
        altitude_band=_range(data["altitude_band"], "altitude_band"),
        instruction_obstacle=data["instruction_obstacle"],
        sha256=hashlib.sha256(raw).hexdigest(), path=str(path),
    )


def load_registry(path: Path = ASSET_REGISTRY) -> AssetRegistry:
    data = json.loads(Path(path).read_text())
    assets = {
        name: AssetEntry(
            name=name, ue_path=a["ue_path"], base_size_m=tuple(float(v) for v in a["base_size_m"]),
            pivot=a["pivot"], footprint=a["footprint"], category=a["category"], role=a["role"], seen=a["seen"],
            measured_extent_cm_at_unit_scale=(tuple(float(v) for v in a["measured_extent_cm_at_unit_scale"])
                                              if a["measured_extent_cm_at_unit_scale"] is not None else None),
        )
        for name, a in data["assets"].items()
    }
    materials = {
        name: MaterialEntry(name=name, kind=m["kind"], ue_path=m["ue_path"], parent=m.get("parent"),
                            parameter=m.get("parameter"),
                            rgba=tuple(float(v) for v in m["rgba"]) if "rgba" in m else None)
        for name, m in data["materials"].items()
    }
    return AssetRegistry(assets=assets, materials=materials)
```

- [ ] **Step 6: Write `autofly_ue5/scenes/generate.py`**

```python
"""Seeded expansion of a scene file into concrete obstacle instances (spec §6.2)."""

from __future__ import annotations

import math
import random

from autofly_ue5.scenes.model import AssetRegistry, Bounds, Instance, Layout, SceneFile


class UnsupportedPlacementError(ValueError):
    """The placement type or asset footprint is not implemented in Plan 1 (only jittered_grid circles)."""


def jittered_grid(rng: random.Random, bounds: Bounds, count: int, margin_m: float, jitter_m: float) -> list[tuple[float, float]]:
    x0, x1 = bounds.x_min + margin_m, bounds.x_max - margin_m
    y0, y1 = bounds.y_min + margin_m, bounds.y_max - margin_m
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError(f"margin {margin_m} m leaves no placement area")
    cols = max(1, math.ceil(math.sqrt(count * width / height)))
    rows = math.ceil(count / cols)
    cell_w, cell_h = width / cols, height / rows
    if jitter_m > min(cell_w, cell_h) / 2:
        raise ValueError(f"jitter {jitter_m} m exceeds half a grid cell ({min(cell_w, cell_h) / 2:.2f} m)")
    cells = sorted(rng.sample([(r, c) for r in range(rows) for c in range(cols)], count))
    return [
        (x0 + (c + 0.5) * cell_w + rng.uniform(-jitter_m, jitter_m), y0 + (r + 0.5) * cell_h + rng.uniform(-jitter_m, jitter_m))
        for r, c in cells
    ]


def generate_layout(scene: SceneFile, registry: AssetRegistry, seed: int | None = None) -> Layout:
    used_seed = scene.seed if seed is None else seed
    rng = random.Random(used_seed)
    instances: list[Instance] = []
    for group in scene.obstacle_groups:
        if group.asset not in registry.assets:
            raise ValueError(f"asset {group.asset!r} is not in the registry")
        asset = registry.assets[group.asset]
        if asset.footprint != "circle" or asset.pivot != "center":
            raise UnsupportedPlacementError(f"asset {group.asset!r} footprint={asset.footprint} pivot={asset.pivot} is not supported in Plan 1")
        for material in group.palette:
            if material not in registry.materials:
                raise ValueError(f"material {material!r} is not in the registry")
        placement = group.placement["type"]
        if placement != "jittered_grid":
            raise UnsupportedPlacementError(f"placement '{placement}' is not implemented in Plan 1 (planned for M4)")
        points = jittered_grid(rng, scene.bounds, group.count, group.placement["margin_m"], group.placement["jitter_m"])
        for x, y in points:
            s_xy = round(rng.uniform(*group.scale_xy), 4)
            s_z = round(rng.uniform(*group.scale_z), 4)
            material = rng.choice(group.palette)
            height = round(asset.base_size_m[2] * s_z, 4)
            instances.append(Instance(
                tag=f"obs_{len(instances):04d}", asset=group.asset, x=round(x, 4), y=round(y, 4),
                z_center=round(-height / 2.0, 4), yaw=0.0, scale=(s_xy, s_xy, s_z), material=material,
                radius_m=round(asset.base_size_m[0] * s_xy / 2.0, 4), height_m=height,
            ))
    return Layout(scene_id=scene.id, seed=used_seed, bounds=scene.bounds, instances=tuple(instances))
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_scene_model.py tests/test_generate.py`
Expected: `12 passed`.

- [ ] **Step 8: Commit**

```bash
git -C /home/nvidiasims/research_uav/autofly_ue5 add autofly_ue5/scenes/__init__.py autofly_ue5/scenes/scene.schema.json autofly_ue5/scenes/model.py autofly_ue5/scenes/generate.py assets/registry.json scenes/s01_white_pillars.json tests/test_scene_model.py tests/test_generate.py && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Add the scene file schema, the s01 white-pillar scene, the primitive asset registry and the jittered_grid generator" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Reachability check and the scene build CLI

**Files:**
- Create: `autofly_ue5/scenes/reachability.py`, `scripts/build_scenes.py`
- Test: `tests/test_reachability.py`

**Interfaces:**
- Consumes: `autofly_ue5.scenes.model.{Bounds, Instance, Layout, load_scene_file, load_registry}`, `autofly_ue5.scenes.generate.generate_layout`, `autofly_ue5.paths.RUNS_DIR`.
- Produces:
  - `autofly_ue5.scenes.reachability`: `EDGES = ("x_min", "x_max", "y_min", "y_max")`; `@dataclass(frozen=True) ReachabilityResult(ok: bool, resolution_m: float, inflate_m: float, free_start_cells: int, free_target_cells: int, unreachable_start_cells: int, crossing_unreachable_cells: dict[str, int], reason: str)` with `to_json() -> dict`; `cell_centers(bounds: Bounds, resolution_m: float) -> tuple[np.ndarray, np.ndarray]`; `occupancy(layout: Layout, resolution_m: float, inflate_m: float) -> np.ndarray` (bool, index `[ix, iy]`); `edge_distances(bounds: Bounds, resolution_m: float) -> dict[str, np.ndarray]` (distance of every cell centre to each edge); `band_mask(bounds: Bounds, resolution_m: float, d_min: float, d_max: float) -> np.ndarray`; `reachable_from(free: np.ndarray, seeds: np.ndarray) -> np.ndarray` (4-connected); `crossing_unreachable(free: np.ndarray, bounds: Bounds, resolution_m: float, start_band: tuple[float, float], target_band: tuple[float, float]) -> dict[str, int]`; `check_reachability(layout: Layout, start_band: tuple[float, float], target_band: tuple[float, float], clearance_m: float = 1.0, drone_radius_m: float = 0.4, resolution_m: float = 0.25) -> ReachabilityResult`.
  - Rule (spec §6.2): a cell is blocked when its centre is closer than `radius + drone_radius_m + clearance_m` to an obstacle centre; the spec rule holds when there is at least one free start cell and one free target cell and **every** free start-band cell reaches some free target-band cell.
  - Crossing rule (added on top, see Global Constraints): for each edge, the corridor is the free cells farther than `start_band[1]` from the two perpendicular edges; every free start-band cell of that edge inside the corridor must reach a free target-band cell of the opposite edge inside the same corridor. Measured on the plan's own code: the spec rule alone gives `unreachable_start_cells = 0` both for s01 (seed 1001) and for an empty layout (15872 free start cells, 12864 free target cells each), so it cannot tell them apart; a full-width wall across x = 0 passes the spec rule but leaves 3712 start cells on each of the `x_min` and `x_max` edges unable to cross, while s01 crosses from all four edges.
  - The layout is accepted iff both rules hold.
  - CLI: `python scripts/build_scenes.py <scene.json> [--out-dir runs/levels]` writes `<out-dir>/<id>.layout.json` (`scene_path`, `scene_sha256`, `layout`, `reachability`); exit 1 if rejected. (Task 14 extends it to also write the level spec.)

- [ ] **Step 1: Write the failing test**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_reachability.py`:
```python
import numpy as np

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.generate import generate_layout
from autofly_ue5.scenes.model import Bounds, Instance, Layout, load_registry, load_scene_file
from autofly_ue5.scenes.reachability import (
    EDGES,
    band_mask,
    cell_centers,
    check_reachability,
    occupancy,
    reachable_from,
)

BOUNDS = Bounds(-35.0, 35.0, -35.0, 35.0)


def _post(tag: str, x: float, y: float, radius: float = 0.2) -> Instance:
    return Instance(tag=tag, asset="cylinder", x=x, y=y, z_center=-5.0, yaw=0.0, scale=(2 * radius, 2 * radius, 10.0),
                    material="white", radius_m=radius, height_m=10.0)


def test_cell_centers_and_band_mask():
    xs, ys = cell_centers(BOUNDS, 0.5)
    assert len(xs) == 140 and xs[0] == -34.75 and ys[-1] == 34.75
    band = band_mask(BOUNDS, 0.5, 2.0, 6.0)
    ix = int(np.argmin(np.abs(xs - (-31.0))))
    iy = int(np.argmin(np.abs(ys - 0.0)))
    assert band[ix, iy]  # about 4 m inside the south (x_min) edge
    assert not band[int(np.argmin(np.abs(xs - 0.0))), iy]  # centre of the scene


def test_occupancy_inflates_by_radius():
    layout = Layout("t", 0, BOUNDS, (_post("obs_0000", 0.0, 0.0, radius=0.5),))
    occ = occupancy(layout, 0.25, inflate_m=0.0)
    xs, ys = cell_centers(BOUNDS, 0.25)
    assert occ[int(np.argmin(np.abs(xs - 0.125))), int(np.argmin(np.abs(ys - 0.125)))]
    assert not occ[int(np.argmin(np.abs(xs - 1.125))), int(np.argmin(np.abs(ys - 0.125)))]
    assert occupancy(layout, 0.25, inflate_m=1.4).sum() > occ.sum()


def test_reachable_from_grows_through_free_cells_only():
    free = np.ones((5, 5), dtype=bool)
    free[2, :] = False
    seeds = np.zeros((5, 5), dtype=bool)
    seeds[0, 0] = True
    reach = reachable_from(free, seeds)
    assert reach[1, 4] and not reach[3, 0]


def test_empty_layout_is_reachable():
    result = check_reachability(Layout("t", 0, BOUNDS, ()), (2.0, 6.0), (0.0, 3.0))
    assert result.ok and result.unreachable_start_cells == 0 and result.free_start_cells > 0
    assert result.crossing_unreachable_cells == {edge: 0 for edge in EDGES}


def test_full_width_wall_passes_the_spec_rule_but_fails_the_crossing_rule():
    wall = tuple(_post(f"w{k}", 0.0, -35.0 + 0.25 * k) for k in range(281))  # x = 0, from y = -35 to y = 35
    result = check_reachability(Layout("t", 0, BOUNDS, wall), (2.0, 6.0), (0.0, 3.0))
    assert result.unreachable_start_cells == 0  # every start cell still reaches its own edge's target ring
    assert not result.ok and "cross" in result.reason
    assert result.crossing_unreachable_cells["x_min"] > 0 and result.crossing_unreachable_cells["x_max"] > 0
    assert result.crossing_unreachable_cells["y_min"] == 0 and result.crossing_unreachable_cells["y_max"] == 0


def test_enclosed_start_pocket_is_rejected():
    posts = []
    for k in range(0, 41):  # vertical walls at x=-33 and x=-25 from y=-5 to y=5
        y = -5.0 + 0.25 * k
        posts += [_post(f"w{k}a", -33.0, y), _post(f"w{k}b", -25.0, y)]
    for k in range(0, 33):  # horizontal walls at y=-5 and y=5 from x=-33 to x=-25
        x = -33.0 + 0.25 * k
        posts += [_post(f"h{k}a", x, -5.0), _post(f"h{k}b", x, 5.0)]
    result = check_reachability(Layout("t", 0, BOUNDS, tuple(posts)), (2.0, 6.0), (0.0, 3.0))
    assert not result.ok and result.unreachable_start_cells > 0
    assert "cannot reach" in result.reason


def test_s01_layout_is_reachable():
    scene = load_scene_file(SCENES_DIR / "s01_white_pillars.json")
    layout = generate_layout(scene, load_registry())
    result = check_reachability(layout, scene.start_band, scene.target_band)
    assert result.ok, result.reason
    assert result.inflate_m == 1.4
    assert result.crossing_unreachable_cells == {edge: 0 for edge in EDGES}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_reachability.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'autofly_ue5.scenes.reachability'`.

- [ ] **Step 3: Write `autofly_ue5/scenes/reachability.py`**

```python
"""2D occupancy reachability (spec §6.2) plus a crossing rule.

Spec rule: every free start-band cell must reach a free target-band cell. Crossing rule: every free start-band cell
must also reach the target band of the opposite edge through the obstacle field (a corridor that excludes the
perimeter ring along the two perpendicular edges). The check only guarantees solvable layouts; it never produces actions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from autofly_ue5.scenes.model import Bounds, Layout

EDGES = ("x_min", "x_max", "y_min", "y_max")
OPPOSITE = {"x_min": "x_max", "x_max": "x_min", "y_min": "y_max", "y_max": "y_min"}
PERPENDICULAR = {"x_min": ("y_min", "y_max"), "x_max": ("y_min", "y_max"), "y_min": ("x_min", "x_max"),
                 "y_max": ("x_min", "x_max")}


@dataclass(frozen=True)
class ReachabilityResult:
    ok: bool
    resolution_m: float
    inflate_m: float
    free_start_cells: int
    free_target_cells: int
    unreachable_start_cells: int
    crossing_unreachable_cells: dict[str, int]
    reason: str

    def to_json(self) -> dict:
        return asdict(self)


def cell_centers(bounds: Bounds, resolution_m: float) -> tuple[np.ndarray, np.ndarray]:
    nx = int(round(bounds.width / resolution_m))
    ny = int(round(bounds.height / resolution_m))
    xs = bounds.x_min + (np.arange(nx) + 0.5) * resolution_m
    ys = bounds.y_min + (np.arange(ny) + 0.5) * resolution_m
    return xs, ys


def occupancy(layout: Layout, resolution_m: float, inflate_m: float) -> np.ndarray:
    b = layout.bounds
    xs, ys = cell_centers(b, resolution_m)
    occ = np.zeros((len(xs), len(ys)), dtype=bool)
    for inst in layout.instances:
        r = inst.radius_m + inflate_m
        ix0 = max(0, int(math.floor((inst.x - r - b.x_min) / resolution_m)))
        ix1 = min(len(xs), int(math.ceil((inst.x + r - b.x_min) / resolution_m)) + 1)
        iy0 = max(0, int(math.floor((inst.y - r - b.y_min) / resolution_m)))
        iy1 = min(len(ys), int(math.ceil((inst.y + r - b.y_min) / resolution_m)) + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        gx, gy = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1], indexing="ij")
        occ[ix0:ix1, iy0:iy1] |= (gx - inst.x) ** 2 + (gy - inst.y) ** 2 < r * r
    return occ


def edge_distances(bounds: Bounds, resolution_m: float) -> dict[str, np.ndarray]:
    xs, ys = cell_centers(bounds, resolution_m)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return {"x_min": gx - bounds.x_min, "x_max": bounds.x_max - gx, "y_min": gy - bounds.y_min, "y_max": bounds.y_max - gy}


def band_mask(bounds: Bounds, resolution_m: float, d_min: float, d_max: float) -> np.ndarray:
    d = np.minimum.reduce(list(edge_distances(bounds, resolution_m).values()))
    return (d >= d_min) & (d <= d_max)


def reachable_from(free: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    reach = seeds & free
    while True:
        grown = reach.copy()
        grown[1:, :] |= reach[:-1, :]
        grown[:-1, :] |= reach[1:, :]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= free
        if np.array_equal(grown, reach):
            return reach
        reach = grown


def crossing_unreachable(
    free: np.ndarray, bounds: Bounds, resolution_m: float, start_band: tuple[float, float], target_band: tuple[float, float]
) -> dict[str, int]:
    d = edge_distances(bounds, resolution_m)
    counts: dict[str, int] = {}
    for edge in EDGES:
        side_a, side_b = PERPENDICULAR[edge]
        corridor = free & (d[side_a] > start_band[1]) & (d[side_b] > start_band[1])
        start = corridor & (d[edge] >= start_band[0]) & (d[edge] <= start_band[1])
        far = OPPOSITE[edge]
        target = corridor & (d[far] >= target_band[0]) & (d[far] <= target_band[1])
        counts[edge] = int((start & ~reachable_from(corridor, target)).sum())
    return counts


def check_reachability(
    layout: Layout,
    start_band: tuple[float, float],
    target_band: tuple[float, float],
    clearance_m: float = 1.0,
    drone_radius_m: float = 0.4,
    resolution_m: float = 0.25,
) -> ReachabilityResult:
    inflate = drone_radius_m + clearance_m
    free = ~occupancy(layout, resolution_m, inflate)
    start = band_mask(layout.bounds, resolution_m, *start_band) & free
    target = band_mask(layout.bounds, resolution_m, *target_band) & free
    n_start, n_target = int(start.sum()), int(target.sum())
    if n_start == 0 or n_target == 0:
        return ReachabilityResult(False, resolution_m, inflate, n_start, n_target, n_start, {},
                                  "no free start cell" if n_start == 0 else "no free target cell")
    unreachable = int((start & ~reachable_from(free, target)).sum())
    crossing = crossing_unreachable(free, layout.bounds, resolution_m, start_band, target_band)
    if unreachable:
        reason = f"{unreachable} free start cells cannot reach any target cell"
    elif sum(crossing.values()):
        reason = f"free start cells cannot cross the obstacle field to the opposite edge: {crossing}"
    else:
        reason = "ok"
    ok = unreachable == 0 and sum(crossing.values()) == 0
    return ReachabilityResult(ok, resolution_m, inflate, n_start, n_target, unreachable, crossing, reason)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_reachability.py`
Expected: `7 passed`.

- [ ] **Step 5: Write `scripts/build_scenes.py`**

```python
"""Expand a scene file into a layout, check reachability and write <out-dir>/<id>.layout.json.

env -u PYTHONPATH .venv/bin/python scripts/build_scenes.py scenes/s01_white_pillars.json
"""

import argparse
import json
import sys
from pathlib import Path

from autofly_ue5.paths import RUNS_DIR
from autofly_ue5.scenes.generate import generate_layout
from autofly_ue5.scenes.model import load_registry, load_scene_file
from autofly_ue5.scenes.reachability import check_reachability


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR / "levels")
    args = parser.parse_args()
    scene = load_scene_file(args.scene)
    layout = generate_layout(scene, load_registry())
    reach = check_reachability(layout, scene.start_band, scene.target_band)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{scene.id}.layout.json"
    out.write_text(json.dumps({"scene_path": str(args.scene), "scene_sha256": scene.sha256,
                               "layout": layout.to_json(), "reachability": reach.to_json()}, indent=2))
    print(json.dumps({"scene": scene.id, "instances": len(layout.instances), "reachability": reach.to_json(), "layout": str(out)}))
    return 0 if reach.ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the CLI on s01**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python scripts/build_scenes.py scenes/s01_white_pillars.json; echo "exit=$?"`
Expected: a JSON line with `"scene": "s01"`, `"instances": 80`, `"ok": true`, `"unreachable_start_cells": 0`, `"crossing_unreachable_cells": {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0}`, and `exit=0`; file `runs/levels/s01.layout.json` exists.

- [ ] **Step 7: Commit**

```bash
git -C /home/nvidiasims/research_uav/autofly_ue5 add autofly_ue5/scenes/reachability.py scripts/build_scenes.py tests/test_reachability.py && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Add the 2D occupancy reachability check with 1.0 m clearance, the crossing rule and the scene build CLI" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: UE level builder for s01

**Files:**
- Create: `autofly_ue5/scenes/level_spec.py`, `autofly_ue5/scenes/ue/build_level.py`, `autofly_ue5/scenes/ue/verify_level.py`, `scripts/build_level.sh`
- Modify: `scripts/build_scenes.py` (full new content below; also writes the level spec); `assets/registry.json` (Step 11 writes the live-measured bounds)
- Test: `tests/test_level_spec.py` + live commandlet build

**Interfaces:**
- Consumes: `autofly_ue5.frames.{ned_to_ue_cm, ned_yaw_deg_to_ue_yaw_deg}`; `autofly_ue5.scenes.model.{Layout, SceneFile, AssetRegistry, load_scene_file, load_registry}`; `autofly_ue5.scenes.generate.generate_layout`; `autofly_ue5.scenes.reachability.check_reachability`; UE facts proven in Task 4 Step 8 (commandlet subsystems, `SunSky_C` path, `Color` parameter, material-instance parent and colour set and read back from Python; `set_material_instance_vector_parameter_value` itself always returns `False` in 5.7.4, `MaterialEditingLibrary.cpp:1253-1262`, so its result is never used as success).
- Produces:
  - `autofly_ue5.scenes.level_spec`: constants `MAP_ROOT = "/Game/AutoFly/Maps"`, `GROUND_SIZE_M = 200.0`, `GROUND_THICKNESS_M = 0.2`, `SUNSKY_CLASS = "/SunPosition/SunSky.SunSky_C"`, `GAME_MODE_CLASS = "/Script/ProjectAirSim.ProjectAirSimGameMode"`; `map_path_for(scene_id: str) -> str` (`"s01"` → `"/Game/AutoFly/Maps/S01"`); `layout_to_level_spec(layout: Layout, scene: SceneFile, registry: AssetRegistry) -> dict`.
  - Level spec JSON (UE centimetres): `version`, `scene_id`, `scene_sha256`, `layout_seed`, `map_path`, `game_mode_class`, `sunsky_class`, `materials: [{name, kind, ue_path, parent, parameter, rgba}]`, `ground: {tag: "AF_Ground", mesh, location_cm, yaw_deg, scale, material, expected_extent_cm}`, `actors: [{tag, mesh, location_cm, yaw_deg, scale, material, expected_extent_cm}]`.
  - `bash scripts/build_level.sh <scene_id>` → `ue_project/Content/AutoFly/Maps/<ID>.umap`, `ue_project/Content/AutoFly/Materials/MI_White.uasset`, reports `runs/levels/<id>.build.json` and `runs/levels/<id>.verify.json` (both `"pass": true`).
  - `scripts/build_scenes.py` now also writes `runs/levels/<id>.level.json`.
  - `assets/registry.json` entries `cylinder` and `cube` carry `measured_extent_cm_at_unit_scale` (UE bounds half-extent / actor scale, expected `[50.0, 50.0, 50.0]`) and `measured_by`.

- [ ] **Step 1: Write the failing test**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_level_spec.py`:
```python
import pytest

from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.generate import generate_layout
from autofly_ue5.scenes.level_spec import GAME_MODE_CLASS, SUNSKY_CLASS, layout_to_level_spec, map_path_for
from autofly_ue5.scenes.model import Bounds, Instance, Layout, load_registry, load_scene_file

SCENE = load_scene_file(SCENES_DIR / "s01_white_pillars.json")
REGISTRY = load_registry()


def test_map_path():
    assert map_path_for("s01") == "/Game/AutoFly/Maps/S01"


def test_single_pillar_converts_to_ue_centimetres():
    pillar = Instance(tag="obs_0000", asset="cylinder", x=1.0, y=-2.0, z_center=-5.0, yaw=0.0, scale=(1.0, 1.0, 10.0),
                      material="white", radius_m=0.5, height_m=10.0)
    spec = layout_to_level_spec(Layout("s01", 1001, Bounds(-35, 35, -35, 35), (pillar,)), SCENE, REGISTRY)
    actor = spec["actors"][0]
    assert actor["location_cm"] == [100.0, -200.0, 500.0]
    assert actor["mesh"] == "/Engine/BasicShapes/Cylinder" and actor["yaw_deg"] == 0.0
    assert actor["scale"] == [1.0, 1.0, 10.0] and actor["expected_extent_cm"] == [50.0, 50.0, 500.0]
    assert actor["material"] == "white"


def test_ground_top_is_at_ue_zero():
    spec = layout_to_level_spec(Layout("s01", 1001, SCENE.bounds, ()), SCENE, REGISTRY)
    ground = spec["ground"]
    assert ground["tag"] == "AF_Ground" and ground["mesh"] == "/Engine/BasicShapes/Cube"
    assert ground["location_cm"] == pytest.approx([0.0, 0.0, -10.0])
    assert ground["expected_extent_cm"] == pytest.approx([10000.0, 10000.0, 10.0])
    assert ground["location_cm"][2] + ground["expected_extent_cm"][2] == pytest.approx(0.0)
    assert ground["material"] == "grid"


def test_full_s01_spec():
    layout = generate_layout(SCENE, REGISTRY)
    spec = layout_to_level_spec(layout, SCENE, REGISTRY)
    assert spec["map_path"] == "/Game/AutoFly/Maps/S01" and spec["scene_sha256"] == SCENE.sha256
    assert spec["game_mode_class"] == GAME_MODE_CLASS and spec["sunsky_class"] == SUNSKY_CLASS
    assert len(spec["actors"]) == 80 and spec["actors"][79]["tag"] == "obs_0079"
    names = {m["name"] for m in spec["materials"]}
    assert names == {"grid", "white"}
    white = next(m for m in spec["materials"] if m["name"] == "white")
    assert white["ue_path"] == "/Game/AutoFly/Materials/MI_White" and white["rgba"] == [0.9, 0.9, 0.9, 1.0]
    for actor, inst in zip(spec["actors"], layout.instances):
        assert actor["location_cm"][2] == pytest.approx(inst.height_m * 50.0, abs=0.01)  # base on the ground
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_level_spec.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'autofly_ue5.scenes.level_spec'`.

- [ ] **Step 3: Write `autofly_ue5/scenes/level_spec.py`**

```python
"""Layout (NED metres) -> UE level spec (centimetres, Z up) consumed by scenes/ue/build_level.py."""

from __future__ import annotations

import math

from autofly_ue5.frames import ned_to_ue_cm, ned_yaw_deg_to_ue_yaw_deg
from autofly_ue5.scenes.model import AssetRegistry, Layout, SceneFile

MAP_ROOT = "/Game/AutoFly/Maps"
GROUND_SIZE_M = 200.0
GROUND_THICKNESS_M = 0.2
SUNSKY_CLASS = "/SunPosition/SunSky.SunSky_C"
GAME_MODE_CLASS = "/Script/ProjectAirSim.ProjectAirSimGameMode"


def map_path_for(scene_id: str) -> str:
    return f"{MAP_ROOT}/{scene_id.upper()}"


def _extent_cm(base_size_m: tuple[float, float, float], scale: tuple[float, float, float]) -> list[float]:
    return [round(b * s * 50.0, 4) for b, s in zip(base_size_m, scale)]


def layout_to_level_spec(layout: Layout, scene: SceneFile, registry: AssetRegistry) -> dict:
    cube = registry.assets["cube"]
    ground_scale = (GROUND_SIZE_M / cube.base_size_m[0], GROUND_SIZE_M / cube.base_size_m[1], GROUND_THICKNESS_M / cube.base_size_m[2])
    ground = {
        "tag": "AF_Ground",
        "mesh": cube.ue_path,
        "location_cm": list(ned_to_ue_cm(0.0, 0.0, GROUND_THICKNESS_M / 2.0)),
        "yaw_deg": 0.0,
        "scale": list(ground_scale),
        "material": scene.ground,
        "expected_extent_cm": _extent_cm(cube.base_size_m, ground_scale),
    }
    actors = []
    for inst in layout.instances:
        asset = registry.assets[inst.asset]
        actors.append({
            "tag": inst.tag,
            "mesh": asset.ue_path,
            "location_cm": [round(v, 4) for v in ned_to_ue_cm(inst.x, inst.y, inst.z_center)],
            "yaw_deg": ned_yaw_deg_to_ue_yaw_deg(math.degrees(inst.yaw)),
            "scale": list(inst.scale),
            "material": inst.material,
            "expected_extent_cm": _extent_cm(asset.base_size_m, inst.scale),
        })
    used = sorted({scene.ground} | {inst.material for inst in layout.instances})
    materials = []
    for name in used:
        m = registry.materials[name]
        materials.append({"name": name, "kind": m.kind, "ue_path": m.ue_path, "parent": m.parent,
                          "parameter": m.parameter, "rgba": list(m.rgba) if m.rgba is not None else None})
    return {
        "version": 1,
        "scene_id": scene.id,
        "scene_sha256": scene.sha256,
        "layout_seed": layout.seed,
        "map_path": map_path_for(scene.id),
        "game_mode_class": GAME_MODE_CLASS,
        "sunsky_class": SUNSKY_CLASS,
        "materials": materials,
        "ground": ground,
        "actors": actors,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_level_spec.py`
Expected: `4 passed`.

- [ ] **Step 5: Replace `scripts/build_scenes.py` so it also writes the level spec**

```python
"""Expand a scene file into a layout, check reachability, write <id>.layout.json and <id>.level.json.

env -u PYTHONPATH .venv/bin/python scripts/build_scenes.py scenes/s01_white_pillars.json
"""

import argparse
import json
import sys
from pathlib import Path

from autofly_ue5.paths import RUNS_DIR
from autofly_ue5.scenes.generate import generate_layout
from autofly_ue5.scenes.level_spec import layout_to_level_spec
from autofly_ue5.scenes.model import load_registry, load_scene_file
from autofly_ue5.scenes.reachability import check_reachability


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR / "levels")
    args = parser.parse_args()
    scene = load_scene_file(args.scene)
    registry = load_registry()
    layout = generate_layout(scene, registry)
    reach = check_reachability(layout, scene.start_band, scene.target_band)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    layout_out = args.out_dir / f"{scene.id}.layout.json"
    layout_out.write_text(json.dumps({"scene_path": str(args.scene), "scene_sha256": scene.sha256,
                                      "layout": layout.to_json(), "reachability": reach.to_json()}, indent=2))
    summary = {"scene": scene.id, "instances": len(layout.instances), "reachability": reach.to_json(), "layout": str(layout_out)}
    if reach.ok:
        level_out = args.out_dir / f"{scene.id}.level.json"
        level_out.write_text(json.dumps(layout_to_level_spec(layout, scene, registry), indent=2))
        summary["level_spec"] = str(level_out)
    print(json.dumps(summary))
    return 0 if reach.ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Write the UE-side builder `autofly_ue5/scenes/ue/build_level.py`**

```python
"""UE-side map builder (UnrealEditor-Cmd -run=pythonscript; UE embedded Python; stdlib + unreal only).

Reads the level spec (env AUTOFLY_LEVEL_SPEC), creates the map with ground, obstacle actors, SunSky
lighting and the ProjectAirSim GameMode override, saves it and writes a report (env AUTOFLY_BUILD_OUT).
Every failed check raises, so the commandlet logs "Python script executed with errors".
"""
import json
import os

import unreal

SPEC = json.load(open(os.environ["AUTOFLY_LEVEL_SPEC"]))
OUT = os.environ["AUTOFLY_BUILD_OUT"]
TOL_CM = 1.0
REPORT = {"map_path": SPEC["map_path"], "materials": {}, "actors": [], "pass": False}


def check(ok, what):
    if not ok:
        raise RuntimeError("AUTOFLY CHECK FAILED: " + what)


def make_material(entry, ass):
    if entry["kind"] == "engine":
        material = unreal.load_asset(entry["ue_path"])
        check(material is not None, "engine material " + entry["ue_path"])
        return material
    check(entry["kind"] == "color_instance", "unknown material kind " + str(entry["kind"]))
    folder, name = entry["ue_path"].rsplit("/", 1)
    parent = unreal.load_asset(entry["parent"])
    check(parent is not None, "parent material " + entry["parent"])
    params = [str(n) for n in unreal.MaterialEditingLibrary.get_vector_parameter_names(parent)]
    REPORT["materials"][entry["name"]] = {"parent_vector_params": params}
    check(entry["parameter"] in params, "parameter %s not in %s" % (entry["parameter"], params))
    if ass.does_asset_exist(entry["ue_path"]):
        mic = ass.load_asset(entry["ue_path"])
    else:
        mic = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, folder, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    check(mic is not None, "create or load " + entry["ue_path"])
    unreal.MaterialEditingLibrary.set_material_instance_parent(mic, parent)
    got_parent = mic.get_editor_property("parent")
    check(got_parent is not None and got_parent.get_path_name() == parent.get_path_name(),
          "parent of %s is %s" % (entry["ue_path"], got_parent.get_path_name() if got_parent is not None else None))
    # The setter's bool is never set in UE 5.7.4 and is always False (MaterialEditingLibrary.cpp:1253-1262): read back.
    unreal.MaterialEditingLibrary.set_material_instance_vector_parameter_value(
        mic, entry["parameter"], unreal.LinearColor(*entry["rgba"]))
    unreal.MaterialEditingLibrary.update_material_instance(mic)
    got = unreal.MaterialEditingLibrary.get_material_instance_vector_parameter_value(mic, entry["parameter"])
    readback = [got.r, got.g, got.b, got.a]
    REPORT["materials"][entry["name"]]["readback_rgba"] = readback
    check(all(abs(a - b) < 1e-4 for a, b in zip(readback, entry["rgba"])),
          "set %s on %s: read back %s" % (entry["parameter"], entry["ue_path"], readback))
    check(ass.save_loaded_asset(mic, False), "save " + entry["ue_path"])
    return mic


def spawn_mesh(a, eas, materials):
    mesh = unreal.load_asset(a["mesh"])
    check(mesh is not None, "mesh " + a["mesh"])
    actor = eas.spawn_actor_from_object(mesh, unreal.Vector(*a["location_cm"]), unreal.Rotator(roll=0.0, pitch=0.0, yaw=a["yaw_deg"]))
    check(actor is not None, "spawn " + a["tag"])
    actor.set_actor_scale3d(unreal.Vector(*a["scale"]))
    component = actor.get_editor_property("static_mesh_component")
    component.set_mobility(unreal.ComponentMobility.MOVABLE)
    component.set_collision_profile_name("BlockAll")
    component.set_material(0, materials[a["material"]])
    actor.set_actor_label(a["tag"])
    actor.set_editor_property("tags", [a["tag"]])
    origin, extent = actor.get_actor_bounds(False)
    got_origin = [origin.x, origin.y, origin.z]
    got_extent = [extent.x, extent.y, extent.z]
    origin_err = max(abs(g - w) for g, w in zip(got_origin, a["location_cm"]))
    extent_err = max(abs(g - w) for g, w in zip(got_extent, a["expected_extent_cm"]))
    REPORT["actors"].append({"tag": a["tag"], "origin_cm": got_origin, "extent_cm": got_extent,
                             "origin_err_cm": origin_err, "extent_err_cm": extent_err})
    check(origin_err <= TOL_CM, "%s bounds origin %s != %s (mesh pivot not centred?)" % (a["tag"], got_origin, a["location_cm"]))
    check(extent_err <= TOL_CM, "%s bounds extent %s != %s (base size not 1 m?)" % (a["tag"], got_extent, a["expected_extent_cm"]))
    return actor


def main():
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    ass = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)
    ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    check(None not in (les, eas, ass, ues), "an editor subsystem is unavailable in this commandlet")
    check(les.new_level(SPEC["map_path"], False), "new_level " + SPEC["map_path"])
    materials = {m["name"]: make_material(m, ass) for m in SPEC["materials"]}
    spawn_mesh(SPEC["ground"], eas, materials)
    for a in SPEC["actors"]:
        spawn_mesh(a, eas, materials)
    sunsky_class = unreal.load_class(None, SPEC["sunsky_class"])
    check(sunsky_class is not None, "SunSky class " + SPEC["sunsky_class"])
    # below the ground slab so the SunSky helper meshes can never block or appear in front of the camera
    sunsky = eas.spawn_actor_from_class(sunsky_class, unreal.Vector(0.0, 0.0, -500.0))
    check(sunsky is not None, "spawn SunSky")
    sunsky.set_actor_label("SunSky")
    sunsky.set_editor_property("tags", ["SunSky"])
    game_mode = unreal.load_class(None, SPEC["game_mode_class"])
    check(game_mode is not None, "GameMode class " + SPEC["game_mode_class"])
    ues.get_editor_world().get_world_settings().set_editor_property("default_game_mode", game_mode)
    check(les.save_current_level(), "save_current_level")
    REPORT["actor_count"] = len(REPORT["actors"])
    REPORT["pass"] = True


try:
    main()
finally:
    with open(OUT, "w") as handle:
        json.dump(REPORT, handle, indent=2)
```

- [ ] **Step 7: Write the UE-side verifier `autofly_ue5/scenes/ue/verify_level.py`**

```python
"""UE-side reload check of a saved map (fresh commandlet process; stdlib + unreal only).

env AUTOFLY_LEVEL_SPEC (level spec JSON), AUTOFLY_VERIFY_OUT (report JSON).
"""
import json
import os

import unreal

SPEC = json.load(open(os.environ["AUTOFLY_LEVEL_SPEC"]))
OUT = os.environ["AUTOFLY_VERIFY_OUT"]
TOL_CM = 1.0
REPORT = {"map_path": SPEC["map_path"], "pass": False}


def check(ok, what):
    if not ok:
        raise RuntimeError("AUTOFLY CHECK FAILED: " + what)


def main():
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    check(les.load_level(SPEC["map_path"]), "load_level " + SPEC["map_path"])
    by_tag = {}
    for actor in eas.get_all_level_actors():
        for tag in actor.get_editor_property("tags"):
            by_tag.setdefault(str(tag), []).append(actor)
    expected = [SPEC["ground"]] + SPEC["actors"]
    worst = 0.0
    for a in expected:
        found = by_tag.get(a["tag"], [])
        check(len(found) == 1, "tag %s found %d times" % (a["tag"], len(found)))
        loc = found[0].get_actor_location()
        err = max(abs(g - w) for g, w in zip([loc.x, loc.y, loc.z], a["location_cm"]))
        worst = max(worst, err)
        check(err <= TOL_CM, "%s location error %.3f cm" % (a["tag"], err))
    obstacle_tags = sorted(t for t in by_tag if t.startswith("obs_"))
    check(len(obstacle_tags) == len(SPEC["actors"]), "obstacle tag count %d != %d" % (len(obstacle_tags), len(SPEC["actors"])))
    check(len(by_tag.get("SunSky", [])) == 1, "SunSky actor missing")
    game_mode = ues.get_editor_world().get_world_settings().get_editor_property("default_game_mode")
    game_mode_path = game_mode.get_path_name() if game_mode is not None else None
    check(game_mode_path == SPEC["game_mode_class"], "GameMode override is %s" % game_mode_path)
    REPORT.update({"obstacles": len(obstacle_tags), "worst_location_error_cm": worst, "game_mode": game_mode_path, "pass": True})


try:
    main()
finally:
    with open(OUT, "w") as handle:
        json.dump(REPORT, handle, indent=2)
```

- [ ] **Step 8: Write `scripts/build_level.sh`**

```bash
#!/usr/bin/env bash
# Build ue_project/Content/AutoFly/Maps/<ID>.umap from runs/levels/<id>.level.json, then verify it in a fresh process.
set -euo pipefail
ROOT=/home/nvidiasims/research_uav/autofly_ue5
ID="${1:?usage: build_level.sh <scene_id>}"
LEVELS=$ROOT/runs/levels
SPEC=$LEVELS/$ID.level.json
test -f "$SPEC" || { echo "missing $SPEC (run scripts/build_scenes.py first)"; exit 1; }
MAP=$(jq -r .map_path "$SPEC")
UMAP=$ROOT/ue_project/Content/${MAP#/Game/}.umap
CMD=$ROOT/engine/Engine/Binaries/Linux/UnrealEditor-Cmd
PROJECT=$ROOT/ue_project/Blocks.uproject
rm -f "$UMAP" "$LEVELS/$ID.build.json" "$LEVELS/$ID.verify.json"

run_script() {  # $1 script, $2 log name, remaining: env assignments
  local script=$1 log=$2
  shift 2
  set +e
  env -u PYTHONPATH DISPLAY=:1 SDL_VIDEODRIVER=x11 "UE-ZenDataPath=$ROOT/ue_project/DerivedDataCache/Zen" \
    "UE-LocalDataCachePath=$ROOT/ue_project/DerivedDataCache" AUTOFLY_LEVEL_SPEC="$SPEC" "$@" \
    "$CMD" "$PROJECT" -run=pythonscript -script="$script" -unattended -nop4 -nosplash -nullrhi -notraceserver \
    -stdout -FullStdOutLogOutput -abslog="$LEVELS/$log" > "$LEVELS/$log.stdout" 2>&1
  local code=$?
  set -e
  # information only: commandlets exit 1 whenever any error was logged (LaunchEngineLoop.cpp:4160-4166)
  echo "$log exit code: $code (information), Error lines: $(grep -c "Error:" "$LEVELS/$log" || true)"
  grep -E "Python script executed (successfully|with errors)" "$LEVELS/$log" || true
  grep -q "Python script executed successfully" "$LEVELS/$log" || { grep "AUTOFLY CHECK FAILED" "$LEVELS/$log" | head -5; exit 1; }
}

run_script "$ROOT/autofly_ue5/scenes/ue/build_level.py" "$ID.build.log" AUTOFLY_BUILD_OUT="$LEVELS/$ID.build.json"
test -f "$UMAP" || { echo "map file $UMAP was not written"; exit 1; }
jq -e '.pass == true' "$LEVELS/$ID.build.json" > /dev/null
run_script "$ROOT/autofly_ue5/scenes/ue/verify_level.py" "$ID.verify.log" AUTOFLY_VERIFY_OUT="$LEVELS/$ID.verify.json"
jq -e '.pass == true' "$LEVELS/$ID.verify.json" > /dev/null
echo "level $MAP built and verified: $(jq -c '{obstacles, worst_location_error_cm, game_mode}' "$LEVELS/$ID.verify.json")"
```

- [ ] **Step 9: Generate the s01 level spec**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python scripts/build_scenes.py scenes/s01_white_pillars.json && jq '{map_path, actors: (.actors|length), ground: .ground.location_cm, materials: [.materials[].name]}' runs/levels/s01.level.json`
Expected: summary JSON with `"level_spec": ".../runs/levels/s01.level.json"`, then `{"map_path": "/Game/AutoFly/Maps/S01", "actors": 80, "ground": [0, 0, -10], "materials": ["grid", "white"]}`.

- [ ] **Step 10: Build and verify the map headless as a job (no GPU; a few minutes; total wait budget 40 minutes)**

Run: `chmod +x /home/nvidiasims/research_uav/autofly_ue5/scripts/build_level.sh && cd /home/nvidiasims/research_uav/autofly_ue5 && bash scripts/run_job.sh start build_level_s01 -- bash scripts/build_level.sh s01 && bash scripts/run_job.sh wait build_level_s01 540; echo "wait=$?"` (repeat `wait` while it returns 124, within the budget).
Expected: in the job log `s01.build.log exit code: … (information)` and `s01.verify.log exit code: … (information)` (0 or 1, both acceptable), `Python script executed successfully` twice, final line `level /Game/AutoFly/Maps/S01 built and verified: {"obstacles":80,"worst_location_error_cm":<≤1>,"game_mode":"/Script/ProjectAirSim.ProjectAirSimGameMode"}`, `job build_level_s01 finished: exit=0`, `wait=0`. This verifies live the UNCONFIRMED items: `new_level`/`spawn_actor_from_object`/`save_current_level` in a commandlet, engine `Cylinder` and `Cube` are 100 cm with centred pivots (bounds assertion), `MaterialInstanceConstant` creation with its parent and `Color` read back after setting (the setter's return value is not used), `Rotator(roll, pitch, yaw)` keywords, and the GameMode override surviving a reload. On any failure: stop and report `runs/levels/s01.build.json` (or `.verify.json`), the `AUTOFLY CHECK FAILED` line and `grep -iE "error" runs/levels/s01.build.log | head -40`; do not switch APIs without the user.

- [ ] **Step 11: Write the measured bounds into `assets/registry.json`**

Run:
```bash
cd /home/nvidiasims/research_uav/autofly_ue5 && jq '{materials, ground: .actors[0], first_pillar: .actors[1], max_extent_err_cm: ([.actors[].extent_err_cm] | max)}' runs/levels/s01.build.json && ls -la ue_project/Content/AutoFly/Maps ue_project/Content/AutoFly/Materials && env -u PYTHONPATH .venv/bin/python - <<'EOF'
import json
import statistics
from pathlib import Path

build = json.loads(Path("runs/levels/s01.build.json").read_text())
spec = json.loads(Path("runs/levels/s01.level.json").read_text())
placed = {a["tag"]: a for a in [spec["ground"]] + spec["actors"]}
registry_path = Path("assets/registry.json")
registry = json.loads(registry_path.read_text())
for name, entry in registry["assets"].items():
    rows = [[e / s for e, s in zip(actor["extent_cm"], placed[actor["tag"]]["scale"])]
            for actor in build["actors"] if placed[actor["tag"]]["mesh"] == entry["ue_path"]]
    if rows:
        entry["measured_extent_cm_at_unit_scale"] = [round(statistics.median(r[i] for r in rows), 2) for i in range(3)]
        entry["measured_by"] = f"Task 14 build of s01: UE get_actor_bounds half-extent / actor scale, median of {len(rows)} actors"
registry_path.write_text(json.dumps(registry, indent=2) + "\n")
print({name: entry["measured_extent_cm_at_unit_scale"] for name, entry in registry["assets"].items()})
EOF
env -u PYTHONPATH .venv/bin/python -m pytest tests/test_scene_model.py tests/test_generate.py tests/test_level_spec.py
jq -e '[.assets.cylinder, .assets.cube] | all(.measured_extent_cm_at_unit_scale | (type == "array" and length == 3))' assets/registry.json
```
Expected: `materials.white.parent_vector_params` contains `"Color"` and `materials.white.readback_rgba` ≈ `[0.9, 0.9, 0.9, 1.0]`; ground extent `[10000, 10000, 10]`; `max_extent_err_cm` ≤ 1; `S01.umap` and `MI_White.uasset` listed; then `{'cylinder': [50.0, 50.0, 50.0], 'cube': [50.0, 50.0, 50.0]}`, `16 passed` (the registry test now checks the measured values), and `true` printed once by `jq` (exit 0; if either entry lacks a 3-value measurement it prints `false` and exits 1, meaning no built actor matched that mesh path: stop and report). A measured value outside 49–51 cm: stop and report it (the 1 m base-size assumption of the registry is contradicted).

- [ ] **Step 12: Commit**

```bash
git -C /home/nvidiasims/research_uav/autofly_ue5 add autofly_ue5/scenes/level_spec.py autofly_ue5/scenes/ue/build_level.py autofly_ue5/scenes/ue/verify_level.py scripts/build_level.sh scripts/build_scenes.py tests/test_level_spec.py assets/registry.json && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Build the s01 UE map headless from its level spec (ground, white pillars, SunSky, GameMode), verify it on reload and record measured asset bounds" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: Packaged simulator with the S01 map

**Files:**
- Create: `scripts/package_sim.sh`, `configs/scene_autofly_s01.jsonc`, `autofly_ue5/sim/check_map.py`
- Create (captured): `docs/gates/m1_check_map.json`, `docs/gates/m1_package_manifest.json`
- Modify: `tests/test_pas_configs.py` (full new content below)
- Test: `tests/test_check_map.py` + live packaging and launch

**Interfaces:**
- Consumes: `ue_project/Content/AutoFly/Maps/S01.umap` (Task 14); `runs/levels/s01.layout.json` (Task 13/14); `configs/robot_autofly_quadrotor.jsonc` (Task 7); `scripts/launch_sim.py` / `scripts/stop_sim.py` (Task 5); `autofly_ue5.sim.process.ports_for_instance`; `autofly_ue5.paths.{CONFIGS_DIR, PACKAGED_BINARY}`.
- Produces:
  - `ROOT/ue_project/Packaged/Development/Linux/Blocks/Binaries/Linux/Blocks` (= `autofly_ue5.paths.PACKAGED_BINARY`), a Development Linux package containing `/Game/AutoFly/Maps/S01` and `/Game/BlocksMap`.
  - `configs/scene_autofly_s01.jsonc` (scene id `SceneAutoFlyS01`, drone origin `-33.0 0.0 -2.0`, i.e. 2 m inside the south (x_min, NED) edge, in the start band).
  - `autofly_ue5.sim.check_map.bbox_error(instance: dict, bbox: dict) -> dict` (`found`, `center_error_m`, `size_error_m`); CLI `python -m autofly_ue5.sim.check_map --instance N --layout <layout.json> --out <json>` (exit 0 iff pass; client log `<out>.client.log`). Its report `runs/m1/check_map.json` is required by the M1 gate (Task 16).
  - `runs/package/package_manifest.json` (`binary` {path, sha256}, `paks` {path: sha256}, `git_head`, `level_spec_sha256`, `level_verify`, `engine_build_version`), embedded by the M1 gate report (Task 16); RunUAT logs in `runs/package/uat_logs/`.

- [ ] **Step 1: Write the s01 scene config**

`configs/scene_autofly_s01.jsonc`:
```jsonc
{
  "id": "SceneAutoFlyS01",
  "actors": [
    {
      "type": "robot",
      "name": "Drone1",
      "origin": { "xyz": "-33.0 0.0 -2.0", "rpy-deg": "0 0 0" },
      "robot-config": "robot_autofly_quadrotor.jsonc"
    }
  ],
  "clock": { "type": "steppable", "step-ns": 5000000, "real-time-update-rate": 3000000, "pause-on-start": true },
  "home-geo-point": { "latitude": 47.641468, "longitude": -122.140165, "altitude": 122.0 },
  "scene-type": "UnrealNative"
}
```

- [ ] **Step 2: Replace `tests/test_pas_configs.py` to cover the s01 scene and write `tests/test_check_map.py`**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_pas_configs.py`:
```python
import pytest
from projectairsim.utils import load_scene_config_as_dict

from autofly_ue5.paths import CONFIGS_DIR


@pytest.mark.parametrize("scene", ["scene_autofly_m0.jsonc", "scene_autofly_m0_fast.jsonc", "scene_autofly_s01.jsonc"])
def test_scene_validates_against_project_airsim_schema(scene):
    config, _paths = load_scene_config_as_dict(scene, str(CONFIGS_DIR))
    assert config["clock"]["type"] == "steppable"
    assert config["clock"]["step-ns"] == 5_000_000
    assert config["clock"]["pause-on-start"] is True
    robot = config["actors"][0]["robot-config"]
    camera = next(s for s in robot["sensors"] if s["id"] == "FrontCamera")
    assert camera["capture-interval"] == 0.001
    settings = {c["image-type"]: c for c in camera["capture-settings"]}
    assert set(settings) == {0, 1}
    for c in settings.values():
        assert (c["width"], c["height"], c["fov-degrees"]) == (256, 256, 90)
        assert c["capture-enabled"] is True and c["compress"] is False
    assert camera["origin"]["xyz"] == "0.40 0.0 0.0"


def test_s01_start_is_in_the_south_start_band_at_2m():
    config, _paths = load_scene_config_as_dict("scene_autofly_s01.jsonc", str(CONFIGS_DIR))
    x, y, z = (float(v) for v in config["actors"][0]["origin"]["xyz"].split())
    assert 2.0 <= x - (-35.0) <= 6.0 and z == -2.0
```

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_check_map.py`:
```python
import pytest

from autofly_ue5.sim.check_map import bbox_error

INSTANCE = {"tag": "obs_0000", "x": -24.1, "y": 3.2, "z_center": -5.0, "radius_m": 0.5, "height_m": 10.0}


def test_matching_bbox():
    bbox = {"center": {"x": -24.1, "y": 3.2, "z": -5.0}, "size": {"x": 1.0, "y": 1.0, "z": 10.0}}
    result = bbox_error(INSTANCE, bbox)
    assert result["found"] is True
    assert result["center_error_m"] == pytest.approx(0.0) and result["size_error_m"] == pytest.approx(0.0)


def test_offset_bbox_reports_errors():
    bbox = {"center": {"x": -24.0, "y": 3.2, "z": -5.0}, "size": {"x": 1.0, "y": 1.2, "z": 10.0}}
    result = bbox_error(INSTANCE, bbox)
    assert result["center_error_m"] == pytest.approx(0.1) and result["size_error_m"] == pytest.approx(0.2)


def test_missing_actor():
    assert bbox_error(INSTANCE, {}) == {"found": False}
```

- [ ] **Step 3: Run them to verify the new test fails**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_check_map.py; env -u PYTHONPATH .venv/bin/python -m pytest tests/test_pas_configs.py`
Expected: the first run FAILS with `ModuleNotFoundError: No module named 'autofly_ue5.sim.check_map'`; the second run passes (`4 passed`, the s01 config from Step 1 validates).

- [ ] **Step 4: Write `autofly_ue5/sim/check_map.py`**

```python
"""Confirm a running simulator has the generated map loaded by comparing obstacle bounding boxes with the layout.

env -u PYTHONPATH .venv/bin/python -m autofly_ue5.sim.check_map --instance 0 --layout runs/levels/s01.layout.json --out runs/m1/check_map.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from autofly_ue5.paths import CONFIGS_DIR
from autofly_ue5.sim.process import ports_for_instance, route_client_log

TOL_M = 0.05


def bbox_error(instance: dict, bbox: dict) -> dict:
    if not bbox:
        return {"found": False}
    center = (bbox["center"]["x"], bbox["center"]["y"], bbox["center"]["z"])
    center_error = math.dist(center, (instance["x"], instance["y"], instance["z_center"]))
    expected = (2 * instance["radius_m"], 2 * instance["radius_m"], instance["height_m"])
    size = (bbox["size"]["x"], bbox["size"]["y"], bbox["size"]["z"])
    return {"found": True, "center_error_m": center_error, "size_error_m": max(abs(a - b) for a, b in zip(size, expected))}


def main(argv: list[str] | None = None) -> int:
    from projectairsim import ProjectAirSimClient, World
    from projectairsim.types import BoxAlignment

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--scene-config", default="scene_autofly_s01.jsonc")
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    instances = json.loads(args.layout.read_text())["layout"]["instances"]
    chosen = [instances[0], instances[len(instances) // 2], instances[-1]]
    ports = ports_for_instance(args.instance)
    report: dict = {"layout": str(args.layout), "obstacles": []}
    route_client_log(args.out.with_suffix(".client.log"))
    client = ProjectAirSimClient(port_topics=ports.topics, port_services=ports.services)
    try:
        client.connect()
        world = World(client, args.scene_config, delay_after_load_sec=0, sim_config_path=str(CONFIGS_DIR))
        for inst in chosen:
            result = bbox_error(inst, world.get_3d_bounding_box(inst["tag"], BoxAlignment.WORLD_AXIS))
            result["tag"] = inst["tag"]
            report["obstacles"].append(result)
        ground = world.get_3d_bounding_box("AF_Ground", BoxAlignment.WORLD_AXIS)
        report["ground_top_z_ned"] = ground["center"]["z"] - ground["size"]["z"] / 2.0 if ground else None
    except Exception as err:
        report["error"] = f"{type(err).__name__}: {err}"
    finally:
        client.disconnect()
    report["pass"] = (
        "error" not in report
        and all(o["found"] and o["center_error_m"] < TOL_M and o["size_error_m"] < TOL_M for o in report["obstacles"])
        and report.get("ground_top_z_ned") is not None and abs(report["ground_top_z_ned"]) < 0.02
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_pas_configs.py tests/test_check_map.py`
Expected: `7 passed`.

- [ ] **Step 6: Write `scripts/package_sim.sh`**

```bash
#!/usr/bin/env bash
# Package the Development Linux game with /Game/AutoFly/Maps/S01 and /Game/BlocksMap into ue_project/Packaged/Development.
set -euo pipefail
ROOT=/home/nvidiasims/research_uav/autofly_ue5
OUT=$ROOT/ue_project/Packaged/Development
LOG=$ROOT/runs/package/package_dev.log
# Dedicated folder: UAT clears its log folder at startup and would otherwise use ~/Documents/Unreal Engine/LocalBuildLogs
# on an installed engine (CommandEnvironment.cs:104-122).
UAT_LOGS=$ROOT/runs/package/uat_logs
BIN=$OUT/Linux/Blocks/Binaries/Linux/Blocks
mkdir -p "$(dirname "$LOG")" "$UAT_LOGS"
test -f "$ROOT/ue_project/Content/AutoFly/Maps/S01.umap" || { echo "S01.umap missing (run scripts/build_level.sh s01)"; exit 1; }
FREE_GB=$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc 0-9)
test "$FREE_GB" -ge 80 || { echo "only ${FREE_GB} GB free, need 80"; exit 1; }
set +e
env -u PYTHONPATH DISPLAY=:1 SDL_VIDEODRIVER=x11 "UE-ZenDataPath=$ROOT/ue_project/DerivedDataCache/Zen" \
  "UE-LocalDataCachePath=$ROOT/ue_project/DerivedDataCache" "uebp_LogFolder=$UAT_LOGS" \
  "$ROOT/engine/Engine/Build/BatchFiles/RunUAT.sh" BuildCookRun \
  -project="$ROOT/ue_project/Blocks.uproject" -noP4 -utf8output -unattended \
  -platform=Linux -clientconfig=Development -build -cook -stage -pak -compressed -archive \
  -archivedirectory="$OUT" -map=/Game/AutoFly/Maps/S01+/Game/BlocksMap \
  -AdditionalCookerOptions=-notraceserver \
  -nocompileeditor -skipbuildeditor > "$LOG" 2>&1
CODE=$?
set -e
echo "RunUAT exit code: $CODE"
grep -E "BUILD SUCCESSFUL|AutomationTool exiting with ExitCode" "$LOG" | tail -2 || true
test "$CODE" -eq 0
if [ ! -x "$BIN" ]; then
  echo "expected binary $BIN not found; executables under $OUT:"
  find "$OUT" -maxdepth 6 -type f -perm -u+x -name 'Blocks*' | head
  exit 1
fi
echo "packaged binary: $BIN"
```

- [ ] **Step 7: Package as a job (20–60 minutes on first cook; total wait budget 120 minutes)**

Run: `chmod +x /home/nvidiasims/research_uav/autofly_ue5/scripts/package_sim.sh && cd /home/nvidiasims/research_uav/autofly_ue5 && bash scripts/run_job.sh start package_sim -- bash scripts/package_sim.sh && bash scripts/run_job.sh wait package_sim 540; echo "wait=$?"` (repeat `bash scripts/run_job.sh wait package_sim 540` while it returns 124, within the budget; if the budget is used up, `bash scripts/run_job.sh stop package_sim` and report `tail -50 runs/package/package_dev.log`).
Expected: in the job log `RunUAT exit code: 0`, a line `AutomationTool exiting with ExitCode=0 (Success)` (wording UNCONFIRMED; the RunUAT exit code is authoritative), `packaged binary: /home/nvidiasims/research_uav/autofly_ue5/ue_project/Packaged/Development/Linux/Blocks/Binaries/Linux/Blocks`, `job package_sim finished: exit=0`, `wait=0`. Afterwards `ls -d ~/.config/Epic/UnrealEngine/Common/Zen/Data 2>/dev/null || echo zen-default-absent; ls -d ~/"Documents/Unreal Engine" ~/UnrealEngine /tmp/UnrealTraceServer.pid 2>/dev/null || echo out-of-root-absent; ls runs/package/uat_logs | head` prints `zen-default-absent` (the cook used the ROOT cache), `out-of-root-absent` (UAT honoured `uebp_LogFolder` and the cooker `-notraceserver`; a listed path is reported, not deleted) and the UAT log files. The archive layout `Linux/Blocks/Binaries/Linux/Blocks` and the effect of `-nocompileeditor -skipbuildeditor` are UNCONFIRMED: if the binary is elsewhere, the script prints the executables found; stop and report them (do not edit `PACKAGED_BINARY` without the user). On a cook error, stop and report `grep -nE "Error:|error:" runs/package/package_dev.log | head -40` together with `ls runs/package/uat_logs` (the cooker's own log is there).

- [ ] **Step 8: Confirm the maps were cooked and fingerprint the package**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && grep -cE "S01" runs/package/package_dev.log && ls ue_project/Packaged/Development/Linux/Blocks/Content/Paks/`
Expected: a non-zero count and at least one `.pak` (`Blocks-Linux.pak` or similar, name UNCONFIRMED) listed.

Then write the package manifest that ties the M1 gate to this build:
```bash
cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python - <<'EOF'
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


package = Path("ue_project/Packaged/Development/Linux/Blocks")
binary = package / "Binaries" / "Linux" / "Blocks"
manifest = {
    "binary": {"path": str(binary), "sha256": sha256(binary)},
    "paks": {str(p): sha256(p) for p in sorted((package / "Content" / "Paks").glob("*.pak"))},
    "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
    "level_spec_sha256": sha256("runs/levels/s01.level.json"),
    "level_verify": json.loads(Path("runs/levels/s01.verify.json").read_text()),
    "engine_build_version": json.loads(Path("engine/Engine/Build/Build.version").read_text()),
}
assert manifest["paks"], "no .pak files found"
Path("runs/package/package_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps({"binary_sha256": manifest["binary"]["sha256"][:12], "paks": len(manifest["paks"]), "git_head": manifest["git_head"][:7]}))
EOF
```
Expected: one JSON line with a 12-character binary hash prefix, `"paks"` ≥ 1 and the current commit; `runs/package/package_manifest.json` exists. An `AssertionError` or a missing file: stop and report.

- [ ] **Step 9: Launch the packaged S01 simulator (job; total wait budget 20 minutes) and check the map**

Run:
```bash
cd /home/nvidiasims/research_uav/autofly_ue5 && nvidia-smi --query-gpu=memory.used --format=csv,noheader && bash scripts/run_job.sh start launch_s01 -- env -u PYTHONPATH .venv/bin/python scripts/launch_sim.py --mode packaged --map /Game/AutoFly/Maps/S01 --instance 0 --timeout 600 && bash scripts/run_job.sh wait launch_s01 540; echo "wait=$?"
```
Repeat `wait` while it returns 124, within the budget. When `wait=0`, run the map check as a job (it loads the scene config once and can wait up to 60 s for topic info; total wait budget 10 minutes):
```bash
cd /home/nvidiasims/research_uav/autofly_ue5 && bash scripts/run_job.sh start check_map_s01 -- env -u PYTHONPATH .venv/bin/python -m autofly_ue5.sim.check_map --instance 0 --layout runs/levels/s01.layout.json --out runs/m1/check_map.json && bash scripts/run_job.sh wait check_map_s01 540; echo "wait=$?"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
```
If it returns 124, call `bash scripts/run_job.sh wait check_map_s01 60` once more; if the budget is used up, run `bash scripts/run_job.sh stop check_map_s01`, then `env -u PYTHONPATH .venv/bin/python scripts/stop_sim.py --instance 0`, and report.
Expected: launch JSON line with `job launch_s01 finished: exit=0`; in the `check_map_s01` job log a check-map JSON with three obstacles `"found": true`, center and size errors < 0.05 m, `"ground_top_z_ned"` ≈ 0, `"pass": true`, then `job check_map_s01 finished: exit=0` and `wait=0`. This verifies live that the packaged binary takes the map as its second argument (after the project name), that actor tags survive cooking and are matched by Project AirSim's `FindActor`, and that the UE↔NED conversion has no origin offset. If `found` is false for all obstacles, or `wait` is 1 or 3, check `grep -m3 LoadMap runs/sim/inst0/sim.log` for the loaded map, stop instance 0 (`scripts/stop_sim.py --instance 0`) and report (include `runs/m1/check_map.json` and `tail -40 runs/m1/check_map.client.log`).

- [ ] **Step 10: Stop the packaged instance**

Run (Bash tool `timeout` 600000 ms): `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python scripts/stop_sim.py --instance 0 && sleep 30; pgrep -a -f "$(pwd)/ue_project/Packaged/" || echo none`
Expected: `terminated`, then `none` (survivors after another 60 s are reported, not killed).

- [ ] **Step 11: Commit**

```bash
mkdir -p /home/nvidiasims/research_uav/autofly_ue5/docs/gates && cp /home/nvidiasims/research_uav/autofly_ue5/runs/m1/check_map.json /home/nvidiasims/research_uav/autofly_ue5/docs/gates/m1_check_map.json && cp /home/nvidiasims/research_uav/autofly_ue5/runs/package/package_manifest.json /home/nvidiasims/research_uav/autofly_ue5/docs/gates/m1_package_manifest.json
git -C /home/nvidiasims/research_uav/autofly_ue5 add scripts/package_sim.sh configs/scene_autofly_s01.jsonc autofly_ue5/sim/check_map.py tests/test_pas_configs.py tests/test_check_map.py docs/gates/m1_check_map.json docs/gates/m1_package_manifest.json && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Package the Development Linux simulator with the S01 map, fingerprint the package and confirm the cooked map from the client" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: M1 gate — live checks of spec §11 on the packaged s01 simulator

**Files:**
- Create: `autofly_ue5/validate/geometry.py`, `autofly_ue5/validate/live_m1.py`
- Create (captured): `docs/gates/m1_gate.json`
- Test: `tests/test_geometry.py`, `tests/test_live_m1_fake.py` + live gate run

**Interfaces:**
- Consumes: `autofly_ue5.sim.airsim_backend.ProjectAirSimSimulator` (Task 11) through the `Simulator` methods only; `autofly_ue5.sim.types.Pose`; `autofly_ue5.sim.fake.FakeSimulator` (tests); `autofly_ue5.sim.process.{instance_dir, route_client_log}`; `autofly_ue5.scenes.model.{Bounds, load_scene_file, load_registry}`; `autofly_ue5.scenes.generate.generate_layout` (tests); `autofly_ue5.frames.wrap_pi`; `autofly_ue5.gpu.gpu_memory_mib`; `autofly_ue5.paths.{RUNS_DIR, PACKAGED_BINARY}`; `autofly_ue5.validate.engine_check.{xid_count, boot_id, count_device_lost}`; `runs/levels/s01.layout.json`; `runs/m1/check_map.json` and `runs/package/package_manifest.json` (Task 15); the packaged binary (Task 15); `scripts/run_job.sh`.
- Produces:
  - `autofly_ue5.validate.geometry`: `CAMERA_OFFSET_M = 0.40`; frozen dataclasses `Pillar(tag: str, x: float, y: float, radius: float, height: float)` and `DepthProbe(tag: str, distance_m: float, pose: Pose, pillar: Pillar, expected_depth_m: float)`; `pillars_from_layout_json(data: dict) -> list[Pillar]`; `ray_circle_distance(ox: float, oy: float, yaw: float, cx: float, cy: float, r: float) -> float | None`; `expected_center_depth(pose: Pose, pillar: Pillar, camera_offset_m: float = CAMERA_OFFSET_M) -> float | None`; `choose_depth_probes(pillars: list[Pillar], bounds: Bounds, distances: tuple[float, ...] = (3.0, 6.0, 12.0), altitude_m: float = 2.0, camera_offset_m: float = CAMERA_OFFSET_M, body_clearance_m: float = 1.5, ray_clearance_m: float = 0.3, edge_margin_m: float = 2.0) -> list[DepthProbe]`; `depth_agreement(a: np.ndarray, b: np.ndarray, far_m: float = 30.0) -> dict` (`mask_agreement`, `pixels_compared`, `median_relative_diff`); `tracking_error(commanded: list[float], measured: list[float], settle: int) -> dict`; `yaw_rates_from_yaws(yaws: list[float], dt: float) -> list[float]`; `speed_along_heading(velocity_ned: tuple[float, float, float], yaw: float) -> float`; `mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float`; `pillar_width_px(axis_distance_m: float, radius_m: float, image_width: int = 256, hfov_deg: float = 90.0) -> float` (pinhole image width of a vertical cylinder's silhouette seen from `axis_distance_m`); `contiguous_width(mask_row: np.ndarray, center: int = 128) -> int`.
  - `autofly_ue5.validate.live_m1`: `ORANGE_MATERIAL = "/Game/Geometry/Materials/M_Orange"`; `check_rgb_changes(sim: Simulator) -> dict`, `check_depth(sim: Simulator, probes: list[DepthProbe]) -> dict`, `check_crash(sim: Simulator, probe: DepthProbe) -> dict`, `check_tracking(sim: Simulator) -> dict`, `check_one_step(sim: Simulator) -> dict`, `check_spawn_destroy(sim: Simulator) -> dict` (each with `pass`).
  - CLI `python -m autofly_ue5.validate.live_m1 [--map /Game/AutoFly/Maps/S01] [--instance 0] [--layout runs/levels/s01.layout.json] [--check-map runs/m1/check_map.json] [--package-manifest runs/package/package_manifest.json] [--out runs/m1/m1_gate.json]`; report `checks.{rgb_changes_with_pose, depth_matches_geometry, crash_raises_collision, velocity_tracking, one_step_per_record, spawn_destroy_packaged}` each with `pass`, plus `check_map` (the Task 15 report), `package_manifest` (the Task 15 manifest), `package_matches_manifest` (sha256 of the packaged binary equals the manifest's), `faults` (`xid_since`, `xid_before`, `xid_after` counted from the run's start across reboots, `boot_id`, `device_lost` per `runs/sim/inst<N>/sim*.log`), `faults_ok`, `launch_s`, `vram_used_mib`, `pass`; client log `<out>.client.log`; exit 0 iff pass.
  - Gate thresholds (stated here, recorded in the report): RGB mean absolute difference between a pillar view and an empty view > 10 grey levels and brightness in (20, 235); centre planar depth within max(0.15 m, 2 % of distance) of the ray–cylinder distance at 3, 6 and 12 m, and the pillar's silhouette width in depth row 128 within max(2 px, 10 %) of the 90° HFOV pinhole prediction at each distance; a forward flight at 2 m/s into a pillar yields a collision within 25 steps whose impact point is within radius + 0.5 m of the pillar axis, and the following `reset` to the pillar view restores the camera (stamped camera position within 0.1 m of kinematics + mount offset; depth pixels under 30 m agree with the pre-crash view on ≥ 98 % of the image with median relative difference ≤ 5 %); steady-state mean absolute tracking error ≤ 20 % of the command for forward 2.0 m/s, vertical ±0.5 m/s and yaw rate 0.5 rad/s (first 5 steps excluded); 20 of 20 records advance exactly one step and 200 ms, with RGB, depth and kinematics timestamps all equal to the returned step time; a runtime-spawned cooked `1M_Cube` with the runtime base material `M_Orange`, 3 m ahead, reads 2.1 ± 0.15 m of centre depth, its centre 16 × 16 RGB patch is orange-dominant (mean R > G > B and R − B > 30), and it is gone after `destroy`; `runs/m1/check_map.json` passed; the packaged binary's sha256 matches `runs/package/package_manifest.json`; no new Xid and no `VK_ERROR_DEVICE_LOST` during the run.

- [ ] **Step 1: Write the failing test**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_geometry.py`:
```python
import math

import numpy as np
import pytest

from autofly_ue5.scenes.model import Bounds
from autofly_ue5.sim.types import Pose
from autofly_ue5.validate.geometry import (
    Pillar,
    choose_depth_probes,
    contiguous_width,
    depth_agreement,
    expected_center_depth,
    mean_abs_diff,
    pillar_width_px,
    pillars_from_layout_json,
    ray_circle_distance,
    speed_along_heading,
    tracking_error,
    yaw_rates_from_yaws,
)

BOUNDS = Bounds(-35.0, 35.0, -35.0, 35.0)


def test_ray_circle_distance():
    assert ray_circle_distance(0.0, 0.0, 0.0, 5.0, 0.0, 1.0) == pytest.approx(4.0)
    assert ray_circle_distance(0.0, 0.0, math.pi / 2, 5.0, 0.0, 1.0) is None
    assert ray_circle_distance(0.0, 0.0, 0.0, -5.0, 0.0, 1.0) is None
    assert ray_circle_distance(5.0, 0.0, 0.0, 5.0, 0.0, 1.0) == pytest.approx(1.0)


def test_expected_center_depth_subtracts_camera_offset():
    pillar = Pillar("obs_0000", 5.0, 0.0, 0.5, 10.0)
    assert expected_center_depth(Pose(0.0, 0.0, -2.0, 0.0), pillar) == pytest.approx(4.1)


def test_probe_on_a_lone_pillar_matches_requested_distance():
    probes = choose_depth_probes([Pillar("obs_0000", 0.0, 0.0, 0.5, 10.0)], BOUNDS, distances=(3.0, 6.0))
    assert [p.distance_m for p in probes] == [3.0, 6.0]
    for probe in probes:
        assert probe.expected_depth_m == pytest.approx(probe.distance_m)
        assert probe.pose.z == -2.0


def test_probe_avoids_an_occluded_bearing():
    target = Pillar("obs_0000", 0.0, 0.0, 0.5, 10.0)
    blocker = Pillar("obs_0001", -2.0, 0.0, 0.3, 10.0)  # sits on the first (yaw 0) approach
    probe = choose_depth_probes([target, blocker], BOUNDS, distances=(3.0,))[0]
    assert probe.tag == "obs_0000" and probe.pose.yaw != 0.0
    assert expected_center_depth(probe.pose, target) == pytest.approx(3.0)


def test_no_probe_possible_raises():
    with pytest.raises(ValueError, match="12.0"):
        choose_depth_probes([Pillar("obs_0000", 0.0, 0.0, 0.5, 10.0)], Bounds(-5.0, 5.0, -5.0, 5.0), distances=(12.0,))


def test_tracking_error():
    result = tracking_error([2.0] * 5, [0.0, 1.0, 1.8, 1.8, 1.8], settle=2)
    assert result["mean_abs_error"] == pytest.approx(0.2) and result["relative_error"] == pytest.approx(0.1)
    assert result["samples"] == 3
    with pytest.raises(ValueError):
        tracking_error([1.0], [1.0, 2.0], settle=0)


def test_yaw_rates_wrap_across_pi():
    assert yaw_rates_from_yaws([3.1, -3.1], 0.2)[0] == pytest.approx((2 * math.pi - 6.2) / 0.2)


def test_speed_and_image_difference():
    assert speed_along_heading((0.0, 2.0, 0.0), math.pi / 2) == pytest.approx(2.0)
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    b = np.full((2, 2, 3), 12, dtype=np.uint8)
    assert mean_abs_diff(a, b) == pytest.approx(12.0) and mean_abs_diff(b, a) == pytest.approx(12.0)


def test_pillar_width_and_contiguous_width():
    assert pillar_width_px(3.5, 0.5) == pytest.approx(256 * 0.5 / math.sqrt(3.5**2 - 0.5**2))  # about 36.95 px
    assert pillar_width_px(3.5, 0.5, hfov_deg=60.0) == pytest.approx(pillar_width_px(3.5, 0.5) / math.tan(math.radians(30.0)))
    row = np.array([False, True, True, True, False, True])
    assert contiguous_width(row, 2) == 3 and contiguous_width(row, 5) == 1 and contiguous_width(row, 0) == 0


def test_depth_agreement():
    a = np.array([[1.0, 2.0], [np.inf, 40.0]], dtype=np.float32)
    b = np.array([[1.05, 2.0], [np.inf, np.inf]], dtype=np.float32)
    result = depth_agreement(a, b)
    assert result["mask_agreement"] == 1.0 and result["pixels_compared"] == 2
    assert result["median_relative_diff"] == pytest.approx(0.025, abs=1e-6)
    c = np.array([[np.inf, 2.0], [np.inf, 40.0]], dtype=np.float32)
    assert depth_agreement(a, c)["mask_agreement"] == 0.75
    far = np.full((2, 2), np.inf, dtype=np.float32)
    assert depth_agreement(far, far) == {"mask_agreement": 1.0, "pixels_compared": 0, "median_relative_diff": None}


def test_pillars_from_layout_json():
    data = {"layout": {"instances": [{"tag": "obs_0000", "x": 1.0, "y": 2.0, "radius_m": 0.5, "height_m": 9.0,
                                      "asset": "cylinder", "z_center": -4.5, "yaw": 0.0, "scale": [1, 1, 9], "material": "white"}]}}
    assert pillars_from_layout_json(data) == [Pillar("obs_0000", 1.0, 2.0, 0.5, 9.0)]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_geometry.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'autofly_ue5.validate.geometry'`.

- [ ] **Step 3: Write `autofly_ue5/validate/geometry.py`**

```python
"""Geometry and metric helpers for the M1 live checks (NED metres, yaw radians)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from autofly_ue5.frames import wrap_pi
from autofly_ue5.scenes.model import Bounds
from autofly_ue5.sim.types import Pose

CAMERA_OFFSET_M = 0.40


@dataclass(frozen=True)
class Pillar:
    tag: str
    x: float
    y: float
    radius: float
    height: float


@dataclass(frozen=True)
class DepthProbe:
    tag: str
    distance_m: float
    pose: Pose
    pillar: Pillar
    expected_depth_m: float


def pillars_from_layout_json(data: dict) -> list[Pillar]:
    return [Pillar(i["tag"], float(i["x"]), float(i["y"]), float(i["radius_m"]), float(i["height_m"]))
            for i in data["layout"]["instances"]]


def ray_circle_distance(ox: float, oy: float, yaw: float, cx: float, cy: float, r: float) -> float | None:
    dx, dy = math.cos(yaw), math.sin(yaw)
    fx, fy = ox - cx, oy - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - r * r
    disc = b * b - c
    if disc < 0:
        return None
    root = math.sqrt(disc)
    for t in (-b - root, -b + root):
        if t > 0:
            return t
    return None


def expected_center_depth(pose: Pose, pillar: Pillar, camera_offset_m: float = CAMERA_OFFSET_M) -> float | None:
    cam_x = pose.x + camera_offset_m * math.cos(pose.yaw)
    cam_y = pose.y + camera_offset_m * math.sin(pose.yaw)
    return ray_circle_distance(cam_x, cam_y, pose.yaw, pillar.x, pillar.y, pillar.radius)


def _segment_point_distance(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / length2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def choose_depth_probes(
    pillars: list[Pillar],
    bounds: Bounds,
    distances: tuple[float, ...] = (3.0, 6.0, 12.0),
    altitude_m: float = 2.0,
    camera_offset_m: float = CAMERA_OFFSET_M,
    body_clearance_m: float = 1.5,
    ray_clearance_m: float = 0.3,
    edge_margin_m: float = 2.0,
) -> list[DepthProbe]:
    probes = []
    for distance in distances:
        found = None
        for pillar in sorted(pillars, key=lambda p: p.tag):
            others = [q for q in pillars if q.tag != pillar.tag]
            for k in range(8):
                yaw = wrap_pi(k * math.pi / 4)
                ux, uy = math.cos(yaw), math.sin(yaw)
                standoff = pillar.radius + distance + camera_offset_m
                x, y = pillar.x - standoff * ux, pillar.y - standoff * uy
                if not (bounds.x_min + edge_margin_m <= x <= bounds.x_max - edge_margin_m
                        and bounds.y_min + edge_margin_m <= y <= bounds.y_max - edge_margin_m):
                    continue
                if any(math.hypot(x - q.x, y - q.y) < q.radius + body_clearance_m for q in others):
                    continue
                cam = (x + camera_offset_m * ux, y + camera_offset_m * uy)
                surface = (pillar.x - pillar.radius * ux, pillar.y - pillar.radius * uy)
                if any(_segment_point_distance(*cam, *surface, q.x, q.y) < q.radius + ray_clearance_m for q in others):
                    continue
                pose = Pose(x, y, -altitude_m, yaw)
                found = DepthProbe(pillar.tag, distance, pose, pillar, expected_center_depth(pose, pillar, camera_offset_m))
                break
            if found is not None:
                break
        if found is None:
            raise ValueError(f"no unobstructed depth probe at {distance} m")
        probes.append(found)
    return probes


def depth_agreement(a: np.ndarray, b: np.ndarray, far_m: float = 30.0) -> dict:
    """Compare two planar-depth images of the same pose: which pixels are nearer than far_m, and by how much they differ."""
    near_a = np.isfinite(a) & (a < far_m)
    near_b = np.isfinite(b) & (b < far_m)
    both = near_a & near_b
    relative = np.abs(a[both] - b[both]) / np.maximum(a[both], 1e-3)
    return {"mask_agreement": float(np.mean(near_a == near_b)), "pixels_compared": int(both.sum()),
            "median_relative_diff": float(np.median(relative)) if both.any() else None}


def tracking_error(commanded: list[float], measured: list[float], settle: int) -> dict:
    if len(commanded) != len(measured) or settle >= len(commanded):
        raise ValueError("commanded and measured must have equal length greater than settle")
    c, m = commanded[settle:], measured[settle:]
    mae = sum(abs(a - b) for a, b in zip(m, c)) / len(c)
    scale = sum(abs(v) for v in c) / len(c)
    return {"mean_abs_error": mae, "mean_commanded": sum(c) / len(c), "mean_measured": sum(m) / len(m),
            "relative_error": mae / scale if scale > 0 else None, "samples": len(c)}


def yaw_rates_from_yaws(yaws: list[float], dt: float) -> list[float]:
    return [wrap_pi(b - a) / dt for a, b in zip(yaws, yaws[1:])]


def speed_along_heading(velocity_ned: tuple[float, float, float], yaw: float) -> float:
    return velocity_ned[0] * math.cos(yaw) + velocity_ned[1] * math.sin(yaw)


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def pillar_width_px(axis_distance_m: float, radius_m: float, image_width: int = 256, hfov_deg: float = 90.0) -> float:
    """Pinhole image width of a vertical cylinder whose axis lies on the optical axis at axis_distance_m.

    The silhouette's tangent rays make asin(r / D) with the axis, i.e. an image half-width of f * r / sqrt(D^2 - r^2),
    with focal length f = (image_width / 2) / tan(hfov / 2)."""
    return image_width * radius_m / (math.sqrt(axis_distance_m**2 - radius_m**2) * math.tan(math.radians(hfov_deg) / 2.0))


def contiguous_width(mask_row: np.ndarray, center: int = 128) -> int:
    """Length of the run of True values in mask_row that contains index center (0 when center is False)."""
    if not mask_row[center]:
        return 0
    left = center
    while left > 0 and mask_row[left - 1]:
        left -= 1
    right = center
    while right < len(mask_row) - 1 and mask_row[right + 1]:
        right += 1
    return right - left + 1
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_geometry.py`
Expected: `11 passed`.

- [ ] **Step 5: Write the failing offline test of the gate checks**

`/home/nvidiasims/research_uav/autofly_ue5/tests/test_live_m1_fake.py`:
```python
from autofly_ue5.paths import SCENES_DIR
from autofly_ue5.scenes.generate import generate_layout
from autofly_ue5.scenes.model import load_registry, load_scene_file
from autofly_ue5.sim.fake import FakeSimulator
from autofly_ue5.validate.geometry import choose_depth_probes, pillars_from_layout_json
from autofly_ue5.validate.live_m1 import check_crash, check_one_step, check_rgb_changes, check_tracking

SCENE = load_scene_file(SCENES_DIR / "s01_white_pillars.json")
LAYOUT_JSON = {"layout": generate_layout(SCENE, load_registry()).to_json()}


def _probes():
    return choose_depth_probes(pillars_from_layout_json(LAYOUT_JSON), SCENE.bounds)


def test_depth_probes_exist_for_the_generated_s01_layout():
    probes = _probes()
    assert [p.distance_m for p in probes] == [3.0, 6.0, 12.0]
    assert all(abs(p.expected_depth_m - p.distance_m) < 1e-6 for p in probes)


def test_gate_checks_pass_on_the_fake_simulator():
    pillars = pillars_from_layout_json(LAYOUT_JSON)
    sim = FakeSimulator(obstacles=[(p.x, p.y, p.radius) for p in pillars], image_size=32)
    sim.launch("/Game/AutoFly/Maps/S01", 0)
    assert check_rgb_changes(sim)["pass"] is True
    crash = check_crash(sim, _probes()[1])
    assert crash["pass"] is True and crash["reset_after_crash"]["depth"]["mask_agreement"] == 1.0
    one_step = check_one_step(sim)
    assert one_step["pass"] is True and all(r["frame_times_match"] for r in one_step["rows"])
    assert check_tracking(sim)["pass"] is True
```
(The fake has perfect tracking, pose-seeded RGB, exact steps and circle collisions; this pins the check logic offline. `check_spawn_destroy` needs rendered depth and is exercised only live.)

- [ ] **Step 6: Run it to verify it fails**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_live_m1_fake.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'autofly_ue5.validate.live_m1'`.

- [ ] **Step 7: Write `autofly_ue5/validate/live_m1.py`**

```python
"""M1 gate: live checks of spec section 11 on the packaged s01 simulator, through the Simulator interface.

env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.live_m1 --out runs/m1/m1_gate.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from autofly_ue5.gpu import gpu_memory_mib
from autofly_ue5.paths import PACKAGED_BINARY, RUNS_DIR
from autofly_ue5.scenes.model import Bounds
from autofly_ue5.sim.airsim_backend import ProjectAirSimSimulator
from autofly_ue5.sim.process import instance_dir, route_client_log
from autofly_ue5.sim.protocol import Simulator
from autofly_ue5.sim.types import Pose
from autofly_ue5.validate.engine_check import boot_id, count_device_lost, xid_count
from autofly_ue5.validate.geometry import (
    CAMERA_OFFSET_M,
    DepthProbe,
    choose_depth_probes,
    contiguous_width,
    depth_agreement,
    expected_center_depth,
    mean_abs_diff,
    pillar_width_px,
    pillars_from_layout_json,
    speed_along_heading,
    tracking_error,
    yaw_rates_from_yaws,
)

DT = 0.2
# Opaque base UMaterial cooked with /Game/Geometry; M0 (Task 7 Step 11) showed it renders orange at runtime.
ORANGE_MATERIAL = "/Game/Geometry/Materials/M_Orange"
OPEN_POSE = Pose(-31.0, -25.0, -2.0, math.pi / 2)  # south start band (x_min edge), facing east along the band
PILLAR_VIEW = Pose(-31.0, 0.0, -2.0, 0.0)          # facing north into the pillar field
EMPTY_VIEW = Pose(-31.0, 0.0, -2.0, math.pi)        # facing south (-x), out of the scene over bare ground
SPAWN_POSE = Pose(-31.0, -10.0, -2.0, 0.0)          # runtime cube in the free start band (pillars start at x = -28)
SPAWN_VIEW = Pose(-31.0, -13.0, -2.0, math.pi / 2)  # 3 m west of the cube, facing east at it


def _finite(v: float | None) -> float | None:
    return v if v is not None and math.isfinite(v) else None


def check_rgb_changes(sim: Simulator) -> dict:
    a = sim.reset(PILLAR_VIEW).rgb
    b = sim.reset(EMPTY_VIEW).rgb
    a2 = sim.reset(PILLAR_VIEW).rgb
    out = {"diff_pillar_vs_empty": mean_abs_diff(a, b), "diff_pillar_repeat": mean_abs_diff(a, a2),
           "brightness_pillar_view": float(a.mean()), "brightness_empty_view": float(b.mean())}
    out["pass"] = out["diff_pillar_vs_empty"] > 10.0 and 20.0 < out["brightness_pillar_view"] < 235.0
    return out


def check_depth(sim: Simulator, probes: list[DepthProbe]) -> dict:
    rows = []
    for probe in probes:
        obs = sim.reset(probe.pose)
        measured = float(np.median(obs.depth[126:130, 126:130]))
        expected = expected_center_depth(obs.pose, probe.pillar)
        tolerance = max(0.15, 0.02 * probe.distance_m)
        error = abs(measured - expected) if expected is not None and math.isfinite(measured) else None
        # Field of view (spec §4.1): the pillar silhouette in row 128 (camera height) against the 90° pinhole prediction.
        # Silhouette pixels have planar depth below the axis distance; background pixels are farther or +inf.
        cam_x = obs.pose.x + CAMERA_OFFSET_M * math.cos(obs.pose.yaw)
        cam_y = obs.pose.y + CAMERA_OFFSET_M * math.sin(obs.pose.yaw)
        axis_distance = math.hypot(probe.pillar.x - cam_x, probe.pillar.y - cam_y)
        expected_width = pillar_width_px(axis_distance, probe.pillar.radius)
        row = obs.depth[128]
        width = contiguous_width(np.isfinite(row) & (row < axis_distance + 0.3))
        width_tolerance = max(2.0, 0.1 * expected_width)
        hfov_consistent = abs(width - expected_width) <= width_tolerance
        rows.append({"tag": probe.tag, "distance_m": probe.distance_m, "measured_m": _finite(measured), "expected_m": expected,
                     "error_m": error, "tolerance_m": tolerance, "width_px": width, "expected_width_px": expected_width,
                     "width_tolerance_px": width_tolerance, "hfov_consistent": hfov_consistent,
                     "pose": [obs.pose.x, obs.pose.y, obs.pose.z, obs.pose.yaw],
                     "pass": error is not None and error <= tolerance and hfov_consistent})
    return {"probes": rows, "pass": len(rows) == len(probes) and all(r["pass"] for r in rows)}


def check_crash(sim: Simulator, probe: DepthProbe) -> dict:
    reference = sim.reset(PILLAR_VIEW)
    sim.reset(probe.pose)
    out: dict = {"pass": False, "reason": "no collision within 25 steps"}
    for step in range(25):
        sim.command_velocity(2.0, 0.0, 0.0)
        sim.step(DT)
        obs = sim.observe()
        if obs.step_collisions:
            event = obs.step_collisions[0]
            axis_distance = math.hypot(event.impact_point[0] - probe.pillar.x, event.impact_point[1] - probe.pillar.y)
            out = {"steps_to_collision": step + 1, "object_name": event.object_name, "impact_to_pillar_axis_m": axis_distance,
                   "pillar_radius_m": probe.pillar.radius, "collided_flag": obs.collided,
                   "camera_pose_error_at_collision_m": obs.camera_pose_error_m,
                   "pass": obs.collided and axis_distance <= probe.pillar.radius + 0.5}
            break
    # A reset after a crash must bring the Unreal actor and its camera back (research gotcha 6); the backend raises
    # CameraPoseError when it cannot.
    try:
        after = sim.reset(PILLAR_VIEW)
    except Exception as err:
        out["reset_after_crash"] = {"error": f"{type(err).__name__}: {err}", "pass": False}
        out["pass"] = False
        return out
    depth = depth_agreement(reference.depth, after.depth)
    reset_ok = (
        after.camera_pose_error_m < 0.1 and after.collided is False and depth["mask_agreement"] >= 0.98
        and (depth["median_relative_diff"] is None or depth["median_relative_diff"] <= 0.05)
    )
    out["reset_after_crash"] = {"camera_pose_error_m": after.camera_pose_error_m, "depth": depth,
                                "rgb_mean_abs_diff": mean_abs_diff(reference.rgb, after.rgb), "pass": reset_ok}
    out["pass"] = bool(out["pass"]) and reset_ok
    return out


def check_tracking(sim: Simulator) -> dict:
    sim.reset(OPEN_POSE)
    forward = []
    for _ in range(15):
        sim.command_velocity(2.0, 0.0, 0.0)
        sim.step(DT)
        obs = sim.observe()
        forward.append(speed_along_heading(obs.velocity_ned, obs.pose.yaw))
    sim.reset(OPEN_POSE)
    up, down = [], []
    for v_z, sink in [(0.5, up)] * 10 + [(-0.5, down)] * 10:
        sim.command_velocity(0.0, 0.0, v_z)
        sim.step(DT)
        sink.append(-sim.observe().velocity_ned[2])
    obs = sim.reset(OPEN_POSE)
    yaws, body_rates = [obs.pose.yaw], []
    for _ in range(15):
        sim.command_velocity(0.0, 0.5, 0.0)
        sim.step(DT)
        obs = sim.observe()
        yaws.append(obs.pose.yaw)
        body_rates.append(obs.yaw_rate)
    axes = {
        "forward": tracking_error([2.0] * 15, forward, settle=5),
        "vertical_up": tracking_error([0.5] * 10, up, settle=5),
        "vertical_down": tracking_error([-0.5] * 10, down, settle=5),
        "yaw_rate": tracking_error([0.5] * 15, yaw_rates_from_yaws(yaws, DT), settle=5),
        "yaw_rate_body_angular_z": tracking_error([0.5] * 15, body_rates, settle=5),
    }
    gated = ("forward", "vertical_up", "vertical_down", "yaw_rate")
    return {"axes": axes, "threshold_relative_error": 0.2,
            "pass": all(axes[k]["relative_error"] is not None and axes[k]["relative_error"] <= 0.2 for k in gated)}


def check_one_step(sim: Simulator) -> dict:
    """Each record: one clock step of 200 ms, and the RGB, depth and kinematics that built it all carry that step's time."""
    sim.reset(OPEN_POSE)
    rows = []
    start = time.monotonic()
    for _ in range(20):
        steps_before, time_before = sim.steps_taken, sim.observe().sim_time_ns
        sim.command_velocity(1.0, 0.0, 0.0)
        t = sim.step(DT)
        obs = sim.observe()
        rows.append({"steps": sim.steps_taken - steps_before, "dt_ns": t - time_before,
                     "times_ns": [obs.rgb_time_ns, obs.depth_time_ns, obs.kinematics_time_ns],
                     "frame_times_match": obs.rgb_time_ns == obs.depth_time_ns == obs.kinematics_time_ns == t})
    wall = time.monotonic() - start
    ok = all(r["steps"] == 1 and r["dt_ns"] == 200_000_000 and r["frame_times_match"] for r in rows)
    return {"records": len(rows), "records_per_s": len(rows) / wall, "rows": rows, "pass": ok}


def check_spawn_destroy(sim: Simulator) -> dict:
    """Runtime spawn of a cooked Blocks mesh with a runtime material in the packaged build (spec §4.1), seen in depth and
    colour, then destroyed. The backend raises if set_object_material returns False."""
    name = sim.spawn("AF_M1_Cube", "1M_Cube", SPAWN_POSE, (1.0, 1.0, 1.0), ORANGE_MATERIAL)
    with_obs = sim.reset(SPAWN_VIEW)
    with_cube = float(np.median(with_obs.depth[126:130, 126:130]))
    red, green, blue = (float(v) for v in with_obs.rgb[120:136, 120:136].reshape(-1, 3).mean(axis=0))
    orange_dominant = red > green > blue and red - blue > 30.0
    sim.destroy(name)
    without_cube = float(np.median(sim.reset(SPAWN_VIEW).depth[126:130, 126:130]))
    expected = 3.0 - 0.5 - CAMERA_OFFSET_M
    gone = math.isinf(without_cube) or without_cube > expected + 1.0
    return {"spawned_name": name, "material": ORANGE_MATERIAL, "center_depth_with_cube_m": _finite(with_cube),
            "expected_m": expected, "center_rgb_with_cube": [red, green, blue], "orange_dominant": orange_dominant,
            "center_depth_after_destroy_m": _finite(without_cube),
            "pass": name.startswith("AF_M1_Cube") and abs(with_cube - expected) <= 0.15 and orange_dominant and gone}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="/Game/AutoFly/Maps/S01")
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--layout", type=Path, default=RUNS_DIR / "levels" / "s01.layout.json")
    parser.add_argument("--check-map", type=Path, default=RUNS_DIR / "m1" / "check_map.json")
    parser.add_argument("--package-manifest", type=Path, default=RUNS_DIR / "package" / "package_manifest.json")
    parser.add_argument("--out", type=Path, default=RUNS_DIR / "m1" / "m1_gate.json")
    args = parser.parse_args(argv)
    data = json.loads(args.layout.read_text())
    b = data["layout"]["bounds"]
    probes = choose_depth_probes(pillars_from_layout_json(data), Bounds(b["x_min"], b["x_max"], b["y_min"], b["y_max"]))
    report: dict = {"map": args.map, "layout": str(args.layout), "scene_sha256": data["scene_sha256"], "checks": {},
                    "check_map": json.loads(args.check_map.read_text()) if args.check_map.exists() else {},
                    "package_manifest": json.loads(args.package_manifest.read_text()) if args.package_manifest.exists() else {}}
    binary_digest = hashlib.sha256()
    with open(PACKAGED_BINARY, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            binary_digest.update(chunk)
    report["package_matches_manifest"] = (
        report["package_manifest"].get("binary", {}).get("sha256") == binary_digest.hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    route_client_log(args.out.with_suffix(".client.log"))
    sim = ProjectAirSimSimulator(scene_config="scene_autofly_s01.jsonc")
    checks = [
        ("rgb_changes_with_pose", lambda: check_rgb_changes(sim)),
        ("depth_matches_geometry", lambda: check_depth(sim, probes)),
        ("crash_raises_collision", lambda: check_crash(sim, probes[1])),
        ("velocity_tracking", lambda: check_tracking(sim)),
        ("one_step_per_record", lambda: check_one_step(sim)),
        ("spawn_destroy_packaged", lambda: check_spawn_destroy(sim)),
    ]
    # Xid lines are counted from this run's start without journalctl's -k (which implies -b, the current boot only).
    run_started = time.strftime("%Y-%m-%d %H:%M:%S")
    xid_before = xid_count(run_started)
    try:
        start = time.monotonic()
        sim.launch(args.map, args.instance)
        report["launch_s"] = time.monotonic() - start
        for name, run in checks:
            report["checks"][name] = run()
            args.out.write_text(json.dumps(report, indent=2))
        report["vram_used_mib"] = gpu_memory_mib()[0]
    except Exception as err:
        report["error"] = f"{type(err).__name__}: {err}"
    finally:
        sim.close()
    logs = sorted(instance_dir(args.instance).glob("sim*.log"))
    report["faults"] = {"xid_since": run_started, "xid_before": xid_before, "xid_after": xid_count(run_started),
                        "boot_id": boot_id(), "device_lost": count_device_lost(logs)}
    report["faults_ok"] = (report["faults"]["xid_after"] == xid_before and len(logs) > 0
                           and sum(report["faults"]["device_lost"].values()) == 0)
    report["pass"] = ("error" not in report and len(report["checks"]) == len(checks)
                      and all(c["pass"] for c in report["checks"].values())
                      and report["check_map"].get("pass") is True and report["package_matches_manifest"]
                      and report["faults_ok"])
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"pass": report["pass"], "error": report.get("error"), "check_map": report["check_map"].get("pass"),
                      "package_matches_manifest": report["package_matches_manifest"], "faults_ok": report["faults_ok"],
                      "checks": {k: v["pass"] for k, v in report["checks"].items()}}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8: Run the offline gate-check test to verify it passes**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_live_m1_fake.py`
Expected: `2 passed`. If `choose_depth_probes` raises `ValueError: no unobstructed depth probe`, stop and report: the s01 layout leaves no unobstructed approach, which contradicts the scene design.

- [ ] **Step 9: Run the full offline suite**

Run: `cd /home/nvidiasims/research_uav/autofly_ue5 && env -u PYTHONPATH .venv/bin/python -m pytest && grep -rln "projectairsim" autofly_ue5 --include=*.py | grep -v "^autofly_ue5/sim/"`
Expected: `124 passed`; grep prints nothing.

- [ ] **Step 10: Run the M1 gate live as a job (launches and stops the packaged simulator itself; total wait budget 40 minutes)**

Run (Bash tool `timeout` 600000 ms): `cd /home/nvidiasims/research_uav/autofly_ue5 && nvidia-smi --query-gpu=memory.used --format=csv,noheader && jq .pass runs/m1/check_map.json && jq -r .binary.sha256 runs/package/package_manifest.json && bash scripts/run_job.sh start live_m1 -- env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.live_m1 --out runs/m1/m1_gate.json && bash scripts/run_job.sh wait live_m1 540; echo "wait=$?"` (repeat `bash scripts/run_job.sh wait live_m1 540` while it returns 124, within the budget; if the budget is used up: `bash scripts/run_job.sh stop live_m1`, then `env -u PYTHONPATH .venv/bin/python scripts/stop_sim.py --instance 0`, and report).
Expected: GPU below 2000 MiB, `true` and a 64-character hash before the run; last log line `{"pass": true, "error": null, "check_map": true, "package_matches_manifest": true, "faults_ok": true, "checks": {"rgb_changes_with_pose": true, "depth_matches_geometry": true, "crash_raises_collision": true, "velocity_tracking": true, "one_step_per_record": true, "spawn_destroy_packaged": true}}`; `job live_m1 finished: exit=0`; `wait=0`. Then run `cd /home/nvidiasims/research_uav/autofly_ue5 && ls runs/sim/inst0/ && jq '.faults, .checks.crash_raises_collision.reset_after_crash, .checks.spawn_destroy_packaged' runs/m1/m1_gate.json && jq '.checks.velocity_tracking.axes | map_values({relative_error, mean_measured})' runs/m1/m1_gate.json && jq '.checks.depth_matches_geometry.probes[] | {distance_m, error_m, width_px, expected_width_px}' runs/m1/m1_gate.json`: no `pid.json` in `runs/sim/inst0/` (the backend stopped its own process); `xid_after` equal to `xid_before` and every `device_lost` count `0`; `spawn_destroy_packaged.center_rgb_with_cube` orange-dominant; per-probe silhouette widths near 37, 20 and 10 px for 3, 6 and 12 m (radius 0.5 m; exact values depend on each pillar's radius); yaw-rate sign and rad/s units are confirmed when `yaw_rate.mean_measured` ≈ +0.5. Any `false` or `error` (including `CameraPoseError` or `CommandTimeoutError`): stop and report the full `runs/m1/m1_gate.json`, `tail -80 runs/sim/inst0/sim.log` and `tail -40 runs/m1/m1_gate.client.log`; if `runs/sim/inst0/pid.json` still exists, stop it with `scripts/stop_sim.py --instance 0` first. Do not loosen thresholds without the user.

- [ ] **Step 11: Commit**

```bash
cp /home/nvidiasims/research_uav/autofly_ue5/runs/m1/m1_gate.json /home/nvidiasims/research_uav/autofly_ue5/docs/gates/m1_gate.json
git -C /home/nvidiasims/research_uav/autofly_ue5 add autofly_ue5/validate/geometry.py autofly_ue5/validate/live_m1.py tests/test_geometry.py tests/test_live_m1_fake.py docs/gates/m1_gate.json && git -C /home/nvidiasims/research_uav/autofly_ue5 commit -m "Pass the M1 gate on packaged s01: RGB vs pose, depth vs geometry, crash collision and reset, velocity tracking, one step per record, runtime spawn, map check and GPU faults" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 12: Milestone stop (spec §12)**

Report to the user: `docs/gates/m1_gate.json` (tracking errors per axis, depth errors per distance, reset-after-crash camera error, records/s and VRAM of the packaged build, Xid/`VK_ERROR_DEVICE_LOST` counts), `docs/gates/m1_check_map.json`, the M1 package manifest `docs/gates/m1_package_manifest.json`, and the reachability crossing rule added on top of spec §6.2 (Task 13; as confirmed at the M0 stop), and wait for the go-ahead before any M2 planning.

---

## Not in Plan 1

Each later milestone gets its own plan, written after the previous gate passes and the user gives the go-ahead (spec §12):

- **M2 — SAC expert on s01** (`autofly_ue5/expert/`): environment wrapper, CNN+MLP network, stable-baselines3 SAC, vectorised simulator instances, reward shaping, ≥ 95 % success over 200 episodes, throughput and instance-count measurements.
- **M3 — Collector, dataset writer, validator** (`autofly_ue5/collect/`, `autofly_ue5/dataset/`, `autofly_ue5/validate/` dataset checks): episode protocol with targets/distractors and a0, instruction templates, TFDS/RLDS shards, provenance and manifest, state[9] decoding against the two real episodes, 100-episode pilot. Runtime target colouring via `set_object_material` with base materials is designed there.
- **M4 — Asset library and scenes s02–s12, s05r, s06r**: target and distractor assets (Fab/UE sample content with licences), full `assets/registry.json`, `poisson`/`clusters`/`stacks` placements, non-circle footprints, per-scene builds and live checks.
- **M5 — Experts for s02–s10, full collection, Grounding-DINO rebalancing.**
- **M6 — Test splits and evaluation harness** (SR, CR, PER, d_col choice, scripted baseline).
- **AutoFly-checkpoint pilot**: no public checkpoint exists (checked 2026-09-15); nothing in Plan 1 depends on it.
- Also out of Plan 1: exposure locking for dataset images, multi-instance training orchestration, any change of platform (Cosys-AirSim fallback) — the latter only by user decision after an M0/M1 failure report.
