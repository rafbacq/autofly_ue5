"""M0 gate: VRAM samples, GPU-fault and ~/.config/Epic records, and the assembled gate report.

env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.m0_gate vram baseline
env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.m0_gate faults
env -u PYTHONPATH .venv/bin/python -m autofly_ue5.validate.m0_gate assemble
"""

import json
import sys
import time
from pathlib import Path

from autofly_ue5.gpu import gpu_memory_mib
from autofly_ue5.paths import ROOT, RUNS_DIR
from autofly_ue5.validate.engine_check import (
    boot_id,
    count_device_lost,
    epic_config_usage,
    instance_logs_since,
    kernel_journal_readable,
    out_of_root_state,
    xid_count,
)

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


def m0_fault_logs(since_epoch: float, sim_root: Path = RUNS_DIR / "sim", m0_dir: Path = M0_DIR) -> list[Path]:
    """The simulator and editor-probe logs written since the M0 run started. Scanning every sim*.log ever written
    (as before) would pull every later M1/M2 run's logs into an M0 re-verification (C7)."""
    sim_logs = [p for d in sorted(Path(sim_root).glob("inst*")) for p in instance_logs_since(d, since_epoch)]
    return sim_logs + instance_logs_since(m0_dir, since_epoch, pattern="editor_open*.log")


def fault_record(engine: dict, xid_now: int, device_lost: dict[str, int], epic_after: dict, boot_id_now: str,
                 out_of_root_after: dict[str, bool], journal_readable: bool = True) -> dict:
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
        "kernel_journal_readable": journal_readable,
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
            and faults.get("kernel_journal_readable") is True
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
        engine = _load(M0_DIR / "engine_check.json")
        started = engine.get("started_at")
        since_epoch = time.mktime(time.strptime(started, "%Y-%m-%d %H:%M:%S")) if started else 0.0
        logs = m0_fault_logs(since_epoch)
        record = fault_record(engine, xid_count(started), count_device_lost(logs), epic_config_usage(),
                              boot_id(), out_of_root_state(), journal_readable=kernel_journal_readable())
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
