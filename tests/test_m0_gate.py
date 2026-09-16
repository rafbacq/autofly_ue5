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


def test_gate_fails_on_failing_engine_check():
    gate = _gate(engine={"pass": False})
    assert gate["engine_check"] is False
    assert gate["pass"] is False
    # the failure came from engine_check specifically, not a wholesale collapse of the assembled dict
    assert gate["plugin_manifest"] is True and gate["blocks_editor_build"] is True
    assert gate["editor_python_probe"] is True and gate["smoke_single_instance"] is True
    assert gate["second_instance"] is True and gate["gpu_faults"] is True


def test_gate_fails_on_failing_single_instance_smoke():
    gate = _gate(smoke=_smoke(False))
    assert gate["smoke_single_instance"] is False
    assert gate["pass"] is False
    # the failure came from smoke_single_instance specifically, not a wholesale collapse
    assert gate["engine_check"] is True and gate["plugin_manifest"] is True
    assert gate["blocks_editor_build"] is True and gate["editor_python_probe"] is True
    assert gate["second_instance"] is True and gate["gpu_faults"] is True


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
