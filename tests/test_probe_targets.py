"""scripts/probe_targets.py: the live check of the target pool on the packaged simulator (plan 5 C6), run in the window
after the editor import. Offline here, against a FakeSimulator that renders spawned objects as their registry boxes:
the probe must pass a simulator that lists, spawns, sizes, renders and paints every target, and fail one that misses a
cooked asset, paints a base-material target grey, reports bounds in the wrong unit, or renders nothing in depth."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from autofly_ue5.paths import ROOT
from autofly_ue5.sim.fake import FakeSimulator

PATCH = ROOT / "assets" / "pending" / "runtime_materials.registry.json"
SPAWN_TABLE = ("Cylinder", "Cube", "Cone", "Sphere", "tgt_box")


def _write_world(tmp_path: Path) -> tuple[Path, Path]:
    """A registry (the real one, the runtime colours merged, one imported base-pivot target) and a pool over it."""
    registry = json.loads((ROOT / "assets" / "registry.json").read_text())
    registry["materials"].update({k: {**v, "roughness": 0.5, "made_by": "test"} for k, v in json.loads(PATCH.read_text())["materials"].items()})
    registry["assets"]["tgt_box"] = {"ue_path": "/Game/AutoFly/Targets/tgt_box/tgt_box", "base_size_m": [1.0, 0.5, 0.8],
                                     "pivot": "base", "footprint": "circle", "category": "other", "role": "target", "seen": True,
                                     "measured_extent_cm_at_unit_scale": [50.0, 25.0, 40.0], "bounds_origin_cm": [0.0, 0.0, 40.0]}
    src = {"source": "polyhaven", "id": "box", "licence": "CC0-1.0", "page": "https://polyhaven.com/a/box"}

    def entry(name, asset, scale, material, status="primitive", source=None):
        return {"name": name, "category": "geometry", "seen": True, "asset": asset, "scale": scale, "material": material,
                "status": status, "source": source, "height_m": None, "stand_in": status == "placeholder"}

    pool = {"version": 1, "targets": [
        entry("orange cylinder", "cylinder", [1.0, 1.0, 2.0], "orange"),
        entry("red cube", "cube", [1.2, 1.2, 1.2], "runtime_red"),
        entry("blue cone", "cone", [1.0, 1.0, 1.5], "runtime_blue"),
        entry("grey box", "tgt_box", [1.0, 1.0, 1.0], None, "imported", src),
        entry("yellow box", "tgt_box", [2.0, 2.0, 2.0], "runtime_yellow", "imported", src),
        entry("red van", "tgt_box", [1.5, 1.5, 1.5], None, "placeholder", src),
        entry("blue hatchback", None, None, None, "candidate", src)]}
    reg_path, pool_path = tmp_path / "registry.json", tmp_path / "targets.json"
    reg_path.write_text(json.dumps(registry))
    pool_path.write_text(json.dumps(pool))
    return reg_path, pool_path


class BoxRenderer(FakeSimulator):
    """Renders the ground (z = 0) and every spawned object as its registry box: planar depth from a 90 deg pinhole
    camera 0.4 m ahead of the body (as FakeSimulator's own pillar depth), and a flat RGB with the object painted its base
    material's colour. Variants switch one behaviour off."""

    N = 64

    def __init__(self, registry, *, assets=SPAWN_TABLE, paint=True, bounds_scale=1.0, visible=True):
        super().__init__(image_size=self.N, assets=assets)
        self.registry, self.paint, self.bounds_scale, self.visible = registry, paint, bounds_scale, visible
        self.by_short = {}
        for a in registry.assets.values():
            self.by_short.setdefault(a.ue_path.rsplit("/", 1)[-1], a)
        self.rgba = {m.ue_path: m.rgba for m in registry.materials.values() if m.rgba is not None}

    def box(self, name):
        asset_name, pose, scale, _material = self._objects[name]
        a = self.by_short[asset_name]
        o = a.bounds_origin_cm or (0.0, 0.0, 0.0)
        size = tuple(b * s for b, s in zip(a.base_size_m, scale))
        return (pose.x + o[0] * scale[0] / 100, pose.y + o[1] * scale[1] / 100, pose.z - o[2] * scale[2] / 100), size

    def object_bounds(self, name):
        if name in self._objects:
            centre, size = self.box(name)
            return centre, tuple(v * self.bounds_scale for v in size)
        return super().object_bounds(name)

    def observe(self):
        obs = super().observe()
        n, p = self.N, obs.pose
        a = (np.arange(n) + 0.5 - n / 2) / (n / 2)
        b = (np.arange(n) + 0.5 - n / 2) / (n / 2)
        A, B = np.meshgrid(a, b)
        fwd, right = np.array([math.cos(p.yaw), math.sin(p.yaw), 0.0]), np.array([-math.sin(p.yaw), math.cos(p.yaw), 0.0])
        d = fwd[None, None, :] + A[..., None] * right[None, None, :] + B[..., None] * np.array([0, 0, 1.0])[None, None, :]
        c = np.array([p.x, p.y, p.z]) + 0.4 * fwd
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(B > 0, -c[2] / B, np.inf)
            rgb = np.zeros((n, n, 3), np.uint8)
            rgb[:] = (90, 110, 80)
            for name, (_asset, _pose, _scale, material) in self._objects.items():
                if not self.visible:
                    continue
                (cx, cy, cz), (sx, sy, sz) = self.box(name)
                lo, hi = np.array([cx - sx / 2, cy - sy / 2, cz - sz / 2]), np.array([cx + sx / 2, cy + sy / 2, cz + sz / 2])
                t1, t2 = (lo - c) / d, (hi - c) / d
                t_near = np.nanmax(np.minimum(t1, t2), axis=-1)
                t_far = np.nanmin(np.maximum(t1, t2), axis=-1)
                hit = (t_near <= t_far) & (t_far > 0) & (t_near < depth)
                depth = np.where(hit, t_near, depth)
                colour = (np.array(self.rgba[material][:3]) * 220).astype(np.uint8) if (self.paint and material in self.rgba) else (128, 128, 128)
                rgb[hit] = colour
        return dataclasses.replace(obs, depth=depth.astype(np.float32), rgb=rgb)


class LateRenderer(BoxRenderer):
    """Draws a spawned object only once it has existed for `appear_after` steps: live, a target spawned for the first
    time in a session drew nothing on step 1 and its whole outline by step 5 (runs/m4/targets_settle_diag/settle.json)."""

    def __init__(self, registry, *, appear_after: int, **kwargs):
        super().__init__(registry, **kwargs)
        self.appear_after = appear_after
        self.spawned_at: dict[str, int] = {}

    def spawn(self, name, asset, pose, scale, material=None):
        unique = super().spawn(name, asset, pose, scale, material)
        self.spawned_at[unique] = self.steps_taken
        return unique

    def observe(self):
        hidden = {n: v for n, v in self._objects.items() if self.steps_taken - self.spawned_at[n] < self.appear_after}
        for name in hidden:
            del self._objects[name]
        try:
            return super().observe()
        finally:
            self._objects.update(hidden)


def _setup(tmp_path, **kwargs):
    from autofly_ue5.dataset.targets import TargetPool
    from autofly_ue5.scenes.model import load_registry
    from autofly_ue5.scenes.resolve import resolve_scene

    reg_path, pool_path = _write_world(tmp_path)
    registry = load_registry(reg_path)
    pool = TargetPool.load(pool_path, registry=registry)
    sim = BoxRenderer(registry, **kwargs)
    sim.launch("/Game/AutoFly/Maps/S01", 5)
    return sim, pool, registry, resolve_scene("s01").layout


def _rows(summary):
    return {row["name"]: row for row in summary["targets"]}


def test_a_simulator_that_lists_spawns_sizes_renders_and_paints_every_target_passes(tmp_path):
    from scripts.probe_targets import run_probe

    sim, pool, registry, layout = _setup(tmp_path)
    summary = run_probe(sim, pool, registry, layout, out_dir=tmp_path / "out")
    rows = _rows(summary)
    assert summary["pass"] is True, [(r["name"], r["failed"]) for r in summary["targets"]]
    assert list(rows) == ["orange cylinder", "red cube", "blue cone", "grey box", "yellow box", "red van"], \
        "every entry that can spawn, the stand-in too; never one without an asset"
    assert rows["red van"]["drawn"] is False and rows["red cube"]["drawn"] is True
    for row in rows.values():
        assert row["listed"] and row["spawned"] and row["bounds"]["ok"] and row["depth"]["ok"], row
        assert row["depth"]["hit_share"] > 0.3 and abs(row["depth"]["centre_depth_m"] - row["depth"]["expected_near_m"]) < 0.2
        assert (tmp_path / "out" / row["rgb_png"]).is_file() and (tmp_path / "out" / row["depth_png"]).is_file()
    assert rows["red cube"]["material"] == "/Game/AutoFly/Materials/Runtime/M_Red" and rows["red cube"]["colour"]["ok"]
    assert rows["yellow box"]["colour"]["ok"] and rows["grey box"]["colour"] is None
    assert rows["orange cylinder"]["colour"] is None, "M_Orange's colour is not in the registry: nothing to compare"
    assert rows["yellow box"]["bounds"]["expected_size"] == pytest.approx([2.0, 1.0, 1.6])
    assert rows["grey box"]["bounds"]["expected_centre"][2] == pytest.approx(-0.4), "a base pivot stands on the ground"
    assert sim._objects == {}, "every probe object is destroyed"
    assert summary["spawn_table"]["missing"] == []


def test_an_asset_missing_from_the_cook_fails_its_row_and_the_others_still_run(tmp_path):
    from scripts.probe_targets import run_probe

    sim, pool, registry, layout = _setup(tmp_path, assets=("Cylinder", "Cube", "Cone", "Sphere"))
    summary = run_probe(sim, pool, registry, layout, out_dir=tmp_path / "out")
    rows = _rows(summary)
    assert summary["pass"] is False and summary["spawn_table"]["missing"] == ["tgt_box"]
    assert rows["grey box"]["listed"] is False and rows["grey box"]["spawned"] is False and "listed" in rows["grey box"]["failed"]
    assert rows["red cube"]["pass"] is True


def test_a_base_material_that_does_not_show_fails_the_colour_check(tmp_path):
    from scripts.probe_targets import run_probe

    sim, pool, registry, layout = _setup(tmp_path, paint=False)
    rows = _rows(run_probe(sim, pool, registry, layout, out_dir=tmp_path / "out"))
    for name in ("red cube", "blue cone", "yellow box"):
        assert rows[name]["colour"]["ok"] is False and "colour" in rows[name]["failed"], rows[name]
    assert rows["grey box"]["pass"] is True


def test_bounds_in_the_wrong_unit_fail_the_bounds_check(tmp_path):
    from scripts.probe_targets import run_probe

    sim, pool, registry, layout = _setup(tmp_path, bounds_scale=100.0)
    rows = _rows(run_probe(sim, pool, registry, layout, out_dir=tmp_path / "out"))
    assert all(not r["bounds"]["ok"] and "bounds" in r["failed"] for r in rows.values())


def test_a_target_the_depth_camera_does_not_see_fails_the_depth_check(tmp_path):
    from scripts.probe_targets import run_probe

    sim, pool, registry, layout = _setup(tmp_path, visible=False)
    rows = _rows(run_probe(sim, pool, registry, layout, out_dir=tmp_path / "out"))
    assert all(not r["depth"]["ok"] and "depth" in r["failed"] for r in rows.values())


def _late_setup(tmp_path, appear_after):
    from autofly_ue5.dataset.targets import TargetPool
    from autofly_ue5.scenes.model import load_registry
    from autofly_ue5.scenes.resolve import resolve_scene

    tmp_path.mkdir(parents=True, exist_ok=True)
    reg_path, pool_path = _write_world(tmp_path)
    registry = load_registry(reg_path)
    sim = LateRenderer(registry, appear_after=appear_after)
    sim.launch("/Game/AutoFly/Maps/S01", 5)
    return sim, TargetPool.load(pool_path, registry=registry), registry, resolve_scene("s01").layout


def test_the_probe_judges_a_target_after_the_env_s_settle_steps_and_records_its_first_step(tmp_path):
    # Plan 6 step 2a: the env's reset holds PREWARM_HOLD_STEPS hover steps for an object new to the session, so the
    # probe judges what the first recorded frame would show: the capture after those steps. Step 1's is recorded too.
    from autofly_ue5.expert.env import PREWARM_HOLD_STEPS
    from scripts.probe_targets import run_probe

    sim, pool, registry, layout = _late_setup(tmp_path, appear_after=2)
    summary = run_probe(sim, pool, registry, layout, out_dir=tmp_path / "out")
    assert summary["pass"] is True and summary["settle_steps"] == PREWARM_HOLD_STEPS
    for row in summary["targets"]:
        assert row["first_step_visible"] is False and row["first_step_depth"]["ok"] is False, row["name"]
        assert row["depth"]["ok"] and row["settle_steps"] == PREWARM_HOLD_STEPS
    # without settle steps the probe judges step 1, as it did before
    summary = run_probe(*_late_setup(tmp_path / "b", appear_after=2)[:4], out_dir=tmp_path / "out_b", settle_steps=0)
    assert summary["pass"] is False and all("depth" in r["failed"] for r in summary["targets"])
    # and a target still invisible after the settle steps fails
    summary = run_probe(*_late_setup(tmp_path / "c", appear_after=1 + PREWARM_HOLD_STEPS + 1)[:4], out_dir=tmp_path / "c_out")
    assert summary["pass"] is False and all("depth" in r["failed"] for r in summary["targets"])


def test_a_target_visible_at_once_is_first_step_visible(tmp_path):
    from scripts.probe_targets import run_probe

    sim, pool, registry, layout = _setup(tmp_path)
    summary = run_probe(sim, pool, registry, layout, out_dir=tmp_path / "out", settle_steps=2)
    assert summary["pass"] is True and all(r["first_step_visible"] is True for r in summary["targets"])


def test_main_takes_the_settle_steps(tmp_path):
    from autofly_ue5.scenes.model import load_registry
    from scripts.probe_targets import main

    reg_path, pool_path = _write_world(tmp_path)
    base = ["--scene", "s01", "--instance", "5", "--targets", str(pool_path), "--registry", str(reg_path)]
    assert main([*base, "--out", str(tmp_path / "p0"), "--settle-steps", "0"],
                sim_factory=lambda: BoxRenderer(load_registry(reg_path))) == 0
    assert json.loads((tmp_path / "p0" / "summary.json").read_text())["settle_steps"] == 0
    with pytest.raises(SystemExit):
        main([*base, "--out", str(tmp_path / "p1"), "--settle-steps", "-1"], sim_factory=lambda: pytest.fail("launched"))


def test_the_view_looks_along_a_clear_line_on_open_ground():
    from autofly_ue5.scenes.resolve import resolve_scene
    from scripts.probe_targets import REACH_M, VIEW_CLEARANCE_M, choose_view

    layout = resolve_scene("s01").layout
    view = choose_view(layout)
    assert view.z == pytest.approx(-2.0) and round(view.yaw / (math.pi / 2)) * (math.pi / 2) == pytest.approx(view.yaw)
    ux, uy = math.cos(view.yaw), math.sin(view.yaw)
    for inst in layout.instances:
        t = max(0.0, min(REACH_M, (inst.x - view.x) * ux + (inst.y - view.y) * uy))
        gap = math.hypot(inst.x - (view.x + t * ux), inst.y - (view.y + t * uy)) - inst.radius_m
        assert gap >= VIEW_CLEARANCE_M


def test_main_writes_the_summary_and_the_contact_sheet_and_ends_every_object(tmp_path):
    from autofly_ue5.scenes.model import load_registry
    from scripts.probe_targets import main

    reg_path, pool_path = _write_world(tmp_path)
    made = []

    def factory():
        made.append(BoxRenderer(load_registry(reg_path)))
        return made[-1]

    out = tmp_path / "probe"
    code = main(["--scene", "s01", "--instance", "5", "--targets", str(pool_path), "--registry", str(reg_path),
                 "--out", str(out)], sim_factory=factory)
    summary = json.loads((out / "summary.json").read_text())
    assert code == 0 and summary["pass"] is True and summary["instance"] == 5 and summary["map_path"] == "/Game/AutoFly/Maps/S01"
    assert summary["pool"]["sha256"] and len(summary["targets"]) == 6
    import cv2

    sheet = cv2.imread(str(out / "contact_sheet.png"))
    assert sheet is not None and sheet.shape[0] > 100 and sheet.shape[1] > 100
    assert made[0]._launched is None, "closed"


def test_main_refuses_a_pool_that_does_not_load_before_it_launches_anything(tmp_path):
    from scripts.probe_targets import main

    reg_path, pool_path = _write_world(tmp_path)
    pool = json.loads(pool_path.read_text())
    pool["targets"][1]["material"] = "red"  # the palette's instance: not runtime-spawnable
    pool_path.write_text(json.dumps(pool))
    launched = []
    code = main(["--scene", "s01", "--instance", "5", "--targets", str(pool_path), "--registry", str(reg_path),
                 "--out", str(tmp_path / "probe")], sim_factory=lambda: launched.append(1))
    assert code == 2 and launched == []
    assert "color_instance" in json.loads((tmp_path / "probe" / "summary.json").read_text())["error"]


def test_a_reset_that_fails_for_one_target_fails_that_row_and_the_probe_goes_on(tmp_path):
    """The live backend's reset raises typed faults (ResetPoseError, CameraPoseError) that a fresh reset recovers."""
    from autofly_ue5.sim.types import ResetPoseError
    from scripts.probe_targets import run_probe

    sim, pool, registry, layout = _setup(tmp_path)
    resets = {"n": 0}
    real_reset = sim.reset

    def flaky_reset(pose):
        resets["n"] += 1
        if resets["n"] == 2:
            raise ResetPoseError("settled 0.5 m off")
        return real_reset(pose)

    sim.reset = flaky_reset
    rows = _rows(run_probe(sim, pool, registry, layout, out_dir=tmp_path / "out"))
    assert rows["red cube"]["spawned"] is False and "ResetPoseError" in rows["red cube"]["spawn_error"]
    assert rows["blue cone"]["pass"] is True and rows["orange cylinder"]["pass"] is True
