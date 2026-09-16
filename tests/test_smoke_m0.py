import pytest

import autofly_ue5.sim.smoke_m0 as smoke_m0


def test_main_refuses_a_zero_warmup(tmp_path, monkeypatch, capsys):
    """R8 (controller ruling): frame 0 of a session is captured before the depth buffer has stabilised
    (measured: no_hit_px=14, min_finite_m=0.0425, max_finite_m=0.998, vs. no_hit_px~24,690, min_finite_m~1.4
    on every later frame), so --warmup-steps < 1 must be refused before anything is launched.

    connect() is patched to blow up immediately (never dialling a real, possibly nonexistent, server) so this
    stays a pure offline parse/validation test: if the refusal doesn't happen first, this fails loudly and
    fast instead of hanging.
    """

    def _boom(self):
        raise AssertionError("client.connect() must not be reached: --warmup-steps 0 must be refused first")

    monkeypatch.setattr(smoke_m0.ProjectAirSimClient, "connect", _boom)
    out = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exc:
        smoke_m0.main(["--warmup-steps", "0", "--out", str(out)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--warmup-steps" in err and "0.0425" in err
