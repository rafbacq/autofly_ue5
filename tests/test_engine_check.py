import pytest

from autofly_ue5.validate.engine_check import (
    clang_version,
    count_device_lost,
    count_xid,
    find_fatal_lines,
    find_zen_redirect_line,
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


ZEN_DIR = "/home/nvidiasims/research_uav/autofly_ue5/ue_project/DerivedDataCache/Zen"


def test_find_zen_redirect_line_matches_env_var_line():
    log = (
        "LogZenServiceInstance: Found Zen config default=/home/nvidiasims/.config/Epic/UnrealEngine/Common/Zen/Data\n"
        f"LogZenServiceInstance: Log: Found environment variable UE_ZenDataPath={ZEN_DIR}\n"
    )
    assert find_zen_redirect_line(log, ZEN_DIR) == f"LogZenServiceInstance: Log: Found environment variable UE_ZenDataPath={ZEN_DIR}"


def test_find_zen_redirect_line_matches_command_line_override():
    log = f"LogZenServiceInstance: Log: Found command line override ZenDataPath={ZEN_DIR}\n"
    assert find_zen_redirect_line(log, ZEN_DIR) == log.rstrip("\n")


def test_find_zen_redirect_line_returns_none_for_default_config():
    log = "LogZenServiceInstance: Found Zen config default=/home/nvidiasims/.config/Epic/UnrealEngine/Common/Zen/Data\n"
    assert find_zen_redirect_line(log, ZEN_DIR) is None


# ------------------------------------------------------------------------------------------------------
# C7 (2026-09-24 review): the GPU-fault audit scans every log a run wrote and fails closed.
# ------------------------------------------------------------------------------------------------------
class _Completed:
    def __init__(self, returncode, stdout):
        self.returncode, self.stdout = returncode, stdout


def _fake_journalctl(monkeypatch, result):
    import autofly_ue5.validate.engine_check as ec

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "journalctl"
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(ec.subprocess, "run", fake_run)


def test_kernel_journal_is_readable_when_it_returns_a_kernel_line(monkeypatch):
    from autofly_ue5.validate.engine_check import kernel_journal_readable

    _fake_journalctl(monkeypatch, _Completed(0, "Sep 24 18:20:01 host kernel: audit: ...\n"))
    assert kernel_journal_readable() is True


@pytest.mark.parametrize("result", [_Completed(0, ""), _Completed(1, "Hint: You are not seeing messages\n"),
                                    FileNotFoundError("journalctl")])
def test_kernel_journal_is_unreadable_when_empty_failing_or_missing(monkeypatch, result):
    # An unreadable journal made xid_count() return 0 before and after -- a "no new Xid" that proved nothing.
    from autofly_ue5.validate.engine_check import kernel_journal_readable

    _fake_journalctl(monkeypatch, result)
    assert kernel_journal_readable() is False


def test_instance_logs_since_includes_rotated_in_run_logs_and_excludes_earlier_ones(tmp_path):
    # UE renames sim.log to sim-backup-<UTC time>.log at each launch, keeping its mtime: after N relaunches the run's
    # evidence is spread over N backups, and only filtering on mtime (not on the name) finds exactly those.
    import os

    from autofly_ue5.validate.engine_check import instance_logs_since

    run_start = 1_800_000_000.0
    before = tmp_path / "sim-backup-2026.09.16-10.00.00.log"
    during = tmp_path / "sim-backup-2026.09.16-20.00.00.log"
    current = tmp_path / "sim.log"
    other = tmp_path / "client.log"
    for path, mtime in ((before, run_start - 600), (during, run_start + 60), (current, run_start + 120), (other, run_start + 5)):
        path.write_text("x\n")
        os.utime(path, (mtime, mtime))
    assert instance_logs_since(tmp_path, run_start) == [during, current]


def test_audit_fails_on_a_device_lost_line_in_a_rotated_log_or_an_unreadable_journal(tmp_path, monkeypatch):
    import autofly_ue5.validate.engine_check as ec

    (tmp_path / "inst0").mkdir()
    (tmp_path / "inst0" / "sim.log").write_text("clean\n")
    (tmp_path / "inst0" / "sim-backup-2026.09.16-20.00.00.log").write_text("LogVulkanRHI: Error: VK_ERROR_DEVICE_LOST\n")
    monkeypatch.setattr(ec, "xid_count", lambda since=None: 3)
    monkeypatch.setattr(ec, "boot_id", lambda: "boot-a")
    monkeypatch.setattr(ec, "kernel_journal_readable", lambda: True)
    args = dict(since="2026-09-16 14:00:00", since_epoch=0.0, xid_before=3, boot_before="boot-a",
                log_dirs=[tmp_path / "inst0"])

    audit = ec.audit_engine_faults(**args)
    assert audit["logs_scanned"] == 2 and audit["ok"] is False
    assert sum(audit["device_lost"].values()) == 1

    (tmp_path / "inst0" / "sim-backup-2026.09.16-20.00.00.log").write_text("clean\n")
    assert ec.audit_engine_faults(**args)["ok"] is True
    monkeypatch.setattr(ec, "kernel_journal_readable", lambda: False)
    unreadable = ec.audit_engine_faults(**args)
    assert unreadable["kernel_journal_readable"] is False and unreadable["ok"] is False
