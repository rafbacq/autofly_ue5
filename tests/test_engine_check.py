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
