import pytest

from autofly_ue5.gpu import GpuBusyError, check_gpu_for_launch, own_process_gpu_mib, parse_gpu_memory


def test_parse_gpu_memory():
    assert parse_gpu_memory("622, 24564\n") == (622, 24564)


def test_idle_gpu_allows_first_instance():
    check_gpu_for_launch(622, 24564, own_running=0, own_process_mib=0)


def test_foreign_load_blocks_first_instance():
    with pytest.raises(GpuBusyError, match="no simulator of ours"):
        check_gpu_for_launch(3000, 24564, own_running=0, own_process_mib=0)


def test_second_instance_needs_headroom_only():
    check_gpu_for_launch(9000, 24564, own_running=1, own_process_mib=0)
    with pytest.raises(GpuBusyError, match="free"):
        check_gpu_for_launch(20000, 24564, own_running=1, own_process_mib=0)


# --------------------------------------------------------------------------------------------------------
# Live finding (Task 8 shakedown2): a training process relaunching its own simulator instance still holds
# its own torch/CUDA context. Before this fix, check_gpu_for_launch's "own_running == 0 and used_mib over
# idle_max_mib" rule mistook that for a foreign job and killed the run at the first fault escalation.
# --------------------------------------------------------------------------------------------------------
def test_own_process_memory_is_excluded_from_the_idle_judgment():
    # 2600 MiB used, all of it this process's own (e.g. a training process's torch/CUDA context) -- must
    # not look like a foreign job just because own_running == 0 (the simulator instance was just torn down
    # for a relaunch).
    check_gpu_for_launch(2600, 24564, own_running=0, own_process_mib=2600)


def test_a_foreign_processs_memory_still_blocks_even_with_our_own_excluded():
    # 5000 MiB used, 2600 of which is ours -- the remaining 2400 is still over idle_max_mib (2000) and must
    # still be treated as a foreign job. The exclusion must not become a blanket bypass.
    with pytest.raises(GpuBusyError, match="no simulator of ours"):
        check_gpu_for_launch(5000, 24564, own_running=0, own_process_mib=2600)


def test_own_process_memory_never_makes_attributable_usage_negative():
    # A defensive floor: if own_process_mib is ever over-counted relative to used_mib (e.g. a stale
    # reading), attributable usage must clamp at 0, not go negative and vacuously pass everything upstream.
    check_gpu_for_launch(1000, 24564, own_running=0, own_process_mib=5000)


def test_check_gpu_for_launch_computes_own_process_mib_by_default(monkeypatch):
    # The real (closed) call site in autofly_ue5/sim/airsim_backend.py never passes own_process_mib, so the
    # fix has to hold with the default too -- verified here via a spy rather than a real nvidia-smi call.
    import autofly_ue5.gpu as gpu_module

    monkeypatch.setattr(gpu_module, "own_process_gpu_mib", lambda: 2600)
    check_gpu_for_launch(2600, 24564, own_running=0)  # no own_process_mib passed


def test_own_process_gpu_mib_sums_only_pids_in_our_tree(monkeypatch):
    import autofly_ue5.gpu as gpu_module

    def fake_run(cmd, **kwargs):
        class _Result:
            pass

        result = _Result()
        if cmd[0] == "ps":
            # our tree: 100 (root) -> 200 (child) -> 300 (grandchild); 999 is unrelated (a foreign job)
            result.stdout = "100 1\n200 100\n300 200\n999 1\n"
        elif "nvidia-smi" in cmd[0] and "--query-compute-apps" in cmd[1]:
            result.stdout = "100, 500\n300, 700\n999, 9000\n"
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return result

    monkeypatch.setattr(gpu_module.subprocess, "run", fake_run)

    assert own_process_gpu_mib(root_pid=100) == 1200  # 500 (self) + 700 (grandchild); 999 excluded


def test_own_process_gpu_mib_is_zero_with_no_compute_apps(monkeypatch):
    import autofly_ue5.gpu as gpu_module

    def fake_run(cmd, **kwargs):
        class _Result:
            stdout = ""

        return _Result()

    monkeypatch.setattr(gpu_module.subprocess, "run", fake_run)

    assert own_process_gpu_mib(root_pid=100) == 0
