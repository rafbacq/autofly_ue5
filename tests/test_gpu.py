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
