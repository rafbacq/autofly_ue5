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
