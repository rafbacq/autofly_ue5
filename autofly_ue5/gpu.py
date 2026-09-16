"""GPU memory reading and the rule applied before launching a simulator."""

import os
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


def _own_process_tree_pids(root_pid: int | None = None) -> set[int]:
    """`root_pid` (default: this process) and every descendant PID, from one `ps` snapshot.

    Used to tell "this process's own GPU memory" apart from "a foreign job's": a training process that
    relaunches its own simulator instance still holds its own torch/CUDA context (and, for N>1, its
    SubprocVecEnv worker children may hold theirs), and none of that is a foreign job.
    """
    root_pid = root_pid if root_pid is not None else os.getpid()
    out = subprocess.run(["ps", "-eo", "pid,ppid", "--no-headers"], capture_output=True, text=True).stdout
    children: dict[int, list[int]] = {}
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        pid, ppid = int(parts[0]), int(parts[1])
        children.setdefault(ppid, []).append(pid)
    tree = {root_pid}
    frontier = [root_pid]
    while frontier:
        pid = frontier.pop()
        for child in children.get(pid, ()):
            if child not in tree:
                tree.add(child)
                frontier.append(child)
    return tree


def own_process_gpu_mib(root_pid: int | None = None) -> int:
    """Sum of GPU memory (MiB), per nvidia-smi's own per-process accounting, attributable to `root_pid`
    (default: this process) and its descendants. 0 if nvidia-smi reports no compute apps, or none of them
    are in our tree."""
    tree = _own_process_tree_pids(root_pid)
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout
    total = 0
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid_str, mem_str = (v.strip() for v in line.split(","))
        if int(pid_str) in tree:
            total += int(mem_str)
    return total


def check_gpu_for_launch(
    used_mib: int,
    total_mib: int,
    own_running: int,
    idle_max_mib: int = 2000,
    min_free_mib: int = 6000,
    own_process_mib: int | None = None,
) -> None:
    """Refuse to launch if the GPU looks like it belongs to someone else, or lacks headroom.

    `own_process_mib`: GPU memory already attributable to this process's own tree (its own torch/CUDA
    context, e.g.), subtracted from `used_mib` before judging whether the GPU looks idle. Live finding
    (Task 8): this rule was written in M0, when nothing but the packaged simulator itself ever touched the
    GPU; once a training process shares the GPU with the simulator it relaunches, `used_mib` legitimately
    includes several GiB that is *this process's own* memory, not a foreign job's -- without this
    exclusion, every post-fault relaunch during training falsely concludes "another job is using it" and
    aborts the run. Defaults to `None`, which computes it for real via `own_process_gpu_mib()` -- callers
    that already know the figure (or want a hermetic test) can pass it directly. The separate headroom
    check below is intentionally NOT adjusted by `own_process_mib`: a new simulator instance still cannot
    use VRAM this process is genuinely holding, foreign job or not.
    """
    if own_process_mib is None:
        own_process_mib = own_process_gpu_mib()
    attributable_used_mib = max(used_mib - own_process_mib, 0)
    if own_running == 0 and attributable_used_mib > idle_max_mib:
        raise GpuBusyError(
            f"GPU has {attributable_used_mib} MiB in use (of {used_mib} MiB total, excluding this "
            f"process's own {own_process_mib} MiB) and no simulator of ours is running; another job is "
            f"using it"
        )
    free = total_mib - used_mib
    if free < min_free_mib:
        raise GpuBusyError(f"only {free} MiB free on the GPU, need {min_free_mib} MiB")
