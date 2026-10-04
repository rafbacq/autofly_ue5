"""Records that survive a power cut whole or not at all.

Four boots of this host between 2026-09-25 and 2026-10-03 ended the same way: Wi-Fi driver errors (mt7921e, "driver
own failed"), then nothing, with no shutdown sequence. On 2026-10-03 at 17:38 the freeze landed mid-relaunch:
os.replace() had put runs/sim/inst1/pid.json in place, but ext4 had not yet written the new file's data, and the record
came back from the reboot as 0 bytes. A rename is atomic, not
durable: without an fsync of the data first, the rename can reach the disk before the bytes it names. ext4 forces the
data out first only when the rename replaces an existing file (auto_da_alloc), and launch_process unlinks a dead
record before writing the new one.
"""

from __future__ import annotations

import os
from pathlib import Path


def write_text_durably(path: Path, text: str) -> None:
    """Replace `path` with `text` so that after a crash it holds the old content or the new, never a zero-length or
    partial file: write a sibling `<name>.tmp`, fsync it, rename it over `path`, then fsync the directory so the
    rename itself is on disk."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
