import os

from autofly_ue5.durable import write_text_durably


def test_the_data_is_synced_before_the_rename_and_the_directory_after(tmp_path, sync_events):
    # 2026-10-03 17:38: the host froze after os.replace() had put runs/sim/inst1/pid.json in place but before ext4 had
    # written the file's data, and the record came back from the reboot as 0 bytes.
    target = tmp_path / "pid.json"
    write_text_durably(target, '{"pid": 1}')
    tmp, final, directory = (os.path.realpath(p) for p in (tmp_path / "pid.json.tmp", target, tmp_path))
    assert sync_events == [("fsync", tmp), ("replace", tmp, final), ("fsync", directory)]
    assert target.read_text() == '{"pid": 1}'
    assert not (tmp_path / "pid.json.tmp").exists()


def test_an_existing_file_is_replaced_whole(tmp_path):
    target = tmp_path / "sessions.json"
    target.write_text("old content that is longer than the new one")
    write_text_durably(target, "new")
    assert target.read_text() == "new"
