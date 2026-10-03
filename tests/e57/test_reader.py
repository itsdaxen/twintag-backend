from pathlib import Path

import pytest

from twintag_backend.e57.reader import E57Reader, E57ReadError


def test_reads_scan_metadata(sample_e57: Path) -> None:
    metadata = E57Reader().read(sample_e57)

    assert metadata.filename == "sample.e57"
    assert metadata.size_bytes > 0
    assert metadata.image_count == 0
    assert metadata.point_count == 2
    assert len(metadata.sweeps) == 1

    sweep = metadata.sweeps[0]
    assert sweep.index == 0
    assert sweep.name == "Test sweep"
    assert sweep.point_count == 2
    assert (sweep.position.x, sweep.position.y, sweep.position.z) == (1.0, 2.0, 3.0)
    assert (sweep.rotation.x, sweep.rotation.y, sweep.rotation.z, sweep.rotation.w) == (
        0.0,
        0.0,
        0.0,
        1.0,
    )


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(E57ReadError, match="not found"):
        E57Reader().read(tmp_path / "missing.e57")


def test_rejects_wrong_file_type(tmp_path: Path) -> None:
    source = tmp_path / "scan.txt"
    source.write_text("not an E57 file")

    with pytest.raises(E57ReadError, match=r"Expected an \.e57 file"):
        E57Reader().read(source)


def test_rejects_invalid_e57(tmp_path: Path) -> None:
    source = tmp_path / "invalid.e57"
    source.write_text("not an E57 file")

    with pytest.raises(E57ReadError, match="Could not open E57 file"):
        E57Reader().read(source)
