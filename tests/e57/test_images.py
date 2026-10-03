from pathlib import Path

import pytest

from twintag_backend.e57.images import E57ImageExtractor
from twintag_backend.e57.reader import E57ReadError


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(E57ReadError, match="not found"):
        E57ImageExtractor().extract(tmp_path / "missing.e57", tmp_path / "images")


def test_rejects_wrong_file_type(tmp_path: Path) -> None:
    source = tmp_path / "scan.txt"
    source.write_text("not an E57 file")

    with pytest.raises(E57ReadError, match=r"Expected an \.e57 file"):
        E57ImageExtractor().extract(source, tmp_path / "images")


def test_extracts_no_images_from_scan_without_images(
    tmp_path: Path,
    sample_e57: Path,
) -> None:
    images = E57ImageExtractor().extract(sample_e57, tmp_path / "images")

    assert images == ()
    assert list((tmp_path / "images").iterdir()) == []
