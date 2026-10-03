import json
from pathlib import Path

import laspy
import pytest

from twintag_backend.e57.preview import E57PreviewExporter


def test_exports_global_colored_laz(sample_e57: Path, tmp_path: Path) -> None:
    result = E57PreviewExporter().export(sample_e57, tmp_path / "preview", 2)
    cloud = laspy.read(result.path)

    assert result.point_count == 2
    assert list(cloud.x) == pytest.approx([1.0, 2.0])
    assert list(cloud.y) == pytest.approx([2.0, 4.0])
    assert list(cloud.z) == pytest.approx([3.0, 6.0])
    assert list(cloud.red) == [65535, 0]
    assert list(cloud.green) == [0, 65535]


def test_writes_preview_manifest(sample_e57: Path, tmp_path: Path) -> None:
    output = tmp_path / "preview"
    E57PreviewExporter().export(sample_e57, output, 1)

    manifest = json.loads((output / "preview.json").read_text())
    assert manifest["source"] == "sample.e57"
    assert manifest["path"] == "preview.laz"
    assert manifest["point_count"] == 1


def test_rejects_empty_preview(sample_e57: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        E57PreviewExporter().export(sample_e57, tmp_path, 0)
