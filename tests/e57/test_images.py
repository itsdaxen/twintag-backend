from pathlib import Path

import pytest

from twintag_backend.e57.images import E57ImageExtractor, ExtractedImage
from twintag_backend.e57.reader import E57ReadError, Quaternion, Vector3


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


def test_writes_browser_manifest_with_relative_image_path(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    image = ExtractedImage(
        id="sweep-00-face-00",
        sweep_index=0,
        face_index=0,
        name="Skybox 0",
        guid="image-guid",
        associated_sweep_guid="sweep-guid",
        representation="pinholeRepresentation",
        width=4096,
        height=4096,
        media_type="image/jpeg",
        path=tmp_path / "sweep-00-face-00.jpg",
        position=Vector3(1, 2, 3),
        rotation=Quaternion(0, 0, 0, 1),
    )

    E57ImageExtractor._write_manifest(target, [image])

    manifest = target.read_text()
    assert '"path": "sweep-00-face-00.jpg"' in manifest
    assert str(tmp_path) not in manifest
