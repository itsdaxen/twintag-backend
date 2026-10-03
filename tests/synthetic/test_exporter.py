import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from twintag_backend.synthetic.exporter import YoloDatasetExporter


def _device(path: Path, shade: int) -> Path:
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (160, 100), (shade, shade, shade), -1)
    cv2.imwrite(str(path), image)
    return path


def test_exports_disjoint_yolo_splits(tmp_path: Path) -> None:
    sources = [
        _device(tmp_path / f"view-{index}.jpg", 20 + index) for index in range(4)
    ]
    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    for index in range(5):
        cv2.imwrite(
            str(backgrounds / f"background-{index}.jpg"),
            np.full((180, 240, 3), 80 + index, dtype=np.uint8),
        )

    output = tmp_path / "dataset"
    result = YoloDatasetExporter().export(
        sources,
        output,
        count=10,
        validation_fraction=0.2,
        backgrounds=backgrounds,
        seed=4,
        size=320,
    )

    assert result.training_samples == 8
    assert result.validation_samples == 2
    assert set(result.training_backgrounds).isdisjoint(result.validation_backgrounds)
    assert len(list((output / "images" / "train").glob("*.jpg"))) == 8
    assert len(list((output / "labels" / "val").glob("*.txt"))) == 2
    values = (output / "labels" / "train" / "train-00000.txt").read_text().split()
    assert values[0] == "0"
    assert all(0 <= float(value) <= 1 for value in values[1:])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sources"] == [source.name for source in sources]
    assert manifest["image_size"] == 320
    assert (output / "dataset.yaml").is_file()


def test_rejects_invalid_export_configuration(tmp_path: Path) -> None:
    exporter = YoloDatasetExporter()

    with pytest.raises(ValueError, match="four device views"):
        exporter.export([], tmp_path / "out")
