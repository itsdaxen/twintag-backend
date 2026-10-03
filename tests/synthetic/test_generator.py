import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from twintag_backend.synthetic.generator import SyntheticDatasetGenerator


def test_generates_annotated_samples(tmp_path: Path) -> None:
    source = tmp_path / "device.jpg"
    image = np.full((160, 240, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (220, 140), (25, 25, 25), 4)
    cv2.imwrite(str(source), image)

    dataset = SyntheticDatasetGenerator(size=320).generate(
        source, tmp_path / "dataset", count=3, seed=42
    )

    assert len(dataset.samples) == 3
    for sample in dataset.samples:
        assert (tmp_path / "dataset" / sample.image).is_file()
        assert (tmp_path / "dataset" / sample.annotation).is_file()
        box = sample.bounding_box
        assert 0 <= box.left < box.right <= 320
        assert 0 <= box.top < box.bottom <= 320

    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text())
    assert manifest["source"] == "device.jpg"
    assert len(manifest["samples"]) == 3


def test_rejects_invalid_requests(tmp_path: Path) -> None:
    generator = SyntheticDatasetGenerator()

    with pytest.raises(ValueError, match="greater than zero"):
        generator.generate(tmp_path / "missing.jpg", tmp_path / "out", count=0)
    with pytest.raises(ValueError, match="Could not read"):
        generator.generate(tmp_path / "missing.jpg", tmp_path / "out", count=1)
