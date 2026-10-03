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


def test_rejects_scene_photos_without_a_clean_background(tmp_path: Path) -> None:
    source = tmp_path / "scene.jpg"
    random = np.random.default_rng(4)
    cv2.imwrite(str(source), random.integers(0, 255, (240, 320, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="clean background"):
        SyntheticDatasetGenerator().generate(source, tmp_path / "out", count=1)


def test_removes_clean_source_background_before_compositing(tmp_path: Path) -> None:
    source = tmp_path / "device.jpg"
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (80, 60), (220, 140), (30, 30, 30), -1)
    cv2.imwrite(str(source), image)

    dataset = SyntheticDatasetGenerator(size=320).generate(
        source, tmp_path / "dataset", count=1, seed=7
    )
    generated = cv2.imread(str(tmp_path / "dataset" / dataset.samples[0].image))
    box = dataset.samples[0].bounding_box
    crop = generated[int(box.top) : int(box.bottom), int(box.left) : int(box.right)]

    assert crop.mean() < 220


def test_keeps_enclosed_white_device_details() -> None:
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (60, 40), (240, 160), (80, 80, 80), -1)
    cv2.rectangle(image, (120, 80), (180, 120), (255, 255, 255), -1)

    device, mask = SyntheticDatasetGenerator()._extract_device(image)
    center = mask[mask.shape[0] // 2, mask.shape[1] // 2]

    assert center == 255
    assert device[device.shape[0] // 2, device.shape[1] // 2].min() == 255


def test_uses_real_background_images_when_available(tmp_path: Path) -> None:
    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    image = np.full((240, 320, 3), (20, 180, 40), dtype=np.uint8)
    cv2.imwrite(str(backgrounds / "cabinet.jpg"), image)

    background = SyntheticDatasetGenerator(
        size=320, background_directory=backgrounds
    )._background(np.random.default_rng(2))

    assert background.shape == (320, 320, 3)
    assert background[..., 1].mean() > background[..., 0].mean()


def test_prefers_horizontal_sweep_faces_with_panel_views(tmp_path: Path) -> None:
    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    for face in range(6):
        cv2.imwrite(
            str(backgrounds / f"sweep-01-face-{face:02d}.jpg"),
            np.full((160, 160, 3), face * 20, dtype=np.uint8),
        )

    generator = SyntheticDatasetGenerator(background_directory=backgrounds)

    assert {path.stem for path in generator.backgrounds} == {
        "sweep-01-face-01",
        "sweep-01-face-03",
        "sweep-01-face-04",
    }


def test_relighting_varies_scene_lighting_without_changing_shape() -> None:
    generator = SyntheticDatasetGenerator(size=256)
    canvas = np.full((256, 256, 3), 160, dtype=np.uint8)

    relit = [
        generator._relight(canvas, np.random.default_rng(seed)) for seed in range(8)
    ]

    assert all(image.shape == canvas.shape for image in relit)
    assert any(not np.array_equal(image, canvas) for image in relit)
    assert len({round(float(image.mean())) for image in relit}) > 1


def test_casts_shadow_next_to_device_only() -> None:
    generator = SyntheticDatasetGenerator(size=256)
    canvas = np.full((256, 256, 3), 200, dtype=np.uint8)
    mask = np.full((60, 60), 255, dtype=np.uint8)

    shadowed = generator._cast_shadow(canvas, mask, 100, 100, np.random.default_rng(3))

    assert shadowed[130, 130].mean() < 200
    assert shadowed[5, 5].mean() == 200


def test_harmonizes_device_toward_surroundings() -> None:
    device = np.full((40, 40, 3), 240, dtype=np.uint8)
    mask = np.full((40, 40), 255, dtype=np.uint8)
    dark_region = np.full((40, 40, 3), 60, dtype=np.uint8)

    harmonized = SyntheticDatasetGenerator._harmonize(
        device, mask, dark_region, np.random.default_rng(0)
    )

    assert 60 < harmonized.mean() < 240
