import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from twintag_backend.detection.inference import DeviceDetector


class Values:
    def __init__(self, values: list) -> None:
        self.values = values

    def tolist(self) -> list:
        return self.values


class FakeModel:
    def __init__(self, results: list) -> None:
        self.results = results
        self.arguments = None

    def predict(self, **kwargs):
        self.arguments = kwargs
        return self.results


def test_detects_devices_and_writes_structured_report(tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"checkpoint")
    images = tmp_path / "images"
    images.mkdir()
    image = images / "sweep-03-face-01.jpg"
    image.write_bytes(b"image")
    (images / "._sweep-03-face-01.jpg").write_bytes(b"metadata")
    result = SimpleNamespace(
        path=str(image),
        orig_shape=(1000, 2000),
        names={0: "ABB REX615"},
        boxes=SimpleNamespace(
            cls=Values([0.0]),
            conf=Values([0.93456789]),
            xyxy=Values([[200.0, 100.0, 600.0, 500.0]]),
        ),
    )
    fake_model = FakeModel([result])
    output = tmp_path / "detections.json"

    report = DeviceDetector(
        model_path, model_factory=lambda _: fake_model
    ).detect_directory(images, output)

    assert report.image_count == 1
    assert report.detections[0].sweep_index == 3
    assert report.detections[0].face_index == 1
    assert report.detections[0].confidence == 0.934568
    assert report.detections[0].normalized_box.right == pytest.approx(0.3)
    assert fake_model.arguments["conf"] == 0.85
    assert fake_model.arguments["imgsz"] == 1280
    assert fake_model.arguments["stream"] is True
    assert fake_model.arguments["source"] == str(images)
    assert json.loads(output.read_text())["detections"][0]["class_name"] == "ABB REX615"


def test_rejects_invalid_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        DeviceDetector(tmp_path / "best.pt", confidence_threshold=0)


def test_requires_images(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"checkpoint")
    images = tmp_path / "images"
    images.mkdir()

    with pytest.raises(ValueError, match="No supported images"):
        DeviceDetector(model, model_factory=lambda _: FakeModel([])).detect_directory(
            images
        )
