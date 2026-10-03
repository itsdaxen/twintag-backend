import cv2
import numpy as np
from fastapi.testclient import TestClient

from twintag_backend.main import app

client = TestClient(app)


def _files(content: bytes, media_type: str = "image/jpeg") -> list[tuple[str, tuple[str, bytes, str]]]:
    return [
        ("sources", (f"relay-{index}.jpg", content, media_type))
        for index in range(1, 5)
    ]


def test_generates_synthetic_previews() -> None:
    image = np.full((120, 180, 3), 230, dtype=np.uint8)
    cv2.rectangle(image, (15, 15), (165, 105), (30, 30, 30), 3)
    success, encoded = cv2.imencode(".jpg", image)
    assert success

    response = client.post(
        "/api/synthetic-datasets/preview",
        files=_files(encoded.tobytes()),
        data={"device_name": "ABB REX615", "device_type": "Protection relay", "preview_count": 2, "planned_samples": 1_000, "seed": 12},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["device_name"] == "ABB REX615"
    assert payload["device_type"] == "Protection relay"
    assert len(payload["source_filenames"]) == 4
    assert payload["preview_count"] == 2
    assert payload["planned_samples"] == 1_000
    assert len(payload["previews"]) == 2
    assert payload["previews"][0]["image_url"].startswith("data:image/jpeg;base64,")


def test_rejects_unsupported_upload() -> None:
    response = client.post(
        "/api/synthetic-datasets/preview",
        files=_files(b"not an image", "text/plain"),
        data={"device_name": "Relay", "device_type": "Protection relay"},
    )

    assert response.status_code == 415
    assert response.json() == {"detail": "Upload JPEG, PNG or WebP images."}


def test_plans_ten_thousand_samples_by_default() -> None:
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (15, 15), (165, 105), (30, 30, 30), -1)
    success, encoded = cv2.imencode(".jpg", image)
    assert success

    response = client.post(
        "/api/synthetic-datasets/preview",
        files=_files(encoded.tobytes()),
        data={"device_name": "ABB REX615", "device_type": "Protection relay", "preview_count": 1},
    )

    assert response.status_code == 200
    assert response.json()["planned_samples"] == 10_000


def test_identifies_invalid_source_angle() -> None:
    valid = np.full((120, 180, 3), 255, dtype=np.uint8)
    cv2.rectangle(valid, (15, 15), (165, 105), (30, 30, 30), -1)
    success, valid_encoded = cv2.imencode(".jpg", valid)
    assert success
    scene = np.random.default_rng(8).integers(0, 255, (120, 180, 3), dtype=np.uint8)
    success, scene_encoded = cv2.imencode(".jpg", scene)
    assert success
    files = _files(valid_encoded.tobytes())
    files[0] = ("sources", ("front.jpg", scene_encoded.tobytes(), "image/jpeg"))

    response = client.post(
        "/api/synthetic-datasets/preview",
        files=files,
        data={"device_name": "ABB REX615", "device_type": "Protection relay"},
    )

    assert response.status_code == 422
    assert response.json()["detail"].startswith("Front image:")
