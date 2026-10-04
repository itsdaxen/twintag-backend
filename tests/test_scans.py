from fastapi.testclient import TestClient

from twintag_backend.main import app

client = TestClient(app)


def test_list_scans() -> None:
    response = client.get("/api/scans")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "veo-reference",
            "name": "VEO reference facility",
            "status": "ready",
            "mode": "fixture",
            "source": {
                "filename": "cloud_0-001.e57",
                "format": "e57",
                "size_bytes": 2_322_023_424,
            },
            "counts": {
                "sweeps": 18,
                "images": 108,
                "points": 116_640_000,
            },
            "stages": [
                {
                    "key": "metadata",
                    "label": "Read scanner positions",
                    "status": "complete",
                },
                {
                    "key": "images",
                    "label": "Extract camera images",
                    "status": "complete",
                },
                {
                    "key": "recognition",
                    "label": "Recognize equipment",
                    "status": "complete",
                },
                {
                    "key": "positioning",
                    "label": "Calculate 3D positions",
                    "status": "complete",
                },
                {
                    "key": "tags",
                    "label": "Prepare asset tags",
                    "status": "complete",
                },
            ],
        }
    ]


def test_get_scan() -> None:
    response = client.get("/api/scans/veo-reference")

    assert response.status_code == 200
    assert response.json()["id"] == "veo-reference"


def test_get_unknown_scan() -> None:
    response = client.get("/api/scans/missing")

    assert response.status_code == 404
    assert response.json() == {"detail": "Scan not found."}


def test_list_scan_tags_returns_fused_real_detections() -> None:
    response = client.get("/api/scans/veo-reference/tags")

    assert response.status_code == 200
    tags = response.json()
    assert len(tags) == 7
    assert tags[0]["id"] == "asset-001"
    assert tags[0]["asset_type"] == "ABB REX615"
    assert tags[0]["confidence"] == 0.91962
    assert tags[0]["position"] == {"x": -5.767, "y": -1.793, "z": 1.032}
    assert tags[0]["sweep_count"] == 4
    assert tags[0]["evidence"][0] == {
        "image_id": "sweep-03-face-01",
        "sweep_index": 3,
        "face_index": 1,
        "box": {
            "left": 1616.744,
            "top": 1627.476,
            "right": 1857.757,
            "bottom": 1792.263,
        },
    }


def test_list_scan_tags_rejects_unknown_scan() -> None:
    response = client.get("/api/scans/missing/tags")

    assert response.status_code == 404
    assert response.json() == {"detail": "Scan not found."}
