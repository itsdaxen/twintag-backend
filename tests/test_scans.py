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
