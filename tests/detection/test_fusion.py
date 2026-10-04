from twintag_backend.detection.fusion import E57DetectionFusion, SpatialObservation
from twintag_backend.e57.projection import SpatialLocation
from twintag_backend.e57.reader import Vector3


def observation(
    sweep: int,
    position: tuple[float, float, float],
    *,
    confidence: float = 0.9,
    class_name: str = "ABB REX615",
) -> SpatialObservation:
    point = Vector3(*position)
    return SpatialObservation(
        image_id=f"sweep-{sweep:02d}-face-01",
        sweep_index=sweep,
        face_index=1,
        class_name=class_name,
        confidence=confidence,
        box=(10, 20, 30, 40),
        location=SpatialLocation(point, point, point, 2, 100),
    )


def test_fuses_repeated_views_and_keeps_distinct_devices() -> None:
    fusion = E57DetectionFusion(cluster_radius=0.45, minimum_sweeps=2)

    tags = fusion.cluster(
        (
            observation(1, (1.0, 2.0, 3.0), confidence=0.90),
            observation(2, (1.1, 2.0, 3.0), confidence=0.96),
            observation(3, (2.0, 2.0, 3.0), confidence=0.92),
            observation(4, (2.1, 2.0, 3.0), confidence=0.94),
        )
    )

    assert len(tags) == 2
    assert tags[0].id == "asset-001"
    assert tags[0].position == Vector3(1.05, 2.0, 3.0)
    assert tags[0].confidence == 0.93
    assert tags[0].observation_count == 2
    assert tags[0].sweep_count == 2
    assert tags[1].position == Vector3(2.05, 2.0, 3.0)


def test_rejects_single_sweep_clusters() -> None:
    fusion = E57DetectionFusion(cluster_radius=0.45, minimum_sweeps=2)

    tags = fusion.cluster(
        (
            observation(1, (1.0, 2.0, 3.0)),
            observation(1, (1.1, 2.0, 3.0)),
        )
    )

    assert tags == ()


def test_does_not_merge_different_classes() -> None:
    fusion = E57DetectionFusion(cluster_radius=0.45, minimum_sweeps=1)

    tags = fusion.cluster(
        (
            observation(1, (1.0, 2.0, 3.0)),
            observation(2, (1.0, 2.0, 3.0), class_name="Other device"),
        )
    )

    assert len(tags) == 2


def test_complete_link_does_not_chain_nearby_devices() -> None:
    fusion = E57DetectionFusion(cluster_radius=0.30, minimum_sweeps=2)

    tags = fusion.cluster(
        (
            observation(1, (0, 0, 1.10)),
            observation(2, (0, 0, 1.25)),
            observation(3, (0, 0, 1.40)),
            observation(4, (0, 0, 1.55)),
        )
    )

    assert len(tags) == 2
    assert all(tag.sweep_count == 2 for tag in tags)
