import json
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from twintag_backend.e57.images import (
    ExtractedImage,
    PinholeIntrinsics,
)
from twintag_backend.e57.projection import (
    PinholeCamera,
    SpatialLocation,
    apply_visibility,
    locate_detection,
)
from twintag_backend.e57.reader import Quaternion, Vector3


@dataclass(frozen=True)
class SpatialObservation:
    image_id: str
    sweep_index: int
    face_index: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]
    location: SpatialLocation


@dataclass(frozen=True)
class AssetTag:
    id: str
    class_name: str
    position: Vector3
    confidence: float
    observation_count: int
    sweep_count: int
    spatial_spread: float
    observations: tuple[SpatialObservation, ...]


@dataclass(frozen=True)
class FusionReport:
    source: str
    input_detections: int
    positioned_detections: int
    rejected_detections: int
    tags: tuple[AssetTag, ...]


PointProvider = Callable[[int], np.ndarray]


class E57DetectionFusion:
    def __init__(
        self,
        *,
        cluster_radius: float = 0.30,
        minimum_sweeps: int = 2,
        minimum_support: int = 20,
    ) -> None:
        if cluster_radius <= 0:
            raise ValueError("Cluster radius must be greater than zero")
        if minimum_sweeps < 1 or minimum_support < 1:
            raise ValueError("Minimum sweeps and support must be greater than zero")
        self.cluster_radius = cluster_radius
        self.minimum_sweeps = minimum_sweeps
        self.minimum_support = minimum_support

    def fuse(
        self,
        source: Path,
        detections_file: Path,
        images_manifest: Path,
        output: Path,
        *,
        point_provider: PointProvider | None = None,
        max_points_per_sweep: int = 2_000_000,
    ) -> FusionReport:
        detections = json.loads(detections_file.read_text())["detections"]
        images = self._load_images(images_manifest)
        provider = point_provider or self._e57_point_provider(
            source, max_points_per_sweep
        )
        observations = []
        rejected = 0

        detections_by_image = defaultdict(list)
        for detection in detections:
            detections_by_image[detection["image_id"]].append(detection)

        for image_id, image_detections in detections_by_image.items():
            image = images.get(image_id)
            if image is None or image.intrinsics is None:
                rejected += len(image_detections)
                continue
            projected = PinholeCamera.from_image(image).project(
                provider(image.sweep_index)
            )
            visible_projection = apply_visibility(projected, image.width, image.height)
            for detection in image_detections:
                box = detection["box"]
                coordinates = tuple(
                    float(box[key]) for key in ("left", "top", "right", "bottom")
                )
                try:
                    location = locate_detection(
                        visible_projection,
                        coordinates,
                        minimum_support=self.minimum_support,
                    )
                except ValueError:
                    rejected += 1
                    continue
                observations.append(
                    SpatialObservation(
                        image_id=image.id,
                        sweep_index=image.sweep_index,
                        face_index=image.face_index,
                        class_name=detection["class_name"],
                        confidence=float(detection["confidence"]),
                        box=coordinates,
                        location=location,
                    )
                )

        tags = self.cluster(tuple(observations))
        report = FusionReport(
            source=source.name,
            input_detections=len(detections),
            positioned_detections=len(observations),
            rejected_detections=rejected,
            tags=tags,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(report), indent=2) + "\n")
        return report

    def cluster(
        self, observations: tuple[SpatialObservation, ...]
    ) -> tuple[AssetTag, ...]:
        groups: list[list[SpatialObservation]] = []
        for class_observations in self._by_class(observations).values():
            groups.extend(self._complete_link_clusters(class_observations))

        accepted = [
            group
            for group in groups
            if len({item.sweep_index for item in group}) >= self.minimum_sweeps
        ]
        accepted.sort(key=lambda group: self._center(group))
        return tuple(
            self._tag(index, group) for index, group in enumerate(accepted, start=1)
        )

    def _complete_link_clusters(
        self, observations: list[SpatialObservation]
    ) -> list[list[SpatialObservation]]:
        clusters = [[observation] for observation in observations]
        while True:
            candidates = [
                (self._cluster_distance(left, right), left_index, right_index)
                for left_index, left in enumerate(clusters)
                for right_index, right in enumerate(
                    clusters[left_index + 1 :], left_index + 1
                )
            ]
            candidates = [item for item in candidates if item[0] <= self.cluster_radius]
            if not candidates:
                return clusters
            _, left_index, right_index = min(candidates)
            clusters[left_index].extend(clusters.pop(right_index))

    @classmethod
    def _cluster_distance(
        cls,
        left: list[SpatialObservation],
        right: list[SpatialObservation],
    ) -> float:
        return max(cls._distance(a, b) for a in left for b in right)

    @staticmethod
    def _by_class(
        observations: tuple[SpatialObservation, ...],
    ) -> dict[str, list[SpatialObservation]]:
        grouped = defaultdict(list)
        for observation in observations:
            grouped[observation.class_name].append(observation)
        return grouped

    @staticmethod
    def _distance(left: SpatialObservation, right: SpatialObservation) -> float:
        a = left.location.position
        b = right.location.position
        return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))

    @staticmethod
    def _center(group: list[SpatialObservation]) -> tuple[float, float, float]:
        points = np.array(
            [
                (
                    item.location.position.x,
                    item.location.position.y,
                    item.location.position.z,
                )
                for item in group
            ]
        )
        return tuple(float(value) for value in np.median(points, axis=0))

    @classmethod
    def _tag(cls, index: int, group: list[SpatialObservation]) -> AssetTag:
        center = cls._center(group)
        distances = [
            math.dist(
                center,
                (
                    item.location.position.x,
                    item.location.position.y,
                    item.location.position.z,
                ),
            )
            for item in group
        ]
        ordered = tuple(
            sorted(group, key=lambda item: (item.sweep_index, item.face_index))
        )
        return AssetTag(
            id=f"asset-{index:03d}",
            class_name=group[0].class_name,
            position=Vector3(*center),
            confidence=round(sum(item.confidence for item in group) / len(group), 6),
            observation_count=len(group),
            sweep_count=len({item.sweep_index for item in group}),
            spatial_spread=round(max(distances, default=0), 6),
            observations=ordered,
        )

    @staticmethod
    def _load_images(manifest: Path) -> dict[str, ExtractedImage]:
        payload = json.loads(manifest.read_text())
        images = {}
        for item in payload["images"]:
            intrinsics = item.get("intrinsics")
            image = ExtractedImage(
                id=item["id"],
                sweep_index=int(item["sweep_index"]),
                face_index=int(item["face_index"]),
                name=item["name"],
                guid=item["guid"],
                associated_sweep_guid=item["associated_sweep_guid"],
                representation=item["representation"],
                width=int(item["width"]),
                height=int(item["height"]),
                media_type=item["media_type"],
                path=manifest.parent / item["path"],
                position=Vector3(**item["position"]),
                rotation=Quaternion(**item["rotation"]),
                intrinsics=PinholeIntrinsics(**intrinsics) if intrinsics else None,
            )
            images[image.id] = image
        return images

    @staticmethod
    def _e57_point_provider(source: Path, max_points: int) -> PointProvider:
        from twintag_backend.e57.calibration import E57CalibrationExporter

        cached_sweep = None
        cached_points = None

        def points(sweep_index: int) -> np.ndarray:
            nonlocal cached_sweep, cached_points
            if sweep_index != cached_sweep:
                cached_points = E57CalibrationExporter._read_points(
                    source, sweep_index, max_points
                )[0]
                cached_sweep = sweep_index
            return cached_points

        return points
