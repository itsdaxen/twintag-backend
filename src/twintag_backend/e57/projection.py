from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twintag_backend.e57.images import ExtractedImage, PinholeIntrinsics
from twintag_backend.e57.reader import Quaternion, Vector3

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ProjectedPoints:
    pixels: FloatArray
    depths: FloatArray
    world_points: FloatArray
    source_indexes: NDArray[np.int64]


@dataclass(frozen=True)
class SpatialLocation:
    position: Vector3
    minimum: Vector3
    maximum: Vector3
    depth: float
    supporting_points: int


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    intrinsics: PinholeIntrinsics
    position: Vector3
    rotation: Quaternion

    @classmethod
    def from_image(cls, image: ExtractedImage) -> "PinholeCamera":
        if image.intrinsics is None:
            raise ValueError("Image does not use a pinhole representation")
        return cls(
            width=image.width,
            height=image.height,
            intrinsics=image.intrinsics,
            position=image.position,
            rotation=image.rotation,
        )

    def project(self, world_points: FloatArray) -> ProjectedPoints:
        points = _points(world_points)
        rotation = quaternion_matrix(self.rotation)
        origin = np.array(
            [self.position.x, self.position.y, self.position.z], dtype=np.float64
        )
        camera_points = (points - origin) @ rotation
        depths = -camera_points[:, 2]
        valid = depths > 0

        pixels = np.empty((len(points), 2), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels[:, 0] = (
                self.intrinsics.principal_point_x
                + camera_points[:, 0] * self.intrinsics.focal_x_pixels / depths
            )
            pixels[:, 1] = (
                self.intrinsics.principal_point_y
                + camera_points[:, 1] * self.intrinsics.focal_y_pixels / depths
            )
        valid &= np.isfinite(pixels).all(axis=1)
        valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < self.width)
        valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < self.height)
        return ProjectedPoints(
            pixels[valid], depths[valid], points[valid], np.flatnonzero(valid)
        )

    def unproject(self, pixels: FloatArray, depths: FloatArray) -> FloatArray:
        pixels = np.asarray(pixels, dtype=np.float64)
        depths = np.asarray(depths, dtype=np.float64)
        camera_points = np.column_stack(
            (
                (pixels[:, 0] - self.intrinsics.principal_point_x)
                * depths
                / self.intrinsics.focal_x_pixels,
                (pixels[:, 1] - self.intrinsics.principal_point_y)
                * depths
                / self.intrinsics.focal_y_pixels,
                -depths,
            )
        )
        rotation = quaternion_matrix(self.rotation)
        origin = np.array(
            [self.position.x, self.position.y, self.position.z], dtype=np.float64
        )
        return camera_points @ rotation.T + origin


def quaternion_matrix(quaternion: Quaternion) -> FloatArray:
    values = np.array(
        [quaternion.w, quaternion.x, quaternion.y, quaternion.z], dtype=np.float64
    )
    norm = np.linalg.norm(values)
    if norm == 0:
        raise ValueError("Camera quaternion cannot be zero")
    w, x, y, z = values / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def visible_points(
    projected: ProjectedPoints,
    width: int,
    height: int,
    scale: float = 1,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return output x/y pixels and source indexes after nearest-point z-buffering."""
    if not 0 < scale <= 1:
        raise ValueError("scale must be greater than zero and at most one")
    output_width = max(1, round(width * scale))
    output_height = max(1, round(height * scale))
    xy = np.floor(projected.pixels * scale).astype(np.int64)
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < output_width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < output_height)
    )
    source_indexes = np.flatnonzero(inside)
    xy = xy[inside]
    linear = xy[:, 1] * output_width + xy[:, 0]
    order = np.lexsort((projected.depths[source_indexes], linear))
    sorted_linear = linear[order]
    first = np.empty(len(order), dtype=bool)
    first[0:] = True
    if len(order) > 1:
        first[1:] = sorted_linear[1:] != sorted_linear[:-1]
    selected = source_indexes[order[first]]
    selected_xy = xy[order[first]]
    return selected_xy[:, 0], selected_xy[:, 1], selected


def apply_visibility(
    projected: ProjectedPoints, width: int, height: int
) -> ProjectedPoints:
    """Discard points hidden by a nearer point at the same image pixel."""
    _, _, selected = visible_points(projected, width, height)
    return ProjectedPoints(
        pixels=projected.pixels[selected],
        depths=projected.depths[selected],
        world_points=projected.world_points[selected],
        source_indexes=projected.source_indexes[selected],
    )


def locate_detection(
    projected: ProjectedPoints,
    bounding_box: tuple[float, float, float, float],
    minimum_support: int = 20,
) -> SpatialLocation:
    """Locate a 2D detection in a visibility-filtered point projection."""
    left, top, right, bottom = bounding_box
    within = (
        (projected.pixels[:, 0] >= left)
        & (projected.pixels[:, 0] <= right)
        & (projected.pixels[:, 1] >= top)
        & (projected.pixels[:, 1] <= bottom)
    )
    depths = projected.depths[within]
    points = projected.world_points[within]
    if len(points) < minimum_support:
        raise ValueError("Detection has too little 3D support")

    order = np.argsort(depths)
    depths = depths[order]
    points = points[order]
    gaps = np.diff(depths)
    threshold = np.maximum(0.15, depths[:-1] * 0.04)
    boundaries = np.flatnonzero(gaps > threshold) + 1
    groups = np.split(np.arange(len(depths)), boundaries)
    foreground = next(
        (group for group in groups if len(group) >= minimum_support), None
    )
    if foreground is None:
        raise ValueError("Detection has no coherent 3D surface")

    selected = points[foreground]
    center = np.median(selected, axis=0)
    minimum = np.percentile(selected, 5, axis=0)
    maximum = np.percentile(selected, 95, axis=0)
    return SpatialLocation(
        position=Vector3(*center),
        minimum=Vector3(*minimum),
        maximum=Vector3(*maximum),
        depth=float(np.median(depths[foreground])),
        supporting_points=len(selected),
    )


def _points(values: FloatArray) -> FloatArray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("world_points must have shape (n, 3)")
    return points
