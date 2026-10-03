import numpy as np
import pytest

from twintag_backend.e57.images import PinholeIntrinsics
from twintag_backend.e57.projection import (
    PinholeCamera,
    ProjectedPoints,
    apply_visibility,
    locate_detection,
    visible_points,
)
from twintag_backend.e57.reader import Quaternion, Vector3


@pytest.fixture
def camera() -> PinholeCamera:
    return PinholeCamera(
        width=100,
        height=100,
        intrinsics=PinholeIntrinsics(
            focal_length=1,
            pixel_width=0.02,
            pixel_height=0.02,
            principal_point_x=50,
            principal_point_y=50,
        ),
        position=Vector3(0, 0, 0),
        rotation=Quaternion(0, 0, 0, 1),
    )


def test_projects_e57_camera_coordinates(camera: PinholeCamera) -> None:
    projected = camera.project(
        np.array([[0, 0, -2], [1, 0, -2], [0, 1, -2], [0, 0, 2]])
    )

    np.testing.assert_allclose(projected.pixels, [[50, 50], [75, 50], [50, 75]])
    np.testing.assert_allclose(projected.depths, [2, 2, 2])


def test_projection_round_trips_through_camera(camera: PinholeCamera) -> None:
    points = np.array([[0.2, -0.3, -2], [-0.4, 0.1, -3]])
    projected = camera.project(points)

    np.testing.assert_allclose(
        camera.unproject(projected.pixels, projected.depths), points
    )


def test_projects_with_camera_rotation_and_translation(camera: PinholeCamera) -> None:
    rotated = PinholeCamera(
        width=camera.width,
        height=camera.height,
        intrinsics=camera.intrinsics,
        position=Vector3(10, 2, 3),
        rotation=Quaternion(0, 2**-0.5, 0, 2**-0.5),
    )

    projected = rotated.project(np.array([[8, 2, 3]]))

    np.testing.assert_allclose(projected.pixels, [[50, 50]])
    np.testing.assert_allclose(projected.depths, [2])


def test_z_buffer_keeps_nearest_point(camera: PinholeCamera) -> None:
    projected = camera.project(np.array([[0, 0, -5], [0, 0, -2]]))

    x, y, indexes = visible_points(projected, 100, 100)

    np.testing.assert_array_equal(x, [50])
    np.testing.assert_array_equal(y, [50])
    np.testing.assert_array_equal(indexes, [1])
    visible = apply_visibility(projected, 100, 100)
    np.testing.assert_allclose(visible.depths, [2])


def test_locates_detection_on_nearest_supported_surface() -> None:
    foreground = np.column_stack((np.linspace(10, 20, 30), np.linspace(10, 20, 30)))
    background = np.column_stack((np.linspace(10, 20, 40), np.linspace(20, 10, 40)))
    projected = ProjectedPoints(
        pixels=np.concatenate((foreground, background)),
        depths=np.concatenate((np.full(30, 2.0), np.full(40, 5.0))),
        world_points=np.concatenate(
            (
                np.tile([1.0, 2.0, 3.0], (30, 1)),
                np.tile([8.0, 9.0, 10.0], (40, 1)),
            )
        ),
        source_indexes=np.arange(70),
    )

    location = locate_detection(projected, (5, 5, 25, 25))

    assert location.position == Vector3(1, 2, 3)
    assert location.depth == 2
    assert location.supporting_points == 30
