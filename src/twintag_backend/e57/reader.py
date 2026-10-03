from dataclasses import dataclass
from pathlib import Path

import pye57


class E57ReadError(ValueError):
    """Raised when a scan cannot be read as an E57 file."""


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float


@dataclass(frozen=True)
class SweepMetadata:
    index: int
    name: str
    guid: str
    point_count: int
    position: Vector3
    rotation: Quaternion


@dataclass(frozen=True)
class E57Metadata:
    filename: str
    size_bytes: int
    image_count: int
    point_count: int
    sweeps: tuple[SweepMetadata, ...]


class E57Reader:
    def read(self, source: Path) -> E57Metadata:
        source = source.resolve()
        self._validate_source(source)

        try:
            scan = pye57.E57(str(source))
        except Exception as exc:
            raise E57ReadError(f"Could not open E57 file: {source.name}") from exc

        try:
            sweeps = tuple(
                self._read_sweep(scan, index) for index in range(scan.scan_count)
            )
            image_count = self._image_count(scan)
        except Exception as exc:
            raise E57ReadError(f"Could not read E57 metadata: {source.name}") from exc
        finally:
            scan.close()

        return E57Metadata(
            filename=source.name,
            size_bytes=source.stat().st_size,
            image_count=image_count,
            point_count=sum(sweep.point_count for sweep in sweeps),
            sweeps=sweeps,
        )

    @staticmethod
    def _validate_source(source: Path) -> None:
        if not source.is_file():
            raise E57ReadError(f"E57 file not found: {source}")
        if source.suffix.lower() != ".e57":
            raise E57ReadError(f"Expected an .e57 file: {source.name}")

    @staticmethod
    def _read_sweep(scan: pye57.E57, index: int) -> SweepMetadata:
        header = scan.get_header(index)
        position = tuple(float(value) for value in header.translation)
        rotation_wxyz = tuple(float(value) for value in header.rotation)
        w, x, y, z = rotation_wxyz

        return SweepMetadata(
            index=index,
            name=str(header["name"].value()).strip(),
            guid=header.guid,
            point_count=header.point_count,
            position=Vector3(*position),
            rotation=Quaternion(x=x, y=y, z=z, w=w),
        )

    @staticmethod
    def _image_count(scan: pye57.E57) -> int:
        if not scan.root.isDefined("images2D"):
            return 0
        return scan.root["images2D"].childCount()
