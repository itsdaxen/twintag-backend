import json
from dataclasses import asdict, dataclass
from pathlib import Path

import laspy
import numpy as np
import pye57

from twintag_backend.e57.reader import E57ReadError, Vector3


@dataclass(frozen=True)
class PointCloudPreview:
    source: str
    path: Path
    point_count: int
    minimum: Vector3
    maximum: Vector3


class E57PreviewExporter:
    def export(
        self,
        source: Path,
        output_directory: Path,
        target_points: int = 2_000_000,
    ) -> PointCloudPreview:
        source = source.resolve()
        if not source.is_file() or source.suffix.lower() != ".e57":
            raise E57ReadError(f"Expected an existing .e57 file: {source}")
        if target_points < 1:
            raise ValueError("target_points must be greater than zero")

        output_directory.mkdir(parents=True, exist_ok=True)
        scan = self._open(source)
        try:
            xyz, rgb = self._sample_scan(scan, target_points)
        except Exception as exc:
            raise E57ReadError(f"Could not export E57 preview: {source.name}") from exc
        finally:
            scan.close()

        target = output_directory / "preview.laz"
        self._write_laz(target, xyz, rgb)
        preview = self._preview(source, target, xyz)
        self._write_manifest(output_directory / "preview.json", preview)
        return preview

    @staticmethod
    def _open(source: Path) -> pye57.E57:
        try:
            return pye57.E57(str(source))
        except Exception as exc:
            raise E57ReadError(f"Could not open E57 file: {source.name}") from exc

    def _sample_scan(
        self, scan: pye57.E57, target_points: int
    ) -> tuple[np.ndarray, np.ndarray]:
        counts = np.array(
            [scan.get_header(index).point_count for index in range(scan.scan_count)]
        )
        total = int(counts.sum())
        allocations = np.maximum(1, np.rint(target_points * counts / total).astype(int))
        xyz_parts = []
        rgb_parts = []

        for index, allocation in enumerate(allocations):
            data = scan.read_scan(
                index,
                colors=True,
                transform=True,
                ignore_missing_fields=True,
            )
            available = len(data["cartesianX"])
            selected = np.linspace(
                0, available - 1, min(available, allocation), dtype=np.int64
            )
            xyz_parts.append(
                np.column_stack(
                    (
                        data["cartesianX"][selected],
                        data["cartesianY"][selected],
                        data["cartesianZ"][selected],
                    )
                )
            )
            rgb_parts.append(self._colors(data, selected))

        return np.concatenate(xyz_parts), np.concatenate(rgb_parts)

    @staticmethod
    def _colors(data: dict, selected: np.ndarray) -> np.ndarray:
        names = ("colorRed", "colorGreen", "colorBlue")
        if not all(name in data for name in names):
            return np.full((len(selected), 3), 32768, dtype=np.uint16)

        colors = np.column_stack(tuple(data[name][selected] for name in names)).astype(
            np.float64
        )
        maximum = float(colors.max(initial=0))
        if maximum <= 1:
            colors = colors * 65535
        elif maximum <= 255:
            colors = colors * 257
        return np.clip(colors, 0, 65535).astype(np.uint16)

    @staticmethod
    def _write_laz(target: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.001, 0.001, 0.001])
        header.offsets = xyz.min(axis=0)
        cloud = laspy.LasData(header)
        cloud.x, cloud.y, cloud.z = xyz.T
        cloud.red, cloud.green, cloud.blue = rgb.T
        cloud.write(target)

    @staticmethod
    def _preview(source: Path, target: Path, xyz: np.ndarray) -> PointCloudPreview:
        minimum = xyz.min(axis=0)
        maximum = xyz.max(axis=0)
        return PointCloudPreview(
            source=source.name,
            path=target,
            point_count=len(xyz),
            minimum=Vector3(*minimum),
            maximum=Vector3(*maximum),
        )

    @staticmethod
    def _write_manifest(target: Path, preview: PointCloudPreview) -> None:
        payload = asdict(preview)
        payload["path"] = preview.path.name
        target.write_text(json.dumps(payload, indent=2) + "\n")
