import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from twintag_backend.e57.images import E57ImageExtractor
from twintag_backend.e57.preview import E57PreviewExporter
from twintag_backend.e57.projection import (
    PinholeCamera,
    SpatialLocation,
    apply_visibility,
    locate_detection,
    visible_points,
)
from twintag_backend.e57.reader import E57ReadError


@dataclass(frozen=True)
class CalibrationReport:
    source: str
    sweep_index: int
    face_index: int
    image: str
    input_points: int
    projected_points: int
    visible_pixels: int
    minimum_depth: float
    maximum_depth: float
    output_width: int
    output_height: int
    location: SpatialLocation | None


class E57CalibrationExporter:
    def export(
        self,
        source: Path,
        output_directory: Path,
        sweep_index: int,
        face_index: int,
        max_points: int = 1_000_000,
        max_size: int = 1024,
        images_directory: Path | None = None,
        bounding_box: tuple[float, float, float, float] | None = None,
    ) -> CalibrationReport:
        if max_points < 1 or max_size < 1:
            raise ValueError("max_points and max_size must be greater than zero")
        output_directory.mkdir(parents=True, exist_ok=True)
        images = E57ImageExtractor().extract(
            source, images_directory or output_directory / "images"
        )
        try:
            image = next(
                item
                for item in images
                if item.sweep_index == sweep_index and item.face_index == face_index
            )
        except StopIteration as exc:
            raise E57ReadError(
                f"No image for sweep {sweep_index}, face {face_index}"
            ) from exc

        xyz, rgb = self._read_points(source, sweep_index, max_points)
        camera = PinholeCamera.from_image(image)
        projected = camera.project(xyz)
        visible_projection = apply_visibility(projected, image.width, image.height)
        location = (
            locate_detection(visible_projection, bounding_box)
            if bounding_box is not None
            else None
        )
        scale = min(1.0, max_size / max(image.width, image.height))
        x, y, visible = visible_points(
            projected, image.width, image.height, scale=scale
        )
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        colors = (rgb[projected.source_indexes] / 257).astype(np.uint8)

        source_image = Image.open(image.path).convert("RGB").resize((width, height))
        source_image.save(output_directory / "original.jpg", quality=92)
        rendered, mask = self._render(width, height, x, y, colors[visible])
        rendered.save(output_directory / "projected-points.png")
        Image.composite(
            rendered, source_image, mask.point(lambda value: value // 2)
        ).save(output_directory / "overlay.jpg", quality=92)
        self._depth_image(width, height, x, y, projected.depths[visible]).save(
            output_directory / "depth.png"
        )

        report = CalibrationReport(
            source=source.name,
            sweep_index=sweep_index,
            face_index=face_index,
            image=image.path.name,
            input_points=len(xyz),
            projected_points=len(projected.pixels),
            visible_pixels=len(visible),
            minimum_depth=float(projected.depths.min()),
            maximum_depth=float(projected.depths.max()),
            output_width=width,
            output_height=height,
            location=location,
        )
        (output_directory / "report.json").write_text(
            json.dumps(asdict(report), indent=2) + "\n"
        )
        return report

    @staticmethod
    def _read_points(
        source: Path, sweep_index: int, max_points: int
    ) -> tuple[np.ndarray, np.ndarray]:
        scan = E57PreviewExporter._open(source.resolve())
        try:
            data = scan.read_scan(
                sweep_index,
                colors=True,
                transform=True,
                ignore_missing_fields=True,
            )
        finally:
            scan.close()
        available = len(data["cartesianX"])
        selected = np.linspace(
            0, available - 1, min(available, max_points), dtype=np.int64
        )
        xyz = np.column_stack(
            tuple(
                data[name][selected]
                for name in ("cartesianX", "cartesianY", "cartesianZ")
            )
        )
        return xyz, E57PreviewExporter._colors(data, selected)

    @staticmethod
    def _render(
        width: int,
        height: int,
        x: np.ndarray,
        y: np.ndarray,
        colors: np.ndarray,
    ) -> tuple[Image.Image, Image.Image]:
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        pixels[y, x] = colors
        mask[y, x] = 255
        rendered = Image.fromarray(pixels).filter(ImageFilter.MaxFilter(3))
        expanded_mask = Image.fromarray(mask).filter(ImageFilter.MaxFilter(3))
        return rendered, expanded_mask

    @staticmethod
    def _depth_image(
        width: int,
        height: int,
        x: np.ndarray,
        y: np.ndarray,
        depths: np.ndarray,
    ) -> Image.Image:
        output = np.zeros((height, width), dtype=np.uint8)
        if len(depths):
            low, high = np.percentile(depths, (2, 98))
            normalized = 1 - np.clip((depths - low) / max(high - low, 1e-9), 0, 1)
            output[y, x] = np.rint(normalized * 255).astype(np.uint8)
        return Image.fromarray(output).filter(ImageFilter.MaxFilter(3))
