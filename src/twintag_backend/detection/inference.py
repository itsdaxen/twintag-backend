import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_ID = re.compile(r"^sweep-(?P<sweep>\d+)-face-(?P<face>\d+)$")


@dataclass(frozen=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True)
class DeviceDetection:
    image_id: str
    image: str
    sweep_index: int | None
    face_index: int | None
    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox
    normalized_box: BoundingBox


@dataclass(frozen=True)
class DetectionReport:
    model: str
    confidence_threshold: float
    image_count: int
    detections: tuple[DeviceDetection, ...]


class DeviceDetector:
    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float = 0.85,
        image_size: int = 1280,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not 0 < confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        if image_size < 32:
            raise ValueError("Image size must be at least 32 pixels")

        self.model_path = model_path.resolve()
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self._model_factory = model_factory

    def detect_directory(
        self, images_directory: Path, output: Path | None = None
    ) -> DetectionReport:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not images_directory.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_directory}")

        images = tuple(
            path
            for path in sorted(images_directory.iterdir())
            if not path.name.startswith("._")
            and path.suffix.lower() in SUPPORTED_IMAGES
        )
        if not images:
            raise ValueError(f"No supported images found in {images_directory}")

        model = self._load_model()
        results = model.predict(
            source=str(images_directory),
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            verbose=False,
            stream=True,
        )
        detections = tuple(
            detection
            for result in results
            for detection in self._result_detections(result)
        )
        report = DetectionReport(
            model=self.model_path.name,
            confidence_threshold=self.confidence_threshold,
            image_count=len(images),
            detections=detections,
        )
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(asdict(report), indent=2) + "\n")
        return report

    def _load_model(self) -> Any:
        if self._model_factory:
            return self._model_factory(str(self.model_path))
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Detector dependencies are missing; run `uv sync --extra inference`."
            ) from exc
        return YOLO(str(self.model_path))

    @staticmethod
    def _result_detections(result: Any) -> Sequence[DeviceDetection]:
        image = Path(result.path)
        image_id = image.stem
        match = IMAGE_ID.match(image_id)
        height, width = result.orig_shape
        names = result.names
        detections = []

        for class_id, confidence, coordinates in zip(
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.xyxy.tolist(),
            strict=True,
        ):
            left, top, right, bottom = (float(value) for value in coordinates)
            integer_class_id = int(class_id)
            detections.append(
                DeviceDetection(
                    image_id=image_id,
                    image=image.name,
                    sweep_index=int(match.group("sweep")) if match else None,
                    face_index=int(match.group("face")) if match else None,
                    class_id=integer_class_id,
                    class_name=str(names[integer_class_id]),
                    confidence=round(float(confidence), 6),
                    box=BoundingBox(left, top, right, bottom),
                    normalized_box=BoundingBox(
                        left / width,
                        top / height,
                        right / width,
                        bottom / height,
                    ),
                )
            )
        return detections
