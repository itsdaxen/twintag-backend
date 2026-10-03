import json
from dataclasses import asdict, dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True)
class SyntheticSample:
    image: str
    annotation: str
    width: int
    height: int
    class_id: int
    bounding_box: BoundingBox


@dataclass(frozen=True)
class SyntheticDataset:
    source: str
    samples: tuple[SyntheticSample, ...]


class SyntheticDatasetGenerator:
    def __init__(self, size: int = 640) -> None:
        if size < 128:
            raise ValueError("size must be at least 128 pixels")
        self.size = size

    def validate_source(self, source: Path) -> None:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read source image: {source}")
        self._extract_device(image)

    def generate(
        self,
        source: Path,
        output_directory: Path,
        count: int,
        seed: int | None = None,
    ) -> SyntheticDataset:
        if count < 1:
            raise ValueError("count must be greater than zero")
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read source image: {source}")
        device, device_mask = self._extract_device(image)

        images_directory = output_directory / "images"
        labels_directory = output_directory / "labels"
        images_directory.mkdir(parents=True, exist_ok=True)
        labels_directory.mkdir(parents=True, exist_ok=True)
        random = np.random.default_rng(seed)
        samples = []

        for index in range(count):
            sample_seed = int(random.integers(0, np.iinfo(np.uint32).max))
            canvas, bounding_box = self._compose(device, device_mask, random)
            augmented, transformed_box = self._augment(
                canvas, bounding_box, sample_seed
            )
            image_name = f"sample-{index:05d}.jpg"
            annotation_name = f"sample-{index:05d}.txt"
            image_path = images_directory / image_name
            annotation_path = labels_directory / annotation_name
            cv2.imwrite(str(image_path), augmented, [cv2.IMWRITE_JPEG_QUALITY, 90])
            annotation_path.write_text(
                "0 " + " ".join(f"{value:.2f}" for value in transformed_box) + "\n"
            )
            samples.append(
                SyntheticSample(
                    image=f"images/{image_name}",
                    annotation=f"labels/{annotation_name}",
                    width=self.size,
                    height=self.size,
                    class_id=0,
                    bounding_box=BoundingBox(*transformed_box),
                )
            )

        dataset = SyntheticDataset(source=source.name, samples=tuple(samples))
        (output_directory / "manifest.json").write_text(
            json.dumps(asdict(dataset), indent=2) + "\n"
        )
        return dataset

    def _compose(
        self,
        device: np.ndarray,
        device_mask: np.ndarray,
        random: np.random.Generator,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        canvas = self._background(random)
        height, width = device.shape[:2]
        maximum_side = int(self.size * random.uniform(0.38, 0.72))
        scale = maximum_side / max(height, width)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized_device = cv2.resize(
            device, (resized_width, resized_height), interpolation=cv2.INTER_AREA
        )
        resized_mask = cv2.resize(
            device_mask,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        left = int(random.integers(0, self.size - resized_width + 1))
        top = int(random.integers(0, self.size - resized_height + 1))
        right = left + resized_width
        bottom = top + resized_height
        alpha = resized_mask.astype(np.float32)[..., None] / 255
        region = canvas[top:bottom, left:right].astype(np.float32)
        composite = resized_device.astype(np.float32) * alpha + region * (1 - alpha)
        canvas[top:bottom, left:right] = np.clip(composite, 0, 255).astype(np.uint8)
        return canvas, (left, top, right, bottom)

    def _extract_device(self, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = source.shape[:2]
        if min(height, width) < 64:
            raise ValueError("Source image must be at least 64 pixels on each side")

        border = np.concatenate(
            [source[0], source[-1], source[:, 0], source[:, -1]], axis=0
        ).astype(np.float32)
        background = np.median(border, axis=0)
        if float(np.mean(np.linalg.norm(border - background, axis=1))) > 18:
            raise ValueError("Source image must show one device on a clean background")

        distance = np.linalg.norm(source.astype(np.float32) - background, axis=2)
        background_candidates = np.where(distance <= 8, 1, 0).astype(np.uint8)
        _, components = cv2.connectedComponents(background_candidates, connectivity=8)
        border_labels = np.unique(
            np.concatenate(
                [components[0], components[-1], components[:, 0], components[:, -1]]
            )
        )
        connected_background = np.isin(components, border_labels)
        mask = np.where(connected_background, 0, 255).astype(np.uint8)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
        )
        points = cv2.findNonZero(mask)
        if points is None:
            raise ValueError("No device could be separated from the background")
        left, top, crop_width, crop_height = cv2.boundingRect(points)
        coverage = crop_width * crop_height / (width * height)
        if coverage > 0.95:
            raise ValueError("Source image must be tightly framed on a clean background")

        padding = max(2, round(max(crop_width, crop_height) * 0.01))
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(width, left + crop_width + 2 * padding)
        bottom = min(height, top + crop_height + 2 * padding)
        cropped_device = source[top:bottom, left:right]
        cropped_mask = mask[top:bottom, left:right]
        cropped_mask = cv2.GaussianBlur(cropped_mask, (5, 5), 0)
        return cropped_device, cropped_mask

    def _background(self, random: np.random.Generator) -> np.ndarray:
        base = random.integers(45, 180, size=3)
        vertical = np.linspace(-25, 25, self.size, dtype=np.float32)[:, None, None]
        noise = random.normal(0, 9, (self.size, self.size, 3))
        background = base.astype(np.float32) + vertical + noise

        for _ in range(int(random.integers(3, 9))):
            x = int(random.integers(0, self.size))
            color = tuple(int(value) for value in random.integers(35, 210, size=3))
            cv2.line(background, (x, 0), (x, self.size), color, 2)
        return np.clip(background, 0, 255).astype(np.uint8)

    def _augment(
        self,
        image: np.ndarray,
        bounding_box: tuple[float, float, float, float],
        seed: int,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        transform = A.Compose(
            [
                A.Perspective(scale=(0.02, 0.1), keep_size=True, p=0.75),
                A.Affine(
                    scale=(0.85, 1.1),
                    translate_percent=(-0.08, 0.08),
                    rotate=(-15, 15),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    p=0.8,
                ),
                A.RandomBrightnessContrast(0.3, 0.3, p=0.8),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.GaussNoise(std_range=(0.01, 0.08), p=0.4),
                A.ImageCompression(quality_range=(45, 95), p=0.35),
                A.CoarseDropout(
                    num_holes_range=(1, 5),
                    hole_height_range=(0.03, 0.12),
                    hole_width_range=(0.03, 0.12),
                    fill="random_uniform",
                    p=0.35,
                ),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.25,
                clip=True,
            ),
            seed=seed,
        )
        result = transform(
            image=image,
            bboxes=[bounding_box],
            class_labels=[0],
        )
        boxes = result["bboxes"]
        if not boxes:
            return image, bounding_box
        return result["image"], tuple(float(value) for value in boxes[0][:4])
