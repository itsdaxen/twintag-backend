import json
from dataclasses import asdict, dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

DEFAULT_BACKGROUNDS_DIRECTORY = Path(__file__).parent / "backgrounds"


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
    def __init__(
        self, size: int = 640, background_directory: Path | None = None
    ) -> None:
        if size < 128:
            raise ValueError("size must be at least 128 pixels")
        self.size = size
        backgrounds = (
            tuple(
                path
                for path in sorted(background_directory.iterdir())
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            if background_directory and background_directory.is_dir()
            else ()
        )
        curated = tuple(
            path
            for path in backgrounds
            if "-face-" not in path.stem
            or path.stem.endswith(("face-01", "face-03", "face-04"))
        )
        self.backgrounds = curated or backgrounds
        self._background_cache: dict[Path, np.ndarray] = {}

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
        maximum_side = int(self.size * random.uniform(0.16, 0.48))
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
        left, top = self._panel_position(canvas, resized_width, resized_height, random)
        right = left + resized_width
        bottom = top + resized_height
        region = canvas[top:bottom, left:right]
        resized_device = self._harmonize(resized_device, resized_mask, region, random)
        if random.random() < 0.6:
            canvas = self._cast_shadow(canvas, resized_mask, left, top, random)
        alpha = resized_mask.astype(np.float32)[..., None] / 255
        region = canvas[top:bottom, left:right].astype(np.float32)
        composite = resized_device.astype(np.float32) * alpha + region * (1 - alpha)
        canvas[top:bottom, left:right] = np.clip(composite, 0, 255).astype(np.uint8)
        canvas = self._relight(canvas, random)
        return canvas, (left, top, right, bottom)

    @staticmethod
    def _harmonize(
        device: np.ndarray,
        mask: np.ndarray,
        region: np.ndarray,
        random: np.random.Generator,
    ) -> np.ndarray:
        """Pull the device's brightness and tint part-way toward its surroundings."""
        weights = mask.astype(np.float32) / 255
        if float(weights.sum()) < 1:
            return device
        device_lab = cv2.cvtColor(device, cv2.COLOR_BGR2LAB).astype(np.float32)
        region_lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
        device_mean = (device_lab * weights[..., None]).sum((0, 1)) / weights.sum()
        region_mean = region_lab.reshape(-1, 3).mean(axis=0)
        strength = np.array(
            [random.uniform(0.1, 0.35), *([random.uniform(0.05, 0.25)] * 2)],
            dtype=np.float32,
        )
        device_lab += (region_mean - device_mean) * strength
        device_lab = np.clip(device_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(device_lab, cv2.COLOR_LAB2BGR)

    def _cast_shadow(
        self,
        canvas: np.ndarray,
        mask: np.ndarray,
        left: int,
        top: int,
        random: np.random.Generator,
    ) -> np.ndarray:
        """Darken a blurred, offset copy of the device silhouette behind it."""
        height, width = mask.shape
        offset = max(2, round(max(height, width) * random.uniform(0.01, 0.06)))
        direction = random.uniform(0, 2 * np.pi)
        dx = round(np.cos(direction) * offset)
        dy = round(abs(np.sin(direction)) * offset)  # light mostly from above
        shadow = np.zeros(canvas.shape[:2], dtype=np.float32)
        shadow_left, shadow_top = left + dx, top + dy
        x0, y0 = max(0, shadow_left), max(0, shadow_top)
        x1 = min(self.size, shadow_left + width)
        y1 = min(self.size, shadow_top + height)
        if x1 <= x0 or y1 <= y0:
            return canvas
        shadow[y0:y1, x0:x1] = mask[
            y0 - shadow_top : y1 - shadow_top, x0 - shadow_left : x1 - shadow_left
        ]
        kernel = max(3, (offset * 4) | 1)
        shadow = cv2.GaussianBlur(shadow / 255, (kernel, kernel), 0)
        darkness = 1 - shadow[..., None] * random.uniform(0.15, 0.45)
        return np.clip(canvas.astype(np.float32) * darkness, 0, 255).astype(np.uint8)

    def _relight(self, canvas: np.ndarray, random: np.random.Generator) -> np.ndarray:
        """Apply scene-wide lighting: gradients, spotlights, colour cast, vignette."""
        gain = np.ones((self.size, self.size), dtype=np.float32)
        ys, xs = np.mgrid[0 : self.size, 0 : self.size].astype(np.float32) / self.size

        if random.random() < 0.7:
            angle = random.uniform(0, 2 * np.pi)
            ramp = (xs - 0.5) * np.cos(angle) + (ys - 0.5) * np.sin(angle)
            gain *= 1 + ramp * 2 * random.uniform(0.05, 0.3)

        if random.random() < 0.4:
            cx, cy = random.uniform(0, 1, size=2)
            sigma = random.uniform(0.2, 0.6)
            blob = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
            gain *= 1 + blob * random.uniform(-0.25, 0.3)

        if random.random() < 0.3:
            radius = np.sqrt((xs - 0.5) ** 2 + (ys - 0.5) ** 2) / np.sqrt(0.5)
            gain *= 1 - radius**2 * random.uniform(0.1, 0.35)

        lit = canvas.astype(np.float32) * gain[..., None]

        if random.random() < 0.6:
            temperature = random.uniform(-0.12, 0.12)  # >0 warm, <0 cool (BGR)
            lit *= np.array([1 - temperature, 1, 1 + temperature], dtype=np.float32)

        return np.clip(lit, 0, 255).astype(np.uint8)

    @staticmethod
    def _panel_position(
        canvas: np.ndarray,
        width: int,
        height: int,
        random: np.random.Generator,
    ) -> tuple[int, int]:
        """Prefer bright, neutral, low-texture regions typical of cabinet faces."""
        hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_32F)
        candidates = []
        minimum_top = round(canvas.shape[0] * 0.15)
        maximum_top = min(
            canvas.shape[0] - height,
            round(canvas.shape[0] * 0.78) - height,
        )
        if maximum_top < minimum_top:
            minimum_top = 0
            maximum_top = canvas.shape[0] - height
        for _ in range(24):
            left = int(random.integers(0, canvas.shape[1] - width + 1))
            top = int(random.integers(minimum_top, maximum_top + 1))
            region = hsv[top : top + height, left : left + width]
            texture = np.mean(np.abs(edges[top : top + height, left : left + width]))
            score = float(region[..., 2].mean() - region[..., 1].mean() - texture)
            candidates.append((score, left, top))
        _, left, top = max(candidates)
        return left, top

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
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
        points = cv2.findNonZero(mask)
        if points is None:
            raise ValueError("No device could be separated from the background")
        left, top, crop_width, crop_height = cv2.boundingRect(points)
        coverage = crop_width * crop_height / (width * height)
        if coverage > 0.95:
            raise ValueError(
                "Source image must be tightly framed on a clean background"
            )

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
        if self.backgrounds:
            return self._real_background(random)

        base = random.integers(45, 180, size=3)
        vertical = np.linspace(-25, 25, self.size, dtype=np.float32)[:, None, None]
        noise = random.normal(0, 9, (self.size, self.size, 3))
        background = base.astype(np.float32) + vertical + noise

        for _ in range(int(random.integers(3, 9))):
            x = int(random.integers(0, self.size))
            color = tuple(int(value) for value in random.integers(35, 210, size=3))
            cv2.line(background, (x, 0), (x, self.size), color, 2)
        return np.clip(background, 0, 255).astype(np.uint8)

    def _real_background(self, random: np.random.Generator) -> np.ndarray:
        path = self.backgrounds[int(random.integers(0, len(self.backgrounds)))]
        image = self._background_cache.get(path)
        if image is None:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read background image: {path}")
            self._background_cache[path] = image

        height, width = image.shape[:2]
        crop_size = max(64, round(min(height, width) * random.uniform(0.35, 1)))
        left = int(random.integers(0, width - crop_size + 1))
        top = int(random.integers(0, height - crop_size + 1))
        crop = image[top : top + crop_size, left : left + crop_size]
        background = cv2.resize(
            crop, (self.size, self.size), interpolation=cv2.INTER_AREA
        )
        if random.random() < 0.4:
            background = cv2.GaussianBlur(background, (5, 5), 0)
        return background

    def _augment(
        self,
        image: np.ndarray,
        bounding_box: tuple[float, float, float, float],
        seed: int,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        transform = A.Compose(
            [
                A.RandomBrightnessContrast(0.15, 0.15, p=0.7),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5)),
                        A.MotionBlur(blur_limit=(3, 7)),
                        A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.3)),
                    ],
                    p=0.3,
                ),
                A.Downscale(scale_range=(0.5, 0.9), p=0.2),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.45), p=0.35),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.35),
                A.ImageCompression(quality_range=(45, 95), p=0.35),
                A.CoarseDropout(
                    num_holes_range=(1, 3),
                    hole_height_range=(0.02, 0.06),
                    hole_width_range=(0.02, 0.06),
                    fill=0,
                    p=0.15,
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
