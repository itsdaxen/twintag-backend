import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from twintag_backend.synthetic.generator import (
    DEFAULT_BACKGROUNDS_DIRECTORY,
    BoundingBox,
    SyntheticDatasetGenerator,
)


@dataclass(frozen=True)
class DatasetExport:
    output: str
    training_samples: int
    validation_samples: int
    training_backgrounds: tuple[str, ...]
    validation_backgrounds: tuple[str, ...]


class YoloDatasetExporter:
    def export(
        self,
        sources: list[Path],
        output: Path,
        *,
        count: int = 10_000,
        validation_fraction: float = 0.2,
        backgrounds: Path = DEFAULT_BACKGROUNDS_DIRECTORY,
        seed: int = 615,
        class_name: str = "ABB REX615",
        size: int = 768,
    ) -> DatasetExport:
        if len(sources) != 4:
            raise ValueError("Exactly four device views are required")
        if count < 2:
            raise ValueError("Dataset must contain at least two samples")
        if not 0 < validation_fraction < 1:
            raise ValueError("Validation fraction must be between zero and one")

        background_paths = sorted(
            path
            for path in backgrounds.iterdir()
            if not path.name.startswith("._")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if len(background_paths) < 2:
            raise ValueError("At least two background images are required")

        validator = SyntheticDatasetGenerator(background_paths=background_paths)
        for source in sources:
            validator.validate_source(source)

        random = np.random.default_rng(seed)
        shuffled = list(background_paths)
        random.shuffle(shuffled)
        validation_background_count = min(
            len(shuffled) - 1,
            max(1, round(len(shuffled) * validation_fraction)),
        )
        validation_backgrounds = tuple(shuffled[:validation_background_count])
        training_backgrounds = tuple(shuffled[validation_background_count:])

        validation_count = max(1, round(count * validation_fraction))
        training_count = count - validation_count
        if training_count < 1:
            raise ValueError("Training split must contain at least one sample")

        output.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            (output / "images" / split).mkdir(parents=True, exist_ok=True)
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)

        self._export_split(
            sources,
            output,
            "train",
            training_count,
            training_backgrounds,
            seed,
            size,
        )
        self._export_split(
            sources,
            output,
            "val",
            validation_count,
            validation_backgrounds,
            seed + 1_000_000,
            size,
        )

        (output / "dataset.yaml").write_text(
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            f"  0: {json.dumps(class_name)}\n"
        )
        result = DatasetExport(
            output=str(output),
            training_samples=training_count,
            validation_samples=validation_count,
            training_backgrounds=tuple(path.name for path in training_backgrounds),
            validation_backgrounds=tuple(path.name for path in validation_backgrounds),
        )
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    **asdict(result),
                    "sources": [source.name for source in sources],
                    "seed": seed,
                    "class_name": class_name,
                    "image_size": size,
                },
                indent=2,
            )
            + "\n"
        )
        return result

    @staticmethod
    def _export_split(
        sources: list[Path],
        output: Path,
        split: str,
        count: int,
        backgrounds: tuple[Path, ...],
        seed: int,
        size: int,
    ) -> None:
        generator = SyntheticDatasetGenerator(size=size, background_paths=backgrounds)
        per_source, remainder = divmod(count, len(sources))
        sample_index = 0
        with TemporaryDirectory(prefix=f"twintag-{split}-") as temporary:
            workspace = Path(temporary)
            for source_index, source in enumerate(sources):
                source_count = per_source + (source_index < remainder)
                if source_count == 0:
                    continue
                generated = generator.generate(
                    source,
                    workspace / f"source-{source_index}",
                    count=source_count,
                    seed=seed + source_index,
                )
                for sample in generated.samples:
                    stem = f"{split}-{sample_index:05d}"
                    shutil.copyfile(
                        workspace / f"source-{source_index}" / sample.image,
                        output / "images" / split / f"{stem}.jpg",
                    )
                    (output / "labels" / split / f"{stem}.txt").write_text(
                        _yolo_annotation(
                            sample.bounding_box, sample.width, sample.height
                        )
                    )
                    sample_index += 1


def _yolo_annotation(box: BoundingBox, width: int, height: int) -> str:
    center_x = (box.left + box.right) / (2 * width)
    center_y = (box.top + box.bottom) / (2 * height)
    box_width = (box.right - box.left) / width
    box_height = (box.bottom - box.top) / height
    return f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n"
