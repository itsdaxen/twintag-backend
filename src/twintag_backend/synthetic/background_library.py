from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from twintag_backend.synthetic.generator import DEFAULT_BACKGROUNDS_DIRECTORY

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class TrainingBackground:
    id: str
    name: str
    source: str
    path: Path


class TrainingBackgroundLibrary:
    """Keeps bundled backgrounds immutable and stores admin overrides separately."""

    def __init__(self, storage_directory: Path, default_directory: Path | None = None):
        self.storage_directory = storage_directory
        self.default_directory = default_directory or DEFAULT_BACKGROUNDS_DIRECTORY
        self.custom_directory = storage_directory / "custom"
        self.configuration_path = storage_directory / "configuration.json"

    def list(self) -> tuple[TrainingBackground, ...]:
        disabled = set(self._configuration()["disabled_defaults"])
        defaults = (
            TrainingBackground(
                id=f"default:{path.name}",
                name=path.stem.replace("-", " ").title(),
                source="default",
                path=path,
            )
            for path in self._images(self.default_directory)
            if path.name not in disabled
        )
        custom = (
            TrainingBackground(
                id=f"custom:{path.name}",
                name=path.stem.split("--", 1)[-1].replace("-", " ").title(),
                source="custom",
                path=path,
            )
            for path in self._images(self.custom_directory)
        )
        return (*defaults, *custom)

    def add(self, filename: str, content: bytes) -> TrainingBackground:
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError("Upload a JPEG, PNG or WebP image.")
        self.custom_directory.mkdir(parents=True, exist_ok=True)
        safe_stem = "-".join(
            part for part in Path(filename).stem.lower().replace("_", "-").split("-") if part
        ) or "background"
        path = self.custom_directory / f"{uuid.uuid4().hex[:12]}--{safe_stem}{suffix}"
        path.write_bytes(content)
        return TrainingBackground(
            id=f"custom:{path.name}",
            name=Path(filename).stem,
            source="custom",
            path=path,
        )

    def remove(self, background_id: str) -> None:
        if len(self.list()) <= 1:
            raise ValueError("Keep at least one active training background.")
        source, separator, filename = background_id.partition(":")
        if not separator or Path(filename).name != filename:
            raise KeyError(background_id)
        if source == "default":
            if not (self.default_directory / filename).is_file():
                raise KeyError(background_id)
            configuration = self._configuration()
            disabled = set(configuration["disabled_defaults"])
            disabled.add(filename)
            self._write_configuration(sorted(disabled))
            return
        if source == "custom":
            path = self.custom_directory / filename
            if not path.is_file():
                raise KeyError(background_id)
            path.unlink()
            return
        raise KeyError(background_id)

    def restore_defaults(self) -> None:
        self._write_configuration([])

    def find(self, background_id: str) -> TrainingBackground:
        match = next((item for item in self.list() if item.id == background_id), None)
        if match is None:
            raise KeyError(background_id)
        return match

    def active_paths(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self.list())

    @staticmethod
    def _images(directory: Path) -> tuple[Path, ...]:
        if not directory.is_dir():
            return ()
        return tuple(
            path
            for path in sorted(directory.iterdir())
            if not path.name.startswith("._") and path.suffix.lower() in IMAGE_SUFFIXES
        )

    def _configuration(self) -> dict[str, list[str]]:
        if not self.configuration_path.is_file():
            return {"disabled_defaults": []}
        try:
            payload = json.loads(self.configuration_path.read_text())
            return {"disabled_defaults": list(payload.get("disabled_defaults", []))}
        except (json.JSONDecodeError, OSError, TypeError):
            return {"disabled_defaults": []}

    def _write_configuration(self, disabled_defaults: list[str]) -> None:
        self.storage_directory.mkdir(parents=True, exist_ok=True)
        self.configuration_path.write_text(
            json.dumps({"disabled_defaults": disabled_defaults}, indent=2) + "\n"
        )


BACKGROUND_STORAGE_DIRECTORY = Path(
    os.environ.get("TWINTAG_DATA_DIR", Path.cwd() / "data")
) / "training-backgrounds"

BACKGROUND_LIBRARY = TrainingBackgroundLibrary(BACKGROUND_STORAGE_DIRECTORY)
