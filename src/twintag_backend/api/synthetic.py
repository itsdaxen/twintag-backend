import base64
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from twintag_backend.schemas.synthetic import (
    BoundingBoxResponse,
    SyntheticDatasetPreviewResponse,
    SyntheticPreviewResponse,
    TrainingBackgroundListResponse,
    TrainingBackgroundResponse,
)
from twintag_backend.synthetic.background_library import BACKGROUND_LIBRARY
from twintag_backend.synthetic.generator import SyntheticDatasetGenerator

router = APIRouter(prefix="/api/synthetic-datasets", tags=["synthetic datasets"])

ALLOWED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SOURCE_BYTES = 15 * 1024 * 1024
AUGMENTATIONS = [
    "real substation backgrounds",
    "scale and panel placement",
    "colour harmonisation",
    "contact shadows",
    "lighting gradients and spotlights",
    "colour temperature",
    "brightness, contrast and gamma",
    "blur and sensor noise",
    "compression",
    "partial occlusion",
]
SOURCE_ANGLES = ("Front", "Front-left", "Front-right", "Side")


def _background_response(background) -> TrainingBackgroundResponse:
    return TrainingBackgroundResponse(
        id=background.id,
        name=background.name,
        source=background.source,
        image_url=f"/api/synthetic-datasets/backgrounds/{background.id}/image",
    )


@router.get("/backgrounds", response_model=TrainingBackgroundListResponse)
def list_backgrounds() -> TrainingBackgroundListResponse:
    backgrounds = BACKGROUND_LIBRARY.list()
    return TrainingBackgroundListResponse(
        backgrounds=[_background_response(item) for item in backgrounds],
        default_count=sum(item.source == "default" for item in backgrounds),
        custom_count=sum(item.source == "custom" for item in backgrounds),
    )


@router.get("/backgrounds/{background_id}/image")
def get_background_image(background_id: str) -> FileResponse:
    try:
        background = BACKGROUND_LIBRARY.find(background_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Background not found.") from exc
    return FileResponse(background.path)


@router.post("/backgrounds", response_model=TrainingBackgroundResponse, status_code=201)
async def add_background(background: Annotated[UploadFile, File()]) -> TrainingBackgroundResponse:
    if background.content_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG or WebP image.")
    content = await background.read(MAX_SOURCE_BYTES + 1)
    if not content or len(content) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Background must be between 1 byte and 15 MB.")
    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=422, detail="Background image could not be read.")
    try:
        created = BACKGROUND_LIBRARY.add(background.filename or "background.jpg", content)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    return _background_response(created)


@router.delete("/backgrounds/{background_id}", status_code=204)
def remove_background(background_id: str) -> None:
    try:
        BACKGROUND_LIBRARY.remove(background_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Background not found.") from exc


@router.post("/backgrounds/restore-defaults", response_model=TrainingBackgroundListResponse)
def restore_default_backgrounds() -> TrainingBackgroundListResponse:
    BACKGROUND_LIBRARY.restore_defaults()
    return list_backgrounds()


@router.post("/preview", response_model=SyntheticDatasetPreviewResponse)
async def create_preview(
    sources: Annotated[list[UploadFile], File()],
    device_name: Annotated[str, Form(min_length=1, max_length=100)],
    device_type: Annotated[str, Form(min_length=1, max_length=100)],
    preview_count: Annotated[int, Form(ge=1, le=12)] = 10,
    planned_samples: Annotated[int, Form(ge=1, le=100_000)] = 10_000,
    seed: Annotated[int | None, Form()] = None,
) -> SyntheticDatasetPreviewResponse:
    if len(sources) != 4:
        raise HTTPException(
            status_code=422, detail="Upload exactly four device images."
        )
    if planned_samples < preview_count:
        raise HTTPException(
            status_code=422,
            detail="Planned samples cannot be smaller than the preview count.",
        )

    with TemporaryDirectory(prefix="twintag-synthetic-") as temporary:
        workspace = Path(temporary)
        source_paths = []
        for index, source in enumerate(sources):
            if source.content_type not in ALLOWED_MEDIA_TYPES:
                raise HTTPException(
                    status_code=415, detail="Upload JPEG, PNG or WebP images."
                )
            content = await source.read(MAX_SOURCE_BYTES + 1)
            if len(content) > MAX_SOURCE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Each source image must be 15 MB or smaller.",
                )
            if not content:
                raise HTTPException(
                    status_code=422, detail="Source images cannot be empty."
                )
            suffix = Path(source.filename or "device.jpg").suffix or ".jpg"
            source_path = workspace / f"source-{index + 1}{suffix}"
            source_path.write_bytes(content)
            source_paths.append(source_path)

        generator = SyntheticDatasetGenerator(
            background_paths=BACKGROUND_LIBRARY.active_paths()
        )
        for angle, source_path in zip(SOURCE_ANGLES, source_paths, strict=True):
            try:
                generator.validate_source(source_path)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail=f"{angle} image: {exc}"
                ) from exc

        previews = []
        for index in range(preview_count):
            source_index = index % len(source_paths)
            output = workspace / f"preview-{index + 1}"
            try:
                dataset = generator.generate(
                    source_paths[source_index],
                    output,
                    count=1,
                    seed=None if seed is None else seed + index,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            sample = dataset.samples[0]
            previews.append(
                SyntheticPreviewResponse(
                    id=f"preview-{index + 1}",
                    image_url=_data_url(output / sample.image),
                    bounding_box=BoundingBoxResponse.model_validate(
                        sample.bounding_box, from_attributes=True
                    ),
                )
            )

    return SyntheticDatasetPreviewResponse(
        device_name=device_name.strip(),
        device_type=device_type.strip(),
        source_filenames=[source.filename or "device image" for source in sources],
        preview_count=len(previews),
        planned_samples=planned_samples,
        augmentations=AUGMENTATIONS,
        previews=previews,
    )


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
