import base64
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from twintag_backend.schemas.synthetic import (
    BoundingBoxResponse,
    SyntheticDatasetPreviewResponse,
    SyntheticPreviewResponse,
)
from twintag_backend.synthetic.generator import SyntheticDatasetGenerator

router = APIRouter(prefix="/api/synthetic-datasets", tags=["synthetic datasets"])

ALLOWED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SOURCE_BYTES = 15 * 1024 * 1024
AUGMENTATIONS = [
    "perspective",
    "rotation",
    "scale",
    "brightness and contrast",
    "blur and noise",
    "compression",
    "partial occlusion",
]
SOURCE_ANGLES = ("Front", "Back", "Left", "Right")


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
        raise HTTPException(status_code=422, detail="Upload exactly four device images.")
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
                    status_code=413, detail="Each source image must be 15 MB or smaller."
                )
            if not content:
                raise HTTPException(status_code=422, detail="Source images cannot be empty.")
            suffix = Path(source.filename or "device.jpg").suffix or ".jpg"
            source_path = workspace / f"source-{index + 1}{suffix}"
            source_path.write_bytes(content)
            source_paths.append(source_path)

        generator = SyntheticDatasetGenerator()
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
