from pydantic import BaseModel


class BoundingBoxResponse(BaseModel):
    left: float
    top: float
    right: float
    bottom: float


class SyntheticPreviewResponse(BaseModel):
    id: str
    image_url: str
    bounding_box: BoundingBoxResponse


class SyntheticDatasetPreviewResponse(BaseModel):
    device_name: str
    device_type: str
    source_filenames: list[str]
    preview_count: int
    planned_samples: int
    augmentations: list[str]
    previews: list[SyntheticPreviewResponse]
