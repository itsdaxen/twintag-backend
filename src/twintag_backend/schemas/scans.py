from typing import Literal

from pydantic import BaseModel, Field

ScanStatus = Literal["queued", "processing", "ready", "failed"]
ScanMode = Literal["fixture", "processed"]
StageStatus = Literal["pending", "running", "complete", "failed"]


class ScanSource(BaseModel):
    filename: str
    format: Literal["e57"]
    size_bytes: int = Field(ge=0)


class ScanCounts(BaseModel):
    sweeps: int = Field(ge=0)
    images: int = Field(ge=0)
    points: int = Field(ge=0)


class ProcessingStage(BaseModel):
    key: str
    label: str
    status: StageStatus


class Scan(BaseModel):
    id: str
    name: str
    status: ScanStatus
    mode: ScanMode
    source: ScanSource
    counts: ScanCounts
    stages: list[ProcessingStage]


class TagPosition(BaseModel):
    x: float
    y: float
    z: float


class TagBox(BaseModel):
    left: float
    top: float
    right: float
    bottom: float


class TagEvidence(BaseModel):
    image_id: str
    sweep_index: int = Field(ge=0)
    face_index: int = Field(ge=0)
    box: TagBox


class ExtractedText(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)
    evidence_image_id: str


class AssetContext(BaseModel):
    model: str
    inference: Literal["precomputed"]
    official_labels: list[ExtractedText] = Field(default_factory=list)
    inspection_markings: list[ExtractedText] = Field(default_factory=list)
    field_notes: list[ExtractedText] = Field(default_factory=list)


class AssetTag(BaseModel):
    id: str
    asset_type: str
    label: str
    source: Literal["model"]
    status: Literal["detected", "reviewed"]
    confidence: float = Field(ge=0, le=1)
    position: TagPosition
    observation_count: int = Field(ge=1)
    sweep_count: int = Field(ge=1)
    spatial_spread: float = Field(ge=0)
    evidence: list[TagEvidence]
    context: AssetContext | None = None
