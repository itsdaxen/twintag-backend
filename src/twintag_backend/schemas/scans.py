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
