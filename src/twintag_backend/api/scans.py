from fastapi import APIRouter, HTTPException

from twintag_backend.fixtures.scans import REFERENCE_SCAN
from twintag_backend.schemas.scans import Scan

router = APIRouter(prefix="/api/scans", tags=["scans"])


@router.get("", response_model=list[Scan])
def list_scans() -> list[Scan]:
    return [REFERENCE_SCAN]


@router.get("/{scan_id}", response_model=Scan)
def get_scan(scan_id: str) -> Scan:
    if scan_id != REFERENCE_SCAN.id:
        raise HTTPException(status_code=404, detail="Scan not found.")

    return REFERENCE_SCAN
