from twintag_backend.schemas.scans import (
    ProcessingStage,
    Scan,
    ScanCounts,
    ScanSource,
)

REFERENCE_SCAN = Scan(
    id="veo-reference",
    name="VEO reference facility",
    status="ready",
    mode="fixture",
    source=ScanSource(
        filename="cloud_0-001.e57",
        format="e57",
        size_bytes=2_322_023_424,
    ),
    counts=ScanCounts(
        sweeps=18,
        images=108,
        points=116_640_000,
    ),
    stages=[
        ProcessingStage(
            key="metadata", label="Read scanner positions", status="complete"
        ),
        ProcessingStage(key="images", label="Extract camera images", status="complete"),
        ProcessingStage(
            key="recognition", label="Recognize equipment", status="complete"
        ),
        ProcessingStage(
            key="positioning", label="Calculate 3D positions", status="complete"
        ),
        ProcessingStage(key="tags", label="Prepare asset tags", status="complete"),
    ],
)
