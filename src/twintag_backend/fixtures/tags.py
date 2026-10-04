import json
from pathlib import Path

from twintag_backend.schemas.scans import (
    AssetContext,
    AssetTag,
    ExtractedText,
    TagBox,
    TagEvidence,
    TagPosition,
)

View = tuple[int, int, tuple[float, float, float, float]]

_TAG_DATA: tuple[
    tuple[str, tuple[float, float, float], float, float, tuple[View, ...]], ...
] = (
    (
        "asset-001",
        (-5.767, -1.793, 1.032),
        0.919620,
        0.117270,
        (
            (3, 1, (1616.744, 1627.476, 1857.757, 1792.263)),
            (4, 1, (2255.910, 1709.344, 2447.758, 1857.898)),
            (13, 1, (3843.457, 426.093, 4096.000, 1000.966)),
            (14, 1, (862.375, 658.038, 1593.416, 1167.061)),
        ),
    ),
    (
        "asset-002",
        (-5.761, -2.795, 1.017),
        0.937681,
        0.091599,
        (
            (3, 1, (648.035, 1581.511, 922.430, 1759.841)),
            (4, 1, (1456.200, 1648.180, 1689.979, 1813.784)),
            (12, 1, (3591.297, 531.611, 4096.000, 1065.508)),
            (13, 1, (768.514, 541.947, 1564.432, 1095.442)),
        ),
    ),
    (
        "asset-003",
        (-5.752, -4.790, 1.097),
        0.906216,
        0.144180,
        (
            (5, 3, (1305.515, 1640.993, 1538.330, 1801.521)),
            (6, 3, (2222.377, 1658.526, 2451.857, 1814.513)),
            (7, 3, (2832.398, 1679.511, 3059.427, 1837.057)),
            (8, 3, (3966.533, 1649.461, 4095.282, 1824.671)),
            (10, 1, (4002.260, 455.235, 4095.921, 1079.826)),
            (11, 1, (613.283, 520.102, 1436.800, 1076.654)),
        ),
    ),
    (
        "asset-004",
        (-5.749, -3.770, 1.097),
        0.930647,
        0.136104,
        (
            (4, 1, (379.141, 1561.255, 699.310, 1755.281)),
            (5, 3, (2168.604, 1666.414, 2389.798, 1820.365)),
            (6, 3, (3095.512, 1668.870, 3338.170, 1828.697)),
            (7, 3, (3675.647, 1703.780, 3904.875, 1861.297)),
            (11, 1, (3595.310, 560.578, 4096.000, 1085.920)),
            (12, 1, (560.818, 473.415, 1411.703, 1059.572)),
        ),
    ),
    (
        "asset-005",
        (-5.747, -5.853, 1.051),
        0.937056,
        0.095404,
        (
            (5, 3, (356.041, 1608.897, 595.575, 1775.067)),
            (6, 3, (1329.798, 1650.164, 1553.546, 1806.111)),
            (7, 3, (1916.679, 1652.941, 2145.885, 1813.712)),
            (8, 3, (2914.469, 1650.837, 3163.715, 1812.766)),
            (9, 1, (3699.118, 544.699, 4096.000, 1079.033)),
            (10, 1, (681.545, 480.341, 1520.745, 1057.148)),
        ),
    ),
    (
        "asset-006",
        (-2.749, -5.920, 1.467),
        0.906000,
        0.089000,
        (
            (6, 2, (894.757, 1828.187, 1151.721, 2235.934)),
            (7, 1, (1952.744, 1738.608, 2647.905, 2409.366)),
            (8, 1, (1.290, 1772.784, 280.944, 2428.061)),
            (8, 1, (0.516, 1770.564, 176.334, 2427.086)),
            (8, 4, (3774.082, 1808.020, 4096.000, 2451.677)),
            (9, 3, (1233.772, 2005.909, 1414.733, 2170.865)),
            (10, 3, (2271.328, 2021.379, 2444.782, 2187.252)),
        ),
    ),
    (
        "asset-007",
        (-2.720, -5.940, 1.214),
        0.937000,
        0.096000,
        (
            (7, 1, (1780.128, 596.648, 2777.958, 1280.901)),
            (8, 1, (0.219, 752.268, 273.444, 1458.722)),
            (9, 3, (1202.554, 1726.524, 1458.425, 1902.457)),
            (10, 3, (2241.260, 1753.091, 2483.123, 1914.593)),
            (11, 3, (3108.611, 1776.162, 3336.801, 1939.734)),
        ),
    ),
)

_OCR_RESULTS_PATH = Path(__file__).with_name("data") / "context_ocr_qwen3_vl_8b.json"
_OCR_RESULTS = json.loads(_OCR_RESULTS_PATH.read_text())
_OCR_BY_ASSET = {
    key.split("-sweep-", 1)[0]: (key, value) for key, value in _OCR_RESULTS.items()
}


def _tag(
    tag_id: str,
    position: tuple[float, float, float],
    confidence: float,
    spread: float,
    views: tuple[View, ...],
) -> AssetTag:
    number = int(tag_id.rsplit("-", 1)[1])
    evidence = [
        TagEvidence(
            image_id=f"sweep-{sweep:02d}-face-{face:02d}",
            sweep_index=sweep,
            face_index=face,
            box=TagBox(left=box[0], top=box[1], right=box[2], bottom=box[3]),
        )
        for sweep, face, box in views
    ]
    result_key, context_data = _OCR_BY_ASSET[tag_id]
    evidence_image_id = result_key.split("-", 2)[2]
    context = AssetContext(
        model="Qwen/Qwen3-VL-8B-Instruct",
        inference="precomputed",
        **{
            category: [
                ExtractedText(
                    text=item["text"],
                    confidence=item["confidence"],
                    evidence_image_id=evidence_image_id,
                )
                for item in values
            ]
            for category, values in context_data.items()
        },
    )
    return AssetTag(
        id=tag_id,
        asset_type="ABB REX615",
        label=f"ABB REX615 · {number:02d}",
        source="model",
        status="detected",
        confidence=confidence,
        position=TagPosition(x=position[0], y=position[1], z=position[2]),
        observation_count=len(evidence),
        sweep_count=len({sweep for sweep, _, _ in views}),
        spatial_spread=spread,
        evidence=evidence,
        context=context,
    )


REFERENCE_TAGS = [_tag(*data) for data in _TAG_DATA]
