from pathlib import Path

import numpy as np
import pye57
import pytest


@pytest.fixture
def sample_e57(tmp_path: Path) -> Path:
    path = tmp_path / "sample.e57"
    scan = pye57.E57(str(path), mode="w")
    scan.write_scan_raw(
        {
            "cartesianX": np.array([0.0, 1.0]),
            "cartesianY": np.array([0.0, 2.0]),
            "cartesianZ": np.array([0.0, 3.0]),
        },
        name="Test sweep",
        translation=np.array([1.0, 2.0, 3.0]),
    )
    scan.close()
    return path
