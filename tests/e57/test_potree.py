from pathlib import Path
from subprocess import CompletedProcess

import pytest

from twintag_backend.e57.potree import PotreeConverter
from twintag_backend.e57.reader import E57ReadError


def test_converts_preview_and_copies_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "PotreeConverter"
    executable.touch()
    source_directory = tmp_path / "source"
    source_directory.mkdir()
    source = source_directory / "preview.laz"
    source.touch()
    (source_directory / "preview.json").write_text('{"point_count": 10}\n')
    output = tmp_path / "browser"

    def convert(command: list[str], check: bool) -> CompletedProcess:
        assert command[0] == str(executable)
        assert command[-2:] == ["-m", "random"]
        assert check is True
        output.mkdir(exist_ok=True)
        (output / "metadata.json").write_text("{}\n")
        return CompletedProcess(command, 0)

    monkeypatch.setattr("twintag_backend.e57.potree.subprocess.run", convert)

    metadata = PotreeConverter(executable).convert(source, output)

    assert metadata == output / "metadata.json"
    assert (output / "preview.json").read_text() == '{"point_count": 10}\n'


def test_rejects_missing_converter(tmp_path: Path) -> None:
    with pytest.raises(E57ReadError, match="was not found"):
        PotreeConverter(tmp_path / "missing").convert(
            tmp_path / "preview.laz", tmp_path / "output"
        )
