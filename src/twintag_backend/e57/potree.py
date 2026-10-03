import shutil
import subprocess
from pathlib import Path

from twintag_backend.e57.reader import E57ReadError


class PotreeConverter:
    def __init__(self, executable: Path | None = None) -> None:
        self.executable = executable

    def convert(self, source: Path, output_directory: Path) -> Path:
        executable = self._resolve_executable()
        output_directory.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [
                    str(executable),
                    str(source.resolve()),
                    "-o",
                    str(output_directory.resolve()),
                    "-m",
                    "random",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise E57ReadError("Potree conversion failed") from exc

        metadata = output_directory / "metadata.json"
        if not metadata.is_file():
            raise E57ReadError("Potree conversion did not produce metadata.json")

        manifest = source.with_name("preview.json")
        if manifest.is_file():
            shutil.copyfile(manifest, output_directory / manifest.name)
        return metadata

    def _resolve_executable(self) -> Path:
        if self.executable is not None:
            executable = self.executable.expanduser().resolve()
            if executable.is_file():
                return executable
            raise E57ReadError(f"PotreeConverter was not found: {executable}")

        discovered = shutil.which("PotreeConverter")
        if discovered:
            return Path(discovered)
        raise E57ReadError("PotreeConverter is not installed or available on PATH")
