import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pye57

from twintag_backend.e57.reader import E57ReadError, Quaternion, Vector3

REPRESENTATIONS = (
    "pinholeRepresentation",
    "sphericalRepresentation",
    "cylindricalRepresentation",
    "visualReferenceRepresentation",
)
IMAGE_BLOBS = {
    "jpegImage": ("jpg", "image/jpeg"),
    "pngImage": ("png", "image/png"),
}


@dataclass(frozen=True)
class PinholeIntrinsics:
    focal_length: float
    pixel_width: float
    pixel_height: float
    principal_point_x: float
    principal_point_y: float

    @property
    def focal_x_pixels(self) -> float:
        return self.focal_length / self.pixel_width

    @property
    def focal_y_pixels(self) -> float:
        return self.focal_length / self.pixel_height


@dataclass(frozen=True)
class ExtractedImage:
    id: str
    sweep_index: int
    face_index: int
    name: str
    guid: str
    associated_sweep_guid: str
    representation: str
    width: int
    height: int
    media_type: str
    path: Path
    position: Vector3
    rotation: Quaternion
    intrinsics: PinholeIntrinsics | None


class E57ImageExtractor:
    def extract(
        self, source: Path, output_directory: Path
    ) -> tuple[ExtractedImage, ...]:
        source = source.resolve()
        self._validate_source(source)
        output_directory.mkdir(parents=True, exist_ok=True)

        try:
            scan = pye57.E57(str(source))
        except Exception as exc:
            raise E57ReadError(f"Could not open E57 file: {source.name}") from exc

        try:
            sweep_indexes = {
                scan.get_header(index).guid: index for index in range(scan.scan_count)
            }
            images = self._extract_images(scan, output_directory, sweep_indexes)
            if images:
                self._write_manifest(output_directory / "manifest.json", images)
        except Exception as exc:
            raise E57ReadError(f"Could not extract E57 images: {source.name}") from exc
        finally:
            scan.close()

        return tuple(images)

    @staticmethod
    def _write_manifest(target: Path, images: list[ExtractedImage]) -> None:
        payload = []
        for image in images:
            item = asdict(image)
            item["path"] = image.path.name
            payload.append(item)
        target.write_text(json.dumps({"images": payload}, indent=2) + "\n")

    @staticmethod
    def _validate_source(source: Path) -> None:
        if not source.is_file():
            raise E57ReadError(f"E57 file not found: {source}")
        if source.suffix.lower() != ".e57":
            raise E57ReadError(f"Expected an .e57 file: {source.name}")

    def _extract_images(
        self,
        scan: pye57.E57,
        output_directory: Path,
        sweep_indexes: dict[str, int],
    ) -> list[ExtractedImage]:
        if not scan.root.isDefined("images2D"):
            return []

        images_node = scan.root["images2D"]
        face_indexes: defaultdict[str, int] = defaultdict(int)
        extracted = []

        for image_index in range(images_node.childCount()):
            image = pye57.libe57.StructureNode(images_node.get(image_index))
            associated_guid = self._string(image, "associatedData3DGuid")
            if associated_guid not in sweep_indexes:
                raise E57ReadError(
                    f"Image {image_index} references an unknown sweep: {associated_guid}"
                )

            sweep_index = sweep_indexes[associated_guid]
            face_index = face_indexes[associated_guid]
            face_indexes[associated_guid] += 1
            extracted.append(
                self._extract_image(
                    image=image,
                    output_directory=output_directory,
                    sweep_index=sweep_index,
                    face_index=face_index,
                    associated_guid=associated_guid,
                )
            )

        return extracted

    def _extract_image(
        self,
        image: pye57.libe57.StructureNode,
        output_directory: Path,
        sweep_index: int,
        face_index: int,
        associated_guid: str,
    ) -> ExtractedImage:
        representation_name, representation = self._representation(image)
        blob_name, extension, media_type = self._image_blob(representation)
        blob = pye57.libe57.BlobNode(representation.get(blob_name))
        image_id = f"sweep-{sweep_index:02d}-face-{face_index:02d}"
        target = output_directory / f"{image_id}.{extension}"
        target.write_bytes(blob.read_buffer().tobytes())
        position, rotation = self._pose(image)
        intrinsics = self._intrinsics(representation_name, representation)

        return ExtractedImage(
            id=image_id,
            sweep_index=sweep_index,
            face_index=face_index,
            name=self._string(image, "name"),
            guid=self._string(image, "guid"),
            associated_sweep_guid=associated_guid,
            representation=representation_name,
            width=self._integer(representation, "imageWidth"),
            height=self._integer(representation, "imageHeight"),
            media_type=media_type,
            path=target,
            position=position,
            rotation=rotation,
            intrinsics=intrinsics,
        )

    @staticmethod
    def _intrinsics(
        representation_name: str,
        representation: pye57.libe57.StructureNode,
    ) -> PinholeIntrinsics | None:
        if representation_name != "pinholeRepresentation":
            return None

        return PinholeIntrinsics(
            focal_length=E57ImageExtractor._float(representation, "focalLength"),
            pixel_width=E57ImageExtractor._float(representation, "pixelWidth"),
            pixel_height=E57ImageExtractor._float(representation, "pixelHeight"),
            principal_point_x=E57ImageExtractor._float(
                representation, "principalPointX"
            ),
            principal_point_y=E57ImageExtractor._float(
                representation, "principalPointY"
            ),
        )

    @staticmethod
    def _representation(
        image: pye57.libe57.StructureNode,
    ) -> tuple[str, pye57.libe57.StructureNode]:
        for name in REPRESENTATIONS:
            if image.isDefined(name):
                return name, pye57.libe57.StructureNode(image.get(name))
        raise E57ReadError("E57 image has no supported representation.")

    @staticmethod
    def _image_blob(
        representation: pye57.libe57.StructureNode,
    ) -> tuple[str, str, str]:
        for name, (extension, media_type) in IMAGE_BLOBS.items():
            if representation.isDefined(name):
                return name, extension, media_type
        raise E57ReadError("E57 image has no supported JPEG or PNG data.")

    @staticmethod
    def _pose(image: pye57.libe57.StructureNode) -> tuple[Vector3, Quaternion]:
        pose = pye57.libe57.StructureNode(image.get("pose"))
        translation = pye57.libe57.StructureNode(pose.get("translation"))
        rotation = pye57.libe57.StructureNode(pose.get("rotation"))

        return (
            Vector3(
                x=E57ImageExtractor._float(translation, "x"),
                y=E57ImageExtractor._float(translation, "y"),
                z=E57ImageExtractor._float(translation, "z"),
            ),
            Quaternion(
                x=E57ImageExtractor._float(rotation, "x"),
                y=E57ImageExtractor._float(rotation, "y"),
                z=E57ImageExtractor._float(rotation, "z"),
                w=E57ImageExtractor._float(rotation, "w"),
            ),
        )

    @staticmethod
    def _string(node: pye57.libe57.StructureNode, name: str) -> str:
        return str(pye57.libe57.StringNode(node.get(name)).value()).strip()

    @staticmethod
    def _integer(node: pye57.libe57.StructureNode, name: str) -> int:
        return int(pye57.libe57.IntegerNode(node.get(name)).value())

    @staticmethod
    def _float(node: pye57.libe57.StructureNode, name: str) -> float:
        return float(pye57.libe57.FloatNode(node.get(name)).value())
