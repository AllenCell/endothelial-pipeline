"""Data structures for image manifests."""

from dataclasses import field
from pathlib import Path

from mashumaro.config import BaseConfig
from pydantic.dataclasses import dataclass


@dataclass
class ImageLocation:
    """Storage locations for images."""

    path: Path | None = None
    """Local path to image.

    The path can be a template path that uses {{position}} or {{timepoint}} as
    placeholders for dynamic values of position or timepoint, respectively.
    These placeholders will be replaced with given position or timepoint values
    when accessing the location via ``get_image_location_for_dataset``.

    .. code-block:: python

        manifest = ImageManifest(
            name="manifest_name",
            workflow="workflow_name",
            locations={
                "dataset_name": ImageLocation(
                    path=Path("P{{position}}/T{{timepoint}}.ome.tiff")
                )
            },
        )

        location = get_image_location_for_dataset(manifest, "dataset_name", 3, 10)
        # returns location as ImageLocation(path=Path("P3/T10.ome.tiff"))
    """


@dataclass
class ImageManifest:
    """Image manifest for pipeline."""

    name: str
    """Unique name of the image manifest."""

    workflow: str
    """Name of workflow that produced the images."""

    parameters: dict = field(default_factory=dict)
    """Specific workflow parameters used to produce the images."""

    locations: dict[str, ImageLocation] = field(default_factory=dict)
    """Locations of individual images."""

    class Config(BaseConfig):
        """Settings for image manifest."""

        forbid_extra_keys = True
        omit_none = False
