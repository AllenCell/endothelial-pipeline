"""Data structures for segmentation manifests."""

from dataclasses import field
from pathlib import Path

from mashumaro.config import BaseConfig
from pydantic.dataclasses import dataclass


@dataclass
class ImageLocation:
    """Storage locations for segmentations."""

    path: Path | None = None
    """Local path to segmentation.

    The path can be a template path that uses {{position}} or {{timepoint}} as
    placeholders for dynamic values of position or timepoint, respectively.
    These placeholders will be replaced with given position or timepoint values
    when accessing the location via ``get_segmentation_location_for_dataset``.

    .. code-block:: python

        manifest = SegmentationManifest(
            name="manifest_name",
            workflow="workflow_name",
            locations={
                "dataset_name": SegmentationLocation(
                    path=Path("P{{position}}/T{{timepoint}}.ome.tiff")
                )
            },
        )

        location = get_segmentation_location_for_dataset(manifest, "dataset_name", 3, 10)
        # returns location as SegmentationLocation(path=Path("P3/T10.ome.tiff"))
    """


@dataclass
class ImageManifest:
    """Segmentation manifest for pipeline."""

    name: str
    """Unique name of the segmentation manifest."""

    workflow: str
    """Name of workflow that produced the segmentations."""

    parameters: dict = field(default_factory=dict)
    """Specific workflow parameters used to produce the segmentations."""

    locations: dict[str, ImageLocation] = field(default_factory=dict)
    """Locations of individual segmentations."""

    class Config(BaseConfig):
        """Settings for segmentation manifest."""

        forbid_extra_keys = True
        omit_none = False
