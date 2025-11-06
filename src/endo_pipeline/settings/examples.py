from typing import NamedTuple

EXAMPLE_DATASET = {
    "SUPP_FIG_Z_SLICE": "20250428_20X",
    "SUPP_FIG_IMG_PROC": "20250224_20X",
    "SUPP_FIG_SINGLE_TP_BF_OUTLIER": "20250224_20X",
    "SUPP_FIG_SINGLE_TP_GFP_OUTLIER": "20250319_20X",
}
"""Dictionary of example datasets for specific figures."""


class ExampleImage(NamedTuple):
    """Structure for information about an example imag used in a figure."""

    dataset_name: str
    position: int
    timepoint: int
    crop_position: tuple[int, int]


MODEL_QC_EXAMPLES: list[ExampleImage] = [
    ExampleImage(
        dataset_name="20250818_20X",
        position=0,
        timepoint=50,
        crop_position=(400, 400),
    ),
    ExampleImage(
        dataset_name="20250618_20X",
        position=0,
        timepoint=200,
        crop_position=(350, 300),
    ),
    ExampleImage(
        dataset_name="20250319_20X",
        position=0,
        timepoint=200,
        crop_position=(550, 450),
    ),
    ExampleImage(
        dataset_name="20250428_20X",
        position=0,
        timepoint=200,
        crop_position=(150, 100),
    ),
    ExampleImage(
        dataset_name="20250611_20X",
        position=0,
        timepoint=100,
        crop_position=(300, 600),
    ),
    ExampleImage(
        dataset_name="20250611_20X",
        position=0,
        timepoint=50,
        crop_position=(300, 600),
    ),
    ExampleImage(
        dataset_name="20250224_20X",
        position=0,
        timepoint=0,
        crop_position=(100, 100),
    ),
]
"""List of example crops for model QC."""
