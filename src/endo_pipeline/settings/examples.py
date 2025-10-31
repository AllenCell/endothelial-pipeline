from typing import TypedDict

EXAMPLE_DATASET = {
    "SUPP_FIG_Z_SLICE": "20250428_20X",
    "SUPP_FIG_IMG_PROC": "20250224_20X",
    "SUPP_FIG_SINGLE_TP_BF_OUTLIER": "20250224_20X",
    "SUPP_FIG_SINGLE_TP_GFP_OUTLIER": "20250319_20X",
}
"""Dictionary of example datasets for specific figures."""


class ExampleImage(TypedDict):
    """Structure for information about an example imag used in a figure."""

    dataset_name: str
    position: int
    timepoint: int
    crop_position: tuple[int, int]


MODEL_QC_EXAMPLES: list[ExampleImage] = [
    {
        "dataset_name": "20250818_20X",
        "position": 0,
        "timepoint": 50,
        "crop_position": (400, 400),
    },  # middle of no flow
    {
        "dataset_name": "20250618_20X",
        "position": 0,
        "timepoint": 200,
        "crop_position": (350, 300),
    },  # middle of min flow without bright puncta
    {
        "dataset_name": "20250319_20X",
        "position": 0,
        "timepoint": 200,
        "crop_position": (550, 450),
    },  # middle of med flow
    {
        "dataset_name": "20250428_20X",
        "position": 0,
        "timepoint": 200,
        "crop_position": (150, 100),
    },  # middle of min flow with bright puncta
    {
        "dataset_name": "20250611_20X",
        "position": 0,
        "timepoint": 100,
        "crop_position": (300, 600),
    },  # middle of max flow
    {
        "dataset_name": "20250611_20X",
        "position": 0,
        "timepoint": 50,
        "crop_position": (300, 600),
    },  # middle of max flow
    {
        "dataset_name": "20250224_20X",
        "position": 0,
        "timepoint": 0,
        "crop_position": (100, 100),
    },  # start of min flow
]
"""List of example crops for model QC."""
