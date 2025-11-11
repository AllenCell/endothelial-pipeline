from typing import NamedTuple

EXAMPLE_DATASET = {
    "SUPP_FIG_Z_SLICE": "20250428_20X",
    "SUPP_FIG_IMG_PROC": "20250224_20X",
    "SUPP_FIG_SINGLE_TP_BF_OUTLIER": "20250224_20X",
    "SUPP_FIG_SINGLE_TP_GFP_OUTLIER": "20250319_20X",
}
"""Dictionary of example datasets for specific figures."""


class ExampleImage(NamedTuple):
    """Structure for information about an example image used in a figure."""

    dataset_name: str
    description: str
    position: int
    timepoint: int
    crop_x_start: int  # res level 1
    crop_y_start: int  # res level 1


MODEL_QC_EXAMPLES: list[ExampleImage] = [
    ExampleImage(
        dataset_name="20250818_20X",
        position=0,
        timepoint=50,
        x=395,
        y=400,
        description="1. UNALIGNED, middle of no flow",
    ),
    ExampleImage(
        dataset_name="20250818_20X",
        position=1,
        timepoint=0,
        x=545,
        y=445,
        description="2. UNALIGNED, start of no flow",
    ),
    ExampleImage(
        dataset_name="20250224_20X",
        position=0,
        timepoint=0,
        x=100,
        y=95,
        description="3. UNALIGNED, start of min flow",
    ),
    ExampleImage(
        dataset_name="20250428_20X",
        position=0,
        timepoint=4,
        x=285,
        y=625,
        description="4. UNALIGNED, start of low flow",
    ),
    ExampleImage(
        dataset_name="20250818_20X",
        position=5,
        timepoint=173,
        x=395,
        y=20,
        description="5. PARALLEL, middle of no flow without bright puncta",
    ),
    ExampleImage(
        dataset_name="20250618_20X",
        position=0,
        timepoint=200,
        x=345,
        y=295,
        description="6. PARALLEL, middle of min flow without bright puncta",
    ),
    ExampleImage(
        dataset_name="20250428_20X",
        position=0,
        timepoint=234,
        x=255,
        y=190,
        description="7. PARALLEL, middle of min flow with bright puncta",
    ),
    ExampleImage(
        dataset_name="20250428_20X",
        position=5,
        timepoint=293,
        x=285,
        y=485,
        description="8. PARALLEL, middle of min flow with bright puncta",
    ),
    ExampleImage(
        dataset_name="20250319_20X",
        position=0,
        timepoint=200,
        x=550,
        y=450,
        description="9. PARALLEL, middle of med flow, puncta location",
    ),
    ExampleImage(
        dataset_name="20250611_20X",
        position=0,
        timepoint=100,
        x=295,
        y=595,
        description="10. PERPENDICULAR, middle of max flow",
    ),
    ExampleImage(
        dataset_name="20250611_20X",
        position=0,
        timepoint=50,
        x=300,
        y=600,
        description="11. PERPENDICULAR, middle of max flow",
    ),
    ExampleImage(
        dataset_name="20250818_20X",
        position=5,
        timepoint=173,
        x=495,
        y=300,
        description="12. PERPENDICULAR / DIAG, middle of no flow",
    ),
    ExampleImage(
        dataset_name="20250827_20X",
        position=0,
        timepoint=236,
        x=535,
        y=405,
        description="13. PERPENDICULAR, middle of high flow.",
    ),
    ExampleImage(
        dataset_name="20250827_20X",
        position=0,
        timepoint=236,
        x=220,
        y=660,
        description="14. PERPENDICULAR, middle of high flow.",
    ),
]
"""List of example crops for model QC."""


CDH5_SEG_FIG_EXAMPLE: ExampleImage = ExampleImage(
    dataset_name="20250818_20X",
    position=4,
    timepoint=0,
    x=500,
    y=500,
    description="no flow center crop for CDH5 segmentation figure",
)
"""Example image for CDH5 segmentation figure."""
