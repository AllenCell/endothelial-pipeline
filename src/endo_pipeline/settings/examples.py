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
    crop_position: tuple[int, int]  # x_start, y_start of the crop for zarr resolution level 1


MODEL_QC_EXAMPLES: list[ExampleImage] = [
    ExampleImage(
        dataset_name="20250818_20X",
        position=0,
        timepoint=50,
        crop_position=(395, 400),
        description="1. UNALIGNED, middle of no flow",
    ),
    ExampleImage(
        dataset_name="20250818_20X",
        position=1,
        timepoint=0,
        crop_position=(545, 445),
        description="2. UNALIGNED, start of no flow",
    ),
    ExampleImage(
        dataset_name="20250224_20X",
        position=0,
        timepoint=0,
        crop_position=(100, 95),
        description="3. UNALIGNED, start of min flow",
    ),
    ExampleImage(
        dataset_name="20250428_20X",
        position=0,
        timepoint=4,
        crop_position=(285, 625),
        description="4. UNALIGNED, start of low flow",
    ),
    ExampleImage(
        dataset_name="20250818_20X",
        position=5,
        timepoint=173,
        crop_position=(395, 20),
        description="5. PARALLEL, middle of no flow without bright puncta",
    ),
    ExampleImage(
        dataset_name="20250618_20X",
        position=0,
        timepoint=200,
        crop_position=(345, 295),
        description="6. PARALLEL, middle of min flow without bright puncta",
    ),
    ExampleImage(
        dataset_name="20250428_20X",
        position=0,
        timepoint=234,
        crop_position=(255, 190),
        description="7. PARALLEL, middle of min flow with bright puncta",
    ),
    ExampleImage(
        dataset_name="20250428_20X",
        position=5,
        timepoint=293,
        crop_position=(285, 485),
        description="8. PARALLEL, middle of min flow with bright puncta",
    ),
    ExampleImage(
        dataset_name="20250319_20X",
        position=0,
        timepoint=200,
        crop_position=(550, 450),
        description="9. PARALLEL, middle of med flow, puncta location",
    ),
    ExampleImage(
        dataset_name="20250611_20X",
        position=0,
        timepoint=100,
        crop_position=(295, 595),
        description="10. PERPENDICULAR, middle of max flow",
    ),
    ExampleImage(
        dataset_name="20250611_20X",
        position=0,
        timepoint=50,
        crop_position=(300, 600),
        description="11. PERPENDICULAR, middle of max flow",
    ),
    ExampleImage(
        dataset_name="20250818_20X",
        position=5,
        timepoint=173,
        crop_position=(495, 300),
        description="12. PERPENDICULAR / DIAG, middle of no flow",
    ),
    ExampleImage(
        dataset_name="20250827_20X",
        position=0,
        timepoint=236,
        crop_position=(535, 405),
        description="13. PERPENDICULAR, middle of high flow.",
    ),
    ExampleImage(
        dataset_name="20250827_20X",
        position=0,
        timepoint=236,
        crop_position=(220, 660),
        description="14. PERPENDICULAR, middle of high flow.",
    ),
]
"""List of example crops for model QC."""


CDH5_SEG_FIG_EXAMPLE: ExampleImage = ExampleImage(
    dataset_name="20250818_20X",
    position=4,
    timepoint=0,
    crop_position=(500, 500),
    description="no flow center crop for CDH5 segmentation figure",
)
"""Example image for CDH5 segmentation figure."""
