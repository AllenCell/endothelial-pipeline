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


CDH5_SEG_FIG_EXAMPLE: ExampleImage = ExampleImage(
    dataset_name="20250818_20X",
    position=4,
    timepoint=0,
    crop_x_start=500,
    crop_y_start=500,
    description="no flow center crop for CDH5 segmentation figure",
)

MODEL_QC_EXAMPLES_TRAINING_POSITIONS: list[ExampleImage] = [
    ExampleImage(
        dataset_name="20250818_20X",
        position=0,
        timepoint=50,
        crop_x_start=395,
        crop_y_start=400,
        description="1. UNALIGNED, middle of no flow.",
    ),
    ExampleImage(
        dataset_name="20250818_20X",
        position=1,
        timepoint=0,
        crop_x_start=545,
        crop_y_start=445,
        description="2. UNALIGNED, start of no flow.",
    ),
    ExampleImage(
        dataset_name="20250428_20X",
        position=0,
        timepoint=4,
        crop_x_start=285,
        crop_y_start=625,
        description="3. UNALIGNED, start of low flow.",
    ),
    ExampleImage(
        dataset_name="20250818_20X",
        position=5,
        timepoint=173,
        crop_x_start=395,
        crop_y_start=20,
        description="4. PARALLEL, middle of no flow without bright puncta.",
    ),
    ExampleImage(
        dataset_name="20250618_20X",
        position=0,
        timepoint=200,
        crop_x_start=345,
        crop_y_start=295,
        description="5. PARALLEL, middle of min flow without bright puncta.",
    ),
    ExampleImage(
        dataset_name="20250428_20X",
        position=0,
        timepoint=234,
        crop_x_start=255,
        crop_y_start=190,
        description="6. PARALLEL, middle of min flow with bright puncta.",
    ),
    ExampleImage(
        dataset_name="20250428_20X",
        position=5,
        timepoint=293,
        crop_x_start=285,
        crop_y_start=485,
        description="7. PARALLEL, middle of min flow with bright puncta.",
    ),
    ExampleImage(
        dataset_name="20250611_20X",
        position=0,
        timepoint=100,
        crop_x_start=295,
        crop_y_start=595,
        description="8. PERPENDICULAR, middle of max flow.",
    ),
    ExampleImage(
        dataset_name="20250611_20X",
        position=0,
        timepoint=50,
        crop_x_start=300,
        crop_y_start=600,
        description="9. PERPENDICULAR, middle of max flow.8",
    ),
    ExampleImage(
        dataset_name="20250818_20X",
        position=5,
        timepoint=173,
        crop_x_start=495,
        crop_y_start=300,
        description="10. PERPENDICULAR / DIAG, middle of no flow.",
    ),
    ExampleImage(
        dataset_name="20250827_20X",
        position=0,
        timepoint=236,
        crop_x_start=535,
        crop_y_start=405,
        description="11. PERPENDICULAR, middle of high flow.",
    ),
    ExampleImage(
        dataset_name="20250827_20X",
        position=0,
        timepoint=236,
        crop_x_start=220,
        crop_y_start=660,
        description="12. PERPENDICULAR, middle of high flow. ",
    ),
]
"""List of example crops for model QC, positions used in model training."""

MODEL_QC_EXAMPLES_VALIDATION_POSITIONS: list[ExampleImage] = [
    ExampleImage(
        dataset_name="20250428_20X",
        position=2,
        timepoint=15,
        crop_x_start=290,
        crop_y_start=350,
        description="1. UNALIGNED, beginning of low flow.",
    ),
    ExampleImage(
        dataset_name="20250611_20X",
        position=3,
        timepoint=0,
        crop_x_start=120,
        crop_y_start=315,
        description="2. UNALIGNED, mixed morph start of high flow.",
    ),
    ExampleImage(
        dataset_name="20250714_20X",
        position=5,
        timepoint=0,
        crop_x_start=400,
        crop_y_start=470,
        description="3. UNALIGNED, rounded cell at start of low flow.",
    ),
    ExampleImage(
        dataset_name="20250714_20X",
        position=5,
        timepoint=195,
        crop_x_start=325,
        crop_y_start=150,
        description="4. PARALLEL, middle of low flow low density.",
    ),
    ExampleImage(
        dataset_name="20250714_20X",
        position=5,
        timepoint=195,
        crop_x_start=325,
        crop_y_start=195,
        description="5. PARALLEL, middle of low flow higher density, some bright puncta.",
    ),
    ExampleImage(
        dataset_name="20250618_20X",
        position=3,
        timepoint=230,
        crop_x_start=185,
        crop_y_start=300,
        description="6. PARALLEL, middle of low flow some bright puncta.",
    ),
    ExampleImage(
        dataset_name="20250319_20X",
        position=0,
        timepoint=200,
        crop_x_start=550,
        crop_y_start=450,
        description="7. PARALLEL, middle of med flow, puncta location.",
    ),
    ExampleImage(
        dataset_name="20250611_20X",
        position=3,
        timepoint=169,
        crop_x_start=125,
        crop_y_start=100,
        description="8. PERPENDICULAR, middle of high flow",
    ),
    ExampleImage(
        dataset_name="20250611_20X",
        position=4,
        timepoint=31,
        crop_x_start=220,
        crop_y_start=250,
        description="9. PERPENDICULAR, early in high flow",
    ),
    ExampleImage(
        dataset_name="20250611_20X",
        position=4,
        timepoint=252,
        crop_x_start=65,
        crop_y_start=90,
        description="10. PERPENDICULAR / DIAG, later in high flow",
    ),
]
"""List of example crops for model QC, positions held out for validation."""

MODEL_QC_EXAMPLES_REP_2_POSITIONS: list[ExampleImage] = [
    ExampleImage(
        dataset_name="20250728_20X",
        position=0,
        timepoint=0,
        crop_x_start=330,
        crop_y_start=445,
        description="1. UNALIGNED, beginning of no flow.",
    ),
    ExampleImage(
        dataset_name="20250409_20X",
        position=0,
        timepoint=0,
        crop_x_start=140,
        crop_y_start=200,
        description="2. UNALIGNED, beginning of no flow.",
    ),
    ExampleImage(
        dataset_name="20251001_20X",
        position=0,
        timepoint=30,
        crop_x_start=200,
        crop_y_start=335,
        description="3. UNALIGNED, beginning of high flow.",
    ),
    ExampleImage(
        dataset_name="20250224_20X",
        position=0,
        timepoint=0,
        crop_x_start=100,
        crop_y_start=95,
        description="4. UNALIGNED, start of min flow.",
    ),
    ExampleImage(
        dataset_name="20250728_20X",
        position=0,
        timepoint=236,
        crop_x_start=420,
        crop_y_start=375,
        description="5. PARALLEL, middle of no flow.",
    ),
    ExampleImage(
        dataset_name="20250409_20X",
        position=0,
        timepoint=225,
        crop_x_start=20,
        crop_y_start=300,
        description="6. PARALLEL, middle of low flow",
    ),
    ExampleImage(
        dataset_name="20250428_20X",
        position=0,
        timepoint=161,
        crop_x_start=200,
        crop_y_start=30,
        description="7. PARALLEL, middle of no flow.",
    ),
    ExampleImage(
        dataset_name="20250728_20X",
        position=0,
        timepoint=518,
        crop_x_start=250,
        crop_y_start=285,
        description="8. PERPENDICULAR, end of no flow.",
    ),
    ExampleImage(
        dataset_name="20251001_20X",
        position=0,
        timepoint=220,
        crop_x_start=75,
        crop_y_start=75,
        description="9. PERPENDICULAR, middle of high flow.",
    ),
    ExampleImage(
        dataset_name="20251001_20X",
        position=0,
        timepoint=220,
        crop_x_start=300,
        crop_y_start=220,
        description="10. PERPENDICULAR / DIAG middle of high flow",
    ),
]
"""List of example crops for model QC, positions from replicate 2. Datasets not used for model training."""
