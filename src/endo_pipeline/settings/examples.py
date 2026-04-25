from typing import NamedTuple

EXAMPLE_DATASET = {
    "FIGURE_2_LOW_FLOW_DATASET": "20250409_20X",
    "FIGURE_2_HIGH_FLOW_DATASET": "20251001_20X",
    "SUPP_FIG_Z_SLICE": "20250428_20X",
    "SUPP_FIG_IMG_PROC": "20250224_20X",
    "SUPP_FIG_SINGLE_TP_BF_OUTLIER": "20250224_20X",
    "SUPP_FIG_SINGLE_TP_GFP_OUTLIER": "20250319_20X",
}
"""Dictionary of example datasets for specific figures."""

FIGURE_3_RECONSTRUCTION_EXAMPLE_DATASETS = [
    "20260114_20X",  # two fixed points (15 dyn)
    "20260218_20X",  # two fixed points (15 dyn)
    "20260225_20X",  # single fixed point (15 dyn)
    "20260202_20X",  # single fixed point (15 dyn)
]


class ExampleImage(NamedTuple):
    """Structure for information about an example image used in a figure."""

    dataset_name: str
    description: str
    position: int
    timepoint: int
    crop_x_start: int  # res level 1
    crop_y_start: int  # res level 1


FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES: list[ExampleImage] = [
    ExampleImage(
        dataset_name="20250402_20X",
        description="low_flow",
        position=3,
        timepoint=150,
        crop_x_start=0,  # res level 0
        crop_y_start=0,  # res level 0
    ),
    ExampleImage(
        dataset_name="20251001_20X",
        description="high_flow",
        position=0,
        timepoint=200,
        crop_x_start=0,  # res level 0
        crop_y_start=0,  # res level 0
    ),
]

FIGURE_1_PATCH_FT_EXAMPLE_IMAGE: ExampleImage = ExampleImage(
    dataset_name="20250409_20X",
    description="example crop for showing segmentation and feature extraction",
    position=2,
    timepoint=204,
    crop_x_start=471,  # res level 0
    crop_y_start=1123,  # res level 0
)

FIGURE_3_EXAMPLE_IMAGES: list[ExampleImage] = [
    ExampleImage(
        dataset_name="20260304_20X",
        description="example of 12 dyn intermediate dataset with // alignment",
        position=5,
        timepoint=202,
        crop_x_start=0,  # res level 0
        crop_y_start=0,  # res level 0
    ),
    ExampleImage(
        dataset_name="20260121_20X",
        description="example of 12 dyn intermediate dataset with mixed alignment",
        position=1,
        timepoint=380,
        crop_x_start=0,  # res level 0
        crop_y_start=100,  # res level 0
    ),
    ExampleImage(
        dataset_name="20250813_20X",
        description="example of 14 dyn intermediate dataset with mixed alignment",
        position=1,
        timepoint=180,
        crop_x_start=250,  # res level 0
        crop_y_start=500,  # res level 0
    ),
    ExampleImage(
        dataset_name="20250326_20X",
        description="example of 15 dyn intermediate dataset with mixed alignment",
        position=0,
        timepoint=180,
        crop_x_start=0,  # res level 0
        crop_y_start=100,  # res level 0
    ),
]

FIGURE_4_EXAMPLE_IMAGES: list[ExampleImage] = [
    ExampleImage(
        dataset_name="20250402_20X",
        description="parental_line",
        position=0,
        timepoint=70,
        crop_x_start=0,  # res level 0
        crop_y_start=0,  # res level 0
    ),
    ExampleImage(
        dataset_name="20251105_20X",
        description="isogenic_control",
        position=0,
        timepoint=275,
        crop_x_start=0,  # res level 0
        crop_y_start=0,  # res level 0
    ),
    ExampleImage(
        dataset_name="20251029_20X",
        description="knock_down",
        position=0,
        timepoint=165,
        crop_x_start=0,  # res level 0
        crop_y_start=0,  # res level 0
    ),
    ExampleImage(
        dataset_name="20251119_20X",
        description="knock_down",
        position=0,
        timepoint=185,
        crop_x_start=0,  # res level 0
        crop_y_start=0,  # res level 0
    ),
]

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

EXAMPLE_DIFFAE_TRAINING_SCHEMATIC = "20250428_20X"

EXAMPLES_DIFFAE_TRAINING_VALIDATION: list[ExampleImage] = [
    ExampleImage(
        dataset_name="20250428_20X",
        position=0,
        timepoint=234,
        crop_x_start=255,
        crop_y_start=190,
        description="6. PARALLEL, middle of min flow with bright puncta.",
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
        dataset_name="20250728_20X",
        position=0,
        timepoint=236,
        crop_x_start=420,
        crop_y_start=375,
        description="5. PARALLEL, middle of no flow.",
    ),
    ExampleImage(
        dataset_name="20250728_20X",
        position=0,
        timepoint=518,
        crop_x_start=250,
        crop_y_start=285,
        description="PERPENDICULAR, end of no flow.",
    ),
]


class OpticalFlowExample(NamedTuple):
    """Reference to a single (dataset, position, timepoint-pair, crop) example
    used in optical-flow comparison figures.

    The crop is identified by its 1-indexed (row, col) position within the
    regular crop grid built for ``position`` -- row 1 / col 1 is the
    top-left crop, rows increase downward (sorted by ``START_Y``), cols
    increase rightward (sorted by ``START_X``).  Pixel bbox is resolved at
    figure-build time from the position's crop grid.
    """

    dataset_name: str
    description: str
    position: int
    t0: int
    t1: int
    crop_row: int
    crop_col: int


SUPP_FIG_OPTICAL_FLOW_COHERENT_EXAMPLE: OpticalFlowExample = OpticalFlowExample(
    dataset_name="20250409_20X",
    description="coherent (high migration coherence) example crop",
    position=2,
    t0=150,
    t1=151,
    crop_row=5,
    crop_col=4,
)

SUPP_FIG_OPTICAL_FLOW_INCOHERENT_EXAMPLE: OpticalFlowExample = OpticalFlowExample(
    dataset_name="20251001_20X",
    description="incoherent (low migration coherence) example crop",
    position=1,
    t0=198,
    t1=199,
    crop_row=2,
    crop_col=5,
)
FLOW_FIELD_CONSTRUCTION_EXAMPLE_IMAGES: list[ExampleImage] = [
    ExampleImage(
        dataset_name="20250409_20X",
        description="low_flow",
        position=0,
        timepoint=290,
        crop_x_start=0,  # res level 0
        crop_y_start=0,  # res level 0
    ),
    ExampleImage(
        dataset_name="20250409_20X",
        description="low_flow",
        position=0,
        timepoint=291,
        crop_x_start=0,  # res level 0
        crop_y_start=0,  # res level 0
    ),
]
"""Example images for illustrating flow field construction from trajectories."""
