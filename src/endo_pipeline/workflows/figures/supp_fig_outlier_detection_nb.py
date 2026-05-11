"""
Visualize the automatic detection of outlier timepoints in BF and EGFP channels.

#supfig #preprocessing
"""

# %%
import matplotlib.pyplot as plt

from endo_pipeline.configs import (
    TimepointAnnotation,
    get_datasets_in_collection,
    load_dataset_config,
)
from endo_pipeline.io import get_output_path
from endo_pipeline.library.process.single_tp_outlier.bf_timepoint_outlier import detect_bf_outliers
from endo_pipeline.library.process.single_tp_outlier.gfp_timepoint_outlier import (
    detect_egfp_scope_errors,
)
from endo_pipeline.library.process.single_tp_outlier.timepoint_outlier import performance_stats
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.settings.examples import EXAMPLE_DATASET
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH

plt.style.use("endo_pipeline.figure")

# %% Load example datasets
FIGURE_ID = "SUPP_FIG_SINGLE_TP"
dataset_config_bf = load_dataset_config(EXAMPLE_DATASET[f"{FIGURE_ID}_BF_OUTLIER"])
dataset_config_gfp = load_dataset_config(EXAMPLE_DATASET[f"{FIGURE_ID}_GFP_OUTLIER"])
position = 0

# %% Panel A - Auto-detect BF outliers
_ = detect_bf_outliers(dataset_config_bf, position, visualize=True)

# %% Panel B - Auto-detect EGFP scope errors
_ = detect_egfp_scope_errors(dataset_config_gfp, position, visualize=True)

# %% Figure
save_dir = get_output_path("annotate_tp_outliers")
output_path = save_dir / "annotate_tp_outliers.svg"
panels = [
    FigurePanel(
        letter="A",
        path=save_dir / f"bf_outliers_{dataset_config_bf.name}_P{position}.svg",
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0.1,
    ),
    FigurePanel(
        letter="B",
        path=save_dir / f"gfp_outliers_{dataset_config_gfp.name}_P{position}.svg",
        x_position=3.75,
        y_position=0,
        x_offset=0,
        y_offset=0.41,
    ),
]
build_figure_from_panels(panels, output_path, width=MAX_FIGURE_WIDTH, height=3)

# %% Performance statistics reported across datasets in collection
datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")

bf_results = performance_stats(
    datasets=datasets,
    manual_annotations=[
        TimepointAnnotation.BF_SCOPE_ERROR,
        TimepointAnnotation.BF_TEMP_ARTIFACT,
    ],
    auto_annotations=[
        TimepointAnnotation.AUTO_BF_SCOPE_ERROR,
        TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT,
    ],
    annotation_type="brightfield_outlier_detection",
)
print(bf_results)

gfp_results = performance_stats(
    datasets=datasets,
    manual_annotations=[TimepointAnnotation.GFP_SCOPE_ERROR],
    auto_annotations=[TimepointAnnotation.AUTO_GFP_SCOPE_ERROR],
    annotation_type="gfp_outlier_detection",
)
print(gfp_results)
# %%
