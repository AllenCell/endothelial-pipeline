# %%
from endo_pipeline.configs import (
    TimepointAnnotation,
    get_datasets_in_collection,
    load_dataset_config,
)
from endo_pipeline.library.process.single_tp_outlier.bf_timepoint_outlier import detect_bf_outliers
from endo_pipeline.library.process.single_tp_outlier.gfp_timepoint_outlier import (
    detect_egfp_scope_errors,
)
from endo_pipeline.library.process.single_tp_outlier.timepoint_outlier import performance_stats
from endo_pipeline.settings.examples import EXAMPLE_DATASET

# %%
DESCRIPTION = "Visualize the automatic detection of outlier timepoints in BF and EGFP channels."
TAGS = ["supfig", "preprocessing"]

# %% Load example datasets
dataset_config_bf = load_dataset_config(EXAMPLE_DATASET["SUPP_FIG_SINGLE_TP_BF_OUTLIER"])
dataset_config_gfp = load_dataset_config(EXAMPLE_DATASET["SUPP_FIG_SINGLE_TP_GFP_OUTLIER"])
position = 0

# %% Panel A - Auto-detect BF outliers
detect_bf_outliers(dataset_config_bf, position, visualize=True)

# %% Panel B - Auto-detect EGFP scope errors
detect_egfp_scope_errors(dataset_config_gfp, position, visualize=True)

# %% Performance statistics reported across datasets in collection
datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")

performance_stats(
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

performance_stats(
    datasets=datasets,
    manual_annotations=[TimepointAnnotation.GFP_SCOPE_ERROR],
    auto_annotations=[TimepointAnnotation.AUTO_GFP_SCOPE_ERROR],
    annotation_type="gfp_outlier_detection",
)
# %%
