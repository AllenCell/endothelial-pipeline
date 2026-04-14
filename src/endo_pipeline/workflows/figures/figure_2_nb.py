# %%
"""
Main function to create figure panels for Figure 2.
"""
import matplotlib.pyplot as plt

from endo_pipeline.configs import TimepointAnnotation, load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_annotations
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_binned_mean_to_fixed_points,
    add_optical_flow_features,
)
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_dataset_color
from endo_pipeline.library.visualize.migration_coherence import plot_optical_flow_histogram
from endo_pipeline.library.visualize.summary_plot import plot_cross_dataset_summaries
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    METADATA_COLUMNS_TO_KEEP,
)
from endo_pipeline.settings.examples import EXAMPLE_DATASET
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
from endo_pipeline.settings.flow_field_dataframes import (
    DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
    STABILITY_COLUMN_NAME,
)
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

# %%
plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("figure_2")

dataset_low = EXAMPLE_DATASET["FIGURE_2_LOW_FLOW_DATASET"]
dataset_high = EXAMPLE_DATASET["FIGURE_2_HIGH_FLOW_DATASET"]
dataset_summary_list = SUMMARY_PLOT_DATASETS["low_high"]

# %%
# Load diffae features
base_name = (
    f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{MIGRATION_COHERENCE_CROP_PATTERN}"
)
feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

fixed_points_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{base_name}"
fixed_points_dataframe_manifest = load_dataframe_manifest(fixed_points_dataframe_manifest_name)
# %%
# --- Cross-dataset summary plots ---
plot_cross_dataset_summaries(
    dataset_names=dataset_summary_list,
    feature_dataframe_manifest=feature_dataframe_manifest,
    fixed_points_dataframe_manifest=fixed_points_dataframe_manifest,
    output_dir=save_dir,
    x_axis_mode="shear_stress_categorical",
    figure_size=(MAX_FIGURE_WIDTH / 4, 2),
    stable_only=True,
)

# %%
optical_flow_feature = ColumnName.OpticalFlow.UNIT_VECTOR_MEAN
vmax = 1
hist_binwidth = 0.02
fig, ax = plt.subplots(figsize=(2.15, 2), layout="constrained")
for dataset_name in [dataset_low, dataset_high]:
    # get settings
    dataset_config = load_dataset_config(dataset_name)
    shear_stress = round(dataset_config.flow_conditions[0].shear_stress)
    dataset_name_flow = f"{dataset_name}_shear_{shear_stress}"
    ss_label = f"{shear_stress} dyn/cm$\u00b2$"
    hist_color = get_dataset_color(dataset_name)

    # load and filter data
    df = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP["grid"], *DYNAMICS_COLUMN_NAMES]
    df_ = df[columns_to_compute].compute()
    df_steady_state = filter_dataframe_by_annotations(
        df_,
        dataset_config,
        timepoint_annotations=[TimepointAnnotation.NOT_STEADY_STATE],
    )

    df_of = add_optical_flow_features(
        df_steady_state,
        datasets=[dataset_name],
    )

    fixed_points_dataframe_location = get_dataframe_location_for_dataset(
        fixed_points_dataframe_manifest, dataset_name
    )
    fixed_points_dataframe = load_dataframe(fixed_points_dataframe_location, delay=False)
    check_required_columns_in_dataframe(
        fixed_points_dataframe,
        required_columns=[
            *DYNAMICS_COLUMN_NAMES,
            ColumnName.DATASET,
            STABILITY_COLUMN_NAME,
        ],
    )

    # Enrich fixed points with binned mean of the optical flow
    # feature so downstream plots (histogram, 3D) can use it.
    df_flow_no_nan = df_of.dropna(subset=[optical_flow_feature])
    df_fp = add_binned_mean_to_fixed_points(
        fixed_points_dataframe,
        df_flow_no_nan,
        x_col=ColumnName.DiffAEData.POLAR_ANGLE,
        y_col=ColumnName.DiffAEData.POLAR_RADIUS,
        z_col=ColumnName.DiffAEData.PC3_FLIPPED,
        binned_col=optical_flow_feature,
    )

    # save individual histogram for this dataset and flow conditio
    fig = plot_optical_flow_histogram(
        df=df_of,
        optical_flow_feature=optical_flow_feature,
        feature_label="Migration Coherence",
        feature_lim=(0.1, vmax),
        ss_label=ss_label,
        color=hist_color,
        df_fp=None,
        binwidth=hist_binwidth,
        figure=(fig, ax),
        legend_loc=None,
    )
save_plot_to_path(
    fig,
    save_dir,
    "migration_coherence_distribution_high_low_flow_comparison",
    pad_inches=0,
    tight_layout=False,
    file_format=".svg",
)
# %%
