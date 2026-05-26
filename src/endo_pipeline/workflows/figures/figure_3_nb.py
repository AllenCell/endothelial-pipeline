# %%
import logging

import matplotlib.pyplot as plt
import pandas as pd

from endo_pipeline.cli import NUM_GPUS
from endo_pipeline.io import get_output_path, load_dataframe, load_model
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_stability
from endo_pipeline.library.visualize.data_example_figures import create_panel_intermediate_examples
from endo_pipeline.library.visualize.figure_3 import (
    generate_synthetic_images_at_stable_fixed_points,
    make_crop_example_contact_sheet,
)
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.summary_plot import (
    build_dataframe_for_fixed_point_dataset_summary,
    plot_cross_dataset_summaries,
)
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.examples import (
    FIGURE_3_EXAMPLE_IMAGES,
    FIGURE_3_RECONSTRUCTION_EXAMPLE_DATASETS,
)
from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
from endo_pipeline.settings.flow_field_dataframes import (
    BOOTSTRAPPING_MANIFEST_NAMES,
    StabilityLabel,
)
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    FEATURES_FILTERED_MANIFEST_NAMES,
)

logger = logging.getLogger(__name__)

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("figure_3")

# %% Example images of intermediate shear stress condition
create_panel_intermediate_examples(
    examples=FIGURE_3_EXAMPLE_IMAGES,
    save_dir=save_dir,
    figure_size=(MAX_FIGURE_WIDTH * 0.65, 2.2),
)

# %% Load diffae features
feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[MIGRATION_COHERENCE_CROP_PATTERN]
feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

fixed_points_bootstrap_dataframe_manifest_name = BOOTSTRAPPING_MANIFEST_NAMES[
    MIGRATION_COHERENCE_CROP_PATTERN
]
fixed_points_bootstrap_dataframe_manifest = load_dataframe_manifest(
    fixed_points_bootstrap_dataframe_manifest_name
)

dataset_summary_list = SUMMARY_PLOT_DATASETS["intermediate"]

BOOTSTRAP_THRESHOLD = 0.4

# %% Cross-dataset summary plots
columns_for_summary_plots = [
    ColumnName.DiffAEData.POLAR_ANGLE,
    ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
    ColumnName.DiffAEData.POLAR_RADIUS,
    ColumnName.DiffAEData.PC3_FLIPPED,
]
dataset_summary_df = build_dataframe_for_fixed_point_dataset_summary(
    dataset_names=dataset_summary_list,
    feature_dataframe_manifest=feature_dataframe_manifest,
    bootstrap_dataframe_manifest=fixed_points_bootstrap_dataframe_manifest,
    column_names=columns_for_summary_plots,
    convert_angle_to_nematic=True,
    stable_only=True,
    bootstrap_threshold=BOOTSTRAP_THRESHOLD,
)
summary_plot_path = plot_cross_dataset_summaries(
    dataset_summary_df,
    output_dir=save_dir,
    column_names=columns_for_summary_plots,
    axis_mode="shear_stress",
    figure_size=(MAX_FIGURE_WIDTH * 0.6, 1.4),
    jitter_width=0.2,
    subplot_layout="vertical",
)

# %% Reconstruction of example images from stable fixed point coordinates

df_reconstruction_examples = pd.DataFrame()
for dataset_name in FIGURE_3_RECONSTRUCTION_EXAMPLE_DATASETS:
    if dataset_name not in feature_dataframe_manifest.locations:
        logger.warning(
            "No location found in dataframe manifest [ %s ] for dataset [ %s ], skipping visualization.",
            feature_dataframe_manifest_name,
            dataset_name,
        )
        continue

    fp_bootstrap_location = get_dataframe_location_for_dataset(
        fixed_points_bootstrap_dataframe_manifest, dataset_name
    )
    df_bootstrap = load_dataframe(fp_bootstrap_location, delay=False)

    n_total = len(df_bootstrap)
    high_confidence_df = df_bootstrap[
        df_bootstrap[ColumnName.BootstrapAnalysis.DETECTION_RATE] >= BOOTSTRAP_THRESHOLD
    ].copy()
    df_stable_fixed_points = filter_dataframe_by_stability(
        high_confidence_df, stability_label=StabilityLabel.STABLE
    )
    df_reconstruction_examples = pd.concat(
        [df_reconstruction_examples, df_stable_fixed_points], ignore_index=True
    )

# %% load and instantiate model for generating synthetic images
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
model_location = model_manifest.locations[DEFAULT_MODEL_RUN_NAME]
model = load_model(model_location, instantiate=True)

generated_image_list = generate_synthetic_images_at_stable_fixed_points(
    stable_fixed_point_dataframe=df_reconstruction_examples,
    feature_column_names=[
        ColumnName.DiffAEData.POLAR_ANGLE,
        ColumnName.DiffAEData.POLAR_RADIUS,
        ColumnName.DiffAEData.PC3_FLIPPED,
    ],
    model=model,
    num_gpus=NUM_GPUS,
    random_seed=4,
)

make_crop_example_contact_sheet(
    stable_fixed_point_dataframe=df_reconstruction_examples,
    generated_image_list=generated_image_list,
    fig_savedir=save_dir,
    fig_filename="reconstructed_fp_crop_examples.svg",
    file_format=".svg",
    gridspec_kwargs={"wspace": 0.01, "hspace": 0.01},
    fig_kwargs={"figsize": (MAX_FIGURE_WIDTH * 0.35, 4.5), "layout": "constrained"},
)

# %%
panels = [
    FigurePanel(
        letter="A",
        path=save_dir / "intermediate_examples_scale_bar_100um.svg",
        x_position=0,
        y_position=0,
        x_offset=0.2,
        y_offset=0,
    ),
    FigurePanel(
        letter="B",
        path=summary_plot_path,
        x_position=0,
        y_position=2.3,
        x_offset=0,
        y_offset=0.1,
    ),
    FigurePanel(
        letter="C",
        path=save_dir / "reconstructed_fp_crop_examples.svg",
        x_position=MAX_FIGURE_WIDTH * 0.6,
        y_position=2.3,
        x_offset=0.1,
        y_offset=0.1,
    ),
]

build_figure_from_panels(
    panels, save_dir / "figure_3.svg", width=MAX_FIGURE_WIDTH, height=MAX_FIGURE_HEIGHT
)
# %%
