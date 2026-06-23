# %%

import matplotlib.pyplot as plt

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_optical_flow_features,
)
from endo_pipeline.library.process.image_processing import (
    load_processed_bf_std_dev_image_crop,
    load_processed_egfp_image_crop,
)
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.spatial_feature_grid import create_panel_spatial_feature_grid
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    METADATA_COLUMNS_TO_KEEP,
)
from endo_pipeline.settings.examples import FIGURE_3_EXAMPLE_IMAGES
from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

# %%
plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("figure_3")

# %%
feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[MIGRATION_COHERENCE_CROP_PATTERN]
feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)
columns_to_compute = [*METADATA_COLUMNS_TO_KEEP["grid"], *DYNAMICS_COLUMN_NAMES]

# %%
example_dfs = []
example_labels = []
example_gfp_images = []
example_bf_images = []

for i, example in enumerate(FIGURE_3_EXAMPLE_IMAGES):
    dataset_name = example.dataset_name
    dataset_config = load_dataset_config(dataset_name)

    # Load VE-cadherin MIP image
    gfp_mip = load_processed_egfp_image_crop(
        dataset_config,
        example.position,
        example.timepoint,
        example.crop_x_start,
        example.crop_y_start,
        crop_size=768,
    )
    example_gfp_images.append(gfp_mip)

    # Load BF std dev projection
    bf_std = load_processed_bf_std_dev_image_crop(
        dataset_config,
        example.position,
        example.timepoint,
        example.crop_x_start,
        example.crop_y_start,
        crop_size=768,
    )
    example_bf_images.append(bf_std)

    # Load selected columns from feature dataframe
    df_delay = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
    df_features = df_delay[columns_to_compute].compute()
    df_features = add_optical_flow_features(df_features, datasets=[dataset_name])
    df_example = df_features[df_features[f"{ColumnName.POSITION}"] == example.position]
    df_example = df_example[df_example[f"{ColumnName.TIMEPOINT}"] == example.timepoint]

    example_dfs.append(df_example)
    shear_stress = dataset_config.flow_conditions[0].shear_stress_bin
    example_labels.append(f"{shear_stress} dyn/cm²\nExample {i+1}")

# %%
feature_columns = [*DYNAMICS_COLUMN_NAMES, ColumnName.OpticalFlow.UNIT_VECTOR_MEAN]
fig = create_panel_spatial_feature_grid(
    example_dataframes=example_dfs,
    feature_columns=feature_columns,
    example_labels=example_labels,
    image_rows={
        "VE-cadherin\nMIP": example_gfp_images,
        "BF\nstd. dev. proj.": example_bf_images,
    },
    crop_size=256,
    start_x_col=ColumnName.DiffAEData.START_X,
    start_y_col=ColumnName.DiffAEData.START_Y,
    grid_start_xy=(128, 128),
    grid_dimensions=(3, 3),
    save_dir=save_dir,
    filename="spatial_feature_grid_examples",
    figure_size=(MAX_FIGURE_WIDTH, 5.5),
)

# %%
panels = [
    FigurePanel(
        letter="A",
        path=save_dir / "spatial_feature_grid_examples.svg",
        x_position=0,
        y_position=0,
        x_offset=0,
        y_offset=0,
    ),
]

build_figure_from_panels(
    panels, save_dir / "figure_3.svg", width=MAX_FIGURE_WIDTH, height=MAX_FIGURE_HEIGHT
)
# %%
