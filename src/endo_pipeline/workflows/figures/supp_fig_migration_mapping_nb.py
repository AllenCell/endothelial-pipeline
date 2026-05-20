# %%
import matplotlib.pyplot as plt

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_steady_state
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_optical_flow_features,
)
from endo_pipeline.library.visualize.data_example_figures import (
    create_panel_retraction_fiber_blob_example,
)
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_dataset_color
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.migration_coherence import plot_optical_flow_histogram
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import METADATA_COLUMNS_TO_KEEP
from endo_pipeline.settings.examples import EXAMPLE_DATASET, SUPP_FIG_RETRACTION_FIBER_BLOB
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

plt.style.use("endo_pipeline.figure")

# %%
base_output_dir = get_output_path("migration_coherence_blob")

# figure is for grid based crops
crop_pattern = "grid"

dataset_low = EXAMPLE_DATASET["FIGURE_2_LOW_FLOW_DATASET"]
dataset_high = EXAMPLE_DATASET["FIGURE_2_HIGH_FLOW_DATASET"]

feature_column_names = [
    Column.DiffAEData.POLAR_ANGLE,
    Column.DiffAEData.POLAR_RADIUS,
    Column.DiffAEData.PC3_FLIPPED,
]
optical_flow_feature = Column.OpticalFlow.UNIT_VECTOR_MEAN
dataframe_columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[crop_pattern], *feature_column_names]

# load dataframe manifests for diffae features, fixed points, optical flow
# features, and bootstrapped fixed points for this crop pattern, which will be
# used for all visualizations in this figure
base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}"
feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

# %%
fig, ax = plt.subplots(figsize=(2, 2), layout="constrained")
for dataset_name in [dataset_low, dataset_high]:
    # get settings
    dataset_config = load_dataset_config(dataset_name)
    shear_stress = dataset_config.flow_conditions[-1].shear_stress_bin

    # load and filter data
    df = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
    df_ = df[dataframe_columns_to_compute].compute()
    df_steady_state = filter_dataframe_to_steady_state(df_, dataset_config)

    df_of = add_optical_flow_features(
        df_steady_state,
        datasets=[dataset_name],
    )

    fig = plot_optical_flow_histogram(
        df=df_of,
        optical_flow_feature=optical_flow_feature,
        feature_label=COLUMN_METADATA[optical_flow_feature].label,
        feature_lim=(0, 1),
        ss_label=f"{shear_stress} dyn/cm{Unicode.SQUARED}",
        color=get_dataset_color(dataset_name),
        df_fp=None,
        binwidth=0.02,
        figure=(fig, ax),
        legend_loc=None,
    )

save_plot_to_path(
    fig,
    base_output_dir,
    "migration_coherence_distribution_high_low_flow_comparison",
    pad_inches=0,
    tight_layout=False,
    file_format=".svg",
)

# %%
t = SUPP_FIG_RETRACTION_FIBER_BLOB.timepoint
create_panel_retraction_fiber_blob_example(
    example=SUPP_FIG_RETRACTION_FIBER_BLOB,
    timepoints=list(range(t, t + 15, 3)),
    save_dir=base_output_dir,
    figure_size=(MAX_FIGURE_WIDTH, 4),
)

# %% --- Assemble all panels into final figure ---
panels = [
    FigurePanel(
        letter="A",
        path=base_output_dir / "migration_coherence_distribution_high_low_flow_comparison.svg",
        x_position=0,
        y_position=0.0,
        x_offset=-0.2,
        y_offset=0,
    ),
    FigurePanel(
        letter="B",
        path=base_output_dir / "retraction_fiber_blob_example.svg",
        x_position=0,
        y_position=2,
        x_offset=0,
        y_offset=0.2,
    ),
]


build_figure_from_panels(
    panels,
    base_output_dir / "supp_fig_migration_coherence_blob.svg",
    width=MAX_FIGURE_WIDTH,
    height=6.2,
)

# %%
