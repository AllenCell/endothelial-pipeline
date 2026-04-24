# %%
import matplotlib.pyplot as plt

from endo_pipeline.cli.demo_mode_defaults import use_default_collection
from endo_pipeline.io import get_output_path, save_plot_to_path
from endo_pipeline.library.analyze.pca import fit_pca
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.library.visualize.diffae_features import feature_viz
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.multi_feature_correlation_viz import (
    get_df_for_feature_correlation_viz,
    visualize_correlation_heatmaps,
)
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_PC_COLUMN_NAME_GROUPS,
    NUM_LATENT_FEATURES,
)
from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
from endo_pipeline.settings.workflow_defaults import (
    DATASET_INFO_COLUMNS,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    SEGMENTATION_FEATURE_COLUMNS,
)

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("supp_fig_features")

# %% plot cumulative explained variance ratio of PCA components
pca = fit_pca(num_pcs=NUM_LATENT_FEATURES)

# %%
fig, _ = feature_viz.plot_explained_variance(pca.explained_variance_ratio_, figsize=(2.1, 2.5))
save_plot_to_path(fig, save_dir, "explained_variance_ratio", file_format=".svg", pad_inches=0)

# %% Correlation heatmap of ML-based features vs. measured features
dataset_name_list = use_default_collection(None, DEFAULT_PCA_DATASET_COLLECTION_NAME)
ml_columns = DIFFAE_PC_COLUMN_NAME_GROUPS["supp_figure"]
measured_feature_columns = SEGMENTATION_FEATURE_COLUMNS["supp_figure"]

# Long operation: takes several minutes
df = get_df_for_feature_correlation_viz(
    dataset_name_list=dataset_name_list,
    dataset_info_columns=DATASET_INFO_COLUMNS,
    segmentation_feature_columns=measured_feature_columns,
    pc_columns=ml_columns,
)

label_column_tuples = [
    ("ML-based Features", [get_label_for_column(col) for col in ml_columns]),
    ("Measured Features", [get_label_for_column(col) for col in measured_feature_columns]),
]

# %%
visualize_correlation_heatmaps(
    dataset_name="aggregate",
    df_dataset=df,
    label_column_tuples=label_column_tuples,
    out_dir=save_dir,
    cross_correlation_only=True,
    figsize_cluster_heatmap=(MAX_FIGURE_WIDTH - 1.6, 2.75),
    y_axis_label_coords=None,
)


# %%
panels = [
    FigurePanel(
        letter="A",
        path=save_dir / "explained_variance_ratio.svg",
        x_position=0,
        y_position=0,
        x_offset=-0.1,
        y_offset=0,
    ),
    FigurePanel(
        letter="B",
        path=save_dir / "correlation_ml-based_features_vs_measured_features_heatmap.svg",
        x_position=2,
        y_position=0,
        x_offset=-0.1,
        y_offset=-0.1,
    ),
]
build_figure_from_panels(
    panels, save_dir / "supp_fig_features.svg", width=MAX_FIGURE_WIDTH, height=3
)
# %%
