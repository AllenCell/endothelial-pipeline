# %%
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d

from endo_pipeline.cli import DEMO_MODE
from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
from endo_pipeline.io import get_output_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.library.analyze.migration_pc.cca_analysis import (
    plot_optical_flow_feature_distribution,
)
from endo_pipeline.library.analyze.migration_pc.optical_flow_feature import (
    add_optical_flow_features,
)
from endo_pipeline.manifests import (
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

logger = logging.getLogger(__name__)

DESCRIPTION = "Migration coherence overlayed on phase portrait"

OPTICAL_FLOW_FEATURE = "optical_flow_mean_unit_vector_dt1"

datasets = get_datasets_in_collection("diffae_model_training") + get_datasets_in_collection(
    "replicate_2_datasets"
)
if DEMO_MODE:
    datasets = datasets[:2]

output_dir = get_output_path("migration_coherence")

# %% Load diffae features
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
pca = fit_pca(num_pcs=80)

# %% Load optical flow features
df_pca_datasets = []
for dataset_name in datasets:
    df_dataset = get_dataframe_for_dynamics_workflows(dataset_name, dataframe_manifest, pca=pca)
    df_pca_datasets.append(df_dataset)

df_pca_all = pd.concat(df_pca_datasets, ignore_index=True)
# %%
df_of = add_optical_flow_features(
    df_pca_all,
    datasets=datasets,
    optical_flow_manifest_name="optical_flow_bf",
)
# %%
for dataset_name in datasets:
    plot_optical_flow_feature_distribution(
        df=df_of,
        optical_flow_feature=OPTICAL_FLOW_FEATURE,
        datasets=[dataset_name],
        binwidth=0.02,
        bins=50,
        kde=True,
    )


# %%
def plot_scatter_and_binned_heatmap(
    df: pd.DataFrame,
    dataset_name: str,
    x_col: str,
    y_col: str,
    color_col: str,
    x_bin_size: float = 0.25,
    y_bin_size: float = 0.25,
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    """Plot scatter (left) and binned mean heatmap (right) side by side."""
    cmap = plt.get_cmap("cool")
    df_plot = df[(df["dataset"] == dataset_name) & df[color_col].notna()]
    x = df_plot[x_col].values
    y = df_plot[y_col].values
    z = df_plot[color_col].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Left: scatter plot
    ax1.scatter(x, y, c=z, cmap=cmap, s=5, vmin=vmin, vmax=vmax)
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)

    # Right: binned heatmap
    x_bins = np.arange(x.min(), x.max() + x_bin_size, x_bin_size)
    y_bins = np.arange(y.min(), y.max() + y_bin_size, y_bin_size)
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        x,
        y,
        z,
        statistic="mean",
        bins=[x_bins, y_bins],
    )
    im = ax2.pcolormesh(
        x_edges,
        y_edges,
        stat.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    fig.colorbar(im, ax=ax2, label=color_col)
    ax2.set_xlabel(x_col)
    ax2.set_ylabel(y_col)

    dataset_config = load_dataset_config(dataset_name)
    flow_conditions = dataset_config.flow_conditions
    shear_stress_values = [fc.shear_stress for fc in flow_conditions]
    shear_stress_label = "-".join(f"{v:g}" for v in shear_stress_values)
    title = f"{dataset_name}, {shear_stress_label} dyn/cm^2"

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close()


# %%
for dataset_name in datasets:
    print(dataset_name)
    plot_scatter_and_binned_heatmap(
        df=df_of,
        dataset_name=dataset_name,
        x_col="polar_r",
        y_col="rho",
        color_col=OPTICAL_FLOW_FEATURE,
        x_bin_size=0.25,
        y_bin_size=0.25,
    )
    plot_scatter_and_binned_heatmap(
        df=df_of,
        dataset_name=dataset_name,
        x_col="polar_r",
        y_col="polar_theta",
        color_col=OPTICAL_FLOW_FEATURE,
        x_bin_size=0.25,
        y_bin_size=0.25,
    )
    plot_scatter_and_binned_heatmap(
        df=df_of,
        dataset_name=dataset_name,
        x_col="rho",
        y_col="polar_theta",
        color_col=OPTICAL_FLOW_FEATURE,
        x_bin_size=0.25,
        y_bin_size=0.25,
    )
