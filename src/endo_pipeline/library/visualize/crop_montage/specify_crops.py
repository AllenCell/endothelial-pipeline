import logging
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.library.analyze.numerics import (
    get_3d_bounds_from_data,
    get_df_by_bin_value,
    get_histogram_by_component,
)
from endo_pipeline.library.visualize.diffae_features.feature_viz import (
    plot_principal_component_histogram,
)
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings import (
    DIFFAE_PC_COLUMN_NAMES,
    NUM_BINS_CROP_HIST,
    NUM_PCS_TO_ANALYZE,
    ColumnName,
)

logger = logging.getLogger(__name__)


def load_data_for_montage(
    dataset_name_list: list[str],
    dataframe_manifest: DataframeManifest,
    include_cell_piling: bool = True,
) -> tuple[pd.DataFrame, PCA]:
    """
    Load Diff AE feature DataFrames for one or more datasets and optionally apply PCA.

    Parameters
    ----------
    dataset_name_list
        List of dataset names to load for montage.
    dataframe_manifest
        Dataframe manifest corresponding to features to load.
    include_cell_piling
        True to include cell piling timepoints, False to exclude them.

    Returns
    -------
    :
        Concatenated feature DataFrame for the specified datasets.
    :
        Fit PCA object for the model.
    """

    pca = fit_pca(
        dataframe_manifest_name=dataframe_manifest.name,
        include_cell_piling=include_cell_piling,
    )

    df_all = pd.concat(
        [
            get_dataframe_for_dynamics_workflows(
                name,
                dataframe_manifest,
                pca,
                include_cell_piling=include_cell_piling,
                include_not_steady_state=True,
            )
            for name in dataset_name_list
        ],
        ignore_index=True,
    )

    return df_all, pca


def filter_dataframe(
    df_all: pd.DataFrame,
    pc_axis: int,
    pc_val: float,
    dataframe_manifest: DataframeManifest,
    dataset_names: list[str],
    pca: PCA,
    fig_savedir: Path,
    frame_range: list[int] | None = None,
    plot_heatmap: bool = False,
) -> pd.DataFrame:
    """
    Filter a DataFrame by a principal component (PC) bin and optional frame range.

    Parameters
    ----------
    df_all
        Combined manifest DataFrame with PCA features.
    pc_axis
        Index of the principal component (0-indexed) to filter by.
    pc_val
        Target bin value (between 0 and 1) of the selected PC to sample crops from.
    dataframe_manifest
        DataframeManifest for model features.
    dataset_names
        List of datasets to be processed.
    pca
        PCA object used to transform the original features.
    fig_savedir
        Directory where the PC histograms will be saved.
    frame_range
        Optional timepoint range [start, end] to further filter the DataFrame.
    plot_heatmap
        If True, generates and saves PC histograms for each dataset.

    Returns
    -------
    :
        DataFrame filtered by the specified PC bin and optional frame range.
    """

    bin_limits = get_3d_bounds_from_data(
        dataset_names, dataframe_manifest, pca, filter_to_valid=False
    )
    hist_array_list, bin_edges, df_with_bins = get_histogram_by_component(
        df_all,
        NUM_BINS_CROP_HIST,
        bin_limits,
        feat_cols=DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE],
    )

    if plot_heatmap:
        for i, dataset_name in enumerate(dataset_names):
            fig, _ = plot_principal_component_histogram(hist_array_list[i], bin_edges)
            fig.suptitle(f"Dataset: {dataset_name}", y=0.95, fontsize=25)
            save_plot_to_path(fig, fig_savedir, f"{dataset_name}_pc_histogram")

    df_filtered = get_df_by_bin_value(df_with_bins, pc_axis, pc_val, bin_edges)

    if frame_range is not None:
        df_filtered = df_filtered[
            (df_filtered[ColumnName.TIMEPOINT] >= frame_range[0])
            & (df_filtered[ColumnName.TIMEPOINT] <= frame_range[1])
        ]

    return df_filtered


def sample_dataframe(
    df_filtered: pd.DataFrame,
    n_num_crops: int = 100,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Randomly sample a subset of rows from a filtered DataFrame.

    Parameters
    ----------
    df_filtered
        DataFrame already filtered by PC value (and optionally, frame range).
    n_num_crops
        Number of samples (rows) to return.
    random_seed
        Seed for reproducibility in random sampling.

    Returns
    -------
    :
        Random sample of the filtered DataFrame.
    """
    return df_filtered.sample(n=n_num_crops, random_state=random_seed, replace=False)
