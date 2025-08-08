import logging
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from cellsmap.util import manifest_io
from src.endo_pipeline.configs import (
    ModelManifest,
    get_available_dataset_collection_names,
    get_available_dataset_names,
    get_datasets_in_collection,
    get_model_manifest,
    get_timelapse_model_manifests,
    load_model_config,
)
from src.endo_pipeline.library.analyze.diffae_manifest import (
    fit_pca,
    get_manifest_for_dynamics_workflows,
)
from src.endo_pipeline.library.analyze.numerics import (
    get_3d_bounds_from_data,
    get_df_by_bin_value,
    get_histogram_by_component,
)
from src.endo_pipeline.library.visualize import viz_base
from src.endo_pipeline.library.visualize.diffae_features.feature_viz import (
    plot_principal_component_histogram,
)

logger = logging.getLogger(__name__)

N_BINS = 40  # number of bins for histogram, hardcoded right now but somewhat arbitrary


def load_data(
    dataset_name: str = "live_20X_objective_3i_microscope", model_name: str = "diffae_04_10"
) -> tuple[pd.DataFrame, Pipeline, list[ModelManifest]]:
    """
    Load manifest DataFrames for one or more datasets and optionally apply PCA.

    Parameters
    ----------
    dataset_name
        Name of the dataset(s) to include, either a single dataset name or
        the name of a dataset collection.
    model_name
        Name of the model for which to load the feature manifest data.

    Returns
    -------
    :
        Concatenated manifest DataFrame for the specified datasets.
    :
        Fit PCA object for the model.
    :
        List of model manifests for the specified datasets.
    """
    model_config = load_model_config(model_name)

    # check if input is a dataset collection or a single dataset name
    if dataset_name is None:
        model_manifest_list = get_timelapse_model_manifests(model_config)
    elif dataset_name in get_available_dataset_collection_names():
        # if it is a dataset collection, load all datasets in the collection
        # and get their model manifests
        model_manifest_list = [
            get_model_manifest(
                dataset_name,
                model_config,
            )
            for dataset_name in get_datasets_in_collection(dataset_name)
        ]
    elif dataset_name in get_available_dataset_names():
        # if it is a single dataset name, load its model manifest
        # as a list with one element
        model_manifest_list = [
            get_model_manifest(
                dataset_name,
                model_config,
            )
        ]
    else:
        logger.error(
            "Dataset name [ %s ] is not a valid dataset or dataset collection name",
            dataset_name,
        )
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    pca = fit_pca(model_name=model_name)

    df_all = pd.concat(
        [
            get_manifest_for_dynamics_workflows(model_manifest, pca=pca, filter_to_valid=False)
            for model_manifest in model_manifest_list
        ],
        ignore_index=True,
    )

    return df_all, pca, model_manifest_list


def filter_dataframe(
    df_all: pd.DataFrame,
    pc_axis: int,
    pc_val: float,
    model_manifest_list: list[ModelManifest],
    pca: Pipeline,
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
    model_manifest_list
        List of model manifests for the datasets being processed.
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
    bin_limits = get_3d_bounds_from_data(model_manifest_list, pca, filter_to_valid=False)
    hist_array_list, bin_edges, df_with_bins = get_histogram_by_component(
        df_all,
        N_BINS,
        bin_limits,
        feat_cols=manifest_io.get_feature_cols(df_all)[:3],
    )

    if plot_heatmap:
        for i, model_manifest in enumerate(model_manifest_list):
            dataset_name = model_manifest.dataset_name
            fig, _ = plot_principal_component_histogram(hist_array_list[i], bin_edges)
            fig.suptitle(f"Dataset: {dataset_name}", y=0.95, fontsize=25)
            viz_base.save_plot(fig, fig_savedir + f"{dataset_name}_pc_histogram")

    df_filtered = get_df_by_bin_value(df_with_bins, pc_axis, pc_val, bin_edges)

    if frame_range is not None:
        df_filtered = df_filtered[
            (df_filtered["frame_number"] >= frame_range[0])
            & (df_filtered["frame_number"] <= frame_range[1])
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
