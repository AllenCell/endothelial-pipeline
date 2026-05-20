from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    include_not_steady_state: Annotated[bool, Parameter(negative="--steady-state-only")] = True,
) -> None:
    """
    Make scatter plots of DiffAE PCA features for specified datasets.

    Parameters
    ----------
    datasets
        Datasets and / or dataset collection(s) to visualize.
    include_not_steady_state
        If true, include timepoints annotated as "not_steady_state" in the
        scatter plots. If false, only exclude these timepoints.

    """
    import logging

    import pandas as pd

    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_steady_state
    from endo_pipeline.library.visualize.diffae_features.feature_viz import plot_pc_scatter
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.diffae_feature_dataframes import (
        DIFFAE_PC_COLUMN_NAMES,
        NUM_PCS_TO_ANALYZE,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_PCA_DATASET_COLLECTION_NAME,
        GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME,
    )

    logger = logging.getLogger(__name__)

    # get dataframe manifest for grid crop-based features
    feature_dataframe_manifest_name = GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    fig_savedir = get_output_path(
        __file__, "including_not_steady_state" if include_not_steady_state else "steady_state_only"
    )

    # get list of dataset names to visualize
    dataset_names = datasets or get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    # initialize list to hold dataframes for each dataset, which will be
    # combined for plotting
    df_list = []

    column_names = DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE]
    columns_to_compute = [*column_names, Column.TIMEPOINT, Column.DATASET]

    # get combined dataframe for all datasets to plot, applying necessary
    # filtering and selecting only necessary columns to compute
    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                f"Dataset {dataset_name} not found in dataframe manifest {feature_dataframe_manifest_name}, skipping."
            )
            continue

        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_[columns_to_compute].compute()

        # if excluding the "not steady state" timepoints, do additional filtering:
        if not include_not_steady_state:
            dataset_config = load_dataset_config(dataset_name)
            df = filter_dataframe_to_steady_state(df, dataset_config)
        df_list.append(df)
    df_combined = pd.concat(df_list, ignore_index=True)

    # scatter plot of pca reference datasets
    plot_pc_scatter(
        df_combined,
        savedir=fig_savedir,
        column_names=column_names,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
