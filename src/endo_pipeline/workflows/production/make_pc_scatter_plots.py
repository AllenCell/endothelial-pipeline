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
    model_manifest_name
        Name of the model manifest to get DiffAE features for.
    run_name
        Name of the model run within the model manifest. If None, uses the most recent run.
    include_cell_piling
        Whether to include cell piling timepoints in the visualization.

    """
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.diffae_features import feature_viz
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        DEFAULT_PCA_DATASET_COLLECTION_NAME,
    )

    # get dataframe manifest for grid crop-based features
    base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_grid"
    feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    fig_savedir = get_output_path(
        __file__, "including_not_steady_state" if include_not_steady_state else "steady_state_only"
    )

    # get list of dataset names to visualize
    dataset_names = datasets or get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    # scatter plot of pca reference datasets
    fig, _ = feature_viz.plot_pc_scatter(
        dataset_names,
        feature_dataframe_manifest,
        scatter_size=1,
        alpha=0.2,
        save_dir=fig_savedir,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
