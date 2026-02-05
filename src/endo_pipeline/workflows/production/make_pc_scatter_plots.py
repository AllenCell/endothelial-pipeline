from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.cli import Datasets
from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    include_cell_piling: Annotated[bool, Parameter(negative="--exclude-cell-piling")] = False,
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
    from endo_pipeline.library.analyze.diffae_dataframe_utils import fit_pca
    from endo_pipeline.library.visualize.diffae_features import feature_viz
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        get_most_recent_run_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import NUM_PCS_TO_ANALYZE
    from endo_pipeline.settings.workflow_defaults import DEFAULT_PCA_DATASET_COLLECTION_NAME

    # get model and dataframe manifests
    model_manifest = load_model_manifest(model_manifest_name)
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name_, crop_pattern="grid"
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    # set up output directory for figures
    include_cell_piling_str = "with_cell_piling" if include_cell_piling else "no_cell_piling"
    include_not_steady_state_str = (
        "including_not_steady_state" if include_not_steady_state else "steady_state_only"
    )
    fig_savedir = get_output_path(
        __file__,
        f"{include_cell_piling_str}_{include_not_steady_state_str}",
    )

    # fit PCA model (using method defaults)
    pca = fit_pca(
        dataset_collection_name=DEFAULT_PCA_DATASET_COLLECTION_NAME,
        dataframe_manifest_name=dataframe_manifest_name,
        num_pcs=NUM_PCS_TO_ANALYZE,
    )

    # get list of dataset names to visualize
    if datasets is None:
        dataset_names = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)
    else:
        dataset_names = datasets.copy()

    # scatter plot of pca reference datasets
    fig, _ = feature_viz.plot_pc_scatter(
        dataset_names,
        dataframe_manifest,
        pca,
        include_cell_piling=include_cell_piling,
        scatter_size=1,
        alpha=0.2,
        save_dir=fig_savedir,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
