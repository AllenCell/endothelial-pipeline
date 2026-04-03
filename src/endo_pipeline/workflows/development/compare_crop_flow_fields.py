# from endo_pipeline.cli import Datasets

# from endo_pipeline.settings.workflow_defaults import (
#     DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED,
#     DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME_FILTERED,
# )

# NOTE THIS WORKFLOW IS CURRENTLY NON-FUNCTIONAL UNTIL I MAINTAIN
# THE FUNCTION `process_dataset_for_track_integration`


def main(
    # datasets: Datasets | None = None,
    # merged_features_manifest_name: str = DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME_FILTERED,
    # diffae_grid_manifest_name: str | None = DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED,
) -> None:
    """
    Make plots comparing cell-centric and grid-based flow fields.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to use for visualization.
    merged_features_manifest_name
        Name of the merged features manifest containing the run to load features from.
    diffae_grid_manifest_name
        Name of the DiffAE PCA feature grid manifest to use for loading grid-based flow fields.
    """
    import matplotlib
    from matplotlib import pyplot as plt

    # from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.integration.track_integration import (  # process_dataset_for_track_integration,
        make_angular_deviation_test,
    )

    # from endo_pipeline.settings.workflow_defaults import DEFAULT_PCA_DATASET_COLLECTION_NAME
    # the below 2 lines are both used to control memory
    # usage problems when making many plots in a loop
    matplotlib.use("Agg")
    plt.ioff()  # turns off interactive mode in matplotlib

    # if datasets is not None:
    #     dataset_name_list = datasets.copy()
    # else:
    #     dataset_name_list = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    # for dataset_name in dataset_name_list:
    #     process_dataset_for_track_integration(
    #         dataset_name=dataset_name,
    #         merged_cellcentric_features_manifest_name=merged_features_manifest_name,
    #         diffae_grid_manifest_name=diffae_grid_manifest_name,
    #         make_integrated_plots=True,
    #     )

    # create a test flow field and test set of vectors
    # to check that the angular deviation calculation
    # works as expected
    out_dir = get_output_path(__file__, include_timestamp=False)
    make_angular_deviation_test(out_dir)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
