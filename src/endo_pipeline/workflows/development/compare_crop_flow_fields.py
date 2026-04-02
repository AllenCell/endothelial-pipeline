from endo_pipeline.cli import Datasets
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
)


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    seg_feature_manifest_name: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
) -> None:
    """
    Make plots comparing cell-centric and grid-based flow fields.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to use for visualization.
    model_manifest_name
        Name of the model manifest containing the run to load features from.
    run_name
        Name of the specific model run to load features for. If None, uses the most recent run.
    seg_feature_manifest_name
        Name of the segmentation feature manifest to use for segmentation features.
    """
    import matplotlib
    from matplotlib import pyplot as plt

    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.integration.track_integration import (
        make_angular_deviation_test,
        process_dataset_for_track_integration,
    )
    from endo_pipeline.settings.workflow_defaults import DEFAULT_PCA_DATASET_COLLECTION_NAME

    # the below 2 lines are both used to control memory
    # usage problems when making many plots in a loop
    matplotlib.use("Agg")
    plt.ioff()  # turns off interactive mode in matplotlib

    if datasets is not None:
        dataset_name_list = datasets.copy()
    else:
        dataset_name_list = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    for dataset_name in dataset_name_list:
        process_dataset_for_track_integration(
            dataset_name=dataset_name,
            make_integrated_plots=True,
        )

    # create a test flow field and test set of vectors
    # to check that the angular deviation calculation
    # works as expected
    out_dir = get_output_path(__file__, include_timestamp=False)
    make_angular_deviation_test(out_dir)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
