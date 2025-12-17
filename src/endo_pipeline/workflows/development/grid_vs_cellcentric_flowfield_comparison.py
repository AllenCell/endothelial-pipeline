from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
)


def main(
    dataset_collection_name: str = "pca_reference",
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    seg_feature_manifest_name: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
) -> None:
    """
    Makes plots comparing cell-centric and grid-based flow fields.
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

    dataset_name_list = get_datasets_in_collection(dataset_collection_name)

    for dataset_name in dataset_name_list:
        process_dataset_for_track_integration(
            dataset_name=dataset_name,
            model_manifest_name=model_manifest_name,
            run_name=run_name,
            seg_feature_manifest_name=seg_feature_manifest_name,
            collection_name_for_pca=DEFAULT_PCA_DATASET_COLLECTION_NAME,
            make_integrated_plots=True,
        )

    # create a test flow field and test set of vectors
    # to check that the angular deviation calculation
    # works as expected
    out_dir = get_output_path(__file__, include_timestamp=False)
    make_angular_deviation_test(out_dir)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
