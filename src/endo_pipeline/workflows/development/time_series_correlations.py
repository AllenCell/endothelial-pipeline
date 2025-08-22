TAGS = ["diffae_features"]


def main(
    dataset_name: str = "3d_flow_field_analysis",
    model_name: str = "diffae_04_10",
) -> None:
    """
    Run auto and cross correlation analysis on DiffAE feature time series data.

    Parameters
    ----------
    dataset_name
        Name of the dataset or dataset collection to analyze.
    model_name
        Name of the DiffAE model to use for feature analysis.
    """
    import logging

    from endo_pipeline.configs import (
        get_available_dataset_collection_names,
        get_available_dataset_names,
        get_datasets_in_collection,
        load_dataset_config,
    )
    from endo_pipeline.library.analyze.diffae_manifest import fit_pca
    from endo_pipeline.library.analyze.numerics import compute_correlation_dict
    from endo_pipeline.library.visualize.diffae_features.correlations import (
        plot_correlation_workflow_outputs,
    )
    from endo_pipeline.manifests import load_dataframe_manifest

    # initialize logger
    logger = logging.getLogger(__name__)

    # check if input is a dataset collection or a single dataset name
    if dataset_name in get_available_dataset_collection_names():
        # if it is a dataset collection, load all datasets in the collection
        dataset_names = get_datasets_in_collection(dataset_name)
    elif dataset_name in get_available_dataset_names():
        # if it is a single dataset name, keep it as is
        dataset_names = [dataset_name]
    else:
        logger.error(
            "Dataset name [ %s ] is not a valid dataset or dataset collection name",
            dataset_name,
        )
        raise ValueError(
            f"Dataset name [ {dataset_name} ] is not a valid",
            "dataset or dataset collection name.",
        )

    # drop any no flow datasets from the list of datasets
    for dataset_name in dataset_names:
        dataset_config = load_dataset_config(dataset_name)
        flow_conditions = dataset_config.flow_conditions
        if int(flow_conditions[0].shear_stress) == 0:
            logger.warning(
                "Dataset [ %s ] has no flow conditions (shear stress = 0). "
                "Skipping this dataset for correlation analysis.",
                dataset_name,
            )
            dataset_names.remove(dataset_name)

    # load dataframe manifest = "{model_name}.yaml"
    dataframe_manifest = load_dataframe_manifest(model_name)

    # fit PCA object for the given model that generates the model manifests
    pca = fit_pca(model_name=model_name)

    # get cross and autocorrelation for pc features for each dataset
    # in the list of model manifests
    correlation_dict = compute_correlation_dict(
        dataset_names,
        dataframe_manifest,
        pca,
    )

    plot_correlation_workflow_outputs(correlation_dict)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
