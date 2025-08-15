TAGS = ["diffae_features"]


def main(dataset_name: str = "3d_flow_field_analysis", model_name="diffae_04_10") -> None:
    """Run correlation analysis on DiffAE feature time series data."""
    import logging
    from typing import cast

    from src.endo_pipeline.configs import (
        CytoDLModelConfig,
        get_available_dataset_collection_names,
        get_available_dataset_names,
        get_datasets_in_collection,
        get_model_manifest,
        load_model_config,
    )
    from src.endo_pipeline.library.analyze.diffae_manifest import fit_pca
    from src.endo_pipeline.library.analyze.numerics import compute_correlation_dict
    from src.endo_pipeline.library.visualize.diffae_features.correlations import (
        plot_correlation_workflow_outputs,
    )

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

    # load model config to get model manifest objects
    model_config = cast(CytoDLModelConfig, load_model_config(model_name))
    model_manifest_list = []
    # get model manifests for each dataset in the list of datasets
    # as long as the dataset exists in the model config
    for dataset_name in dataset_names:
        try:
            model_manifest_list.append(
                get_model_manifest(dataset_name, model_config, model_name=model_name)
            )
        except FileNotFoundError:
            logger.warning(
                "No manifest found for dataset [ %s ] in model config [ %s ].",
                dataset_name,
                model_config.name,
            )
            continue

    # fit PCA object for the given model that generates the model manifests
    pca = fit_pca(model_name=model_name)

    # get cross and autocorrelation for pc features for each dataset
    # in the list of model manifests
    correlation_dict = compute_correlation_dict(
        list_of_model_manifests=model_manifest_list,
        pca=pca,
    )

    plot_correlation_workflow_outputs(correlation_dict)


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
