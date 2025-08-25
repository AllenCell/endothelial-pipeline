TAGS = ["apply_diffae_model", "diffae_features"]


def main(
    model_name: str = "diffae_04_10",
    dataset_name: str = "live_20X_objective_3i_microscope",
    resolution_level: int = 1,
    upload_to_fms: bool = True,
    user_overrides: str | dict | None = None,
    z_stack_offsets: tuple[int, int] | None = None,
    slice_by_global_center: bool = True,
) -> None:
    """
    Apply a trained DiffAE model to grid-based crops of images from multiple datasets.

    Produces a table of latent features from a non-overlapping grid of crops for each dataset.
    The model is applied at the specified resolution level.

    **Workflow testing**

    If testing mode is enabled, the model will only be evaluated on the first few timepoints
    of the first position of the first dataset. Furthermore, the staging environment
    for FMS will be used for file uploads.

    **Z-stack offsets**

    The ``z_stack_offsets`` parameter allows for flexible control over the z-slice loading.
    If ``z_stack_offsets`` is provided, it limits the number of z-slices to load, either
    by slicing about a global center or by using the provided offsets directly. If it
    is ``None``, all z-slices are loaded from the raw brightfield images.

    **Example usage**

    .. code-block:: bash

        endopipe -vg apply-diffae-grid --dataset-name 20250409_20X --z-stack-offsets 0 16 --no-slice-by-global-center


    Parameters
    ----------
    model_name
        Name of the model to apply.
    dataset_name
        Dataset(s) to load images from, either a single dataset name or the name
        of a dataset collection.
    resolution_level
        Resolution level to at which to load images (zarr file format) at.
    upload_to_fms
        True to upload the prediction file for each dataset to FMS, False to only save locally.
    user_overrides
        Optional user overrides to apply to the model config.
    z_stack_offsets
        Lower and upper bounds for z-slicing.
    slice_by_global_center
        Slice about a global center if True, or use z_stack_offsets directly if False.

    Returns
    -------
    :
        Saves the model config with the applied model and model manifest objects.
        The model config is saved to [ endo_pipeline/configs/models/{model_name}.yaml ].
    """  # noqa: E501

    import logging
    from typing import cast

    from endo_pipeline import TESTING_MODE
    from endo_pipeline.configs import (
        CytoDLModelConfig,
        get_available_dataset_collection_names,
        get_available_dataset_names,
        get_datasets_in_collection,
        load_dataset_config,
        load_model_config,
    )
    from endo_pipeline.library.model import apply_model_on_grid_of_crops_from_one_dataset

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

    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_names]

    # load model config
    model_config = cast(CytoDLModelConfig, load_model_config(model_name))

    # apply model to each dataset
    # is there a better way to do this? i.e., load model once
    # and then just loop through datasets...
    # out of scope for this PR but worth doing in a separate PR
    for dataset_config in dataset_config_list:
        apply_model_on_grid_of_crops_from_one_dataset(
            model_config=model_config,
            dataset_config=dataset_config,
            resolution_level=resolution_level,
            upload_to_fms=upload_to_fms,
            user_overrides=user_overrides,
            z_stack_offsets=z_stack_offsets,
            slice_by_global_center=slice_by_global_center,
            testing_mode=TESTING_MODE,
        )
        if TESTING_MODE:
            # if test workflow, only process the first dataset
            logger.debug(
                "Workflow testing is enabled, only processing the first dataset: [ %s ]",
                dataset_config.name,
            )
            break


if __name__ == "__main__":

    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
