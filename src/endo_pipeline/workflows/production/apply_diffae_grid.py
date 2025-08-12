TAGS = ["apply_diffae_model", "diffae_features"]


def main(
    model_name: str = "diffae_04_10",
    dataset_name: str = "live_20X_objective_3i_microscope",
    resolution_level: int = 1,
    upload_to_fms: bool = True,
    user_overrides: str | dict | None = None,
    z_stack_offsets: tuple[int, int] | None = None,
    slice_by_global_center: bool = True,
    test_workflow: bool = False,
) -> None:
    """
    Apply a model to a multiple datasets.

    Example usage:
    ```
    endopipe -v -g apply-diffae-grid \
    --model-name diffae_04_10 \
    --dataset-name 20250409_20X \
    --z-stack-offsets 0 16 \
    --no-slice-by-global-center
    ```
    Apply a trained DiffAE model to grid-based crops of images from multiple datasets.

    Produces a table of latent features from a non-overlapping grid of crops
    for each dataset. The model is applied at the specified resolution level.

    ** Z-stack offsets **
    The `z_stack_offsets` parameter allows for flexible control over the z-slice loading.
    If `z_stack_offsets` is provided, it limits the number of z-slices to load, either
    by slicing about a global center or by using the provided offsets directly. If it
    is `None`, all z-slices are loaded from the raw brightfield images.


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
    test_workflow
        Flag to indicate if this script is being run for testing purposes (e.g., code review).

        If True, only one position and minimal timepoints from each dataset is included for
        loading and performing inferrence on the crops. This speeds up the dataloading process
        during model evaluation.

    Returns
    -------
    :
        Saves the model config with the applied model and model manifest objects.
        The model config is saved to [ endo_pipeline/configs/models/{model_name}.yaml ].
    """

    import logging
    from typing import cast

    from src.endo_pipeline.configs import (
        CytoDLModelConfig,
        get_available_dataset_collection_names,
        get_available_dataset_names,
        get_datasets_in_collection,
        load_dataset_config,
        load_model_config,
        save_model_config,
    )
    from src.endo_pipeline.library.model import apply_model_on_grid_of_crops_from_one_dataset

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
        model_config = apply_model_on_grid_of_crops_from_one_dataset(
            model_config=model_config,
            dataset_config=dataset_config,
            resolution_level=resolution_level,
            upload_to_fms=upload_to_fms,
            user_overrides=user_overrides,
            z_stack_offsets=z_stack_offsets,
            slice_by_global_center=slice_by_global_center,
            test_workflow=test_workflow,
        )
        if test_workflow:
            # if test workflow, only process the first dataset
            logger.warning(
                "Workflow testing is enabled, only processing the first dataset: [ %s ]",
                dataset_config.name,
            )
            break

    # save out updated model config
    save_model_config(model_config)


if __name__ == "__main__":
    import argparse

    from src.endo_pipeline.__main__ import workflow_cli

    parser = argparse.ArgumentParser(description="Run the DiffAE grid application workflow.")
    parser.add_argument(
        "--model-name", type=str, default="diffae_04_10", help="Name of the model to use."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="live_20X_objective_3i_microscope",
        help="Name of the dataset. Defaults to dataset collection.",
    )
    parser.add_argument("--resolution-level", type=int, default=1, help="Resolution level to use.")
    parser.add_argument(
        "--upload-to-fms", type=bool, default=True, help="Whether to upload results to FMS."
    )
    parser.add_argument(
        "--user-overrides", type=str, default=None, help="User overrides as a string or JSON."
    )
    parser.add_argument(
        "--z-stack-offsets",
        type=int,
        nargs=2,
        default=None,
        help="Z-stack offsets as two integers.",
    )
    parser.add_argument(
        "--no-slice-by-global-center",
        dest="slice_by_global_center",
        action="store_false",
        default=True,  # Default is True
        help="Disable slicing by global center (default: enabled).",
    )
    parser.add_argument(
        "--test-workflow", action="store_true", help="Run the workflow in test mode."
    )

    args = parser.parse_args()

    workflow_cli(
        lambda: main(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            resolution_level=args.resolution_level,
            upload_to_fms=args.upload_to_fms,
            user_overrides=args.user_overrides,
            z_stack_offsets=tuple(args.z_stack_offsets) if args.z_stack_offsets else None,
            slice_by_global_center=args.slice_by_global_center,
            test_workflow=args.test_workflow,
        )
    )
