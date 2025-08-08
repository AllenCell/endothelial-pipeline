import logging
from collections.abc import Sequence

import fire

from src.endo_pipeline.configs import (
    get_available_dataset_collection_names,
    get_available_dataset_names,
    get_datasets_in_collection,
    load_dataset_config,
    load_model_config,
    save_model_config,
)
from src.endo_pipeline.library.model import apply_model_on_grid_of_crops_from_one_dataset

logger = logging.getLogger(__name__)


def main(
    model_name: str = "diffae_04_10",
    dataset_names: str | Sequence[str] = "20241016_20X",
    resolution_level: int = 1,
    upload_to_fms: bool = True,
    user_overrides: str | dict | None = None,
    z_stack_offsets: tuple[int, int] | None = None,
    slice_by_global_center: bool = True,
) -> None:
    """
    Apply a model to a multiple datasets.

    Example usage:
    ```
    python src/endo_pipeline/workflows/production/apply_diffae_grid.py \
    --model_name diffae_04_10 \
    --dataset_names '20250409_20X' \
    --z_stack_offsets 0,16 \
    --slice_by_global_center False
    ```

    Parameters
    ----------
    model_name: str
        Name of the model from `model_config.yaml` to apply.
    dataset_names: str | Sequence[str]
        Names of the datasets from `data_config.yaml` to apply the model to.
        If it is a string, it should either be a single dataset name or the name of a
        dataset collection.
    resolution_level: int
        Resolution level to apply the model at. Default is 1 (zarr sample resolution).
    upload_to_fms: bool
        Whether to upload the prediction file to FMS. Default is True.
    save_path: str | Path | None
        Path to save the prediction file. Default is `models/{model_name}/{dataset_name}`.
    user_overrides: str or dict or None
        Additional overrides to apply to the model config. By default, no overrides are applied.
    z_stack_offsets: tuple[int, int] | None
        If None, all z-slices are loaded. Default is None.
        If provided, limits the number of z-slices to load from the raw brightfield images.
        First element is the lower offset, how many slices below the center plane to include, and
        the second element is the upper offset, how many slices above the center plane to include.
    slice_by_global_center: bool
        If true, slice about a global center
        If false, use z_stack_offsets as the upper and lower bounds for z slicing
    """
    # if input is a string, check if it is a dataset collection or a single dataset name
    if isinstance(dataset_names, str):
        if dataset_names in get_available_dataset_collection_names():
            # if it is a dataset collection, load all datasets in the collection
            dataset_names = get_datasets_in_collection(dataset_names)
        elif dataset_names in get_available_dataset_names():
            # if it is a single dataset name, keep it as is
            dataset_names = [dataset_names]
        else:
            logger.error(
                "Dataset name [ %s ] is not a valid dataset or dataset collection name",
                dataset_names,
            )
            raise ValueError(
                f"Dataset name [ {dataset_names} ] is not a valid",
                "dataset or dataset collection name.",
            )

    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_names]

    # load model config
    model_config = load_model_config(model_name)

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
        )

    # save out updated model config
    save_model_config(model_config)


if __name__ == "__main__":
    fire.Fire(main)
