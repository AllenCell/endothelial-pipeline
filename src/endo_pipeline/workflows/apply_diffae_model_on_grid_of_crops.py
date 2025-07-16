from collections.abc import Sequence

import fire

from src.endo_pipeline.configs import (
    get_datasets_in_collection,
    load_dataset_config,
    load_model_config,
    save_model_config,
)
from src.endo_pipeline.library.model import apply_model_on_grid_of_crops_from_one_dataset


def main(
    model_name: str,
    dataset_names: str | Sequence[str] = "reference",
    resolution_level: int = 1,
    upload_to_fms: bool = True,
    overrides: str | dict | None = None,
) -> None:
    """
    Apply a model to a multiple datasets.

    Example usage:
    ```
    uv run src/endo_pipeline/workflows/apply_model.py
    --model_name diffae_04_10 --dataset_names '["20241016_20X","20250224_20X"]'
    ```


    Parameters
    ----------
    model_name: str
        Name of the model from `model_config.yaml` to apply.
    dataset_names: str
        Names of the datasets from `data_config.yaml` to apply the model to.
        If "reference", all reference datasets will be used.
    resolution_level: int
        Resolution level to apply the model at. Default is 1 (zarr sample resolution).
    upload_to_fms: bool
        Whether to upload the prediction file to FMS. Default is True.
    save_path: str | Path | None
        Path to save the prediction file. Default is `models/{model_name}/{dataset_name}`.
    overrides: str or dict or None
        Overrides to apply to the model config. By default, no overrides are applied
    """
    # default is to apply to all reference datasets
    if dataset_names == "reference":
        # get reference dataset names
        dataset_names = get_datasets_in_collection("pca_reference")
    elif isinstance(dataset_names, str):
        # if dataset_names is a single string, convert it to a list
        dataset_names = [dataset_names]
    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_names]

    # load model config
    model_config = load_model_config(model_name)

    # apply model to each dataset
    for dataset_config in dataset_config_list:
        model_config = apply_model_on_grid_of_crops_from_one_dataset(
            model_config=model_config,
            dataset_config=dataset_config,
            resolution_level=resolution_level,
            upload_to_fms=upload_to_fms,
            overrides=overrides,
        )

    # save out updated model config
    save_model_config(model_config)


if __name__ == "__main__":
    fire.Fire(main)
