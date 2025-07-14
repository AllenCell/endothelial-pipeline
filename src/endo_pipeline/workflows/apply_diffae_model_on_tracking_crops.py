from collections.abc import Sequence
from pathlib import Path

import fire

from src.endo_pipeline.configs import load_dataset_config, load_model_config
from src.endo_pipeline.library.model import apply_model_on_tracked_crops_from_one_dataset


def main(
    model_name: str,
    dataset_names: Sequence[str],
    upload_to_fms: bool = True,
    save_path: str | Path | None = None,
    overrides: str | dict | None = None,
) -> None:
    """
    Apply a model to a multiple datasets.

    Example usage:
    ```
    python src/endo_pipeline/workflows/apply_on_crop.py
    --model_name diffae_04_10 --dataset_names '["20241016_20X","20250224_20X"]'
    ```


    Parameters
    ----------
    model_name: str
        Name of the model from `model_config.yaml` to apply.
    dataset_names: str
        Name of the dataset from `data_config.yaml` to apply the model to.
    upload_to_fms: bool
        Whether to upload the prediction file to FMS. Default is True.
    save_path: str | Path | None
        Path to save the prediction file. Default is `models/{model_name}/{dataset_name}`.
    overrides: str or dict or None
        Overrides to apply to the model config. By default, no overrides are applied
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_names]

    # load model config
    model_config = load_model_config(model_name)

    # apply model to each dataset
    for dataset_config in dataset_config_list:
        apply_model_on_tracked_crops_from_one_dataset(
            model_config=model_config,
            dataset_config=dataset_config,
            upload_to_fms=upload_to_fms,
            save_path=save_path,
            overrides=overrides,
        )


if __name__ == "__main__":
    fire.Fire(main)
