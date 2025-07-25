# need to import for type hinting
from pathlib import Path

TAGS = ["apply_diffae_model", "diffae_features"]


def main(
    model_name: str,
    dataset_names: list[str],
    upload_to_fms: bool = True,
    save_path: str | Path | None = None,
    overrides: str | dict | None = None,
) -> None:
    """
    Apply a trained DiffAE model to single-cell-track-based crops of images from multiple datasets.

    Produces a table of latent features from a crops centered on tracked cells
    for each dataset.

    Example usage:
    ... code-block:: bash
    uv run endopipe apply_diffae_tracking --model_name diffae_04_10 --dataset_names 20241120_20X 20241217_20X

    Parameters
    ----------
    model_name
        Name of the model from `model_config.yaml` to apply.
    dataset_names
        Names of the datasets from `data_config.yaml` to apply the model to.
        If it is a string, it should either be a single dataset name or the name of a
        dataset collection.
    upload_to_fms
        Whether to upload the prediction file to FMS.
    save_path
        Path to save the prediction file.
    overrides
        Overrides to apply to the model config.

    Returns
    -------
    None
        Saves the model config with the applied model and model manifest objects.
        The model config is saved to `endo_pipeline/configs/models/{model_name}.yaml`.
    """
    from src.endo_pipeline.configs import load_dataset_config, load_model_config
    from src.endo_pipeline.library.model import apply_model_on_tracked_crops_from_one_dataset

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
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
