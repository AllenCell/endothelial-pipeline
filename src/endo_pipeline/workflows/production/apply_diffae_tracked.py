# need to import for type hinting
from pathlib import Path

TAGS = ["apply_diffae_model", "diffae_features"]


def main(
    model_name: str,
    dataset_names: list[str],
    upload_to_fms: bool = True,
    save_path: str | Path | None = None,
    user_overrides: str | dict | None = None,
) -> None:
    """
    Apply a trained DiffAE model to single-cell-track-based crops of images from multiple datasets.

    Produces a table of latent features from a crops centered on tracked cells
    for each dataset.

    Parameters
    ----------
    model_name
        Name of the model to apply.
    dataset_names
        Names of the datasets from which to load images.
    upload_to_fms
        True to upload the prediction file for each dataset to FMS, False to only save locally.
    save_path
        Path to save the prediction file locally.
    user_overrides
        Optional user overrides to apply to the model config.

    Returns
    -------
    :
        Saves the model config with the applied model and model manifest objects.
        The model config is saved to :code:`endo_pipeline/configs/models/{model_name}.yaml`.
    """
    from typing import cast

    from endo_pipeline.configs import CytoDLModelConfig, load_dataset_config, load_model_config
    from endo_pipeline.library.model import apply_model_on_tracked_crops_from_one_dataset
    from endo_pipeline.library.model.image_loading import get_include_positions
    from endo_pipeline.settings import Z_SLICE_OFFSETS

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_names]

    # load model config
    model_config = cast(CytoDLModelConfig, load_model_config(model_name))

    # apply model to each dataset
    for dataset_config in dataset_config_list:
        only_include_positions = get_include_positions(dataset_config)

        apply_model_on_tracked_crops_from_one_dataset(
            model_config=model_config,
            dataset_config=dataset_config,
            upload_to_fms=upload_to_fms,
            save_path=save_path,
            user_overrides=user_overrides,
            z_slice_offsets=Z_SLICE_OFFSETS,
            only_include_positions=only_include_positions,
        )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
