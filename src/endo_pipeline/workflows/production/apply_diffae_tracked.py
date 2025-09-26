# need to import for type hinting
from pathlib import Path

from endo_pipeline.cli import Datasets

TAGS = ["apply_diffae_model", "diffae_features"]


def main(
    model_manifest_name: str = "diffae_04_10",
    run_name: str | None = None,
    datasets: Datasets | None = None,
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
    model_manifest_name
        Name of the model manifest to load the model from.
    run_name
        Name of the model run to apply. If None, uses the most recent run.
    datasets
        List of datasets or dataset collections to load images from. If not
        provided, workflow runs on the ``20250319_20X`` dataset.
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
    import logging
    from typing import cast

    from endo_pipeline import DEMO_MODE
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import load_model
    from endo_pipeline.library.model import apply_model_on_tracked_crops_from_one_dataset
    from endo_pipeline.library.model.image_loading import get_include_positions
    from endo_pipeline.manifests import get_model_location_for_run, load_model_manifest
    from endo_pipeline.settings import Z_SLICE_OFFSETS

    logger = logging.getLogger(__name__)

    # Default list of datasets if not provided.
    if datasets is None:
        datasets = ["20250319_20X"]

    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in datasets]

    # load model from manifest
    model_manifest = load_model_manifest(model_manifest_name)
    run_name = list(model_manifest.locations.keys())[-1] if run_name is None else run_name
    model_location = get_model_location_for_run(model_manifest, run_name)
    model = load_model(model_location)

    # apply model to each dataset
    for dataset_config in dataset_config_list:
        only_include_positions = get_include_positions(dataset_config)
        if DEMO_MODE:
            only_include_positions = only_include_positions[:1]
            logger.warning(
                "Workflow demo is enabled, only processing first few "
                "timepoints of the first position of dataset: [ %s ]",
                dataset_config.name,
            )

        apply_model_on_tracked_crops_from_one_dataset(
            model=model,
            dataset_config=dataset_config,
            upload_to_fms=upload_to_fms,
            save_path=save_path,
            user_overrides=user_overrides,
            z_slice_offsets=Z_SLICE_OFFSETS,
            only_include_positions=only_include_positions,
        )

        if DEMO_MODE:
            # only apply model to the first dataset in demo mode
            break


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
