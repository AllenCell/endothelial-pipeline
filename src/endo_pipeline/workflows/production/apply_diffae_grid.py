from endo_pipeline.cli import Datasets

TAGS = ["apply_diffae_model", "diffae_features"]


def main(
    model_manifest_name: str = "diffae_04_10",
    run_name: str | None = None,
    datasets: Datasets | None = None,
    resolution_level: int = 1,
    upload_to_fms: bool = True,
    user_overrides: str | dict | None = None,
) -> None:
    """
    Apply a trained DiffAE model to grid-based crops of images from multiple datasets.

    Produces a table of latent features from a non-overlapping grid of crops for each dataset.
    The model is applied at the specified resolution level.

    **Workflow demo**

    If demo mode is enabled, the model will only be evaluated on the first few
    timepoints of the first position of the first dataset.

    **Example usage**

    .. code-block:: bash

        endopipe -vg apply-diffae-grid --datasets 20250409_20X

    Parameters
    ----------
    model_manifest_name
        Name of the model manifest to load the model from.
    run_name
        Name of the model run to apply. If None, uses the most recent run.
    datasets
        List of datasets or dataset collections to load images from. If not
        provided, workflow runs on the ``live_20X_objective_3i_microscope``
        dataset collection.
    resolution_level
        Resolution level to at which to load images (zarr file format) at.
    upload_to_fms
        True to upload the prediction file for each dataset to FMS, False to only save locally.
    user_overrides
        Optional user overrides to apply to the model config.

    Returns
    -------
    :
        Saves the model config with the applied model and model manifest objects.
        The model config is saved to [ endo_pipeline/configs/models/{model_name}.yaml ].
    """
    import logging
    from pathlib import Path

    from endo_pipeline import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import build_fms_annotations, load_model, upload_file_to_fms
    from endo_pipeline.library.model import (
        apply_model_on_grid_of_crops_from_one_dataset,
        upload_prediction_dataframe_to_fms,
    )
    from endo_pipeline.library.model.image_loading import get_include_positions
    from endo_pipeline.manifests import (
        DataframeLocation,
        DataframeManifest,
        get_model_location_for_run,
        load_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings import Z_SLICE_OFFSETS

    logger = logging.getLogger(__name__)

    # Default list of datasets if not provided.
    if datasets is None:
        datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")

    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in datasets]

    # load model from manifest
    model_manifest = load_model_manifest(model_manifest_name)
    run_name = list(model_manifest.locations.keys())[-1] if run_name is None else run_name
    model_location = get_model_location_for_run(model_manifest, run_name)
    model = load_model(model_location)

    # apply model to each dataset
    for dataset_config in dataset_config_list:
        # Get positions to include.
        only_include_positions = get_include_positions(dataset_config)

        # When running workflow in demo mode, only use the first position from each
        # dataset and first two timepoints to speed up the dataloading process (if
        # dataset is not timelapse, then only one timepoint is used). Otherwise, use
        # default frame start and stop values (i.e. all timepoints) and keep all
        # rows in the dataset CSV.
        if DEMO_MODE:
            frame_start = 0
            frame_stop = 1 if dataset_config.is_timelapse else 0
            only_include_positions = only_include_positions[0:1]
            logger.warning(
                "Workflow demo is enabled, only processing first few "
                "timepoints of the first position of dataset: [ %s ]",
                dataset_config.name,
            )
        else:
            frame_start = None
            frame_stop = None

        prediction_path = apply_model_on_grid_of_crops_from_one_dataset(
            model=model,
            dataset_config=dataset_config,
            resolution_level=resolution_level,
            upload_to_fms=upload_to_fms,
            user_overrides=user_overrides,
            z_slice_offsets=Z_SLICE_OFFSETS,
            frame_start=frame_start,
            frame_stop=frame_stop,
            only_include_positions=only_include_positions,
        )

        if upload_to_fms:
            # upload prediction file to FMS
            # Store FMS ID in dataframe manifest.
            upload_prediction_dataframe_to_fms(
                prediction_path,
                dataset_config,
                model_manifest,
                run_name,
                dataframe_manifest_name=model.cfg.run_name,
                workflow_name=Path(__file__).stem,
                parameters={"z_slice_offsets": Z_SLICE_OFFSETS},
            )

        if DEMO_MODE:
            # only apply model to the first dataset in demo mode
            break


if __name__ == "__main__":

    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
