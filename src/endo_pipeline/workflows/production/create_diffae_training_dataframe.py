TAGS = ["diffae_model_training"]


def main(
    resolution_level: int = 1,
    z_stack_offsets: tuple[int, int] | None = None,
    slice_by_global_center: bool = True,
    exclude_cell_piling: bool = False,
) -> None:
    """
    Generate dataframes with paths to zarr files for training a DiffAE model.

    This script collects zarr file paths from multiple datasets, splits them
    into training and validation sets, and saves them as .csv files in a
    specified directory. It also includes metadata such as channel and
    resolution level. These dataframes are accessed by the data loader in the
    DiffAE model training script.


    **Dataset collection**

    The datasets are defined in the ``diffae_model_training`` dataset collection
    configuration.


    **Zarr resolution**

    Zarr files used by training can be used as different resolutions. The
    default resolution of 1 corresponds to downsampling by half.

    **Z-stack offsets**

    The ``z_stack_offsets`` parameter allows for flexible control over the z-slice loading.
    If ``z_stack_offsets`` is provided, it limits the number of z-slices to load, either
    by slicing about a global center or by using the provided offsets directly. If it
    is ``None``, all z-slices are loaded from the raw brightfield images.

    If ``slice_by_global_center`` is set to True, the z-slice range is calculated based on
    the global center plane for the given position. In this case, ``z_stack_offsets`` should
    indicate the number of slices to include below and above the center plane. Else, the
    ``z_stack_offsets`` are used directly as the range bounds.

    **Cell piling exclusion**

    By default, timepoints marked as having cell piling annotations are included in the training
    and validation datasets. This behavior can be changed by setting the ``exclude_cell_piling``
    parameter to True. This allows for toggling between training a model that "sees" cell piling
    versus one that does not.

    **Workflow testing**

    The ``--testing-mode`` (aka ``-x``) flag can be used to run a simplified version of this
    workflow for testing purposes (e.g. during code review). The training and validation datasets
    will only keep one position and minimal timepoints, which speeds up the data loading process
    during model training. Furthermore, the script will use the staging (``stg``) environment
    instead of the production (``prod``) environment for FMS uploads.

    Parameters
    ----------
    resolution_level
        The resolution level of the zarr files to load for training.
    z_stack_offsets
        Lower and upper bounds for z-slicing.
    slice_by_global_center
        Get global center plane per position for z-slicing if True, use offsets directly if False.
    exclude_cell_piling
        Exclude cell piling timepoints if True, include them if False.


    Returns
    -------
    :
        Uploads the training and validation dataframes to FMS and saves a
        DataframeManifest with DatasetLocation objects containing the FMS IDs of
        the uploaded files.
    """

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from endo_pipeline import TESTING_MODE
    from endo_pipeline.configs import load_dataset_collection_config, load_dataset_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.model import (
        build_and_save_dataframe_manifest_for_training,
        build_zarr_image_loading_dataframe,
        get_exclude_frames,
        get_include_positions,
        get_z_slice_bounds_per_position,
    )

    output_savedir = get_output_path("dataframes")

    # load data config
    dataset_name_list = load_dataset_collection_config("diffae_model_training").datasets
    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_name_list]

    zarr_dataframes = []
    for dataset_config in dataset_config_list:
        # generate zarr loading metadata table for each dataset
        # default frame start and stop values are None, i.e., load all timepoints
        frame_start = None
        frame_stop = None

        # parse dataset annotations to get z-slice information,
        # positions to include, and frames to exclude
        z_slice_bounds_per_position = get_z_slice_bounds_per_position(
            dataset_config, z_stack_offsets, slice_by_global_center
        )
        only_include_positions = get_include_positions(dataset_config)
        exclude_frames = get_exclude_frames(dataset_config, exclude_cell_piling=exclude_cell_piling)

        if TESTING_MODE:
            # for workflow testing, only use first position from each dataset
            # and first two timepoints to speed up the dataloading process
            # (if dataset is not timelapse, then only one timepoint is used)
            frame_start = 0
            frame_stop = 1 if dataset_config.is_timelapse else 0
            only_include_positions = only_include_positions[0:1]

        # build zarr loading dataframe for the current dataset
        # and append it to the list of dataframes
        zarr_dataframes.append(
            build_zarr_image_loading_dataframe(
                dataset_config=dataset_config,
                resolution_level=resolution_level,
                channel=[
                    dataset_config.zarr_channel_indices.channel_488,
                    dataset_config.zarr_channel_indices.brightfield,
                ],
                frame_start=frame_start,
                frame_stop=frame_stop,
                z_slice_bounds_per_position=z_slice_bounds_per_position,
                only_include_positions=only_include_positions,
                exclude_frames=exclude_frames,
            )
        )

    # concatenate all dataframes into one
    df = pd.concat(zarr_dataframes, ignore_index=True)

    # split into training and validation sets
    # (percent split is by number of rows, i.e. positions x datasets)
    train, val = train_test_split(df, test_size=0.2, random_state=42)

    # Upload dataframes to FMS, then build and save out DataframeManifest
    # object with FMS IDs to be used in the DiffAE model training script.
    # Note that this can be swapped out with uploading to S3 later on.
    manifest_name = f"diffae_training_dataframe_resolution_{resolution_level}"
    if exclude_cell_piling:
        manifest_name = f"{manifest_name}_exclude_cell_piling"
    if TESTING_MODE:
        manifest_name = f"{manifest_name}_test_workflow"
    build_and_save_dataframe_manifest_for_training(
        train,
        val,
        resolution_level,
        z_stack_offsets,
        slice_by_global_center,
        exclude_cell_piling,
        dataset_config_list,
        output_savedir,
        manifest_name,
        "create_diffae_training_dataframe",
    )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
