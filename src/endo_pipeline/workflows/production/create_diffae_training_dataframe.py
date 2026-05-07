from typing import Annotated

from cyclopts import Parameter


def main(
    include_cell_piling: Annotated[bool, Parameter(negative="--exclude-cell-piling")] = False,
) -> None:
    """
    Generate dataframes with paths to zarr files for training a DiffAE model.

    #diffae #model-training

    This script collects zarr file paths from multiple datasets, splits them
    into training and validation sets, and saves them as .csv files in a
    specified directory. It also includes metadata such as channel and
    resolution level. These dataframes are accessed by the data loader in the
    DiffAE model training script.


    **Dataset collection**

    The datasets are defined in the ``diffae_model_training`` dataset collection
    configuration.

    **Cell piling exclusion**

    By default, timepoints marked as having cell piling annotations are not included in the training
    and validation datasets (``include_cell_piling`` set to ``False``). This behavior can be changed
    by using the command line flag `--include-cell-piling`. This allows for toggling between
    training a model that "sees" cell piling versus one that does not.

    When ``include_cell_piling`` is set to False, the output dataframe manifest name will include
    the suffix ``_exclude_cell_piling``. When set to True, the suffix will be
    ``_include_cell_piling``.

    **Workflow demo**

    The ``--demo-mode`` (aka ``-d``) flag can be used to run a simplified version of this
    workflow for testing purposes (e.g. during code review). The training and validation datasets
    will only keep one position and minimal timepoints, which speeds up the data loading process
    during model training.

    Parameters
    ----------
    include_cell_piling
        True to include timepoints with cell piling in data used for training, False to exclude.
    """

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        TimepointAnnotation,
        get_all_unannotated_timepoints,
        get_subset_of_timepoint_annotations,
        get_unannotated_positions,
        load_dataset_collection_config,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.model import (
        build_and_save_dataframe_manifest_for_training,
        build_zarr_image_loading_dataframe,
        get_z_slice_bounds_per_position,
    )
    from endo_pipeline.settings import DIFFAE_ZARR_RESOLUTION_LEVEL, Z_SLICE_OFFSETS

    output_savedir = get_output_path("dataframes")

    dataset_name_list = load_dataset_collection_config("diffae_model_training").datasets
    dataset_config_list = [load_dataset_config(dataset_name) for dataset_name in dataset_name_list]

    zarr_dataframes = []
    for dataset_config in dataset_config_list:
        # parse dataset annotations to get z-slice information,
        # positions to include, and frames to exclude
        z_slice_bounds_per_position = get_z_slice_bounds_per_position(
            dataset_config, z_slice_offsets=Z_SLICE_OFFSETS
        )
        only_include_positions = get_unannotated_positions(dataset_config)
        # get frames to include based on annotations
        # either including or excluding cell piling timepoints
        # based on the include_cell_piling argument
        # default is to remove all annotations except NOT_STEADY_STATE
        annotations_to_ignore = [TimepointAnnotation.NOT_STEADY_STATE]
        if include_cell_piling:
            # if including cell piling, then ignore that annotation as well
            annotations_to_ignore.append(TimepointAnnotation.CELL_PILING)
        # get list of annotations to filter out
        annotations = get_subset_of_timepoint_annotations(
            annotations_to_ignore=annotations_to_ignore
        )
        # get list of timepoints that do not have any of the annotations
        # for each position (dict of position -> list of timepoints)
        only_include_frames = get_all_unannotated_timepoints(
            dataset_config, annotations=annotations
        )

        # When running workflow in demo mode, only use the first position from each
        # dataset and first two timepoints to speed up the data loading process (if
        # dataset is not timelapse, then only one timepoint is used). Otherwise, use
        # default frame start and stop values (i.e. all timepoints) and keep all
        # rows in the dataset CSV.
        if DEMO_MODE:
            frame_start = 0
            frame_stop = 10 if dataset_config.is_timelapse else 0
            only_include_positions = only_include_positions[0:3]
        else:
            frame_start = None
            frame_stop = None

        # build zarr loading dataframe for the current dataset
        # and append it to the list of dataframes
        zarr_dataframes.append(
            build_zarr_image_loading_dataframe(
                dataset_config=dataset_config,
                resolution_level=DIFFAE_ZARR_RESOLUTION_LEVEL,
                channel=[
                    dataset_config.zarr_channel_indices.channel_488,
                    dataset_config.zarr_channel_indices.brightfield,
                ],
                frame_start=frame_start,
                frame_stop=frame_stop,
                z_slice_bounds_per_position=z_slice_bounds_per_position,
                only_include_positions=only_include_positions,
                only_include_frames=only_include_frames,
            )
        )

    # concatenate all dataframes into one
    df = pd.concat(zarr_dataframes, ignore_index=True)

    # split into training and validation sets
    # (percent split is by number of rows, i.e. positions x datasets)
    train, val = train_test_split(df, test_size=0.2, random_state=42)

    # add "_test_workflow" suffix to manifest name if in demo mode
    name_suffix = "_demo" if DEMO_MODE else ""

    # add include/exclude cell piling suffix to manifest name
    if include_cell_piling:
        name_suffix = f"_include_cell_piling{name_suffix}"
    else:
        name_suffix = f"_exclude_cell_piling{name_suffix}"

    # Upload dataframes to FMS, then build and save out DataframeManifest
    # object with FMS IDs to be used in the DiffAE model training script.
    # Note that this can be swapped out with uploading to S3 later on.
    manifest_name = f"diffae_training_dataframe{name_suffix}"
    build_and_save_dataframe_manifest_for_training(
        train,
        val,
        DIFFAE_ZARR_RESOLUTION_LEVEL,
        Z_SLICE_OFFSETS,
        include_cell_piling,
        dataset_config_list,
        output_savedir,
        manifest_name,
        "create_diffae_training_dataframe",
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
