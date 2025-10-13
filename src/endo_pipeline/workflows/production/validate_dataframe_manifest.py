TAGS = ["validation", "manifests"]


def main(
    dataframe_manifest_name: str,
    for_diffae_evaluation: bool = False,
) -> None:
    """Validate a given dataframe manifest."""
    import logging

    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.library.model.image_loading import get_exclude_frames, get_include_positions
    from endo_pipeline.manifests import (
        get_dataframe_location_for_dataset,
        list_datasets_with_dataframes,
        load_dataframe_manifest,
    )

    logger = logging.getLogger(__name__)

    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    logger.info("Starting validation of dataframe manifest [ %s ]", dataframe_manifest_name)

    for dataset_name, location in dataframe_manifest.locations.items():
        if location.fmsid is None and location.s3uri is None:
            logger.error(
                "Dataset [ %s ] in dataframe manifest [ %s ] does not have a location supplied.",
                dataset_name,
                dataframe_manifest_name,
            )
            raise ValueError(
                f"Dataset {dataset_name} in dataframe manifest {dataframe_manifest_name} "
                "does not have a valid location."
            )
        # confirm we can load the dataframe
        _ = load_dataframe(location)

    # additional (optional) checks specific to DiffAE evaluation dataframes
    if for_diffae_evaluation:
        print(
            "Validating included positions and frames in dataframe manifest: "
            f"[ {dataframe_manifest_name} ]"
        )
        dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
        list_of_datasets = list_datasets_with_dataframes(dataframe_manifest)

        exclude_cell_piling = dataframe_manifest.parameters.get("exclude_cell_piling", True)

        for dataset_name in list_of_datasets:
            dataset_config = load_dataset_config(dataset_name)
            dataset_location = get_dataframe_location_for_dataset(dataframe_manifest, dataset_name)
            df = load_dataframe(dataset_location)
            include_positions = get_include_positions(dataset_config)
            exclude_frames = get_exclude_frames(dataset_config, exclude_cell_piling)
            for position, df_position in df.groupby("position"):

                # check that position should be included
                position_as_int = int(position[1:])  # position in dataframes is 'P[0-9]+'
                if position_as_int not in include_positions:
                    logger.error(
                        "Position [ %s ] in dataframe for dataset [ %s ] in dataframe manifest"
                        "[ %s ] should not be included based on dataset config annotations.",
                        position_as_int,
                        dataset_name,
                        dataframe_manifest_name,
                    )
                    raise ValueError("Dataframe contains invalid positions.")

                # get unique frames in dataframe for this position
                frames_in_df = set(df_position["frame_number"].unique())

                # check that no excluded frames for this position are in the dataframe
                exclude_frames_by_pos = set(exclude_frames.get(position_as_int, []))
                all_possible_frames = set(range(dataset_config.duration))
                frames_should_be_included = all_possible_frames - exclude_frames_by_pos

                # i.e., set difference should be empty
                difference_in_sets = frames_in_df - frames_should_be_included
                if len(difference_in_sets) > 0:
                    logger.error(
                        "Dataframe for dataset [ %s ] from dataframe manifest [ %s ] "
                        "contains invalid frames based on dataset config annotations.",
                        dataset_name,
                        dataframe_manifest_name,
                    )
                    raise ValueError("Dataframe contains invalid frames.")

    logger.info("Finished validation of dataframe manifest [ %s ]", dataframe_manifest_name)
