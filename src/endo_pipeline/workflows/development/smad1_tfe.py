# from endo_pipeline.cli import tags

TAGS = ["immunoflourescence"]


def main(backdrops: bool = False) -> None:
    """
    Generate timelapse feature explorer datasets for immunofluorescence SMAD1 data.

    Parameters
    ----------
    backdrops
        Whether to generate backdrop images for the datasets. If False, it is assumed they were
        generated previously. If True, backdrop images will be created for each dataset and position.
    """
    import logging

    from colorizer_data import convert_colorizer_data

    from endo_pipeline import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.visualize.timelapse_feature_explorer.backdrop_images import (
        add_backdrop_fname_to_manifest,
        generate_backdrops,
    )
    from endo_pipeline.manifests import (
        get_dataframe_location_for_dataset,
        get_image_location_for_dataset,
        load_dataframe_manifest,
        load_image_manifest,
    )

    logger = logging.getLogger(__name__)

    IF_BACKDROP_IMAGES = [
        "bf_slice",
        "bf_std_dev",
        "gfp_max_proj",
        "max_proj_405",
        "max_proj_561",
        "max_proj_640",
    ]

    df_manifest = load_dataframe_manifest("immunofluorescence")
    seg_img_manifest = load_image_manifest("nuclear_stain_seg")
    img_manifest = load_image_manifest("image_zarr")
    datasets = get_datasets_in_collection("smad1")

    if DEMO_MODE:
        datasets = datasets[:1]
        logger.info("Running in demo mode: limiting to first dataset only.")

    for dataset_name in datasets:
        logger.info(f"Processing dataset: {dataset_name}")
        dataset_config = load_dataset_config(dataset_name)

        positions = dataset_config.zarr_positions
        if DEMO_MODE:
            positions = positions[:1]
            logger.info("Running in demo mode: limiting to first position only.")

        for position in positions:
            logger.info(f"  Processing position: {position}")

            seg_img_location = get_image_location_for_dataset(
                seg_img_manifest, dataset_config, position, 0
            )
            img_location = get_image_location_for_dataset(img_manifest, dataset_config, position, 0)

            output_dir = get_output_path("tfe_immunofluorescence", f"{dataset_name}_P{position}")

            df_location = get_dataframe_location_for_dataset(df_manifest, dataset_name)
            df = load_dataframe(df_location)
            df["track_id"] = df["label"]
            df["tid"] = df["track_id"]
            df["image_index"] = 0
            df["seg_image"] = seg_img_location.path

            df = add_backdrop_fname_to_manifest(
                df,
                dataset_name,
                position,
                IF_BACKDROP_IMAGES,
                output_dir=output_dir / "backdrops",
            )

            if backdrops:
                generate_backdrops(
                    dataset_name,
                    position,
                    img_location,
                    IF_BACKDROP_IMAGES,
                    output_dir=output_dir / "backdrops",
                    method="percentile",
                )

            convert_colorizer_data(
                data=df,
                output_dir=output_dir,
                object_id_column="label",
                times_column="image_index",
                track_column="track_id",
                image_column="seg_image",
                centroid_x_column="centroid_x",
                centroid_y_column="centroid_y",
                backdrop_column_names=[f"{image}_backdrop" for image in IF_BACKDROP_IMAGES],
                # feature_column_names=list(LABEL_MAP.keys()),
                # feature_info=feature_info,
            )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
