def main():
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.process.lib_tracking import run_tracking
    from endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest

    out_dir = get_output_path("tracking_output")
    dataset_name = "20250618_20X"

    dataset_config = load_dataset_config(dataset_name)
    manifest = load_image_manifest("nuclear_labelfree_seg_zarr")
    nuclei_seg_location = get_image_location_for_dataset(manifest, dataset_config, 0)

    run_tracking(
        image_location=nuclei_seg_location,
        timepoints_to_eval=range(dataset_config.duration),
        out_dir=out_dir,
        out_filename_prefix=f"{dataset_name}_P0",
        tracking_metrics=["centroid"],
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
