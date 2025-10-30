from endo_pipeline.configs.dataset_io import ipython_cli_flexecute


def main():
    from pathlib import Path

    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.configs.dataset_io import extract_t
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.process.lib_tracking import run_tracking
    from endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest

    out_dir = get_output_path("tracking_output")
    dataset_name = "20241120_20X"

    dataset_config = load_dataset_config(dataset_name)
    manifest = load_image_manifest("nuclear_labelfree_seg")
    nuclei_locations = [
        get_image_location_for_dataset(manifest, dataset_config, 0, timepoint)
        for timepoint in range(dataset_config.duration)
    ]
    nuclei_paths = [location.path for location in nuclei_locations if location.path is not None]

    run_tracking(
        in_dir=nuclei_paths,
        out_dir=Path(out_dir),
        out_filename_prefix=f"{dataset_name}_P0",
        tracking_metrics=["centroid"],
        sorting_key=extract_t,
        C=2,
        image_validation_frequency=1,
        verbose=False,
    )


if __name__ == "__main__":
    ipython_cli_flexecute(main)
