from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None) -> None:
    """
    Detect and annotate the in-focus z-plane index for each position.

    #quality-control #preprocessing #test-ready #cpu-only

    Parameters
    ----------
    datasets
        List of dataset names to process. If None, processes all datasets in the
        "shear_stress" collection. If DEMO_MODE is enabled,
        only the first dataset will be processed.
    """

    import logging
    from multiprocessing import Pool

    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        get_datasets_in_collection,
        load_dataset_config,
        save_dataset_config,
    )
    from endo_pipeline.io import load_image
    from endo_pipeline.io.output import get_output_path
    from endo_pipeline.library.process.z_stack_selection import (
        calculate_global_center_plane,
        plot_standard_devs_per_slice,
        visualize_slice_selection,
    )
    from endo_pipeline.manifests import get_zarr_location_for_position

    logger = logging.getLogger(__name__)

    # Default list of datasets if not provided.
    if datasets is None:
        datasets = get_datasets_in_collection("shear_stress")

    if DEMO_MODE:
        logger.info("DEMO_MODE is ON. Processing only the first dataset.")
        datasets = datasets[:1]

    for dataset_name in datasets:
        logger.info(f"Processing dataset: {dataset_name}")
        save_dir = get_output_path(__file__, dataset_name)
        dataset_config = load_dataset_config(dataset_name)

        positions = dataset_config.zarr_positions
        if DEMO_MODE:
            positions = positions[:1]
            logger.info(f"DEMO_MODE is ON. Processing only position: {positions}")

        # Parallelize position processing
        args = [(dataset_config, position, save_dir) for position in positions]
        with Pool() as pool:
            results = pool.starmap(calculate_global_center_plane, args)

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_dir / f"{dataset_name}_global_center_plane.csv", index=False)
        logger.info(f"Results saved to: {save_dir / f'{dataset_name}_global_center_plane.csv'}")

        # Visualize the center plane for the first position
        position, frame = 0, 0
        center_plane = results_df.loc[
            results_df["position"] == position, "mean_center_plane"
        ].values[0]

        zarr_loc = get_zarr_location_for_position(dataset_config, position)
        bf_stack = load_image(zarr_loc, channels=["BF"], timepoints=frame, level=1, squeeze=True)
        cdh5_stack = load_image(
            zarr_loc, channels=["EGFP"], timepoints=frame, level=1, squeeze=True
        )

        global_center_plane = {
            int(row["position"]): int(row["mean_center_plane"]) for _, row in results_df.iterrows()
        }

        for position, center_plane in global_center_plane.items():
            if center_plane > 13:
                logging.warning(
                    f"{dataset_name} P{position} has a high center plane (>13). Less than 11 slices available."
                )

            else:
                visualize_slice_selection(
                    bf_stack,
                    cdh5_stack,
                    center_plane,
                    dataset_name,
                    position,
                    frame,
                    save_dir,
                )

        if DEMO_MODE:
            logger.info("DEMO_MODE is ON. Dont overwrite dataset config with results.")
            continue

        dataset_config.center_z_plane = global_center_plane
        save_dataset_config(dataset_config)

    # Visualize the standard deviations per slice for the first position
    stdevs = [plane.std().compute() for plane in bf_stack]
    center_plane = max(0, np.argmin(stdevs))
    plot_standard_devs_per_slice(stdevs, center_plane, dataset_name, position, frame, save_dir)


if __name__ == "__main__":

    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
