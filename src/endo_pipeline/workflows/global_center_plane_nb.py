# %%
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from endo_pipeline.configs import (
    get_datasets_in_collection,
    get_zarr_file_for_position,
    load_dataset_config,
)
from endo_pipeline.io import load_zarr_as_dask_array
from endo_pipeline.io.output import get_output_path
from endo_pipeline.library.process.z_stack_selection import (
    plot_global_center_plane,
    plot_standard_devs_per_slice,
    visualize_slice_selection,
)


# %%
def calculate_global_center_plane(
    dataset: str, position: int, config: Any, save_dir: Path
) -> dict[str, Any]:
    """Calculate global center plane for single position."""
    zarr_file = get_zarr_file_for_position(config, position)
    bf_stack_all_frames = load_zarr_as_dask_array(zarr_file, channels=["BF"], level=1)

    center_planes = []

    for frame in range(0, config.duration, 1):
        # Extract the BF stack for the current frame
        bf_stack = bf_stack_all_frames[frame].squeeze()

        # Compute standard deviations for all slices in the current frame
        stdevs = bf_stack.std(axis=(1, 2)).compute()

        # Find the center plane with the minimum standard deviation
        center_plane = max(0, np.argmin(stdevs))
        center_planes.append(center_plane)

    mean, std_dev = plot_global_center_plane(center_planes, dataset, position, save_dir)

    return {
        "position": position,
        "mean_center_plane": round(mean),
        "std_dev_center_plane": round(std_dev),
    }


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# %%
if __name__ == "__main__":
    datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")
    for dataset in datasets:
        logging.info(f"Processing dataset: {dataset}")
        save_dir = get_output_path(__file__, dataset)
        config = load_dataset_config(dataset)

        # Parallelize position processing
        args = [(dataset, position, config, save_dir) for position in range(6)]
        with Pool() as pool:
            results = pool.starmap(calculate_global_center_plane, args)

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_dir / f"{dataset}_global_center_plane.csv", index=False)
        logging.info(f"Results saved to: {save_dir / f'{dataset}_global_center_plane.csv'}")

        # Visualize the center plane for the first position
        position, frame = 0, 0
        center_plane = results_df.loc[
            results_df["position"] == position, "mean_center_plane"
        ].values[0]

        zarr_file = get_zarr_file_for_position(config, position)
        bf_stack = load_zarr_as_dask_array(
            zarr_file, channels=["BF"], timepoints=frame, level=1
        ).squeeze()
        cdh5_stack = load_zarr_as_dask_array(
            zarr_file, channels=["EGFP"], timepoints=frame, level=1
        ).squeeze()

        visualize_slice_selection(
            bf_stack, cdh5_stack, center_plane, 5, 10, dataset, position, frame, save_dir
        )

    # Visualize the standard deviations per slice for the first position
    stdevs = [plane.std().compute() for plane in bf_stack]
    center_plane = max(0, np.argmin(stdevs))
    plot_standard_devs_per_slice(stdevs, center_plane, dataset, position, frame, save_dir)
