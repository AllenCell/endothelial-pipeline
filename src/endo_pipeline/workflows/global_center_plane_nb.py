# %%
import logging
from multiprocessing import Pool

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
    calculate_global_center_plane,
    plot_standard_devs_per_slice,
    visualize_slice_selection,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# %%
if __name__ == "__main__":
    datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")
    for dataset in datasets:
        logging.info(f"Processing dataset: {dataset}")
        save_dir = get_output_path(__file__, dataset)
        dataset_config = load_dataset_config(dataset)

        # Parallelize position processing
        args = [(dataset_config, position, save_dir) for position in range(6)]
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

        zarr_file = get_zarr_file_for_position(dataset_config, position)
        bf_stack = load_zarr_as_dask_array(
            zarr_file, channels=["BF"], timepoints=frame, level=1
        ).squeeze()
        cdh5_stack = load_zarr_as_dask_array(
            zarr_file, channels=["EGFP"], timepoints=frame, level=1
        ).squeeze()

        visualize_slice_selection(
            bf_stack, cdh5_stack, center_plane, 5, 10, dataset, position, frame, save_dir
        )
        break
    # Visualize the standard deviations per slice for the first position
    stdevs = [plane.std().compute() for plane in bf_stack]
    center_plane = max(0, np.argmin(stdevs))
    plot_standard_devs_per_slice(stdevs, center_plane, dataset, position, frame, save_dir)

# %%
