# %%
import numpy as np
import pandas as pd

from src.endo_pipeline.configs import load_dataset_config
from src.endo_pipeline.io.output import get_output_path
from src.endo_pipeline.library.process.get_images import get_zarr_img_for_dataset
from src.endo_pipeline.library.process.z_stack_selection import (
    plot_global_center_plane,
    plot_standard_devs_per_slice,
    visualize_slice_selection,
)

# %%
DATASET = "20241016_20X"
# DATASET = "20241120_20X"
FRAME = 0
save_dir = get_output_path(
    __file__,
    DATASET,
)
config = load_dataset_config(DATASET)
# %%
for position in range(0, 6):
    img = get_zarr_img_for_dataset(DATASET, position, resolution_level=1)
    bf_stack = img.get_image_dask_data("ZYX", C=1, T=FRAME)
    cdh5_stack = img.get_image_dask_data("ZYX", C=0, T=FRAME)

    stdevs = [plane.std().compute() for plane in bf_stack.squeeze()]
    center_plane = max(0, np.argmin(stdevs))

    # Plot the standard devs vs plane index
    plot_standard_devs_per_slice(stdevs, center_plane, DATASET, position, FRAME, save_dir)

    # Visualize the slice selection
    offset = 6
    visualize_slice_selection(
        bf_stack, cdh5_stack, center_plane, offset, DATASET, position, FRAME, save_dir
    )
    break
# %%
results = []

for position in range(0, 6):
    img = get_zarr_img_for_dataset(DATASET, position, resolution_level=1)
    bf_stack_all_frames = img.get_image_dask_data("TZYX", C=1)

    center_planes = []

    for frame in range(0, config.duration, 1):
        # Extract the BF stack for the current frame
        bf_stack = bf_stack_all_frames[frame].squeeze()

        # Compute standard deviations for all planes in the current frame
        stdevs = bf_stack.std(axis=(1, 2)).compute()

        # Find the center plane with the minimum standard deviation
        center_plane = max(0, np.argmin(stdevs))
        center_planes.append(center_plane)  # Collect center plane values

    # Use the helper function to plot and get statistics
    mean, std_dev = plot_global_center_plane(center_planes, DATASET, position, save_dir)
    print(f"Position {position}: Mean = {mean:.2f}, Std Dev = {std_dev:.2f}")

    results.append(
        {
            "position": position,
            "mean_center_plane": round(mean, 2),
            "std_dev_center_plane": round(std_dev, 2),
        }
    )
# %%
# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
results_df.to_csv(save_dir / f"{DATASET}_global_center_plane.csv", index=False)
# %%
