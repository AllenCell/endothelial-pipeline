# %% Import libraries
import concurrent
from pathlib import Path

from matplotlib.gridspec import GridSpec
from matplotlib.projections.polar import PolarAxes
from pandas import DataFrame

from cellsmap.util.dataset_io import load_config, load_dataset_position_as_dask_array
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.library.analyze import optical_flow_calculator
from src.endo_pipeline.library.process.general_image_preprocessing import (
    get_default_dim_order,
    get_dim_map,
)

# %% Make list of datasets to analzye
dataset_name_list = [
    config_data["name"]
    for config_data in load_config(config_type="data")
    if (config_data["microscope"] == "3i")
]
position = (
    0  # NOTE PLACEHOLDER. WORKFLOW SHOULD BECOME MAIN() WITH position AS AN ARGUMENT
)
dataset_name_list = [dataset_name_list[0]]  # this is just a test

debug = True
ncores = 1  # 30
delta_t = 1
level = 1
out_dir = Path(get_output_path(Path(__file__).stem, verbose=False))
out_path_list = []

# %%
if __name__ == "__main__":
    # run using multiple cores
    if ncores > 1:
        with concurrent.futures.ProcessPoolExecutor(ncores) as executor:
            for dataset_name in dataset_name_list:
                out_path = optical_flow_calculator.compute_and_save_flow_field(
                    out_dir, dataset_name, delta_t, level, executor, ncores, debug
                )
                out_path_list.append(out_path)

    else:
        for dataset_name in dataset_name_list:
            no_executor = None
            out_path = optical_flow_calculator.compute_and_save_flow_field(
                out_dir,
                dataset_name,
                position,
                delta_t,
                level,
                no_executor,
                ncores,
                debug,
            )
            out_path_list.append(out_path)

# %% Save the locations of the outputs
DataFrame(
    {"dataset_name": dataset_name_list, "vector_field_image_paths": out_path_list}
).to_csv(out_dir / "vector_field_image_paths.csv", index=False)


import matplotlib.pyplot as plt
import numpy as np

# %% Example of usage:
from bioio import BioImage
from skimage.exposure import rescale_intensity

dim_order = get_default_dim_order()
dim_map = get_dim_map(dim_order)
# %%
# Get the paths to the vector field images
dataset_list = ["20241016_20X"]

for dataset_name in dataset_list:
    # Load vector field image for a dataset
    img = optical_flow_calculator.load_vector_field_img(out_dir, dataset_name)

    # Get the channel names and their indices from the outputted image
    # (the vector field images have metadata saved with them)
    image_paths = optical_flow_calculator.get_vector_field_image_paths(out_dir)
    im_path = image_paths[dataset_name]
    chan_map = {
        img.channel_names: i
        for i, img.channel_names in enumerate(BioImage(im_path).channel_names)
    }
    print(f"The channel names and their indices are: {chan_map}")

    # Get some random crops of the vector fields (this is randomly sampling in time too)
    roi_shape = (1, 4, 64, 64)  # ROI: region of interest
    num_rois = 10
    rois = optical_flow_calculator.get_random_roi(
        img.shape, roi_shape, num_rois=num_rois, random_seed=42
    )
    # apply these crops to the vector field images
    crops = [img[r] for r in rois]
    # load the roi crops into memory
    crops_in_memory = [c.compute() for c in crops]

    # Get a couple of basic descriptive statistics of the norm and angle channels
    norm_chan_idx = chan_map["norm"]
    mag_means = [c[0, norm_chan_idx, ...].mean() for c in crops_in_memory]
    mag_stds = [c[0, norm_chan_idx, ...].std() for c in crops_in_memory]

    angle_chan_idx = chan_map["theta"]

    # Show the vector magnitude channel from the random crops
    nrows, ncols, figsize = 2, 5, (15, 7)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    [
        ax.imshow(crops_in_memory[i][0, norm_chan_idx, ...], cmap="gray")
        for i, ax in enumerate(axs.flatten())
    ]
    [ax.axis("off") for i, ax in enumerate(axs.flatten())]
    [
        ax.set_title(f"mean: {mag_means[i]:.2f}, std: {mag_stds[i]:.2f}")
        for i, ax in enumerate(axs.flatten())
    ]
    fig.suptitle("Vector norms from random crops", size=20)
    plt.tight_layout()
    plt.show()

    # Show the vector angles channel from the random crops
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    [
        ax.imshow(crops_in_memory[i][0, angle_chan_idx, ...], cmap="gray")
        for i, ax in enumerate(axs.flatten())
    ]
    [ax.axis("off") for i, ax in enumerate(axs.flatten())]
    [
        ax.set_title(f"mean: {mag_means[i]:.2f}, std: {mag_stds[i]:.2f}")
        for i, ax in enumerate(axs.flatten())
    ]
    fig.suptitle("Vector angles from random crops", size=20)
    plt.tight_layout()
    plt.show()

    # Show the image crops from the random roi regions vs the same
    # image at the next timepoint too
    img = load_dataset_position_as_dask_array(
        dataset_name, position, channels=["EGFP"], level=level
    )
    img = img.max(axis=dim_map["Z"], keepdims=False)
    rois_future = [
        (slice(r[0].start + delta_t, r[0].stop + delta_t), *r[1:]) for r in rois
    ]
    crops = [
        np.stack(
            [img[rois[i]], img[rois_future[i]], np.zeros_like(img[rois[i]])], axis=-1
        )
        for i in range(len(rois))
    ]
    crops_in_memory = [c.compute() for c in crops]
    crops_in_memory = [
        rescale_intensity(c, out_range=np.uint8) for c in crops_in_memory
    ]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    [
        ax.imshow(crops_in_memory[i][0, 0, ...], cmap="gray")
        for i, ax in enumerate(axs.flatten())
    ]
    [ax.axis("off") for i, ax in enumerate(axs.flatten())]
    fig.suptitle("CDH5 from random crops (red = t, green = t + delta_t)", size=20)
    plt.tight_layout()
    plt.show()

    # Use the loaded vector information to create the flow field
    vfield = optical_flow_calculator.load_vector_field_img(out_dir, dataset_name)
    vfield_crops = [vfield[r] for r in rois]
    vfield_crops = [c.compute() for c in vfield_crops]
    vx_crops, vy_crops = zip(
        *[
            (c[0, chan_map["vx"], ...].squeeze(), c[0, chan_map["vy"], ...].squeeze())
            for c in vfield_crops
        ]
    )

    img = load_dataset_position_as_dask_array(
        dataset_name, position, channels=["EGFP"], level=level
    )
    img = img.max(axis=dim_map["Z"], keepdims=False)
    img_crops = [img[r] for r in rois]
    img_crops = [c.compute().squeeze() for c in img_crops]

    flow_graphs = [
        optical_flow_calculator.get_trimmed_vector_field_map(
            img, vx, vy, display=False, return_map=True
        )
        for img, vx, vy in zip(img_crops, vx_crops, vy_crops)
    ]

    rois_as_titles = [list(zip(*[(slc.start, slc.stop) for slc in r])) for r in rois]
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=[d * 1.5 for d in figsize]
    )
    [ax.imshow(flow_graphs[i], cmap="gray") for i, ax in enumerate(axs.flatten())]
    [ax.axis("off") for i, ax in enumerate(axs.flatten())]
    [
        ax.set_title(
            f"roi start: {rois_as_titles[i][0]}\nroi stop:{rois_as_titles[i][1]}"
        )
        for i, ax in enumerate(axs.flatten())
    ]
    fig.suptitle("Flow fields", size=20)
    plt.tight_layout()
    plt.show()


# %% Create figures for validation:
# The figure should be of PC1 vs PC2 for different crops with select crops shown
# and linked back to their corresponding data points in the PC1 vs PC2 plot

import pandas as pd

# %% 0. imports and dataset loading
from sklearn.preprocessing import minmax_scale

# create an output directory for the plots
out_dir_val = out_dir / "validation_plots"
out_dir_plots = out_dir_val / f"{dataset_name}"
Path.mkdir(out_dir_plots, exist_ok=True, parents=True)

# Pick a dataset
dataset_name = dataset_list[0]

# Load vector field image for a dataset
flow_feat_img = optical_flow_calculator.load_vector_field_img(out_dir, dataset_name)

# Get the channel names and their indices from the outputted image
# (the vector field images have metadata saved with them)
image_paths = optical_flow_calculator.get_vector_field_image_paths(out_dir)
im_path = image_paths[dataset_name]
chan_map = {
    vec_field_img_chan_name: i
    for i, vec_field_img_chan_name in enumerate(BioImage(im_path).channel_names)
}
print(f"The channel names and their indices are: {chan_map}")


# %% 1. Get a bunch of crops:
# Get some random crops of the vector fields (this is randomly sampling in time too)
# and load the roi crops into memory
roi_shape = (1, 4, 64, 64)
num_rois = 200
rois = optical_flow_calculator.get_random_roi(
    flow_feat_img.shape, roi_shape, num_rois=num_rois, random_seed=42
)


# %% 2. Get flow features of the crops
crops = [flow_feat_img[r] for r in rois]
crops_in_memory = [c.compute() for c in crops]


# %% 3. Compute PCA on the crops
# use the standard deviation of the vector magnitudes and the divergence estimates as features since
# they are conspicuous features of the vector field map
features = pd.DataFrame(
    [
        optical_flow_calculator.FlowCalculator.get_features_from_vector_field_image(
            crop, chan_map
        )
        for crop in crops_in_memory
    ]
)
features_for_PCA = features[["vector_magnitudes_std", "divergence_std"]].to_numpy()
pca, feats_proj = optical_flow_calculator.compute_PCA_on_features(
    features_for_PCA, n_components=2, return_as_dataframe=True
)

# rescale the pca features to be between -1 and 1 so that the points are more evenly distributed
# along the pc1 and pc2 axes (facilitates picking out the most average points in each quadrant)
feats_proj[0] = minmax_scale(feats_proj[0], feature_range=(-1, 1))
feats_proj[1] = minmax_scale(feats_proj[1], feature_range=(-1, 1))

# create a dataframe of the features and PCs
feats_proj.reset_index(inplace=True, names="crop_id")
analysis_info = pd.DataFrame(
    [
        {
            "dataset_name": dataset_name,
            "T": t.start,
            "start_c": 0,
            "start_y": y.start,
            "start_x": x.start,
            "delta_t": t.stop - t.start,
            "size_c": c.stop - c.start,
            "size_y": y.stop - y.start,
            "size_x": x.stop - x.start,
            "vx_chan_index": chan_map["vx"],
            "vy_chan_index": chan_map["vy"],
            "norm_chan_index": chan_map["norm"],
            "theta_chan_index": chan_map["theta"],
        }
        for t, c, y, x in rois
    ]
)
features_and_pcs = pd.concat([analysis_info, feats_proj, features], axis="columns")


# %% 4. Get the most "average" point in each quadrant of PC-space as an example
# get example crops from each quadrant
pc_points = features_and_pcs[[0, 1]].to_numpy()
quadrants_origin = np.mean(pc_points, axis=0)
quadrant_means = optical_flow_calculator.get_quadrant_means(
    pc_points, origin=quadrants_origin
)
quadrant_colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
example_points = {}
for i, quad_mean in enumerate(quadrant_means):
    example_point, example_index = (
        optical_flow_calculator.get_point_closest_to_reference_point(
            pc_points, reference_point=quad_mean
        )
    )
    example_crop = features_and_pcs.iloc[example_index]
    # ensure that the example crop is using the correct points from example_pt
    assert all(example_crop[[0, 1]].to_numpy() == example_point)

    example_points[i] = {
        "color": quadrant_colors[i],
        "quadrant_mean": quad_mean,
        "record": example_crop,
    }


# %% 5. Plot two PCs and the example crops from each of the 4 quadrants
# load the cdh5 channel of the dataset in the crop region
img = load_dataset_position_as_dask_array(
    dataset_name, position, channels=["EGFP"], level=level
)
img = img.max(axis=dim_map["Z"], keepdims=False)

# Use the loaded raw image and vector information and the features and pcs dataframe to create
# the validation plots
optical_flow_calculator.generate_validation_plot(
    out_dir_val,
    img,
    flow_feat_img,
    features_and_pcs,
    quadrants_origin,
    example_points,
    vector_field_channel_map=chan_map,
)


# %% 6. Plot the first two PCs and a single selected random crop with the flow field
# Use the loaded raw image and vector information and the features and pcs dataframe to create
# the validation plots for the first 20 crops:
num_crops_to_plot = 20
# for i, roi in enumerate(rois[:num_crops_to_plot]):
for i in range(num_crops_to_plot):
    print(f"Plotting crop {i+1} / {num_crops_to_plot}")
    example_crop = features_and_pcs.iloc[i]
    example_points = {
        0: {"color": quadrant_colors[1], "quadrant_mean": None, "record": example_crop}
    }
    optical_flow_calculator.generate_validation_plot(
        out_dir_val / f"{dataset_name}",
        img,
        flow_feat_img,
        features_and_pcs,
        quadrants_origin,
        example_points,
    )


# %% Below are some tests:
# first, a quick test of the divergence and curl functions:
diverg_curl_test = (
    optical_flow_calculator.get_divergence_curl_example()
)  # vector field example

# %% Create some synthetic data to test the above vector field plotting:
synth_img = optical_flow_calculator.generate_synthetic_data()

# %% compute flow vectors from the synthetic data
flow_graphs, vx, vy, mean_angle_deg, mean_mag = (
    optical_flow_calculator.compute_synthetic_image_flow_vectors_and_summarize(
        synth_img, delta_t=1
    )
)

# %% Get flow fields from first timepoint of synthetic data
flow_graphs = optical_flow_calculator.get_trimmed_vector_field_map(
    synth_img[0], vx, vy, resolution=10, display=False, return_map=True
)


# %% Plot the synthetic data and the flow field
fig = plt.figure(figsize=(6, 3))
axs = GridSpec(ncols=2, nrows=1, figure=fig)

ax1 = fig.add_subplot(axs[0])
ax1.imshow(flow_graphs, cmap="gray")
ax1.set_title(
    f"""Flow Field (synthetic data)\nMean (True) angle: {mean_angle_deg:.2f} deg (135 deg)\nMean (True) mag: {mean_mag:.2f} px (sqrt(2) px)""",
    fontsize=10,
)

ax2 = fig.add_subplot(axs[1], projection="polar")
assert isinstance(ax2, PolarAxes)
ax2.arrow(
    x=np.deg2rad(mean_angle_deg),
    y=0,
    dx=0,
    dy=0.9,
    head_width=0.1,
    head_length=0.15,
    length_includes_head=True,
    lw=2,
    color="r",
    zorder=10,
)
# create minor ticks on the polar plot
[
    ax2.plot((theta, theta), (0.95, 1), c="k", lw=0.5, zorder=0)
    for theta in np.linspace(0, 2 * np.pi, 36 + 1)
]
ax2.set_ylim(0, 1)
ax2.set_theta_direction(-1)
ax2.set_xlim(-np.pi, np.pi)
ax2.yaxis.set_visible(False)
ax2.set_title("Mean angle orientation", fontsize=10)

plt.tight_layout()
fig.savefig(out_dir_val / f"{dataset_name}_synth_data.png")
plt.close(fig)

# %%
