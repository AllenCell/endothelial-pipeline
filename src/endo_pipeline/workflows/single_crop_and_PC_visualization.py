# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

mpl.rcParams["animation.embed_limit"] = 2**128

from skimage.feature import graycomatrix, graycoprops

from cellsmap.util.manifest_io import get_feature_cols
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import (
    add_crop_index,
    add_zarr_path,
    df_to_array,
    get_manifest_for_dynamics_workflows,
)
from src.endo_pipeline.library.process import crop_io
from src.endo_pipeline.library.visualize import viz_base

# %%
# fit PCA to data
pca = fit_pca()

# %%
# project PCA for one dataset - get numpy array
ds_name = "20250319_20X"
df = get_manifest_for_dynamics_workflows(ds_name, pca=pca)
feat_cols = get_feature_cols(df)
X_t = df_to_array(df, feat_cols)

# for crop visualization, only need dataframe for one dataset
df = df.loc[df["dataset"] == ds_name]
# add zarr path to dataframe
df = add_zarr_path(df)
# add crop index to dataframe
# %%
crop_index = 4

# for labeling the images, get the start_x, start_y coordinates and FOV ID
start_x = df.loc[df["crop_index"] == crop_index]["start_x"].values[0]
start_y = df.loc[df["crop_index"] == crop_index]["start_y"].values[0]
position = df.loc[df["crop_index"] == crop_index]["position"].values[0]

# PC component to visualize
pc = 0
y_lims = [X_t[crop_index, :, pc].min() - 0.02, X_t[crop_index, :, pc].max() + 0.02]

# plot trajectory of PC component PC for selected crop
fig, ax = viz_base.init_plot()
ax.plot(X_t[crop_index, :, pc], "k-", linewidth=2)
ax.set_xlabel("frame number", fontsize=14)
ax.set_ylabel(f"PC{pc+1}", fontsize=14)
ax.set_title(f"Crop {crop_index}: {position}, ({start_x},{start_y})", fontsize=14)

# %%
# get images for crop in the specified time range
frame_range = range(X_t.shape[1])  # timepoints to visualize
imgs = crop_io.get_images_for_crop(df, crop_index, frame_range=frame_range)

# %%
imgs_normed = []
img_max_intensity = []
img_mean_intensity = []
img_standard_deviation = []
haralick_correlation = []
haralick_energy = []
haralick_homogeneity = []
for img in imgs:
    # get maximum raw intensity for each image
    img_max = img.max().compute()
    img_max_intensity.append(img_max)

    # get haralick features
    glcm = graycomatrix(
        img, distances=[1], angles=[0], levels=img_max + 1, symmetric=True, normed=True
    )
    haralick_correlation.append(graycoprops(glcm, "correlation")[0][0])
    haralick_energy.append(graycoprops(glcm, "energy")[0][0])
    haralick_homogeneity.append(graycoprops(glcm, "homogeneity")[0][0])

    # get mean raw intensity for each image
    img_mean = img.mean()
    img_mean_intensity.append(img_mean.compute())

    # get standard deviation for each image
    img_std = img.std()
    img_standard_deviation.append(img_std.compute())

    # normalize image for visualization
    img_ = (img - img_mean) / (img_std)
    imgs_normed.append(img_)
# %%
fig, ax = viz_base.init_plot()
ax.scatter(haralick_correlation, X_t[crop_index, frame_range, PC], c="k", s=5)
ax.set_xlabel("haralick correlation", fontsize=14)
ax.set_ylabel(f"PC{PC+1}", fontsize=14)
ax.set_title(f"Crop {crop_index}: P{FOV_ID}, ({start_x},{start_y})", fontsize=14)

fig, ax = viz_base.init_plot()
ax.scatter(haralick_energy, X_t[crop_index, frame_range, PC], c="k", s=5)
ax.set_xlabel("haralick energy", fontsize=14)
ax.set_ylabel(f"PC{PC+1}", fontsize=14)
ax.set_title(f"Crop {crop_index}: P{FOV_ID}, ({start_x},{start_y})", fontsize=14)

fig, ax = viz_base.init_plot()
ax.scatter(haralick_homogeneity, X_t[crop_index, frame_range, PC], c="k", s=5)
ax.set_xlabel("haralick homogeneity", fontsize=14)
ax.set_ylabel(f"PC{PC+1}", fontsize=14)
ax.set_title(f"Crop {crop_index}: P{FOV_ID}, ({start_x},{start_y})", fontsize=14)

fig, ax = viz_base.init_plot()
ax.scatter(img_mean_intensity, X_t[crop_index, frame_range, PC], c="k", s=5)
ax.set_xlabel("mean image intensity", fontsize=14)
ax.set_ylabel(f"PC{PC+1}", fontsize=14)
ax.set_title(f"Crop {crop_index}: P{FOV_ID}, ({start_x},{start_y})", fontsize=14)

fig, ax = viz_base.init_plot()
ax.scatter(img_standard_deviation, X_t[crop_index, frame_range, PC], c="k", s=5)
ax.set_xlabel("standard deviation of image intensity", fontsize=14)
ax.set_ylabel(f"PC{PC+1}", fontsize=14)
ax.set_title(f"Crop {crop_index}: P{FOV_ID}, ({start_x},{start_y})", fontsize=14)

fig, ax = viz_base.init_plot()
ax.scatter(img_max_intensity, X_t[crop_index, frame_range, PC], c="k", s=5)
ax.set_xlabel("maximum image intensity", fontsize=14)
ax.set_ylabel(f"PC{PC+1}", fontsize=14)
ax.set_title(f"Crop {crop_index}: P{FOV_ID}, ({start_x},{start_y})", fontsize=14)
# %%
# animate specified PC axis trajectory for one crop along with images at each time point
fig, ax = plt.subplots(1, 2, figsize=(15, 5), width_ratios=[1, 2])
subfigs = fig.subfigures(2, 1, wspace=0.07)

ax[1].plot([0, 1], X_t[crop_index, :2, PC], "k-", linewidth=2)
ax[1].set_xlim([frame_range[0], frame_range[-1]])
ax[1].set_ylim(y_lims)
ax[1].set_xlabel("frame number", fontsize=14)
ax[1].set_ylabel(f"PC{PC+1}", fontsize=14)

implot = ax[0].imshow(imgs[0], cmap="gray")
ax[0].set_title(f"Crop {crop_index}: P{FOV_ID}, ({start_x},{start_y})", fontsize=14)


def update(frame):
    ax[1].clear()
    ax[1].plot(
        np.arange(frame_range[0], frame_range[0] + frame),
        X_t[crop_index, frame_range[0] : frame_range[0] + frame, PC],
        "k-",
        linewidth=2,
    )
    ax[1].set_xlim([frame_range[0], frame_range[-1]])
    ax[1].set_ylim(y_lims)
    ax[1].set_xlabel("frame number", fontsize=14)
    ax[1].set_ylabel(f"PC{PC+1}", fontsize=14)

    ax[0].clear()
    implot = ax[0].imshow(imgs[frame], cmap="gray")
    ax[0].set_title(f"Crop {crop_index}: P{FOV_ID}, ({start_x},{start_y})", fontsize=14)

    return ax, implot


# %%
fps = 10
anim = FuncAnimation(fig, update, frames=len(frame_range), interval=fps)
writer = PillowWriter(fps=fps, bitrate=1800)
plt.tight_layout()
plt.show()


anim.save(f"PC{PC+1}_animation_crop{crop_index}.gif", writer=writer)


# %%
