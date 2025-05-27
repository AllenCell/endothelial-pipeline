# %%
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity

from cellsmap.analyses.immunofluorescence.if_support.add_if_cols import (
    add_if_cols_to_df,
    get_channels_for_if_processing,
)
from cellsmap.analyses.immunofluorescence.if_support.if_feature_extraction import (
    background_subtract,
    get_raw_intensity_crop,
    get_segmentation_mask_crop,
    sum_projection,
)
from cellsmap.analyses.immunofluorescence.if_support.plots import (
    plot_intensity_distribution,
    projection_image,
)
from cellsmap.util import manifest_io, set_output
from cellsmap.vis.image_processing import infocus_slice

# %%
DATASET = "20250509_20X_IF9"  # high flow 24 hour fixed point with the closest flow to ref dataset
RES_LEVEL = 0
output_dir = set_output.get_output_path("immunoflourescence_analysis")
df = manifest_io.get_diffae_manifest(DATASET)

channels = get_channels_for_if_processing(DATASET)
# %%
for channel in channels:
    df = add_if_cols_to_df(
        df,
        channel_name=channel,
        resolution_level=RES_LEVEL,
    )
# %% plot intensity distributions
for marker in ["SMAD1"]:  # "SOX17" "NR2F2"
    for feature, xlim in [
        (f"crop_nuc_mean_intensity_{marker}", None),
        (f"crop_cyto_mean_intensity_{marker}", None),
        (f"crop_nuc_to_cyto_mean_ratio_{marker}", None),
    ]:
        plot_intensity_distribution(
            df,
            xlabel=feature,
            dataset=DATASET,
            output_dir=output_dir,
            xlim=xlim,
        )
# %%
index = 5
row = df.iloc[index]

seg_mask = get_segmentation_mask_crop(row, resolution_level=RES_LEVEL, channel=0)

dapi_crop = get_raw_intensity_crop(
    row, resolution_level=RES_LEVEL, channel_name="NucViolet"
)
background_subtracted_dapi_crop = background_subtract(dapi_crop, camera_offset=100)
sum_proj_dapi_img = sum_projection(background_subtracted_dapi_crop)

raw_crop = get_raw_intensity_crop(row, resolution_level=RES_LEVEL, channel_name=marker)
background_subtracted_crop = background_subtract(raw_crop, camera_offset=100)
sum_proj_img = sum_projection(background_subtracted_crop)

projection_image(
    sum_proj_img,  # marker
    seg_mask,
    row.dataset,
    row.position,
    row.start_x,
    row.start_y,
    str(output_dir),
)
projection_image(
    sum_proj_dapi_img,  # DAPI
    seg_mask,
    row.dataset,
    row.position,
    row.start_x,
    row.start_y,
    str(output_dir),
)
# %%
# generate some example images for SAC 2025
# NOTE I (SERGE) WROTE THIS SHORTLY BEFORE SAC SLIDES WERE DUE;
# IT NEEDS TO BE REFACTORED AND BETTER INTEGRATED WITH OTHER CODE
output_dir_SAC = Path(output_dir) / "SAC_2025_IF_example_images"
output_dir_SAC.mkdir(exist_ok=True, parents=True)

fig, ax = plt.subplots()
ax.imshow(sum_proj_img, cmap="gray")
ax.axis("off")
fig.savefig(
    Path(output_dir_SAC) / f"SMAD1_projection_index{index}.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)
plt.close(fig)

fig, ax = plt.subplots()
sum_proj_img_contr_adj = rescale_intensity(
    np.clip(
        sum_proj_img,
        a_min=np.percentile(sum_proj_img, 2),
        a_max=np.percentile(sum_proj_img, 98),
    )
)
overlay = label2rgb(label=seg_mask, image=sum_proj_img_contr_adj)
ax.imshow(overlay)
ax.axis("off")
fig.savefig(
    Path(output_dir_SAC) / f"SMAD1_with_mask_projection_segmentation_index{index}.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)
plt.close(fig)

bf_crop = get_raw_intensity_crop(row, resolution_level=RES_LEVEL, channel_name="BF")
bf_slice = infocus_slice(bf_crop)
fig, ax = plt.subplots()
ax.imshow(bf_slice, cmap="gray")
ax.axis("off")
fig.savefig(
    Path(output_dir_SAC) / f"BF_infocus_slice_segmentation_index{index}.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)
plt.close(fig)

std_proj = np.std(bf_crop, axis=0)
fig, ax = plt.subplots()
ax.imshow(std_proj, cmap="gray")
ax.axis("off")
fig.savefig(
    Path(output_dir_SAC) / f"BF_stdev_proj_segmentation_index{index}.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)
plt.close(fig)


# %%
