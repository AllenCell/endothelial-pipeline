# %%
from cellsmap.analyses.immunofluorescence.add_if_cols import add_if_cols_to_df
from cellsmap.analyses.immunofluorescence.if_feature_extraction import (
    background_subtract,
    get_raw_intensity_crop,
    get_segmentation_mask_crop,
    sum_projection,
)
from cellsmap.analyses.immunofluorescence.plots import (
    plot_intensity_distribution,
    projection_image,
)
from cellsmap.util import manifest_io, set_output

# from cellsmap.util.manifest_preprocessing.fms_upload import

# %%
DATASET = "20250122_SMAD1"
MARKER = "SMAD1"
output_dir = set_output.get_output_path("smad1_analysis")
df_manifest = manifest_io.get_diffae_manifest(DATASET)

# %% Filter FOVs that crop nuclei in Z
df_filtered = df_manifest[df_manifest["position"].isin(["P5", "P6", "P7", "P8", "P9"])]
df_filtered = df_filtered.reset_index(drop=True)  # Reset the index and drop the old one

# %%
df = add_if_cols_to_df(
    df_filtered,
    marker=MARKER,
    nuclear_seg_channel=0,
    antibody_channel=3,
    dapi_channel=2,
)
# %%
# Filter crop outlier with bright puncta outlier
df = df[df[f"cyto_mean_intensity_{MARKER}"] < 3000]
# # Save the updated DataFrame to a new CSV file
# df.to_csv(output_dir + f"{DATASET}_IF_results.csv", index=False)


# %% Visualize the intensity distributions
for feature, xlim in [
    (f"nuc_mean_intensity_{MARKER}", 2600),
    (f"cyto_mean_intensity_{MARKER}", 2600),
    (f"nuc_median_intensity_{MARKER}", 2600),
    (f"cyto_median_intensity_{MARKER}", 2600),
    (f"nuc_to_cyto_mean_ratio_{MARKER}", 4),
    (f"nuc_to_cyto_median_ratio_{MARKER}", 4),
]:
    plot_intensity_distribution(
        df, xlabel=feature, dataset=DATASET, output_dir=output_dir, xlim=xlim, ylim=13
    )
# %% Visualize resulting images and intensity
index = 2
row = df.iloc[index]

seg_mask = get_segmentation_mask_crop(row, resolution_level=0, channel=0, binary=False)

dapi_crop = get_raw_intensity_crop(row, resolution_level=0, channel=2)
background_subtracted_dapi_crop = background_subtract(dapi_crop, camera_offset=100)
sum_proj_dapi_img = sum_projection(background_subtracted_dapi_crop)

raw_crop = get_raw_intensity_crop(row, resolution_level=0, channel=3)
background_subtracted_crop = background_subtract(raw_crop, camera_offset=100)
sum_proj_img = sum_projection(background_subtracted_crop)

projection_image(
    sum_proj_img,
    seg_mask,
    row.dataset,
    row.position,
    row.start_x,
    row.start_y,
    str(output_dir),
)
projection_image(
    sum_proj_dapi_img,
    seg_mask,
    row.dataset,
    row.position,
    row.start_x,
    row.start_y,
    str(output_dir),
)
# %%
