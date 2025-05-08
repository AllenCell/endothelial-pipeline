# %%
from cellsmap.analyses.immunofluorescence.add_if_cols import add_if_cols_to_df
from cellsmap.analyses.immunofluorescence.if_feature_extraction import (
    background_subtract,
    get_raw_intensity_crop,
    get_segmentation_mask_crop,
    sum_projection,
    sum_projection_in_mask,
    sum_projection_not_in_mask,
)
from cellsmap.analyses.immunofluorescence.plots import (
    plot_intensity_distribution,
    projection_image,
)
from cellsmap.util import manifest_io, set_output

# %%
DATASET = "20250122_SMAD1"
MARKER = "SMAD1"
VISUALIZATION = False
SAVE_MANIFEST = False

df_manifest = manifest_io.get_diffae_manifest(DATASET)

# Filter FOVs that crop nuclei in Z
filter_positions = ["P0", "P1", "P2", "P3", "P4"]
df_filtered = df_manifest[~df_manifest["position"].isin(filter_positions)]
df_filtered = df_filtered.reset_index(drop=True)  # Reset the index and drop the old one

df = add_if_cols_to_df(
    df_filtered,
    marker=MARKER,
    nuclear_seg_channel=2,
    antibody_channel=3,
    dapi_channel=2,
)

# Filter crop with bright puncta
df = df[df["norm_cyto_intensity_SMAD1"] < 5]

# # Save the updated DataFrame to a new CSV file
if SAVE_MANIFEST:
    output_dir = set_output.get_output_path("smad1_analysis")
    df.to_csv(output_dir + f"{DATASET}_if_results.csv", index=False)


# %% Visualize resulting images and features
if VISUALIZATION:
    # Plot the distribution for each feature
    for feature in [
        f"norm_crop_intensity_{MARKER}",
        f"norm_nuc_intensity_{MARKER}",
        f"norm_cyto_intensity_{MARKER}",
        f"norm_nuc_to_cyto_ratio_{MARKER}",
    ]:
        plot_intensity_distribution(df, feature)

    # Plot the sum projections for a row of the DataFrame
    row = df.iloc[0]

    seg_mask = get_segmentation_mask_crop(row, resolution_level=0, channel=2)

    dapi_crop = get_raw_intensity_crop(row, resolution_level=0, channel=2)
    background_subtracted_dapi_crop = background_subtract(dapi_crop, camera_offset=100)
    sum_proj_dapi_img = sum_projection(background_subtracted_dapi_crop)
    sum_projection_dapi_in_nuclei = sum_projection_in_mask(sum_proj_dapi_img, seg_mask)
    sum_projection_dapi_not_in_nuclei = sum_projection_not_in_mask(
        sum_proj_dapi_img, seg_mask
    )

    raw_crop = get_raw_intensity_crop(row, resolution_level=0, channel=3)
    background_subtracted_crop = background_subtract(raw_crop, camera_offset=100)
    sum_proj_img = sum_projection(background_subtracted_crop)
    sum_proj_img_in_nuclei = sum_projection_in_mask(sum_proj_img, seg_mask)
    sum_projection_not_in_nuclei = sum_projection_not_in_mask(sum_proj_img, seg_mask)

    projection_image(
        sum_proj_img, seg_mask, sum_proj_img_in_nuclei, sum_projection_not_in_nuclei
    )
    projection_image(
        sum_proj_dapi_img,
        seg_mask,
        sum_projection_dapi_in_nuclei,
        sum_projection_dapi_not_in_nuclei,
    )
# %%
