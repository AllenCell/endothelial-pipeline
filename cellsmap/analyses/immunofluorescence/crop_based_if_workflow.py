# %%
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

# %%
DATASET = "20250509_20X_IF1"
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
# %%
# Filter crop outlier with bright puncta outlier
for marker in ["SMAD1", "SOX17"]:

    for feature, xlim in [
        # (f"crop_nuc_mean_intensity_{marker}", None),
        # (f"crop_cyto_mean_intensity_{marker}", None),
        # (f"crop_nuc_to_cyto_mean_ratio_{marker}", None),
        (f"crop_nuc_median_intensity_{marker}", None),
        (f"crop_cyto_median_intensity_{marker}", None),
        (f"crop_nuc_to_cyto_median_ratio_{marker}", None),
    ]:
        plot_intensity_distribution(
            df,
            xlabel=feature,
            dataset=DATASET,
            output_dir=output_dir,
            xlim=xlim,
        )

    index = 2
    row = df.iloc[index]

    seg_mask = get_segmentation_mask_crop(row, resolution_level=RES_LEVEL, channel=0)

    dapi_crop = get_raw_intensity_crop(
        row, resolution_level=RES_LEVEL, channel_name="NucViolet"
    )
    background_subtracted_dapi_crop = background_subtract(dapi_crop, camera_offset=100)
    sum_proj_dapi_img = sum_projection(background_subtracted_dapi_crop)

    raw_crop = get_raw_intensity_crop(
        row, resolution_level=RES_LEVEL, channel_name=marker
    )
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
