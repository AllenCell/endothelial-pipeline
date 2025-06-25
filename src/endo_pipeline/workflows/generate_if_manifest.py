# %%
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

from cellsmap.util import dataset_io
from src.endo_pipeline.library.process.image_processing import background_subtract, sum_proj

# %%
IF_CHANNELS = ["NucViolet", "SOX17", "SMAD1", "NR2F2"]
TIMEPOINT = 0  # IF data is only at timepoint 0, no timelapse
NUC_SEG_TYPE = "nuclear_stain_seg_path"
# %%
dataset = "20250509_20X_IF1"
n_positions = dataset_io.get_total_number_of_positions(dataset)
channel_names = dataset_io.get_channel_names(dataset)
# %%
df_dataset_list = []

for position in range(n_positions):
    seg_image = dataset_io.load_nuclei_prediction(
        dataset_name=dataset,
        position=position,
        T=TIMEPOINT,
        nuc_seg_type=NUC_SEG_TYPE,
        dim_order="YX",
    )
    label_image = label(seg_image)

    default_props_list = []
    for prop in regionprops(label_image):
        default_props_list.append(
            {
                "dataset": dataset,
                "position": position,
                "label": prop.label,
                "area": prop.area,
                "centroid_y": prop.centroid[0],
                "centroid_x": prop.centroid[1],
                "eccentricity": prop.eccentricity,
            }
        )
    df_position = pd.DataFrame(default_props_list)

    for channel in channel_names:
        if channel in IF_CHANNELS:
            raw_image = dataset_io.load_dataset_position_as_dask_array(
                dataset, position, channels=[channel]
            )
            background_subtracted_image = background_subtract(raw_image)
            sum_projection = sum_proj(background_subtracted_image, axis=2)
            sum_projection_yx = sum_projection[0, 0, :, :]  # Adjust if dims vary

            props = regionprops(label_image, intensity_image=sum_projection_yx)

            channel_props = []
            for prop in props:
                intensity_image = prop.intensity_image
                channel_props.append(
                    {
                        "label": prop.label,
                        f"{channel}_sum_proj_std": np.std(intensity_image),
                        f"{channel}_total_sum_proj": np.sum(intensity_image),
                        f"{channel}_mean_sum_proj": prop.mean_intensity,
                        f"{channel}_max_sum_proj": prop.max_intensity,
                        f"{channel}_min_sum_proj": prop.min_intensity,
                    }
                )

            df_channel = pd.DataFrame(channel_props)

            # Merge channel-level data
            df_position = pd.merge(
                df_position, df_channel, on="label", how="left", validate="one_to_one"
            )

    df_dataset_list.append(df_position)

# Concatenate all position DataFrames
df = pd.concat(df_dataset_list, ignore_index=True)

# %%
df.head()
# %%
