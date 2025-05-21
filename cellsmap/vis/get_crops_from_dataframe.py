# %%
from bioio.writers import OmeTiffWriter
import numpy as np
import pandas as pd
from pathlib import Path
import tifffile

from cellsmap.model_features.generate_image import generate_from_coords
from cellsmap.util import manifest_io, dataset_io, set_output
from cellsmap.vis.get_images import (
    get_zarr_img_for_dataset, 
    get_crop
)
from cellsmap.vis.image_processing import (
    contrast_stretching,
    max_proj,
    std_dev,
)

# %%
def get_original_crops_in_dataframe(df: pd.DataFrame, output_dir: Path) -> None:
    for dataset, df_dataset in df.groupby("dataset"):
        for position, _ in df_dataset.groupby("position"):
            p = dataset_io.extract_P(position)
            img = get_zarr_img_for_dataset(dataset, p)
            for index, row in df.iterrows():
                timepoint = row["frame_number"]
                crop = get_crop(
                    img,
                    channel=None,
                    timepoint=timepoint,
                    start_x=row["start_x"],
                    start_y=row["start_y"],
                    crop_size_x=row["crop_size_x"],
                    crop_size_y=row["crop_size_y"],
                )

                # Extract channels once
                bf_channel = crop[:, 1, :, :, :]  # Brightfield channel
                gfp_channel = crop[:, 0, :, :, :]  # GFP channel

                # Perform operations on the extracted channels
                bf_max_projection = max_proj(bf_channel, 1)
                bf_std_deviation = std_dev(bf_channel, 1)
                gfp_max_projection = max_proj(gfp_channel, 1)

                # contrast stretch
                bf_max_proj_contrast = contrast_stretching(
                    bf_max_projection, method="percentile"
                )
                bf_std_dev_contrast = contrast_stretching(
                    bf_std_deviation, method="percentile"
                )
                gfp_max_proj_contrast = contrast_stretching(
                    gfp_max_projection, method="percentile"
                )

                # Combine the processed images into a multichannel array
                multichannel_image = np.stack(
                    [
                        bf_max_proj_contrast,
                        bf_std_dev_contrast,
                        gfp_max_proj_contrast,
                    ],
                    axis=0,  # Stack along the channel axis
                )

                # Save as a multichannel TIFF
                filename = f"{dataset}_{position}_T{timepoint}_crop_{index}.tiff"
                tifffile.imwrite(output_dir + filename, multichannel_image)

def get_reconstructed_crops_in_dataframe(df: pd.DataFrame, output_dir: Path) -> None:
    '''
    Reconstructs crops from the latent coordinates in the dataframe
    '''
    # get coordinates (feature columns) from the dataframe,
    # convert to list of lists for input into DiffAE model
    num_points = df.shape[0]
    latent_coords = []
    feat_cols = manifest_io.get_feature_cols(df)
    for i in range(num_points):
        latent_coords.append(
            df[feat_cols].iloc[i].tolist()
        )

    # pass into DiffAE model to generate reconstructed crops
    walk_img = generate_from_coords(
        "diffae_04_10", latent_coords
    )  # output is a numpy array: (# coords x 128 x 128), greyscale image
    
    # save out each image in the array as a tiff
    for i in range(num_points):
        # get dataset name frame number, position, and crop index from the dataframe
        ds_name = df["dataset"].iloc[i]
        frame_number = df["frame_number"].iloc[i]
        position = df["position"].iloc[i]
        crop_index = df["crop_index"].iloc[i]
        # filename convention same as get_crops_in_dataframe
        filename = f"{ds_name}_{position}_T{frame_number}_crop_{crop_index}.tiff"
        # save out the reconstructed image
        OmeTiffWriter.save(
            walk_img[i], 
            output_dir + filename, 
            overwrite=True
            )
