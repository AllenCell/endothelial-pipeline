# %%
import matplotlib.pyplot as plt
import numpy as np
from bioio import BioImage

from endo_pipeline.configs import (
    get_available_dataset_names,
    get_datasets_in_collection,
    get_zarr_file_for_position,
    load_dataset_config,
)
from endo_pipeline.configs.dataset_io import get_available_channels


# %%
def test_channel_names_consistency() -> None:
    """Test that all reader.channel_names are the same for a given dataset."""
    for dataset_name in get_available_dataset_names():
        channel_names_dict = get_available_channels(dataset_name)

        # Extract all channel names
        all_channel_names = list(channel_names_dict.values())

        # Assert that all channel names are identical
        assert len(all_channel_names) > 0, "No channel names found."
        first_channel_names = all_channel_names[0]
        for channel_names in all_channel_names:
            assert (
                channel_names == first_channel_names
            ), f"Inconsistent channel names found in {dataset_name}: \
                {channel_names} != {first_channel_names}"

    print("All datasets have consistent channel names.")


test_channel_names_consistency()


# %%
def get_channel_crop(
    img: BioImage, t: int, c: int, crop_size: tuple[int, int] = (128, 128)
) -> np.ndarray:
    """
    Get cropped data for a specific channel.

    Parameters
    ----------
    img : BioImage
        The BioImage object containing the image data.
    t : int
        The timepoint index.
    c : int
        The channel index.
    crop_size : tuple[int, int], optional
        The crop size (height, width). Defaults to (128, 128).

    Returns
    -------
    np.ndarray
        The cropped image data as a NumPy array.
    """
    return img.get_image_dask_data("ZYX", T=t, C=c)[
        :,  # Keep all Z-slices
        0 : crop_size[0],  # Crop along Y-axis
        0 : crop_size[1],  # Crop along X-axis
    ]


# %%
# Quickly visualize crop in first position,
# first timepoint of each zarr to confirm channel order is correct
datasets = get_datasets_in_collection("immunofluorescence")
for dataset_name in datasets:
    config = load_dataset_config(dataset_name)
    fmsid = config.fmsid
    barcode = config.barcode
    print(f"dataset: {dataset_name}")
    print(f"fmsid: {fmsid}")
    print(f"barcode: {barcode}")

    for position in config.zarr_positions:
        position_path = get_zarr_file_for_position(config, position)
        img = BioImage(position_path)
        print(f"image shape: {img.shape}")
        n_channels = img.shape[1]
        channel_names = img.channel_names

        # Compute projections for all channels
        channel_projections = []
        for c in range(n_channels):
            channel = get_channel_crop(img, t=0, c=c)
            if c == 1:  # Special case for Channel 1 (BF): use center slice
                projection = channel[channel.shape[0] // 2, :, :]
            else:  # Default: use max projection
                projection = channel.max(axis=0)
            channel_projections.append(projection)

        # Plot all channels
        fig, axes = plt.subplots(1, n_channels, figsize=(6 * n_channels, 6))
        if n_channels == 1:
            axes = [axes]  # Ensure axes is iterable for a single channel
        for c, ax in enumerate(axes):
            ax.imshow(channel_projections[c], cmap="gray")
            ax.set_title(f"{dataset_name} - Channel {c} ({channel_names[c]})")
        plt.show()

# %%
