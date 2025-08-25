from pathlib import Path

import numpy as np
from bioio import BioImage
from skimage.morphology import dilation, disk
from tqdm import tqdm

from endo_pipeline.configs.dataset_io import (
    get_segmentation_features_manifest,
    ipython_cli_flexecute,
)
from endo_pipeline.io import get_output_path
from endo_pipeline.library.process.general_image_preprocessing import (
    get_default_dim_order,
    save_image_output,
    sequence_to_scalar,
)
from endo_pipeline.library.visualize.timelapse_feature_explorer.generate_tfe_dataset import (
    generate_tfe_dataset,
)


def get_crop_bounds(
    crop_center: tuple[float, ...],
    crop_size: int,
) -> tuple:
    """
    Get the crop bounds for a given centroid and crop size.

    Parameters
    ----------
    crop_center : tuple[float, ...]
        The center of the crop in each dimension.
    crop_size : int
        The size of the crop.

    Returns
    -------
    tuple
        The crop bounds in the format (x_start, x_end, y_start, y_end).
    """
    half_crop_size = crop_size / 2
    crop_bounds = tuple(
        (int(center_of_dim - half_crop_size), int(center_of_dim + half_crop_size))
        for center_of_dim in crop_center
    )
    return crop_bounds


def draw_crop_bounds(
    img_arr: np.ndarray,
    crop_bounds: tuple[slice, ...],
    bounds_value: int = 1,
) -> np.ndarray:
    """
    Draw the crop bounds on the image array.

    Parameters
    ----------
    img_arr : np.ndarray
        The image array to draw the crop bounds on.
    crop_bounds : tuple[slice, ...]
        The crop bounds as a tuple of slices with the same length
        as the number of dimensions in `img_arr` (i.e., `len(bounds) == img_arr.ndim`).
        Format: tuple(slice(start, stop), slice(start, stop), ...).
    bounds_value : int
        The value to assign to the crop bounds in the image array.

    Returns
    -------
    np.ndarray
        The modified image array with the crop bounds drawn on it.
    """
    # make a copy of the image array to avoid modifying the original
    img_arr = img_arr.copy()

    # iterate through the dimensions in crop_bounds
    for i, bounds in enumerate(crop_bounds):
        # get the start and end indices for the crop bounds
        # which will become an edge / line
        for edge in (bounds.start, bounds.stop):
            # if the edge has a value then...
            if edge:
                # create a line
                line = list(crop_bounds)
                line[i] = slice(edge, edge + 1)
                # draw the line on img_arr using the bounds value
                img_arr[tuple(line)] = bounds_value
    return img_arr


def get_out_subdirs(out_dir: Path, dataset_name: str, position: int) -> tuple[Path, Path]:
    """
    Get the output subdirectories for fluorescence images overlaid
    with segmentation and crop box, and for crop box-only output images.

    Parameters
    ----------
    out_dir : Path
        The base output directory where the subdirectories will be created.
    dataset_name : str
        The name of the dataset.
    position : int
        The position index in the dataset.

    Returns
    -------
    tuple[Path, Path]
        A tuple containing:
        - The output directory for images with segmentation and crop box overlays.
        - The output directory for images with crop box-only overlays.
    """
    out_dir_seg_and_box = out_dir / "tfe_example_track" / dataset_name / f"P{position}"
    out_dir_seg_and_box.mkdir(exist_ok=True, parents=True)

    out_dir_box_only = out_dir / "tfe_example_track_box_only" / dataset_name / f"P{position}"
    out_dir_box_only.mkdir(exist_ok=True, parents=True)

    return out_dir_seg_and_box, out_dir_box_only


def generate_crop_outline_images(
    out_dir: Path,
    dataset_name: str = "20241120_20X",
    position: int = 0,
    track_id: int = 1852,
    crop_size: int = 256,
    dim_order: str = "TCZYX",
) -> None:
    """
    Generate crop outline images for a specific track in a dataset.

    This function creates images with crop outlines for a given track ID in a dataset.

    Parameters
    ----------
    out_dir : Path
        The directory where the generated crop outline images will be saved.
    dataset_name : str, optional
        The name of the dataset to process. Defaults to "20241120_20X".
    position : int, optional
        The position index in the dataset to process. Defaults to 0.
    track_id : int, optional
        The track ID for which to generate crop outline images. Defaults to 1852.
    crop_size : int, optional
        The size of the crop (in pixels). Defaults to 256.
    dim_order : str, optional
        The dimension order of the dataset (e.g., "TCZYX"). Defaults to "TCZYX".

    Returns
    -------
    None
    """
    # create the output directories
    dim_order = get_default_dim_order()

    out_dir = get_output_path(__file__)

    out_dir_seg_and_box, out_dir_box_only = get_out_subdirs(out_dir, dataset_name, position)

    # load segmentation features table associated with the track id
    # and subset the position of interest
    seg_feat_df = get_segmentation_features_manifest([dataset_name]).query("position == @position")
    img_shape = {
        "T": 1,
        "C": 1,
        "Z": 1,
        "Y": sequence_to_scalar(seg_feat_df["image_size_y"]),
        "X": sequence_to_scalar(seg_feat_df["image_size_x"]),
    }

    timepoints = seg_feat_df["T"].unique()

    seg_feat_df_subset = seg_feat_df.query("track_id == @track_id")

    seg_feat_df_subset.to_csv(
        out_dir / f"{dataset_name}_P{position}_track_{track_id}.csv", index=False
    )

    # get the image pixel sizes
    img = BioImage(seg_feat_df_subset.iloc[0]["cdh5_classic_segmentation_path"])
    img_arr = img.get_image_data(dim_order)
    phys_px_sizes = img.physical_pixel_sizes

    # when saving the image I need to know what the dimensions of
    # the array being passed to save_imgage_output are, so I will
    # get those from the squeezed image with the line below
    squeezed_img_dim_order = "".join(
        [dim_order[i] for i in range(img_arr.ndim) if img_arr.shape[i] > 1]
    )

    for tp in tqdm(
        timepoints,
        desc=f"{dataset_name} P{position} saving images",
        unit="timepoint",
    ):
        # load_segmentation_image(dataset_name, position, timepoint)
        seg_feat_df_at_t = seg_feat_df_subset.query("T == @tp")

        # initialize a blank image
        example_track_img_arr = np.zeros(
            shape=tuple(img_shape[dim] for dim in dim_order),
            dtype=np.uint16,
        )

        # if there is no segmentation for the track of interest at
        # this timepoint, then the table will be empty, so make an
        # array of zeros
        if seg_feat_df_at_t.empty:
            example_track_img_arr = example_track_img_arr.squeeze()
            crop_box_img_arr = example_track_img_arr.copy().squeeze()
        else:
            # load the segmentation image
            fp = Path(sequence_to_scalar(seg_feat_df_at_t["cdh5_classic_segmentation_path"]))
            img = BioImage(fp)
            img_arr = img.get_image_data(dim_order)

            # get the centroid of the segmentation
            centroid_x = sequence_to_scalar(seg_feat_df_at_t["centroid_X"])
            centroid_y = sequence_to_scalar(seg_feat_df_at_t["centroid_Y"])

            # get the crop bounds
            xlims, ylims = get_crop_bounds(
                crop_center=(centroid_x, centroid_y),
                crop_size=crop_size,
            )
            crop_bounds_map = {
                "T": slice(None),
                "C": slice(None),
                "Z": slice(None),
                "Y": slice(*ylims),
                "X": slice(*xlims),
            }
            crop_bounds = tuple(crop_bounds_map[dim] for dim in dim_order)

            seg_id = sequence_to_scalar(seg_feat_df_at_t["label"])

            # draw the crop bounds on the initially blank image
            example_track_img_arr = dilation(
                draw_crop_bounds(
                    example_track_img_arr,
                    crop_bounds,
                    bounds_value=seg_id,
                ).squeeze(),
                footprint=disk(3),
            )
            # save 2 images: one with just the crop bounds and another
            # with both the crop bounds and the segmentation
            crop_box_img_arr = example_track_img_arr.copy()

            # draw the segmentation on the image with the crop bounds
            example_track_img_arr[(img_arr.squeeze() == seg_id)] = seg_id

        # save the image
        image_name = f"{dataset_name}_P{position}_T{tp}"
        out_filename = f"{image_name}.ome.tiff"
        out_path = out_dir_seg_and_box / out_filename
        save_image_output(
            out_path=out_path,
            images=[example_track_img_arr],
            images_metadata={
                "image_name": image_name,
                "dim_order": squeezed_img_dim_order,
                "channel_names": ["segmentation"],
                "channel_colors": [(255, 255, 255)],
                "physical_pixel_sizes": phys_px_sizes,
            },
        )

        out_path = out_dir_box_only / out_filename
        save_image_output(
            out_path=out_path,
            images=[crop_box_img_arr],
            images_metadata={
                "image_name": image_name,
                "dim_order": squeezed_img_dim_order,
                "channel_names": ["segmentation"],
                "channel_colors": [(255, 255, 255)],
                "physical_pixel_sizes": phys_px_sizes,
            },
        )


def generate_tfe_dataset_of_single_track(
    out_dir: Path | None = None,
    dataset_name: str = "20241120_20X",
    position: int = 0,
    track_id: int = 1852,
    crop_size: int = 256,
    verbose: bool = True,
) -> None:
    """Generate a TFE dataset for a single track in a specified dataset and position.

    Parameters
    ----------
    out_dir : Path | None
        The output directory to save the TFE dataset. If None, uses the default
        output path for the project based on the script name.
    dataset_name : str
        The name of the dataset to generate the TFE dataset for.
    position : int
        The position within the dataset to generate the TFE dataset for.
        Note that this takes an integer but the folder itself has a "P"
        prefix, so position 0 will be saved in a folder called "P0".
    track_id : int
        The track ID to generate the TFE dataset for.
        The TFE dataset folder name will be `"{dataset}_P{position}_tid{track_id}"`.
    crop_size : int
        The size of the crop to generate around the track.
        The default is 256 (a square with side length of 256px)
        and works on images at full resolution (i.e. bin_level 0).
    verbose : bool
        Whether or not to print the output directory.

    Returns
    -------
    Nothing; saves the TFE dataset to the specified output directory.
    This function will take about 20 minutes to complete.
    """
    if out_dir is None:
        out_dir = get_output_path(__file__)

    generate_crop_outline_images(out_dir, dataset_name, position, track_id, crop_size)

    _, out_dir_box_only = get_out_subdirs(out_dir, dataset_name, position)

    generate_tfe_dataset(
        dataset=dataset_name,
        position=position,
        output_dir=out_dir,
        source_dir=out_dir_box_only,
        backdrops=True,
        output_dir_suffix=f"tid{track_id}",
    )


if __name__ == "__main__":
    ipython_cli_flexecute(generate_tfe_dataset_of_single_track)
