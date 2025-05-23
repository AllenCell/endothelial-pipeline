from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from bioio import BioImage
from tqdm import tqdm

from cellsmap.util.dataset_io import get_segmentation_features_manifest
from cellsmap.util.general_image_preprocessing import (
    get_default_dim_order,
    save_image_output,
    sequence_to_scalar,
)
from cellsmap.util.set_output import get_output_path
from cellsmap.vis.timelapse_feature_explorer.generate_tfe_dataset import (
    generate_tfe_dataset,
)

# from skimage.morphology import dilation, disk


def get_crop_bounds(
    crop_center: tuple[float, ...],
    crop_size: int,
) -> tuple:
    """
    Get the crop bounds for a given centroid and crop size.
    Parameters
    ----------
    centroid_x : float
        The x coordinate of the centroid.
    centroid_y : float
        The y coordinate of the centroid.
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
    bounds : tuple[slice, ...]
        The crop bounds as a tuple of slices with same length
        as img_arr has dimensions (len(bounds) == img_arr.ndim)
        Format is tuple(slice(start, stop), slice(start, stop), ...)
    bounds_value : int
        The value to assign to the crop bounds in the image array.

    Returns
    -------
    np.ndarray
        The image array with the crop bounds drawn on it.
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


# define list of datasets, position, and track id to process
dataset_name = "20241120_20X"
position = 0
track_id = 1852
verbose = False
dim_order = get_default_dim_order()
crop_size = 256

out_dir = Path(get_output_path(Path(__file__).stem, verbose=False))

out_dir_seg_and_box = out_dir / "tfe_example_track" / dataset_name / f"P{position}"
out_dir_seg_and_box.mkdir(exist_ok=True, parents=True)

out_dir_box_only = (
    out_dir / "tfe_example_track_box_only" / dataset_name / f"P{position}"
)
out_dir_box_only.mkdir(exist_ok=True, parents=True)


def generate_crop_outline_images() -> None:
    # load segmentation features table associated with the track id
    # and subset the position of interest
    seg_feat_df = get_segmentation_features_manifest([dataset_name]).query(
        "position == @position"
    )
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
    phys_px_sizes = img.physical_pixel_sizes

    for tp in tqdm(timepoints):
        # TODO def draw_crop_bounds_around_segmentation_of_interest(): everything below
        # load_segmentation_image(dataset_name, position, timepoint)
        seg_feat_df_at_T = seg_feat_df_subset.query("T == @tp")

        # initialize a blank image
        example_track_img_arr = np.zeros(
            shape=tuple(img_shape[dim] for dim in dim_order),
            dtype=np.uint16,
        )

        # if there is no segmentation for the track of interest at
        # this timepoint, then the table will be empty, so make an
        # array of zeros
        if seg_feat_df_at_T.empty:
            example_track_img_arr
            crop_box_img_arr = example_track_img_arr.copy()
        else:
            # load the segmentation image
            fp = Path(
                sequence_to_scalar(seg_feat_df_at_T["cdh5_classic_segmentation_path"])
            )
            img = BioImage(fp)
            img_arr = img.get_image_data(dim_order)

            # get the centroid of the segmentation
            centroid_x = sequence_to_scalar(seg_feat_df_at_T["centroid_X"])
            centroid_y = sequence_to_scalar(seg_feat_df_at_T["centroid_Y"])

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

            seg_id = sequence_to_scalar(seg_feat_df_at_T["label"])

            # draw the crop bounds on the initially blank image
            example_track_img_arr = draw_crop_bounds(
                example_track_img_arr,
                crop_bounds,
                bounds_value=seg_id,
            )
            # save 2 images: one with just the crop bounds and another
            # with both the crop bounds and the segmentation
            crop_box_img_arr = example_track_img_arr.copy()

            # draw the segmentation on the image with the crop bounds
            example_track_img_arr[(img_arr == seg_id)] = seg_id
            # when saving the image I need to know what the dimensions of
            # the array being passed to save_imgage_output are, so I will
            # get those from the squeezed image with the line below
            # squeezed_img_dim_order = "".join(
            #     [dim_order[i] for i in range(img_arr.ndim) if img_arr.shape[i] > 1]
            # )

        # save the image
        image_name = f"{dataset_name}_P{position}_T{tp}_track_{track_id}"
        out_filename = f"{image_name}.ome.tif"
        out_path = out_dir_seg_and_box / out_filename
        save_image_output(
            out_path=out_path,
            images=[example_track_img_arr],
            images_metadata={
                "image_name": image_name,
                "dim_order": dim_order,
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
                "dim_order": dim_order,
                "channel_names": ["segmentation"],
                "channel_colors": [(255, 255, 255)],
                "physical_pixel_sizes": phys_px_sizes,
            },
        )


out_dir_tfe = (
    "//allen/aics/endothelial/morphological_features/timelapse_feature_explorer"
)

generate_tfe_dataset(
    dataset=dataset_name,
    position=position,
    output_dir=out_dir,
    source_dir=out_dir_box_only,
    backdrops=True,
)
