# importing required packages
import pickle
from pathlib import Path
from typing import Optional
from pathlib import Path
import tifffile
import napari
import dask.array as da
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import seaborn as sns
import zarr

from tifffile import imread, imwrite
from tqdm import tqdm
from IPython.display import display

from napari.utils.notebook_display import nbscreenshot
from numpy.typing import ArrayLike
from ultrack import track, to_tracks_layer, tracks_to_zarr
from ultrack.utils import labels_to_edges
from ultrack.config import MainConfig

from ultrack.imgproc import normalize
from ultrack.imgproc.segmentation import reconstruction_by_dilation, Cellpose
from ultrack.utils.array import array_apply, create_zarr
from ultrack.utils.cuda import import_module, to_cpu, torch_default_device
from ultrack.imgproc.segmentation import reconstruction_by_dilation, Cellpose
from rich import print
from pyift.shortestpath import watershed_from_minima
from skimage.segmentation import relabel_sequential
from skimage.filters import threshold_otsu
import skimage.morphology as morph
import os
import argparse
try:
    import cupy as xp
except ImportError:
    import numpy as xp

'''
Ultrack script for tracking of nuclei. This expects the different types of segmentations to be saved in different directories
'''


parser = argparse.ArgumentParser()
parser.add_argument("--input_dirs_segs", type=str, default=["/allen/aics/assay-dev/computational/data/endothileal_cell_data/Timelapse_20x_dataset/Version_2/JohnPaul_20240305_flowchange_low_to_high/seg_mips/bf_mip", "/allen/aics/assay-dev/computational/data/endothileal_cell_data/Timelapse_20x_dataset/Version_2/JohnPaul_20240305_flowchange_low_to_high/seg_mips/bf_std"])
parser.add_argument("--input_dir_raw", type=str, default="/allen/aics/assay-dev/computational/data/endothileal_cell_data/Timelapse_20x_dataset/montage_data_trainsets/tubulin_raw")
parser.add_argument("--output_parent_dir", type=str, default="/allen/aics/assay-dev/computational/data/endothileal_cell_data/Timelapse_20x_dataset/processed_datasets")




def remove_background(image: ArrayLike, sigma=15.0) -> ArrayLike:
    """
    Removes background using morphological reconstruction by dilation.
    Reconstruction seeds are an extremely blurred version of the input.

    Parameters
    ----------
    imgs : ArrayLike
        Raw image.

    Returns
    -------
    ArrayLike
        Foreground image.
    """
    image = xp.asarray(image)
    ndi = import_module("scipy", "ndimage")
    seeds = ndi.gaussian_filter(image, sigma=sigma)
    background = reconstruction_by_dilation(seeds, image, iterations=100)
    foreground = np.maximum(image, background) - background
    return to_cpu(foreground)


def watershed_segm(
    frame: ArrayLike,
    aux_labels: ArrayLike,
    min_area: int,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Detects foreground using Otsu threshold and auxiliary labels,
    and execute watershed from minima inside that region.

    Parameters
    ----------
    frame : ArrayLike
        Images as an Y,X array.
    aux_labels : ArrayLike
        Auxiliary labels are used to detect the foreground.
    min_area : int
        Minimum size to be considered a cell.

    Returns
    -------
    ArrayLike
        Watershed segmentation labels.
    """
    disk3 = ndi.generate_binary_structure(frame.ndim, 3)

    frame = frame.astype(np.float32)
    frame = ndi.gaussian_filter(frame, 3.0)
    det = frame > (threshold_otsu(frame) * 0.75)  # making otsu less conservative

    det = np.logical_or(det, np.asarray(aux_labels) > 0)

    det = morph.remove_small_objects(det, min_area)
    det = ndi.binary_closing(det, structure=disk3)

    edt = ndi.distance_transform_edt(det)
    labels = relabel_sequential(watershed_from_minima(-edt, det, H_minima=2.0)[1])[0]

    return labels


def plot_tracks(tracks_df: pd.DataFrame) -> None:
    """Center tracks at their initial position and plot them.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        Tracks datafarame sorted by `track_id` and `t`.

    Returns
    -------
    pd.DataFrame
        Centered dataframe.
    """
    centered_df = tracks_df.copy()
    centered_df[["y", "x"]] = centered_df.groupby(
        "track_id",
        as_index=False,
    )[["y", "x"]].transform(lambda x: x - x.iloc[0])

    # sanity check
    assert (centered_df[centered_df["t"] == 0][["y", "x"]] == 0).all().all()

    pallete = sns.color_palette(["gray"], len(centered_df["track_id"].unique()))
    sns.lineplot(
        data=centered_df,
        x="x",
        y="y",
        hue="track_id",
        palette=pallete,
        legend=False,
        alpha=0.5,
        sort=False,
        estimator=None,
    )
    return centered_df


def created_stacked_timelapse(segmentation_dir):
    filenames = [f for f in os.listdir(segmentation_dir) if f.endswith(".tiff") and not f.startswith(".")]
    ALL_segs=[]
    for i in range(len(filenames)):
        seg = imread(os.path.join(segmentation_dir, filenames[i]))
        ALL_segs.append(seg)
    ALL_segs= np.stack(ALL_segs)
    ALL_segs = np.expand_dims(ALL_segs, axis=-1)
    return ALL_segs


if __name__ == "__main__":

    # dataset_path = "/allen/aics/assay-dev/computational/data/endothileal_cell_data/Timelapse_20x_dataset/montage_data_trainsets/tubulin_FOV_dataset"
    # dataset_raw_path = "/allen/aics/assay-dev/computational/data/endothileal_cell_data/Timelapse_20x_dataset/montage_data_trainsets/tubulin_raw"
    # all_timepoints_segs = [f for f in os.listdir(dataset_path) if f.endswith(".tiff")]
    # all_timepoints_raw = [f for f in os.listdir(dataset_raw_path) if f.endswith(".tiff")]
    # ALL_segs = []
    # create a zarr file using segmentations from all timepoints 
    args = parser.parse_args()
    max_frames = 5
    if len(args.input_dirs_segs)==2:
        filenames_dataset_1 = [f for f in os.listdir(args.input_dirs_segs[0]) if f.endswith(".tiff") and not f.startswith(".")]
        filenames_dataset_2 = [f for f in os.listdir(args.input_dirs_segs[1]) if f.endswith(".tiff") and not f.startswith(".")]
        print("length of filenames dataset 1: ", len(filenames_dataset_1))
        print("length of filenames dataset 2: ", len(filenames_dataset_2))
        assert len(filenames_dataset_1)==len(filenames_dataset_2), "The number of files in the two directories are not the same"
        ALL_seg_timelapse_1 = created_stacked_timelapse(args.input_dirs_segs[0])
        ALL_seg_timelapse_2 = created_stacked_timelapse(args.input_dirs_segs[1])

        chunks = (1, np.shape(ALL_seg_timelapse_1)[1], np.shape(ALL_seg_timelapse_2)[2], 1)
        cellpose_labels_1 = create_zarr(np.shape(ALL_seg_timelapse_1), np.uint16, "cellpose_labels_segnuc_1_v3.zarr", chunks=chunks, overwrite=True) 
        cellpose_labels_2 = create_zarr(np.shape(ALL_seg_timelapse_2), np.uint16, "cellpose_labels_segnuc_2_v3.zarr", chunks=chunks, overwrite=True) 
        cellpose_labels_1[:] = ALL_seg_timelapse_1
        cellpose_labels_2[:] = ALL_seg_timelapse_2
        cellpose_labels_1 = da.from_zarr(cellpose_labels_1, chunks=chunks)
        cellpose_labels_2 = da.from_zarr(cellpose_labels_2, chunks=chunks)

        detection, contours = labels_to_edges([cellpose_labels_1[..., c] for c in range(cellpose_labels_1.shape[-1])] + [cellpose_labels_2[..., c] for c in range(cellpose_labels_2.shape[-1])], sigma=5.0, detection_store_or_path=zarr.TempStore(), edges_store_or_path=zarr.TempStore())




    # for i in range(len(all_timepoints_segs)):
    #     seg = imread(os.path.join(dataset_path, all_timepoints_segs[i]))
    #     ALL_segs.append(seg)
    # ALL_segs= np.stack(ALL_segs)
    # ALL_segs = np.expand_dims(ALL_segs, axis=-1)

    # ALL_raws = []
    # for i in range(len(all_timepoints_raw)):
    #     try:
    #         raw = imread(os.path.join(dataset_raw_path, all_timepoints_raw[i]))[0,:,:]
    #     except:
    #         print("error reading file: ", all_timepoints_raw[i])
    #         NameError
    #     ALL_raws.append(raw)
    # ALL_raws= np.stack(ALL_raws)
    # ALL_raws = np.expand_dims(ALL_raws, axis=-1)




    # raw_zarr = create_zarr(np.shape(ALL_raws), np.uint16, "raw_labels.zarr", chunks=chunks, overwrite=True) 
    # raw_zarr[:] = ALL_raws

    #zarr.save("raw_imgz_std_timelapse_ALL.zarr", raw_zarr)

    # TODO: what is chunks here?
    #array_apply(normalized, out_array=cellpose_labels, func=Cellpose(model_type="cyto2", device=torch_default_device()), axis=(0, 3), tile=False, normalize=False,)  # apply cellpose on images--- axis is dimension of data to apply 
    # cellpose_labels = da.from_zarr(cellpose_labels, chunks=chunks) # convert to dask array # shape is (t, y, x, c)
    # merged_labels = cellpose_labels.max(axis=-1) # # shape is (T, Y, X)
    # ws_labels = create_zarr(imgs.shape, np.int32, "ws_labels.zarr", chunks=chunks, overwrite=True) # 
    # array_apply(
    #     normalized,
    #     cellpose_labels,
    #     out_array=ws_labels,
    #     func=watershed_segm,
    #     min_area=250,
    #     axis=(0, 3),
    # )
    # ws_labels = da.from_zarr(ws_labels, chunks=chunks)
    # # This combines watershed labels and cellpose labels together---- This combines different channel segs w/ watershed---- basically have 2 represented segmentations to choose from now

    # my labels to edges are of shape (5, 1712, 9592)
    #detection, contours = labels_to_edges([cellpose_labels[..., c] for c in range(cellpose_labels.shape[-1])], sigma=5.0, detection_store_or_path=zarr.TempStore(), edges_store_or_path=zarr.TempStore())
    # # labels have a weird channel dont fully understand

    # # [cellpose_labels[..., c] for c in range(cellpose_labels.shape[-1])] + [ws_labels[..., c] for c in range(ws_labels.shape[-1])]




    config = MainConfig()

    n_workers = 4


    # Candidate segmentation parameters
    # config.data_config.database="postgresql"
    # config.data_config.address="5432"
    config.segmentation_config.n_workers = n_workers
    config.segmentation_config.min_area = 250
    config.segmentation_config.min_frontier = 0.99
    config.data_config.address="/allen/aics/assay-dev/users/Goutham/aics_ultrack/ultrack/nuclei_multiseg_big_patch_model_low_to_high_flow_dataset.db"




    # Setting the maximum number of candidate neighbors and maximum spatial distance between cells
    config.linking_config.max_neighbors = 5
    config.linking_config.max_distance = 50
    config.linking_config.n_workers = n_workers

    # Tracking integer linear programming (ILP) parameters
    config.tracking_config.division_weight = -0.01
    config.tracking_config.disappear_weight = -0.01
    config.tracking_config.appear_weight = -0.01

    # ILP processing window size.
    # It reduces memory usage while asserting continuity in the tracks.
    config.tracking_config.window_size = 15
    config.tracking_config.overlap_size = 3
    config.tracking_config.solution_gap = 0.01

    print("ultrack config")
    print(config)

    track(
        config,
        detection=detection,
        edges=contours,
        overwrite=True
    )

    tracks_df, lineage_graph = to_tracks_layer(config) # lineage graph is a dict, tracks_df is a dataframe
    tracking_labels = tracks_to_zarr(config, tracks_df) # 

    tracks_df.to_csv(os.path.join(args.output_parent_dir, "JohnPaul_20240305_flowchange_low_to_high_nuclei.csv"))
    #tracks_df.to_csv("tracks_dataframe_multiseg_nuclei.csv")
    zarr.save(os.path.join(args.output_parent_dir, "JohnPaul_20240305_flowchange_low_to_high_nuclei.zarr"), tracking_labels)
    

    