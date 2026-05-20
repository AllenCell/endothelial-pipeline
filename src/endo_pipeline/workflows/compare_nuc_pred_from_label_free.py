import logging
import re
from collections.abc import Generator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bioio import BioImage
from scipy.ndimage import distance_transform_edt
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label
from skimage.morphology import dilation, disk
from skimage.segmentation import watershed

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, load_image, load_model
from endo_pipeline.library.process import get_sldy_metadata as sldmd
from endo_pipeline.manifests import (
    get_model_location_for_run,
    get_zarr_location_for_position,
    load_model_manifest,
)
from endo_pipeline.settings import DIMENSION_ORDER

logger = logging.getLogger(__name__)

# NOTE
# because we don't have zarr files for the datasets in the
# test_datasets list, the function dataset_io.get_channel_index
# does not work. Therefore this script is currently broken.
use_sldy_data = True


def plot_and_save_overlays(
    overlay_bf: np.ndarray,
    overlay_nuc: np.ndarray,
    out_dir: Path,
    dataset_name: str,
    timepoint: int,
    filename_suffix: str = "",
) -> None:
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.imshow(overlay_bf)
    ax2.imshow(overlay_nuc)
    ax1.axis("off")
    ax2.axis("off")
    ax1.set_title("Brightfield Std Dev Overlay")
    ax2.set_title("DAPI Overlay")
    plt.tight_layout()
    fig.savefig(
        out_dir
        / dataset_name
        / f"{dataset_name}_T{timepoint}_bf_std_nuc_pred{filename_suffix}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def get_image_data_from_original(dataset_name: str, scenes_to_use: list[str]) -> Generator:
    dataset_config = load_dataset_config(dataset_name)
    dim_order = DIMENSION_ORDER
    projection_dim = "Z"
    projection_axis = DIMENSION_ORDER.index(projection_dim)

    img_path = dataset_config.original_path
    img = BioImage(img_path)
    for scene in scenes_to_use:
        img.set_scene(scene)
        scene_index = img.current_scene_index

        logger.info(dataset_name, img.current_scene)
        channel_names = sldmd.get_channel_name(img.metadata)
        channel_names = [chan.split("/")[0] for chan in channel_names]
        nuc_chan = channel_names.index("405")
        bf_chan = channel_names.index("TL")
        img_dask_arr_nuc = img.get_image_dask_data(dim_order, C=[nuc_chan]).max(
            axis=projection_axis, keepdims=True
        )
        img_dask_arr_bf_std = img.get_image_dask_data(dim_order, C=[bf_chan]).std(
            axis=projection_axis, keepdims=True
        )
        img_dask_arr_bf = img.get_image_dask_data(dim_order, C=[bf_chan])
        bf_focus_index = np.argmin(
            [
                img.std()
                for img in np.split(
                    img_dask_arr_bf,
                    img_dask_arr_bf.shape[projection_axis],
                    axis=projection_axis,
                )
            ]
        )
        img_dask_arr_bf_near_focus = img.get_image_dask_data(
            dim_order, C=[bf_chan], Z=[max(bf_focus_index - 2, 0)]
        )
        yield (
            scene_index,
            img_dask_arr_nuc,
            img_dask_arr_bf_std,
            img_dask_arr_bf_near_focus,
        )


def get_image_data_from_zarr(dataset_name: str) -> Generator:
    dataset_config = load_dataset_config(dataset_name)
    projection_dim = "Z"
    projection_axis = DIMENSION_ORDER.index(projection_dim)

    for position in dataset_config.zarr_positions:
        zarr_loc = get_zarr_location_for_position(dataset_config, position)

        img_dask_arr_nuc = load_image(zarr_loc, channels=["DAPI"])
        img_dask_arr_bf = load_image(zarr_loc, channels=["Brightfield"])

        img_dask_arr_nuc_max = img_dask_arr_nuc.max(axis=projection_axis, keepdims=True)
        img_dask_arr_bf_std = img_dask_arr_bf.std(axis=projection_axis, keepdims=True)

        # Note that overload error existed before switch to using new image
        # loading methods.
        bf_focus_index = np.argmin(
            [
                img.std()
                for img in np.split(
                    img_dask_arr_bf,
                    img_dask_arr_bf.shape[projection_axis],
                    axis=projection_axis,
                )
            ]
        )
        Z_crop = [
            (
                slice(max(bf_focus_index - 2, 0), max(bf_focus_index - 2, 0) + 1, None)
                if dim == projection_axis
                else slice(None)
            )
            for dim in range(img_dask_arr_bf.ndim)
        ]
        img_dask_arr_bf_near_focus = img_dask_arr_bf[Z_crop]
        yield (
            scene_index,
            img_dask_arr_nuc_max,
            img_dask_arr_bf_std,
            img_dask_arr_bf_near_focus,
        )


datasets_to_use = ["20240328_T02_001", "20240328_T01_001"]
scenes_to_use = {
    "20240328_T02_001": [
        "20240328_T02_001-1711659785-  8",
        "20240328_T02_001-1711659785- 24",
        "20240328_T02_001-1711659785- 39",
        "20240328_T02_001-1711659785-990",
    ],
    "20240328_T01_001": [
        "20240328_T01_001-1711663662-276",
        "20240328_T01_001-1711663662-293",
        "20240328_T01_001-1711663662-307",
        "20240328_T01_001-1711663662-322",
        "20240328_T01_001-1711663662-337",
    ],
}

get_t_from_path = lambda x: int(re.findall("T_[0-9]+", x.stem)[-1].split("T_")[-1])
get_s_from_path = lambda x: int(re.findall("S[0-9]+", x.stem)[-1].split("S")[-1])

out_dir = get_output_path(__file__)

# Load the retrained CellPose label-free nuclear prediction model
model_manifest = load_model_manifest("nuc_pred_labelfree")
run_name = "finetuned_20250419"
model_location = get_model_location_for_run(model_manifest, run_name)
model_bf_stdproject = load_model(model_location)

# CytoDL nuclei predictions from Benji:
cytodl_nuc_pred_dir = list(Path(out_dir / "raw_seg").glob("*.tif*"))

# create empty list to store nuclei count data:
nuclei_count_data = []

for dataset_name in datasets_to_use:
    # load the dapi and brightfield data
    if use_sldy_data:
        imgs_to_eval = get_image_data_from_original(dataset_name, scenes_to_use[dataset_name])
    else:
        imgs_to_eval = get_image_data_from_zarr(dataset_name)

    # make a folder per dataset to save comparison images
    out_dir_dataset = out_dir / dataset_name
    Path.mkdir(out_dir_dataset, exist_ok=True, parents=True)

    for (
        scene_index,
        img_dask_arr_nuc,
        img_dask_arr_bf_std,
        img_dask_arr_bf_near_focus,
    ) in imgs_to_eval:

        img_arr_nuc = img_dask_arr_nuc.squeeze().compute()
        img_arr_bf_std = img_dask_arr_bf_std.squeeze().compute()
        img_arr_bf_near_focus = img_dask_arr_bf_near_focus.squeeze().compute()

        # fig, ax = plt.subplots()
        # ax.imshow(rescale_intensity(np.clip(img_arr_nuc, 0, np.percentile(img_arr_nuc, 98))), cmap='gray')
        # ax.axis('off')
        # plt.tight_layout()
        # fig.savefig(out_dir_dataset / f'{dataset_name}_S{scene_index}_nuc.png', bbox_inches='tight', pad_inches=0, dpi=300)

        # function to extract the timepoint from the CytoDL output files:
        if use_sldy_data:
            cytodl_nuc_pred_path = [
                fp
                for fp in cytodl_nuc_pred_dir
                if dataset_name in str(fp.stem) and get_s_from_path(fp) == scene_index
            ]
        else:
            cytodl_nuc_pred_path = [
                fp
                for fp in cytodl_nuc_pred_dir
                if dataset_name in str(fp.stem) and get_t_from_path(fp) == scene_index
            ]
        assert (
            len(cytodl_nuc_pred_path) == 1
        ), f"Expected 1 file for {dataset_name} T{scene_index}, found {len(cytodl_nuc_pred_path)}"
        cytodl_nuc_pred = BioImage(cytodl_nuc_pred_path[0]).get_image_data().squeeze()

        # for timepoint in range(len(img_arr)):
        logger.debug(f"Working on dataset {dataset_name}, {scene_index}...")

        # Use the CellPose model to predict nuclei from the brightfield std dev channel:
        masks_bf_std = model_bf_stdproject.eval(
            img_arr_bf_std,
            channels=[0, 0],
            min_size=50,
            flow_threshold=0.6,
            cellprob_threshold=0,
        )

        overlay_bf = label2rgb(
            label=masks_bf_std[0],
            image=rescale_intensity(img_arr_nuc.squeeze()),
            bg_label=0,
        )
        overlay_nuc = label2rgb(
            label=masks_bf_std[0],
            image=rescale_intensity(
                np.clip(img_arr_nuc.squeeze(), 0, np.percentile(img_arr_nuc.squeeze(), 98))
            ),
            bg_label=0,
        )
        plot_and_save_overlays(
            overlay_bf,
            overlay_nuc,
            out_dir,
            dataset_name,
            scene_index,
            filename_suffix="_cellpose",
        )

        overlay_bf = label2rgb(
            label=cytodl_nuc_pred,
            image=rescale_intensity(img_arr_bf_near_focus),
            bg_label=0,
        )
        overlay_nuc = label2rgb(
            label=cytodl_nuc_pred,
            image=rescale_intensity(
                np.clip(img_arr_nuc.squeeze(), 0, np.percentile(img_arr_nuc.squeeze(), 98))
            ),
            bg_label=0,
        )
        plot_and_save_overlays(
            overlay_bf,
            overlay_nuc,
            out_dir,
            dataset_name,
            scene_index,
            filename_suffix="_cytodl",
        )

        overlay_bf = label2rgb(
            label=masks_bf_std[0].astype(bool) * 1 + cytodl_nuc_pred.astype(bool) * 2,
            image=rescale_intensity(img_arr_bf_near_focus),
            bg_label=0,
            colors=["red", "cyan", "yellow"],
        )
        overlay_nuc = label2rgb(
            label=masks_bf_std[0].astype(bool) * 1 + cytodl_nuc_pred.astype(bool) * 2,
            image=rescale_intensity(np.clip(img_arr_nuc, 0, np.percentile(img_arr_nuc, 98))),
            bg_label=0,
            colors=["red", "cyan", "yellow"],
        )
        plot_and_save_overlays(
            overlay_bf,
            overlay_nuc,
            out_dir,
            dataset_name,
            scene_index,
            filename_suffix="_cellpose_vs_cytodl",
        )

        # do a classic watershed segmentation on the DAPI channel for comparison:
        normd_nuc = rescale_intensity(img_arr_nuc.squeeze(), out_range=(0, 1))
        thresh = apply_hysteresis_threshold(
            normd_nuc, np.percentile(normd_nuc, 80), np.percentile(normd_nuc, 85)
        )
        dist = distance_transform_edt(thresh)
        peaks_img = np.zeros(dist.shape, dtype=bool)
        peaks_img[tuple(zip(*peak_local_max(dist, min_distance=15), strict=False))] = True
        peaks_img = label(dilation(peaks_img, footprint=disk(5)))
        ws = watershed(rescale_intensity(dist, out_range=(1, 0)), markers=peaks_img, mask=thresh)
        overlay_bf3 = label2rgb(
            label=ws, image=rescale_intensity(img_arr_bf_near_focus), bg_label=0
        )  # , colors=['orange'])
        overlay_nuc3 = label2rgb(
            label=ws, image=rescale_intensity(np.clip(normd_nuc, 0, 0.1)), bg_label=0
        )  # , colors=['orange'])
        plot_and_save_overlays(
            overlay_bf3,
            overlay_nuc3,
            out_dir,
            dataset_name,
            scene_index,
            filename_suffix="_classic",
        )

        nuclei_count_data.append(
            {
                "dataset_name": dataset_name,
                "S": scene_index,
                "image_id": "_".join([dataset_name, str(scene_index)]),
                "nuclei_count": np.count_nonzero(np.unique(cytodl_nuc_pred)),
                "method": "CytoDL",
                "fraction_wrt_classic": np.count_nonzero(np.unique(cytodl_nuc_pred))
                / np.count_nonzero(np.unique(ws)),
            }
        )
        nuclei_count_data.append(
            {
                "dataset_name": dataset_name,
                "S": scene_index,
                "image_id": "_".join([dataset_name, str(scene_index)]),
                "nuclei_count": np.count_nonzero(np.unique(masks_bf_std[0])),
                "method": "CellPose",
                "fraction_wrt_classic": np.count_nonzero(np.unique(masks_bf_std[0]))
                / np.count_nonzero(np.unique(ws)),
            }
        )
        nuclei_count_data.append(
            {
                "dataset_name": dataset_name,
                "S": scene_index,
                "image_id": "_".join([dataset_name, str(scene_index)]),
                "nuclei_count": np.count_nonzero(np.unique(ws)),
                "method": "classic",
                "fraction_wrt_classic": np.count_nonzero(np.unique(ws))
                / np.count_nonzero(np.unique(ws)),
            }
        )


nuclei_count_df = pd.DataFrame(nuclei_count_data)
fig, ax = plt.subplots()
sns.barplot(data=nuclei_count_df, x="dataset_name", y="nuclei_count", hue="method", ax=ax)
plt.tight_layout()
fig.savefig(out_dir / "nuclei_counts.png", bbox_inches="tight", dpi=180)

for nm, grp in nuclei_count_df.groupby("dataset_name"):
    fig, ax = plt.subplots()
    sns.barplot(data=grp, x="image_id", y="nuclei_count", hue="method", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    ax.set_title(str(nm))
    plt.tight_layout()
    fig.savefig(out_dir / f"{nm}_nuclei_counts.png", bbox_inches="tight", dpi=180)

for nm, grp in nuclei_count_df.groupby("dataset_name"):
    fig, ax = plt.subplots()
    sns.barplot(data=grp, x="image_id", y="fraction_wrt_classic", hue="method", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    ax.set_title(str(nm))
    plt.tight_layout()
    fig.savefig(out_dir / f"{nm}_nuclei_counts_fractions.png", bbox_inches="tight", dpi=180)
