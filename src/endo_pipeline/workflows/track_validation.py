from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bioio import BioImage
from skimage import measure
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from cellsmap.util.dataset_io import (
    fire_parse_generate_dataset_name_list,
    get_cdh5_classic_segmentation_path,
    get_dataset_info,
    get_tracking_data_filtered,
    ipython_cli_flexecute,
)
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.library.process.general_image_preprocessing import (
    build_analysis_queue,
    get_dim_map,
)


def save_validation_images(
    cell_id: int,
    track_id: int,
    crop: tuple[slice, ...],
    img_arr: np.ndarray,
    seg_arr: np.ndarray,
    out_dir: Path,
    dataset_name: str,
    T: int,
    padding: int = 50,
) -> None:
    expanded_bbox = tuple([slice(max(0, sl.start - padding), sl.stop + padding) for sl in crop])

    crop_img = img_arr[expanded_bbox].squeeze()
    crop_seg = seg_arr[expanded_bbox].squeeze()
    crop_seg_outline = find_boundaries(crop_seg)
    track_of_interest = (crop_seg == cell_id * 1) + (crop_seg_outline > 0) * 2
    raw_img_crop = rescale_intensity(
        np.clip(crop_img, 0, np.percentile(crop_img, 98)), out_range=(0, 1)
    )
    overlay = label2rgb(
        label=track_of_interest,
        image=raw_img_crop,
        bg_label=0,
        colors=["magenta", "cyan"],
    )

    fig, ax = plt.subplots()
    ax.imshow(overlay)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(
        out_dir / f"{dataset_name}_track{track_id}_T{T}.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=120,
    )
    plt.close(fig)

    return


def generate_and_save_validation_images(dframe: pd.DataFrame) -> None:

    # unpack needed variables
    dataset_name = dframe["dataset_name"].unique()[0]
    scene_index = int(dframe["scene_index"].unique()[0])
    position = int(dframe["position"].unique()[0])
    T = int(dframe["T"].unique()[0])
    out_dir = dframe["out_dir"].unique()[0] / f"{dataset_name}/P{position}"

    # get the raw image and segmentation image filepaths
    raw_path = Path(get_dataset_info(dataset_name)["original_path"])
    seg_dir = get_cdh5_classic_segmentation_path(dataset_name, position)
    if seg_dir is None:
        print(f"No segmentation directory found for {dataset_name}. Skipping tracking analysis.")
        return
    seg_path = seg_dir / f"{dataset_name}_P{position}_T{T}.ome.tiff"

    if not seg_path.exists():
        print(f"No segmentation file found for {dataset_name} P{position} at T{T}.")
        return
    else:
        dim_order = "TCZYX"
        dim_map = get_dim_map(dim_order)

        # print(f'- loading raw image {dataset_name} P{position} T{T}...')
        img = BioImage(raw_path)
        img.set_scene(scene_index)
        cdh5_channel = int(get_dataset_info(dataset_name)["channel_488_index"])
        img_dask = img.get_image_dask_data(dim_order, T=T, C=cdh5_channel)
        img_arr = img_dask.max(axis=dim_map["Z"], keepdims=True).squeeze().compute()

        # print(f'- loading segmentation image {dataset_name} P{position} T{T}...')
        seg = BioImage(seg_path)
        seg_arr = seg.get_image_dask_data(dim_order, T=0, C=0).squeeze().compute()

        # get the labels and crops around each segmented region
        props = measure.regionprops(label_image=seg_arr)
        cell_id_to_crop_map = dict([(region.label, region.slice) for region in props])

        # associate the cell ids with their track ids
        cell_ids_with_tracks = dframe[dframe["T"] == T]["label"].unique().tolist()
        cell_id_to_track_id_map = dict(zip(dframe["label"], dframe["track_id"], strict=False))

        # iterate through each cell id in the timepoint and create
        # an overlay of the raw image and the segmentation
        # corresponding to that cell id
        # for cell_id in tqdm(cell_ids_with_tracks, total=len(cell_ids_with_tracks), desc=f'{dataset_name} P{position} T{T} saving track overlays'):
        for cell_id in cell_ids_with_tracks:
            # print(f'-- saving validation images for cell {cell_id}...')
            track_id = cell_id_to_track_id_map[cell_id]
            crop = cell_id_to_crop_map[cell_id]
            validation_subfolder = out_dir / str(track_id)
            Path.mkdir(validation_subfolder, exist_ok=True, parents=True)
            save_validation_images(
                cell_id,
                track_id,
                crop,
                img_arr=img_arr,
                seg_arr=seg_arr,
                out_dir=validation_subfolder,
                dataset_name=dataset_name,
                T=T,
                padding=50,
            )
        return


def main(
    n_proc: int = 1,
    dataset_name: str | None = None,
    t_final: int | None = None,
    min_track_duration: int = 120,
    verbose: bool = False,
) -> None:
    """t_final is really only used for testing purposes."""
    out_dir = Path(get_output_path(Path(__file__).stem, verbose=False))

    dataset_name_list = fire_parse_generate_dataset_name_list(dataset_name)

    analysis_queue = build_analysis_queue(
        dataset_name_list,
        t_final=t_final,
        use_sldy_data=True,
        out_dir=out_dir,
        verbose=verbose,
    )
    analysis_queue_df = pd.DataFrame(analysis_queue)
    for dataset_name in dataset_name_list:
        tracking_df = get_tracking_data_filtered([dataset_name], as_dask=False)
        if t_final is not None:
            tracking_df = tracking_df.query("T < @t_final")

        tracking_df = tracking_df[tracking_df["dataset_name"] == dataset_name]
        analysis_queue_sub = analysis_queue_df[analysis_queue_df["dataset_name"] == dataset_name]
        position_scene_map = dict(
            zip(
                analysis_queue_sub["position"],
                analysis_queue_sub["scene_index"],
                strict=False,
            )
        )
        tracking_df["scene_index"] = tracking_df["position"].transform(
            lambda x: position_scene_map[x]
        )

        tracking_df["out_dir"] = out_dir

        tracking_df = tracking_df[tracking_df["track_duration"] >= min_track_duration]

        nm, df_subset_list = list(
            zip(
                *tracking_df.groupby(["dataset_name", "position", "T"])[
                    [
                        "dataset_name",
                        "position",
                        "scene_index",
                        "T",
                        "track_id",
                        "label",
                        "out_dir",
                    ]
                ],
                strict=False,
            )
        )
        if n_proc > 1:
            if __name__ == "__main__":
                print(f"Multiprocessing {dataset_name}...")
                with Pool(processes=n_proc) as pool:
                    list(
                        tqdm(
                            pool.imap(
                                generate_and_save_validation_images,
                                df_subset_list,
                                chunksize=1,
                            ),
                            total=len(df_subset_list),
                            desc="Timepoints complete (MP)",
                        )
                    )
                    pool.close()
                    pool.join()
                print(f"Finished multiprocessing {dataset_name}.")
        else:
            print(f"Single processing {dataset_name}...")
            for df_group in tqdm(
                df_subset_list,
                total=len(df_subset_list),
                desc="Timepoints complete (1P)",
            ):
                generate_and_save_validation_images(df_group)
            print(f"Finished single processing {dataset_name}.")

    print("\N{MICROSCOPE} Done.")


if __name__ == "__main__":
    ipython_cli_flexecute(main)
