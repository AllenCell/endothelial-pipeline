from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from skimage.measure import regionprops
from src.endo_pipeline.library.general_image_preprocessing import (
    build_analysis_queue,
    get_default_dim_order,
)
from tqdm import tqdm

from cellsmap.util.dataset_io import (
    concatenate_and_save_feature_tables,
    fire_parse_generate_dataset_name_list,
    ipython_cli_flexecute,
    load_cdh5_classic_segmentation,
    load_dataset_position_as_dask_array,
    load_nuclei_prediction,
    save_git_versioning_info,
)
from cellsmap.util.set_output import get_output_path


def get_and_save_nuclei_features_arg_unpacker(args: dict) -> None:
    dataset_name = args["dataset_name"]
    position = args["position"]
    T = args["T"]
    out_dir = args["output_dir"]
    save_output = args["save_output"]
    get_and_save_nuclei_features(dataset_name, position, T, out_dir, save_output)


def get_and_save_nuclei_features(
    dataset_name: str,
    position: int,
    T: int,
    out_dir: Path,
    save_output: bool = True,
) -> None:

    nuc_props_df = get_nuclei_features_from_dataset_at_T(dataset_name, position, T)

    out_subdir = out_dir / dataset_name / f"P{position}"
    out_subdir.mkdir(exist_ok=True, parents=True)
    out_path = out_subdir / f"{dataset_name}_P{position}_T{T}_nuclei_features.tsv"
    if save_output:
        nuc_props_df.to_csv(out_path, sep="\t", index=False)


def get_nuclei_features_from_image(
    cdh5_seg: np.ndarray,
    nuc_seg: np.ndarray,
    fluorescence_images: list[np.ndarray],
    fluor_img_names: list[str] | None = None,
    seg_dim_order: str = "YX",
) -> pd.DataFrame:
    """
    Extracts features from nuclei segmentations and their overlap with cell segmentations.

    Parameters
    ----------
    cdh5_seg: ndarray
        Image of the cell segmentations based on Cdh5.
    nuc_seg: ndarray:
        Image of the nuclei segmentations.
    fluorescence_images: list[np.ndarray]:
        List of fluorescence images to get intensity information for each
        of the nuclei segmentation regions. In this workflow each image
        is a channel from the raw image.
    fluor_img_names: list[str] | None:
        Names of the fluorescence images. If None, defaults to "Channel_0", "Channel_1", etc.
    nuclei_ambiguity_threshold (float):
        Threshold for determining if a nucleus segmentation overlaps a cell
        segmentation enough to be kept.
    Returns
    -------
        pd.DataFrame: DataFrame with extracted features.
    """
    # just in case make sure that the number of dimensions provided
    # in seg_dim_order matches that of the images
    for img in [
        cdh5_seg,
        nuc_seg,
        *fluorescence_images,
    ]:
        assert len(seg_dim_order) == img.ndim

    # assign default names to fluorescence images if not provided
    channel_indices = range(len(fluorescence_images))
    if fluor_img_names is None:
        fluor_img_names = [f"Channel{i}" for i in channel_indices]

    # get intensities in the segmented nuclei regions
    # for each channel
    nuc_props_on_intens = dict()
    for i in range(len(fluorescence_images)):
        nuc_props_on_intens[fluor_img_names[i]] = {
            prop.label: prop
            for prop in regionprops(
                label_image=nuc_seg, intensity_image=fluorescence_images[i]
            )
        }

    nuc_seg_size_dict = {prop.label: int(prop.area) for prop in regionprops(nuc_seg)}

    # associate each nuclei with a cdh5 segmentation
    reg_props = regionprops(label_image=cdh5_seg, intensity_image=nuc_seg)

    # Set up some initial data containers to populate
    nuc_feats_ls: list = list()

    feats_with_list_of_lists: dict[str, Callable] = {
        "nuc_seg_intens_means": np.mean,
        "nuc_seg_intens_stds": np.std,
        "nuc_seg_intens_medians": np.median,
        "nuc_seg_intens_pct25s": lambda x: np.percentile(x, 25),
        "nuc_seg_intens_pct75s": lambda x: np.percentile(x, 75),
        "nuc_seg_intens_maxs": np.max,
        "nuc_seg_intens_mins": np.min,
    }

    # Go through the region properties and extract features
    for prop in reg_props:
        nuc_seg_labels = np.unique(
            prop.intensity_image[prop.intensity_image != 0]
        ).tolist()

        nuc_feats = {
            "cdh5_segmentation_label": prop.label,
            "nuclei_segmentation_labels": nuc_seg_labels,
            "nuclei_seg_in_cdh5_seg_frac": [],
        }

        for f in feats_with_list_of_lists.keys():
            [nuc_feats.update({f"{f}_{chan}": []}) for chan in fluor_img_names]

        # add the fraction overlap of the cdh5 segmentation with the segmentation
        # to each of the properties in reg_props
        # also add the label with the most overlap
        for lab in nuc_seg_labels:
            if nuc_seg_labels:
                nuc_seg_in_cdh5_seg_size = np.count_nonzero(prop.intensity_image == lab)
                nuc_seg_total_size = nuc_seg_size_dict[lab]
                nuc_feats["nuclei_seg_in_cdh5_seg_frac"].append(
                    nuc_seg_in_cdh5_seg_size / nuc_seg_total_size
                )

                # summarize intensities in segmented nuclei regions for each channel
                for chan in fluor_img_names:
                    nuc_arr = nuc_props_on_intens[chan][lab].image
                    intens_arr = nuc_props_on_intens[chan][lab].image_intensity

                    for feat, func in feats_with_list_of_lists.items():
                        nuc_feats[f"{feat}_{chan}"].append(func(intens_arr[nuc_arr]))

        nuc_lab_frac_dict = dict(
            zip(nuc_seg_labels, nuc_feats["nuclei_seg_in_cdh5_seg_frac"])
        )
        nuclei_seg_with_most_overlap = [
            lab
            for lab in nuc_lab_frac_dict
            if nuc_lab_frac_dict[lab] == max(nuc_lab_frac_dict.values())
        ]
        for i, nuc_lab_max in enumerate(nuclei_seg_with_most_overlap):
            nuc_feats[f"nuclei_seg_with_most_overlap_{i}"] = nuc_lab_max
            # for dim in ["X", "Y"]:
            for dim_index, dim in enumerate(seg_dim_order):
                nuc_feats[f"nuc_with_most_overlap_{i}_centroid_{dim}"] = float(
                    nuc_props_on_intens["BF"][nuc_lab_max].centroid[::-1][dim_index]
                )

        nuc_feats_ls.append(nuc_feats)

    nuc_feats_df = pd.DataFrame(nuc_feats_ls)

    return nuc_feats_df


def get_nuclei_features_from_dataset_at_T(
    dataset_name: str, position: int, T: int, channel_names: list = ["EGFP", "BF"]
) -> pd.DataFrame:

    # Load segmentations and image
    dim_order = get_default_dim_order()

    nuc_seg = (
        load_nuclei_prediction(
            dataset_name=dataset_name,
            position=position,
            T=T,
            dim_order=dim_order,
        )
        .squeeze()
        .compute()
    )

    cdh5_seg = (
        load_cdh5_classic_segmentation(
            dataset_name=dataset_name,
            position=position,
            T=T,
            dim_order=dim_order,
        )
        .squeeze()
        .compute()
    )

    raw_img = load_dataset_position_as_dask_array(
        dataset_name=dataset_name,
        position=position,
        channels=channel_names,
        time_start=T,
        time_end=T,
    )
    raw_MIP = raw_img.max(axis=dim_order.index("Z"), keepdims=True).compute()

    # split up the image into a list of channels
    channel_arrs = np.split(
        raw_MIP, indices_or_sections=len(channel_names), axis=dim_order.index("C")
    )
    channel_arrs = [channel_arr.squeeze() for channel_arr in channel_arrs]

    # Get the nuclei properties
    nuc_feats_df = get_nuclei_features_from_image(
        cdh5_seg=cdh5_seg,
        nuc_seg=nuc_seg,
        fluorescence_images=channel_arrs,
        fluor_img_names=channel_names,
        seg_dim_order="YX",
    )

    # add the total number of detected nuclei per image to the dataframe
    num_nuclei = np.count_nonzero(np.unique(nuc_seg))
    nuc_feats_df["total_nuclei_count_at_T"] = num_nuclei

    # add the dataset name, position, and T to the dataframe
    nuc_feats_df["dataset_name"] = dataset_name
    nuc_feats_df["position"] = position
    nuc_feats_df["T"] = T

    # move the dataset_name, position, and T columns to the front
    # of the data table
    nuc_feats_df = nuc_feats_df[
        ["dataset_name", "position", "T"]
        + [
            col
            for col in nuc_feats_df.columns
            if col not in ["dataset_name", "position", "T"]
        ]
    ]

    return nuc_feats_df


def main(
    dataset_name: str | None = None,
    save_output: bool = True,
    n_proc: int = 1,
    verbose: bool = False,
    use_original_data: bool = False,
    is_test: bool = False,
) -> None:

    dataset_name_list = fire_parse_generate_dataset_name_list(dataset_name)
    out_dir = Path(get_output_path(Path(__file__).stem, verbose=False))

    # build analysis queue
    analysis_queue = build_analysis_queue(
        dataset_name_list,
        save_output=save_output,
        out_dir=out_dir,
        overwrite=True,
        verbose=verbose,
        is_test=is_test,
        image_validation_frequency=None,
        use_original_data=use_original_data,
    )

    # get and save results from images in analysis queue
    if n_proc > 1:
        with ProcessPoolExecutor(max_workers=n_proc) as executor:
            list(
                tqdm(
                    executor.map(
                        get_and_save_nuclei_features_arg_unpacker, analysis_queue
                    ),
                    total=len(analysis_queue),
                    desc="Getting nuclei features (MP)",
                )
            )
    else:
        for args in tqdm(
            analysis_queue,
            total=len(analysis_queue),
            desc="Getting nuclei features (1P)",
        ):
            get_and_save_nuclei_features_arg_unpacker(args)

    # concatenate the results outputs from above in to a single table
    if save_output:
        for dataset_name in tqdm(
            dataset_name_list, desc="Replacing individual tables with combined table..."
        ):
            concatenate_and_save_feature_tables(
                out_dir=out_dir,
                dataset_name=dataset_name,
                out_file_suffix="nuclei_features",
                file_extension=".tsv",
                remove_initial_files_and_folders=True,
            )
        # save git versioning info
        save_git_versioning_info(
            out_dir=out_dir, filename_prefix=f"{Path(__file__).stem}", verbose=verbose
        )

    print("\N{MICROSCOPE} Done analysis.")


if __name__ == "__main__":
    ipython_cli_flexecute(main)
