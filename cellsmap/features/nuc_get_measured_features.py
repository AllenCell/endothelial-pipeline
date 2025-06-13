import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from bioio import BioImage
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from cellsmap.util.dataset_io import (
    extract_T,
    fire_parse_generate_dataset_name_list,
    get_cdh5_classic_segmentation_path,
    get_dataset_info,
    get_original_path,
    get_zarr_name,
    get_zarr_path,
    ipython_cli_flexecute,
    load_cdh5_classic_segmentation,
    load_dataset_position_as_dask_array,
    load_nuclei_prediction,
)
from cellsmap.util.general_image_preprocessing import (
    build_analysis_queue,
    get_default_dim_order,
)
from cellsmap.util.set_output import get_output_path


def get_nuclei_features_arg_unpacker(args: dict) -> pd.DataFrame:
    dataset_name = args["dataset_name"]
    position = args["position"]
    T = args["T"]
    out_dir = args["output_dir"]
    verbose = args["verbose"]

    nuc_props_df = get_nuclei_features(dataset_name, position, T)

    return nuc_props_df


# def get_overlap_props(prop: regionprops) -> None:
#     seg_overlap_fractions = {}
#     for seg_lab, seg_lab_size in zip(*np.unique(prop.intensity_image, return_counts=True)):
#         if seg_lab != 0:
#             overlap_fraction = seg_lab_size / prop.area
#             seg_overlap_fractions[seg_lab] = overlap_fraction

#     prop.seg_overlap_labels, prop.seg_overlap_fractions = zip(*seg_overlap_fractions.items())

#     seg_most_overlap = max(seg_overlap_fractions, key=lambda x: seg_overlap_fractions[x])
#     seg_most_overlap_val = seg_overlap_fractions[seg_most_overlap]
#     seg_most_overlap = tuple(lab for lab in seg_overlap_fractions.keys() if seg_overlap_fractions[lab] == seg_most_overlap_val)

#     prop.seg_most_overlap = seg_most_overlap


def get_nuclei_features(
    dataset_name: str, position: int, T: int, channels: list = ["EGFP", "BF"]
) -> pd.DataFrame:

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
        channels=channels,
        time_start=T,
        time_end=T,
    )
    raw_MIP = raw_img.max(axis=dim_order.index("Z"), keepdims=True).compute()

    # associate each nuclei with a cdh5 segmentation
    nuc_props_on_seg = regionprops(label_image=nuc_seg, intensity_image=cdh5_seg)

    # add the fraction overlap of the cdh5 segmentation with the segmentation
    # to each of the properties in reg_props
    # also add the label with the most overlap
    # for prop in nuc_props_on_seg:
    #     get_overlap_props(prop)
    # seg_overlap_fractions = {}
    # for seg_lab, seg_lab_size in zip(*np.unique(prop.intensity_image, return_counts=True)):
    #     if seg_lab != 0:
    #         overlap_fraction = seg_lab_size / prop.area
    #         seg_overlap_fractions[seg_lab] = overlap_fraction

    # prop.seg_overlap_labels, prop.seg_overlap_fractions = zip(*seg_overlap_fractions.items())

    # seg_most_overlap = max(seg_overlap_fractions, key=lambda x: seg_overlap_fractions[x])
    # seg_most_overlap_val = seg_overlap_fractions[seg_most_overlap]
    # seg_most_overlap = tuple(lab for lab in seg_overlap_fractions.keys() if seg_overlap_fractions[lab] == seg_most_overlap_val)

    # prop.seg_most_overlap = seg_most_overlap

    nuc_props_on_seg_dict = {prop.label: prop for prop in nuc_props_on_seg}

    # keep only nuclei that have more than half of their area in a
    # single segmented region
    nuclei_ambiguity_threshold = 0.5

    # get intensity properties from each channel of the raw image within the
    # nuclei segmentations
    # if channels is None:
    #     num_channels = raw_MIP.shape[dim_order.index("C")]
    #     channel_indices = range(num_channels)
    # else:
    #     channel_indices = channels.index()

    channel_indices = range(len(channels))

    nuc_props_on_intens = dict()
    for i in channel_indices:
        channel_arr = raw_MIP.take(axis=dim_order.index("C"), indices=[i]).squeeze()
        nuc_props_on_intens[channels[i]] = {
            prop.label: prop
            for prop in regionprops(label_image=nuc_seg, intensity_image=channel_arr)
        }

    nuc_seg_size_dict = {prop.label: int(prop.area) for prop in regionprops(nuc_seg)}

    reg_props = regionprops(label_image=cdh5_seg, intensity_image=nuc_seg)

    nuc_feats_ls: list = list()
    for prop in reg_props:
        nuc_seg_labels = np.unique(
            prop.intensity_image[prop.intensity_image != 0]
        ).tolist()
        nuclei_seg_in_cdh5_seg_frac = list()
        ls_init_per_chan: list = [list() for _ in channels]
        nuc_seg_intens_means = dict(zip(channels, ls_init_per_chan))
        nuc_seg_intens_stds = dict(zip(channels, ls_init_per_chan))
        nuc_seg_intens_medians = dict(zip(channels, ls_init_per_chan))
        nuc_seg_intens_maxs = dict(zip(channels, ls_init_per_chan))
        nuc_seg_intens_mins = dict(zip(channels, ls_init_per_chan))
        for lab in nuc_seg_labels:
            if nuc_seg_labels:
                # print(prop.label, nuc_seg_labels)
                # break
                # break
                nuc_seg_in_cdh5_seg_size = np.count_nonzero(prop.intensity_image == lab)
                nuc_seg_total_size = nuc_seg_size_dict[lab]
                nuclei_seg_in_cdh5_seg_frac.append(
                    nuc_seg_in_cdh5_seg_size / nuc_seg_total_size
                )

                ## TODO WORKING HERE -- ADD NUCLEI INTENSITY PROPERTIES TO NUC_FEATS
                for chan in channels:
                    nuc_arr = nuc_props_on_intens[chan][lab].image
                    intens_arr = nuc_props_on_intens[chan][lab].image_intensity
                    intens_means = intens_arr[nuc_arr].mean()

                # cdh5_seg_index = nuc_props_on_seg_dict[lab].seg_overlap_labels.index(prop.label)
                # nuclei_seg_in_cdh5_seg_frac.append(nuc_props_on_seg_dict[lab].seg_overlap_fractions[cdh5_seg_index])
                # nuclei_seg_with_most_overlap.append(nuc_props_on_seg_dict[lab].)
        nuc_seg_intens_means[chan] = intens_means
        nuc_lab_frac_dict = dict(zip(nuc_seg_labels, nuclei_seg_in_cdh5_seg_frac))
        nuclei_seg_with_most_overlap = [
            lab
            for lab in nuc_lab_frac_dict
            if nuc_lab_frac_dict[lab] == max(nuc_lab_frac_dict.values())
        ]

        nuc_feats = {
            "cdh5_segmentation_label": prop.label,
            "nuclei_segmentation_labels": nuc_seg_labels,
            "nuclei_seg_in_cdh5_seg_frac": nuclei_seg_in_cdh5_seg_frac,
            "nuclei_seg_with_most_overlap": nuc_props_on_seg_dict,
        }

    nuc_feats_df = pd.DataFrame(nuc_feats_ls)

    return nuc_feats_df


def main(
    dataset_name: str | None = None,
    save_output: bool = True,
    n_proc: int = 1,
    verbose: bool = False,
    use_original_data: bool = False,
) -> None:

    dataset_name_list = fire_parse_generate_dataset_name_list(dataset_name)
    out_dir = get_output_path(Path(__file__).stem, verbose=False)

    analysis_queue = build_analysis_queue(
        dataset_name_list,
        save_output=save_output,
        out_dir=out_dir,
        overwrite=True,
        verbose=verbose,
        image_validation_frequency=None,
        use_original_data=use_original_data,
    )

    if n_proc > 1:
        with ProcessPoolExecutor(max_workers=n_proc) as executor:
            nuclei_features = list(
                tqdm(
                    executor.map(get_nuclei_features_arg_unpacker, analysis_queue),
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
            nuclei_features = get_nuclei_features_arg_unpacker(args)
