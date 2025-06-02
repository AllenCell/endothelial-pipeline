from multiprocessing import Pool
from pathlib import Path

from bioio import BioImage
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from cellsmap.util import cdh5_preprocessing as preproc
from cellsmap.util.dataset_io import (
    fire_parse_generate_dataset_name_list,
    get_dataset_info,
    get_original_path,
    get_zarr_name,
    get_zarr_path,
    ipython_cli_flexecute,
    load_dataset_position_as_dask_array,
    load_nuclei_prediction,
)
from cellsmap.util.general_image_preprocessing import (
    build_analysis_queue,
    get_default_dim_order,
    get_dim_map,
    save_image_output,
)
from cellsmap.util.set_output import get_output_path


def generate_results_multiproc_wrapper(args: dict) -> None:
    dataset_name = args["dataset_name"]
    scene_name = args["scene_name"]
    scene_index = args["scene_index"]
    position = args["position"]
    T = args["T"]
    img_bin_level = args["image_bin_level"]
    save_output = args["save_output"]
    out_dir = args["output_dir"]
    verbose = args["verbose"]
    use_original_data = args["use_original_data"]
    create_validation_image = args["validation_image"]
    generate_results(
        out_dir=out_dir,
        dataset_name=dataset_name,
        T=T,
        position=position,
        scene_index=scene_index,
        scene_name=scene_name,
        use_original_data=use_original_data,
        img_bin_level=img_bin_level,
        save_output=save_output,
        create_validation_image=create_validation_image,
        verbose=verbose,
    )


def generate_results(
    out_dir: Path,
    dataset_name: str,
    T: int,
    position: int,
    scene_index: int | None = None,
    scene_name: str | None = None,
    use_original_data: bool = False,
    img_bin_level: int = 0,
    save_output: bool = True,
    create_validation_image: bool = False,
    verbose: bool = True,
) -> None:

    print(f"Working on {dataset_name} -- T={T}...")
    print(f"T={T} -- initializing workflow") if verbose else None
    seg_dir = out_dir / "segmentations"
    val_dir = out_dir / "validations"

    print(f"T={T} -- loading dataset") if verbose else None
    dim_order = get_default_dim_order()
    dim_map = get_dim_map(dim_order)
    if use_original_data:
        original_path = Path(get_original_path(dataset_name))
        img_path = original_path
        img = BioImage(img_path)
        egfp_index = get_dataset_info(dataset_name)["488_channel_index"]
        if scene_index is not None or scene_name is not None:
            scene = (
                scene_index or scene_name or 0
            )  #  the "or 0" is here to silence mypy
            img.set_scene(scene)
            raw_dask_arr = img.get_image_dask_data(dim_order, T=T, C=egfp_index)
        else:
            raise ValueError(
                "When using original data, either scene_index or scene_name must be provided."
            )
    else:
        zarr_name = get_zarr_name(dataset_name, position)
        zarr_path = Path(get_zarr_path(dataset_name)[zarr_name])
        img = BioImage(
            zarr_path
        )  # only using BioImage here to pass pixel sizes to output
        raw_dask_arr = load_dataset_position_as_dask_array(
            dataset_name=dataset_name,
            position=position,
            channels=["EGFP"],
            time_start=T,
            time_end=T,
            level=img_bin_level,
        )

    raw_arr_MIP = raw_dask_arr.max(axis=dim_map["Z"], keepdims=True).compute().squeeze()

    print(f"T={T} -- preprocessing image") if verbose else None
    processed_img = preproc.preprocess(raw_arr_MIP)

    print(f"T={T} -- getting and cleaning image thresholds") if verbose else None
    hyst, hyst_clean, hyst_removed = preproc.get_thresholds(processed_img)

    print(f"T={T} -- getting and cleaning segmentations") if verbose else None
    seg2_lab_no_mask_merge, seg2_lab = preproc.generate_segmentations(
        processed_img, hyst, hyst_clean, hyst_removed
    )

    nuc_pred = (
        load_nuclei_prediction(
            dataset_name=dataset_name,
            position=position,
            T=T,
        )
        .squeeze()
        .compute()
    )

    # NOTE THIS BLOCK FOR AUGMENTING SEGMENTATIONS NEEDS TO BE
    # MOVED TO A SEPARATE FUNCTION IN `cdh5_preprocessing.py`
    # def segmentation_augmentation(segmentation, helper_seeds) -> np.ndarray:
    # segmentation = seg2_lab_no_mask_merge
    # helper_seeds = nuc_pred
    import numpy as np
    from matplotlib import pyplot as plt
    from skimage.color import label2rgb
    from skimage.exposure import rescale_intensity
    from skimage.measure import label, regionprops
    from skimage.morphology import dilation, disk, skeletonize
    from skimage.segmentation import relabel_sequential, watershed

    from cellsmap.util.cdh5_preprocessing import get_watershed_seeds_and_basins

    # if nuclei with different labels are touching then separate them
    # only if a segmentation boundary or the cdh5 threshold would have
    # separated them
    nuc_pred_merge_adjacent = label((nuc_pred * ~hyst_clean).astype(bool))
    nuc_pred_skels = skeletonize(nuc_pred_merge_adjacent) * nuc_pred_merge_adjacent
    # nuc_pred_skels = label(nuc_pred_merge_adjacent)
    reg_props = regionprops(
        label_image=seg2_lab_no_mask_merge, intensity_image=nuc_pred_skels
    )
    nuclei_labels_per_region = {
        prop.label: np.unique(prop.intensity_image)[
            np.unique(prop.intensity_image).nonzero()
        ]
        for prop in reg_props
    }
    num_nuclei_per_region = {
        prop.label: np.count_nonzero(np.unique(prop.intensity_image))
        for prop in reg_props
    }

    seg_skels = (
        # skeletonize(~find_boundaries(seg2_lab_no_mask_merge) * seg2_lab_no_mask_merge)
        skeletonize(~find_boundaries(seg2_lab_no_mask_merge))
        * seg2_lab_no_mask_merge
    )
    anucleate_region_labels = [
        lab for lab in num_nuclei_per_region if num_nuclei_per_region[lab] == 0
    ]
    anucleate_reg_skels = np.isin(seg_skels, anucleate_region_labels) * seg_skels

    multinucleate_region_labels = [
        lab for lab in num_nuclei_per_region if num_nuclei_per_region[lab] > 1
    ]

    mononucleate_region_labels = [
        lab for lab in num_nuclei_per_region if num_nuclei_per_region[lab] == 1
    ]
    mononucleate_reg_skels = np.isin(seg_skels, mononucleate_region_labels) * seg_skels

    # get basins for performing watershed segmentation
    _, basins = get_watershed_seeds_and_basins(~hyst)
    # these basins are based on the geometry of the threshold image
    # so add info about the intensity image to make a
    # "ridges and basins" image for watershed to work on
    ridges = processed_img * hyst
    min_val_in_ridges = ridges[np.nonzero(ridges)].min()
    ridges = np.clip(ridges, min_val_in_ridges, None)
    basins_and_ridges = (basins + rescale_intensity(ridges, out_range=(0, 1))) / 2

    # use the skeletonized regions as seed points for mononucleate
    # and anucleate regions
    seeds = anucleate_reg_skels + mononucleate_reg_skels
    # use the nuclei as seed points for the multinucleate regions
    # multinuc_seeds = np.isin(seg2_lab_no_mask_merge, multinucleate_region_labels) * nuc_pred_skels
    multinuc_seeds = (
        np.isin(seg2_lab_no_mask_merge, multinucleate_region_labels)
        * nuc_pred_merge_adjacent
    )
    multinuc_seeds, _, _ = relabel_sequential(multinuc_seeds, offset=1 + seeds.max())
    seeds = seeds + multinuc_seeds

    # segment cells with watershed
    seg_aug = watershed(image=basins_and_ridges, markers=seeds, mask=~hyst_clean)
    seg_aug = watershed(image=basins_and_ridges, markers=seg_aug)

    # NOTE TRY OUT THE VESSELNESS EDGE ENHANCEMENT OR DIFFERENCE OF
    # GAUSSIANS APPROACHES TO CDH5 FLUORESCENCE IMAGE PREPROCESSING
    # FROM SECOND_DATASET_SEGMENTATION_IMPROVEMENT_EFFORTS BRANCH??

    crop = slice(200, 600), slice(1000, 1400)
    raw_clip_norm = rescale_intensity(
        np.clip(
            raw_arr_MIP, np.percentile(raw_arr_MIP, 3), np.percentile(raw_arr_MIP, 97)
        )
    )
    overlay = label2rgb(
        label=dilation(find_boundaries(seg_aug), disk(3)) * 1
        + nuc_pred_merge_adjacent.astype(bool) * 2,
        image=raw_clip_norm,
        bg_label=0,
        colors=["magenta", "cyan", "yellow"],
        alpha=0.5,
    )
    plt.imshow(overlay)
    # plt.imshow(overlay[crop])

    overlay = label2rgb(
        label=dilation(
            find_boundaries(seg_aug) * 1 + ~hyst_clean * seeds.astype(bool) * 2, disk(3)
        ),
        image=raw_clip_norm,
        bg_label=0,
        colors=["magenta", "cyan", "yellow"],
        alpha=0.5,
    )
    plt.imshow(overlay[crop])

    overlay = label2rgb(
        label=dilation(hyst * 1 + ~hyst_clean * seeds.astype(bool) * 2, disk(3)),
        image=basins_and_ridges,
        bg_label=0,
        colors=["magenta", "cyan", "yellow"],
        alpha=0.5,
    )
    plt.imshow(overlay[crop])

    # overlay = label2rgb(label=dilation(find_boundaries(seg2_lab_no_mask_merge)*1 + ~hyst_clean * seeds.astype(bool)*2, disk(3)),
    #                     image=basins_and_ridges, bg_label=0, colors=["magenta", "cyan", "yellow"], alpha=0.5)
    # plt.imshow(overlay[crop])

    if save_output:
        # save every nth image for validation
        if create_validation_image:
            print(f"T={T} -- saving validation overlay") if verbose else None
            val_path = (
                val_dir
                / dataset_name
                / f"P{position}"
                / f"{dataset_name}_P{position}_T{T}.ome.tiff"
            )
            Path.mkdir(val_path.parent, exist_ok=True, parents=True)

            seg2_lab_no_mask_merge_bounds = find_boundaries(seg2_lab_no_mask_merge)
            seg_aug_bounds = find_boundaries(seg_aug)

            images_out = [
                raw_arr_MIP,
                processed_img,
                hyst_clean,
                seg2_lab,
                # seg2_lab_no_mask_merge,
                seg2_lab_no_mask_merge_bounds,
                nuc_pred,
                seg_aug,  # add the augmented segmentation
                seg_aug_bounds,  # add the augmented segmentation boundaries
            ]
            images_out_metadata = {
                "image_name": dataset_name,
                "channel_names": [
                    "raw",
                    "processed",
                    "hysteresis_threshold",
                    "segmentations_initial",
                    "segmentations_merged",
                    "nuclei_predictions",
                    "segmentations_augmented",  # name for the augmented segmentation
                    "segmentations_augmented_borders",  # name for the augmented segmentation boundaries
                ],
                "channel_colors": [
                    (255, 255, 255),
                    (255, 255, 255),
                    (0, 255, 255),
                    (255, 0, 255),
                    (255, 0, 255),
                    (255, 0, 0),  # color for the nuclei predictions
                    (0, 255, 0),  # color for the augmented segmentation
                    (0, 0, 255),  # color for the augmented segmentation boundaries
                ],
                "physical_pixel_sizes": img.physical_pixel_sizes,  # img_metadata['physical_pixel_sizes'],
                "dim_order": "YX",
                "dtype": None,
            }
            save_image_output(val_path, images_out, images_out_metadata)

        # save just the cdh5 segmentations
        print(f"T={T} -- saving segmentation") if verbose else None
        out_path = (
            seg_dir
            / dataset_name
            / f"P{position}"
            / f"{dataset_name}_P{position}_T{T}.ome.tiff"
        )
        Path.mkdir(out_path.parent, exist_ok=True, parents=True)
        images_out = [
            seg2_lab_no_mask_merge,
        ]
        images_out_metadata = {
            "image_name": dataset_name,
            "channel_names": ["segmentations_merged"],
            "channel_colors": [
                (255, 255, 255),
            ],
            "physical_pixel_sizes": img.physical_pixel_sizes,  # img_metadata['physical_pixel_sizes'],
            "dim_order": "YX",
        }
        save_image_output(out_path, images_out, images_out_metadata)
    else:
        pass


def main(
    n_proc: int = 1,
    dataset_name: str | None = None,
    save_output: bool = True,
    overwrite: bool = True,
    use_original_data: bool = False,
    is_test: bool = False,
    verbose: bool = False,
) -> None:

    out_dir = get_output_path(Path(__file__).stem, verbose=False)

    dataset_name_list = fire_parse_generate_dataset_name_list(dataset_name)

    # TODO if possible it would be good to use parallel processing to build analysis_queue
    analysis_queue = build_analysis_queue(
        dataset_name_list,
        save_output=save_output,
        out_dir=out_dir,
        overwrite=overwrite,
        verbose=verbose,
        is_test=is_test,
        use_original_data=use_original_data,
        image_validation_frequency=20,
    )

    if n_proc > 1:
        if __name__ == "__main__":
            print("Starting multiprocessing...")
            with Pool(processes=n_proc) as pool:
                list(
                    tqdm(
                        pool.imap(
                            generate_results_multiproc_wrapper,
                            analysis_queue,
                            chunksize=5,
                        ),
                        total=len(analysis_queue),
                    )
                )
                pool.close()
                pool.join()
            print("Done multiprocessing.")
    else:
        for dataset_name_and_args in analysis_queue:
            generate_results_multiproc_wrapper(dataset_name_and_args)

    print("\N{MICROSCOPE} Done analysis.")


if __name__ == "__main__":
    # ipython_cli_flexecute runs a function via either
    # the command line or an interactive python shell
    ipython_cli_flexecute(main)
