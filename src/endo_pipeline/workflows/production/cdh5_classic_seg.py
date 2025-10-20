def generate_results_multiproc_wrapper(args: dict) -> None:
    """Produce cdh5 segmentations for a given dataset, position, and timepoint using
    multiprocessing.
    """
    dataset_name = args["dataset_name"]
    position = args["position"]
    timepoint = args["T"]
    img_bin_level = args["image_bin_level"]
    save_output = args["save_output"]
    out_dir = args["output_dir"]
    verbose = args["verbose"]
    create_validation_image = args["is_validation_image"]
    generate_results(
        out_dir=out_dir,
        dataset_name=dataset_name,
        timepoint=timepoint,
        position=position,
        img_bin_level=img_bin_level,
        save_output=save_output,
        create_validation_image=create_validation_image,
        verbose=verbose,
    )


def generate_results(
    out_dir: Path,
    dataset_name: str,
    timepoint: int,
    position: int,
    img_bin_level: int = 0,
    save_output: bool = True,
    create_validation_image: bool = False,
    verbose: bool = True,
) -> None:
    """Produce cdh5 segmentations for a given dataset, position, and timepoint."""
    from bioio import BioImage
    from skimage.segmentation import find_boundaries

    from endo_pipeline.configs import get_zarr_file_for_position, load_dataset_config
    from endo_pipeline.io import load_image, load_image_from_path
    from endo_pipeline.library.process import cdh5_preprocessing as preproc
    from endo_pipeline.library.process.general_image_preprocessing import save_image_output
    from endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest
    from endo_pipeline.settings import DIMENSION_ORDER

    print(f"Working on {dataset_name} -- T={timepoint}...") if verbose else None
    print(f"T={timepoint} -- initializing workflow") if verbose else None
    seg_dir = out_dir / "segmentations"
    val_dir = out_dir / "validations"

    print(f"T={timepoint} -- loading dataset from zarr") if verbose else None
    dataset_config = load_dataset_config(dataset_name)
    zarr_file = get_zarr_file_for_position(dataset_config, position)
    img = BioImage(zarr_file)
    raw_dask_arr = load_image_from_path(
        path=zarr_file, channels=["EGFP"], timepoints=timepoint, level=img_bin_level
    )

    raw_arr_mip = (
        raw_dask_arr.max(axis=DIMENSION_ORDER.index("Z"), keepdims=True).compute().squeeze()
    )

    print(f"T={timepoint} -- preprocessing image") if verbose else None
    processed_img = preproc.preprocess(raw_arr_mip)

    print(f"T={timepoint} -- getting and cleaning image thresholds") if verbose else None
    hyst, hyst_clean, hyst_removed = preproc.get_thresholds(processed_img)

    print(f"T={timepoint} -- getting and cleaning RAG-based segmentations") if verbose else None
    seg2_lab_no_mask_merge, seg2_lab = preproc.generate_segmentations(
        processed_img, hyst, hyst_clean, hyst_removed, 80
    )

    print(f"T={timepoint} -- loading nuclei segmentations") if verbose else None
    seg_manifest = load_image_manifest("nuclear_labelfree_seg")
    seg_location = get_image_location_for_dataset(seg_manifest, dataset_name, position, timepoint)
    nuc_pred = load_image(seg_location, squeeze=True, compute=True)

    (
        print(f"T={timepoint} -- splitting RAG-based segmentations using nuclei predictions")
        if verbose
        else None
    )
    seg_aug, seeds = preproc.split_multinucleate_regions(
        cell_segmentations=seg2_lab_no_mask_merge,
        nuclei_segmentations=nuc_pred,
        cell_boundary_thresh=hyst,
        cell_boundary_image=processed_img,
    )

    if save_output:
        # save every nth image for validation
        if create_validation_image:
            print(f"T={timepoint} -- saving validation overlay") if verbose else None
            val_path = (
                val_dir
                / dataset_name
                / f"P{position}"
                / f"{dataset_name}_P{position}_T{timepoint}.ome.tiff"
            )
            Path.mkdir(val_path.parent, exist_ok=True, parents=True)

            seg2_lab_no_mask_merge_bounds = find_boundaries(seg2_lab_no_mask_merge)
            seg_aug_bounds = find_boundaries(seg_aug)

            images_out = [
                raw_arr_mip,
                processed_img,
                hyst_clean,
                seg2_lab,
                seg2_lab_no_mask_merge_bounds,
                seeds,  # NOTE used to be nuc_pred, remove this comment if done
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
                    "cdh5_segmentations_split_by_nuclei",  # name for augmented segmentation
                    "cdh5_segmentations_split_by_nuclei_borders",  # name for aug seg boundaries
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
                "physical_pixel_sizes": img.physical_pixel_sizes,
                "dim_order": "YX",
                "dtype": None,
            }
            save_image_output(val_path, images_out, images_out_metadata)

        # save just the cdh5 segmentations
        print(f"T={timepoint} -- saving segmentation") if verbose else None
        out_path = (
            seg_dir
            / dataset_name
            / f"P{position}"
            / f"{dataset_name}_P{position}_T{timepoint}.ome.tiff"
        )
        Path.mkdir(out_path.parent, exist_ok=True, parents=True)
        images_out = [
            seg_aug,
        ]
        images_out_metadata = {
            "image_name": dataset_name,
            "channel_names": ["cdh5_segmentations_split_by_nuclei"],
            "channel_colors": [
                (255, 255, 255),
            ],
            "physical_pixel_sizes": img.physical_pixel_sizes,
            "dim_order": "YX",
        }
        save_image_output(out_path, images_out, images_out_metadata)
    else:
        pass


def main(
    datasets: Datasets,
    n_proc: int = 1,
    save_output: bool = True,
    overwrite: bool = True,
    is_test: bool = False,
    verbose: bool = False,
) -> None:
    """Run the cdh5 segmentation workflow on a dataset, list of datasets, or dataset collection."""
    from multiprocessing import Pool

    from tqdm import tqdm

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.process.general_image_preprocessing import build_analysis_queue

    out_dir = get_output_path(__file__)

    # TODO if possible it would be good to use parallel processing to build analysis_queue
    analysis_queue = build_analysis_queue(
        datasets,
        save_output=save_output,
        out_dir=out_dir,
        overwrite=overwrite,
        verbose=verbose,
        is_test=is_test,
        image_validation_frequency=48,
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
                        desc="Segmenting (MP)",
                        total=len(analysis_queue),
                    )
                )
                pool.close()
                pool.join()
            print("Done multiprocessing.")
    else:
        for dataset_name_and_args in tqdm(
            analysis_queue, desc="Segmenting (1P)", total=len(analysis_queue)
        ):
            generate_results_multiproc_wrapper(dataset_name_and_args)

    print("\N{MICROSCOPE} Done analysis.")


if __name__ == "__main__":
    from pathlib import Path

    from endo_pipeline.cli import Datasets
    from endo_pipeline.configs.dataset_io import ipython_cli_flexecute

    # ipython_cli_flexecute runs a function via either
    # the command line or an interactive python shell
    ipython_cli_flexecute(main)
