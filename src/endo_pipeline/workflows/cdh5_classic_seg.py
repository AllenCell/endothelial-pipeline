from multiprocessing import Pool
from pathlib import Path

from bioio import BioImage
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from endo_pipeline.cli import Datasets
from endo_pipeline.configs import (
    get_datasets_in_collection,
    get_zarr_file_for_position,
    load_dataset_config,
)
from endo_pipeline.configs.dataset_io import get_original_path, ipython_cli_flexecute
from endo_pipeline.io import get_output_path, load_image, load_zarr_as_dask_array
from endo_pipeline.library.process import cdh5_preprocessing as preproc
from endo_pipeline.library.process.general_image_preprocessing import (
    build_analysis_queue,
    get_default_dim_order,
    get_dim_map,
    save_image_output,
)
from endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest


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
    use_sldy_data = args["use_sldy_data"]
    create_validation_image = args["validation_image"]
    generate_results(
        out_dir=out_dir,
        dataset_name=dataset_name,
        T=T,
        position=position,
        scene_index=scene_index,
        scene_name=scene_name,
        use_sldy_data=use_sldy_data,
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
    use_sldy_data: bool = False,
    img_bin_level: int = 0,
    save_output: bool = True,
    create_validation_image: bool = False,
    verbose: bool = True,
) -> None:

    print(f"Working on {dataset_name} -- T={T}...") if verbose else None
    print(f"T={T} -- initializing workflow") if verbose else None
    seg_dir = out_dir / "segmentations"
    val_dir = out_dir / "validations"

    dim_order = get_default_dim_order()
    dim_map = get_dim_map(dim_order)
    if use_sldy_data:
        print(f"T={T} -- loading dataset from original") if verbose else None
        original_path = Path(get_original_path(dataset_name))
        img_path = original_path
        img = BioImage(img_path)
        dataset_config = load_dataset_config(dataset_name)
        egfp_index = dataset_config.original_channel_indices.channel_488

        if scene_index is not None or scene_name is not None:
            scene = scene_index or scene_name or 0  #  the "or 0" is here to silence mypy
            img.set_scene(scene)
            raw_dask_arr = img.get_image_dask_data(dim_order, T=T, C=egfp_index)
        else:
            raise ValueError(
                "When using original data, either scene_index or scene_name must be provided."
            )
    else:
        print(f"T={T} -- loading dataset from zarr") if verbose else None
        dataset_config = load_dataset_config(dataset_name)
        zarr_file = get_zarr_file_for_position(dataset_config, position)
        img = BioImage(zarr_file)
        raw_dask_arr = load_zarr_as_dask_array(
            path=zarr_file, channels=["EGFP"], timepoints=T, level=img_bin_level
        )

    raw_arr_MIP = raw_dask_arr.max(axis=dim_map["Z"], keepdims=True).compute().squeeze()

    print(f"T={T} -- preprocessing image") if verbose else None
    processed_img = preproc.preprocess(raw_arr_MIP)

    print(f"T={T} -- getting and cleaning image thresholds") if verbose else None
    hyst, hyst_clean, hyst_removed = preproc.get_thresholds(processed_img)

    print(f"T={T} -- getting and cleaning RAG-based segmentations") if verbose else None
    seg2_lab_no_mask_merge, seg2_lab = preproc.generate_segmentations(
        processed_img, hyst, hyst_clean, hyst_removed, 80
    )

    print(f"T={T} -- loading nuclei segmentations") if verbose else None
    seg_manifest = load_image_manifest("nuclear_labelfree_seg")
    seg_location = get_image_location_for_dataset(seg_manifest, dataset_name, position, T)
    nuc_pred = load_image(seg_location)

    (
        print(f"T={T} -- splitting RAG-based segmentations using nuclei predictions")
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
        print(f"T={T} -- saving segmentation") if verbose else None
        out_path = (
            seg_dir / dataset_name / f"P{position}" / f"{dataset_name}_P{position}_T{T}.ome.tiff"
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
    n_proc: int = 1,
    datasets: Datasets | None = None,
    save_output: bool = True,
    overwrite: bool = True,
    use_sldy_data: bool = False,
    is_test: bool = False,
    verbose: bool = False,
) -> None:

    out_dir = get_output_path(__file__)

    if datasets is None:
        dataset_name_list = get_datasets_in_collection("pca_reference")
    else:
        dataset_name_list = datasets

    # TODO if possible it would be good to use parallel processing to build analysis_queue
    analysis_queue = build_analysis_queue(
        dataset_name_list,
        save_output=save_output,
        out_dir=out_dir,
        overwrite=overwrite,
        verbose=verbose,
        is_test=is_test,
        use_sldy_data=use_sldy_data,
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
    # ipython_cli_flexecute runs a function via either
    # the command line or an interactive python shell
    ipython_cli_flexecute(main)
