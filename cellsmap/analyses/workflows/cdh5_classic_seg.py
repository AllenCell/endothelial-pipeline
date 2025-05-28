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

    import numpy as np

    from cellsmap.util.dataset_io import extract_T, get_nuclear_prediction_path

    def load_nuclei_prediction(
        dataset_name: str,
        position: int,
        T: int,
        dim_order: str = "ZYX",
    ) -> np.ndarray:  # da.Array:
        """
        Load the nuclei prediction for a given dataset, position, and timepoint.
        """
        nuc_dir = Path(get_nuclear_prediction_path(dataset_name, position))
        nuc_path_dict = {extract_T(fp.stem): fp for fp in nuc_dir.glob("*.ome.tif*")}

        if nuc_path.exists():
            # Load the nuclei prediction as a Dask array
            nuc_dask_arr = BioImage(nuc_path).get_image_dask_data(dim_order, T=T)
            return nuc_dask_arr.compute()

        return np.zeros()

    raw_arr_MIP = raw_dask_arr.max(axis=dim_map["Z"], keepdims=True).compute().squeeze()

    print(f"T={T} -- preprocessing image") if verbose else None
    processed_img = preproc.preprocess(raw_arr_MIP)

    print(f"T={T} -- getting and cleaning image thresholds") if verbose else None
    hyst, hyst_clean, hyst_removed = preproc.get_thresholds(processed_img)

    print(f"T={T} -- getting and cleaning segmentations") if verbose else None
    seg2_lab_no_mask_merge, seg2_lab = preproc.generate_segmentations(
        processed_img, hyst, hyst_clean, hyst_removed
    )
    seg2_lab_no_mask_merge_bounds = find_boundaries(seg2_lab_no_mask_merge)

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
            # out_path = seg_dir / dataset_name / f'{dataset_name}_T{T}.ome.tiff'
            # Path.mkdir(seg_dir / dataset_name, exist_ok=True, parents=True)
            images_out = [
                raw_arr_MIP,
                processed_img,
                hyst_clean,
                seg2_lab,
                seg2_lab_no_mask_merge,
                seg2_lab_no_mask_merge_bounds,
            ]
            images_out_metadata = {
                "image_name": dataset_name,
                "channel_names": [
                    "raw",
                    "processed",
                    "hysteresis_threshold",
                    "segmentations_initial",
                    "segmentations_merged",
                    "segmentations_merged_borders",
                ],
                "channel_colors": [
                    (255, 255, 255),
                    (255, 255, 255),
                    (0, 255, 255),
                    (255, 0, 255),
                    (255, 0, 255),
                    (255, 255, 0),
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
