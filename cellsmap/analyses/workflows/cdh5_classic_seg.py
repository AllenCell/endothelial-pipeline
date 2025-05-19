from multiprocessing import Pool
from pathlib import Path

from bioio import BioImage
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from cellsmap.util import cdh5_preprocessing as preproc
from cellsmap.util import get_sldy_metadata as sldmd
from cellsmap.util.dataset_io import (
    get_dataset_info,
    get_original_path,
    get_zarr_path,
    ipython_cli_flexecute,
    load_config,
)
from cellsmap.util.general_image_preprocessing import (
    build_analysis_queue,
    get_dim_map,
    save_image_output,
)
from cellsmap.util.set_output import get_output_path


def generate_results_multiproc_wrapper(args):
    dataset_name = args["dataset_name"]
    scenes = (args["scene_index"],)
    position = args["position"]
    T = args["T"]
    img_bin_level = args["image_bin_level"]
    save_output = args["save_output"]
    out_dir = args["output_dir"]
    verbose = args["verbose"]
    use_original_data = args["use_original_data"]
    create_validation_image = args["validation_image"]
    generate_results(
        dataset_name,
        T,
        scenes,
        position,
        use_original_data,
        img_bin_level,
        out_dir=out_dir,
        save_output=save_output,
        create_validation_image=create_validation_image,
        verbose=verbose,
    )


def generate_results(
    dataset_name,
    T,
    scene_list=None,
    position_name=None,
    use_original_data=False,
    img_bin_level=0,
    out_dir=None,
    save_output=True,
    create_validation_image=False,
    verbose=True,
):

    print(f"Working on {dataset_name} -- T={T}...")
    print(f"T={T} -- initializing workflow") if verbose else None
    # out_dir_list, img_metadata = initialize_workflow(dataset_name, save_output, is_test)
    # out_dir, val_dir = out_dir_list
    seg_dir = out_dir / "segmentations"
    val_dir = out_dir / "validations"

    print(f"T={T} -- loading dataset") if verbose else None
    # get the name of the cadherin channel
    # chan_names = [chan_name for chan_name in dataset_io.get_available_channels(dataset_name) if chan_name in ['CDH5', 'CDH5_Tubulin']]
    # load the raw image data of from the cadherin channel
    # raw_arr = dataset_io.load_dataset(dataset_name, channels=chan_names, time_start=T, time_end=T, level=img_bin_level).compute().squeeze()

    if use_original_data:
        original_path = Path(get_original_path(dataset_name))
        img_path = original_path
        img = BioImage(img_path)
        scene_list = scene_list or img.scenes
    else:
        zarr_path = Path(get_zarr_path(dataset_name))
        img_path = zarr_path
        img_dict = {fp.name: BioImage(fp) for fp in img_path.glob("*.zarr")}
        scene_list = scene_list if scene_list else img_dict.keys()

    dim_map = get_dim_map("TCZYX")
    egfp_index = get_dataset_info(dataset_name)["488_channel_index"]

    for scene in scene_list:
        position_name = scene if position_name == None else position_name
        if use_original_data:
            current_img = img
            current_img.set_scene(scene)
        else:
            current_img = img_dict[scene]
            current_img.set_resolution_level(img_bin_level)

        raw_dask_arr = current_img.get_image_dask_data("TCZYX", T=T, C=egfp_index)
        raw_arr_MIP = (
            raw_dask_arr.max(axis=dim_map["Z"], keepdims=True).compute().squeeze()
        )

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
                    / f"P{position_name}"
                    / f"{dataset_name}_P{position_name}_T{T}.ome.tiff"
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
                    "physical_pixel_sizes": current_img.physical_pixel_sizes,  # img_metadata['physical_pixel_sizes'],
                    "dim_order": "YX",
                    "dtype": None,
                }
                save_image_output(val_path, images_out, images_out_metadata)

            # save just the cdh5 segmentations
            print(f"T={T} -- saving segmentation") if verbose else None
            out_path = (
                seg_dir
                / dataset_name
                / f"P{position_name}"
                / f"{dataset_name}_P{position_name}_T{T}.ome.tiff"
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
                "physical_pixel_sizes": current_img.physical_pixel_sizes,  # img_metadata['physical_pixel_sizes'],
                "dim_order": "YX",
            }
            save_image_output(out_path, images_out, images_out_metadata)
        else:
            pass


def main(
    n_proc=1,
    dataset_name=None,
    save_output=True,
    overwrite=False,
    is_test=False,
    verbose=False,
):

    if dataset_name == None:
        dataset_name_list = [
            config_data["name"]
            for config_data in load_config(config_type="data")
            if (
                config_data["microscope"] == "3i"
                and config_data["live_or_fixed_sample"] == "live"
            )
            and "AICS-126" in config_data["cell_lines"]
        ]
    else:
        dataset_name_list = [dataset_name]

    # TODO if possible it would be good to use parallel processing to build analysis_queue
    analysis_queue = build_analysis_queue(
        dataset_name_list,
        save_output=save_output,
        out_dir=get_output_path(Path(__file__).stem, verbose=False),
        overwrite=overwrite,
        verbose=verbose,
        is_test=is_test,
        use_original_data=True,
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
