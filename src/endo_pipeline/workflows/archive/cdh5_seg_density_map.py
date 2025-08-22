from pathlib import Path

import numpy as np


def get_chan_map(filepath: Path) -> dict:
    from bioio import BioImage

    img = BioImage(filepath)
    return {name: index for index, name in enumerate(img.channel_names)}


def multiproc_wrapper(function, args):
    function(*args)


def initialize_workflow(
    dataset_name: str, SAVE_OUTPUT: bool = True, IS_TEST: bool = False
) -> tuple[Path, dict]:
    from bioio import BioImage

    from src.endo_pipeline.configs.dataset_io import get_time_interval_in_minutes, get_zarr_path

    # NOTE: this function is unique to each script
    SCT_NAME = Path(__file__).stem
    PRJ_DIR = Path("../").resolve() if not IS_TEST else Path("../../tests").resolve()
    assert PRJ_DIR.exists()
    out_dir = PRJ_DIR / f"results/{SCT_NAME}" / dataset_name

    # create output directory if it doesn't exist and get image metadata from the input image
    Path.mkdir(out_dir, exist_ok=True, parents=True) if SAVE_OUTPUT else None

    img = BioImage(Path(get_zarr_path(dataset_name)))
    px_res = img.physical_pixel_sizes
    t_res = get_time_interval_in_minutes(dataset_name)
    img_metadata = {
        "dataset_name": dataset_name,
        "physical_pixel_sizes": px_res,
        "t_res (min)": t_res,
        "t_res (hr)": t_res / 60,
    }

    return out_dir, img_metadata


def get_density_map_from_thresholds(
    dataset_name: str, T: int, density_map_sigma: float, VERBOSE: bool = False
) -> np.ndarray:
    """
    Returns a density map of the cadherin channel of a dataset at a given timepoint T.
    This function will load an image from a timepoint in a dataset at the lowest resolution,
    preprocess the image, get the hysteresis threshold of the image, skeletonize the
    thresholded image, and then apply a gaussian filter to the skeletonized image to
    get an estimate of the local density of cells in that region (i.e. the density map).

    Parameters
    ----------
        dataset_name: the name of the dataset to get the density map from.
        T: the timepoint at which to get the density map.
        density_map_sigma: the sigma value to pass along to the gaussian filter.
        VERBOSE: prints out progress statements if True (default is False).

    Returns
    -------
        density_map: the density map (values are floats and should range from 0 to 1).
    """
    from bioio import BioImage
    from skimage.filters import gaussian
    from skimage.morphology import skeletonize

    from src.endo_pipeline.configs.dataset_io import (
        get_available_channels,
        get_zarr_path,
        load_config,
        load_dataset,
    )
    from src.endo_pipeline.library.process.cdh5_preprocessing import get_thresholds, preprocess

    DATASET_NAME_LIST = [config_data["name"] for config_data in load_config(config_type="data")]
    assert (
        dataset_name in DATASET_NAME_LIST
    ), f"dataset_name must be one of {DATASET_NAME_LIST}, not {dataset_name}"

    print(f"T={T} -- loading dataset") if VERBOSE else None
    # get the binning levels of the dataset so we can always use the lowest resolution
    dataset_filepath = Path(get_zarr_path(dataset_name))
    img_bin_level = sorted([level for level in BioImage(dataset_filepath).resolution_levels])[-1]
    # get the name of the cadherin channel
    chan_names = [
        name for name in get_available_channels(dataset_name) if name in ("CDH5", "CDH5_Tubulin")
    ]
    # load the raw image data of from the cadherin channel
    raw_arr = (
        load_dataset(
            dataset_name,
            channels=chan_names,
            time_start=T,
            time_end=T,
            level=img_bin_level,
        )
        .compute()
        .squeeze()
    )

    print(f"T={T} -- preprocessing image") if VERBOSE else None
    sigma = (
        round(3 * 0.5**img_bin_level) or 1
    )  # 3 was the original value used when processing the highest resolution
    radius = (
        round(20 * 0.5**img_bin_level) or 1
    )  # 20 was the original value used when processing the highest resolution
    processed_img = preprocess(raw_arr, sigma, radius)

    print(f"T={T} -- getting and cleaning image thresholds") if VERBOSE else None
    hyst, hyst_clean, hyst_removed = get_thresholds(processed_img)
    skel = skeletonize(hyst)
    density_map = gaussian(skel, sigma=density_map_sigma)

    return density_map


def get_density_map_from_segmentations(
    dataset_name: str, T: int, density_map_sigma: float = 160, VERBOSE: bool = False
) -> np.ndarray:
    from bioio import BioImage
    from skimage.filters import gaussian
    from skimage.morphology import skeletonize
    from skimage.segmentation import find_boundaries
    from skimage.transform import pyramid_reduce

    from src.endo_pipeline.configs.dataset_io import get_zarr_path, load_config
    from src.endo_pipeline.io import load_segmentation
    from src.endo_pipeline.manifests import (
        get_segmentation_location_for_dataset,
        load_image_manifest,
    )

    DATASET_NAME_LIST = [config_data["name"] for config_data in load_config(config_type="data")]
    assert (
        dataset_name in DATASET_NAME_LIST
    ), f"dataset_name must be one of {DATASET_NAME_LIST}, not {dataset_name}"
    # get the lowest resolution binning level of the dataset so we can downsample the
    # classic cdh5 segmentations (which are done on the native resolution)
    dataset_filepath = Path(get_zarr_path(dataset_name))
    img_bin_level = sorted([level for level in BioImage(dataset_filepath).resolution_levels])[-1]

    print(f"T={T} -- loading segmentation") if VERBOSE else None
    # --------------------------------------------------------------------------
    # WARNING: This block loading the segmentation originally called a now
    # deprecated method. It has been replaced with a partial refactor using
    # newer methods, but has not been fully tested because this workflow is
    # archived. Use with caution!
    manifest = load_image_manifest("cdh5_classic")
    location = get_segmentation_location_for_dataset(manifest, dataset_name, 0, T)
    seg = load_segmentation(location)
    # --------------------------------------------------------------------------

    print(f"T={T} -- getting density map of image") if VERBOSE else None
    seg_borders = pyramid_reduce(skeletonize(find_boundaries(seg)), downscale=2**img_bin_level)
    density_map = gaussian(seg_borders, sigma=density_map_sigma)

    return density_map


def multiproc_workflow(args: tuple[str, int, dict, Path, float, bool]) -> None:
    run_density_workflow(*args)


def run_density_workflow(
    dataset_name: str,
    T: int,
    img_metadata: dict,
    out_dir: str | Path,
    density_map_sigma: float,
    VERBOSE: bool = False,
) -> None:

    from src.endo_pipeline.library.process.general_image_preprocessing import save_image_output

    print(f"Working on {dataset_name}, T={T}...")
    print("- getting density map...") if VERBOSE else None
    density_map = get_density_map_from_thresholds(dataset_name, T, density_map_sigma, VERBOSE)

    data_type = np.uint16
    density_map = (density_map * np.iinfo(data_type).max).astype(data_type)

    out_path = out_dir / f"{dataset_name}_T{T}_density_map.ome.tiff"
    images_out_metadata = {
        "image_name": (
            f"{dataset_name}_T{T}_sigma{density_map_sigma}"
            if "image_name" not in img_metadata
            else img_metadata["image_name"]
        ),
        "channel_names": [
            "density_map",
        ],
        "channel_colors": [
            (255, 255, 255),
        ],
        "physical_pixel_sizes": (
            (1, 1, 1)
            if "physical_pixel_sizes" not in img_metadata
            else img_metadata["physical_pixel_sizes"]
        ),
        "dim_order": "YX",
        "dtype": data_type,
    }
    print("- saving...") if VERBOSE else None
    save_image_output(
        out_path,
        [
            density_map,
        ],
        images_out_metadata,
    )


def main(
    n_proc: int = 1,
    dataset_name: str | None = None,
    save_output: bool = True,
    is_test: bool = False,
    verbose: bool = False,
) -> None:
    from multiprocessing import Pool

    import matplotlib as mpl
    from tqdm import tqdm

    mpl.rc("image", cmap="gray")

    from src.endo_pipeline.configs.dataset_io import (
        fire_parse_generate_dataset_name_list,
        get_dataset_duration_in_frames,
    )

    dataset_name_list = fire_parse_generate_dataset_name_list(dataset_name)

    for dataset_name in dataset_name_list:
        print(f"Initializing workflow for {dataset_name}...")
        out_dir, img_metadata = initialize_workflow(dataset_name, save_output, is_test)

        print(f"Getting timepoints for {dataset_name}...")
        timepoints = range(get_dataset_duration_in_frames(dataset_name))
        timepoints = timepoints[560:] if is_test else timepoints
        density_map_gaussian_kernel_sigma = 40
        analysis_args_queue = list(
            zip(
                [dataset_name] * len(timepoints),
                timepoints,
                *zip(
                    *[
                        (
                            img_metadata,
                            out_dir,
                            density_map_gaussian_kernel_sigma,
                            verbose,
                        )
                    ]
                    * len(timepoints),
                    strict=False,
                ),
                strict=False,
            )
        )

        print(f"Running workflow on {dataset_name}...")
        if n_proc > 1:
            if __name__ == "__main__":
                print("Starting multiprocessing...")
                with Pool(processes=n_proc) as pool:
                    list(
                        tqdm(
                            pool.imap(multiproc_workflow, analysis_args_queue, chunksize=5),
                            total=len(analysis_args_queue),
                        )
                    )
                    pool.close()
                    pool.join()
                print("Done multiprocessing.")
        else:
            print("Starting single processing...")
            for dataset_name_and_args in analysis_args_queue:
                multiproc_workflow(dataset_name_and_args)
            print("Done single processing.")

    print("\N{MICROSCOPE} Done analysis.")


if __name__ == "__main__":
    from src.endo_pipeline.configs.dataset_io import ipython_cli_flexecute

    ipython_cli_flexecute(main)
