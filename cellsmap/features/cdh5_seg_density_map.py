import numpy as np
from pathlib import Path
from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from cellsmap.util.io import load_dataset, load_config, get_dataset_info, get_zarr_path, get_time_interval_in_minutes, get_dataset_duration_in_frames
from bioio import BioImage
from cellsmap.util.cdh5_preprocessing import get_cdh5_classic_segmentation_paths, preprocess, get_thresholds, save_image_output, extract_T
import matplotlib as mpl
mpl.rc('image', cmap='gray')
from tqdm import tqdm
from multiprocessing import Pool

try:
    from IPython import get_ipython
except ModuleNotFoundError:
    pass
import fire



def get_chan_map(filepath: Path) -> dict:
    img = BioImage(filepath)
    return {name:index for index, name in enumerate(img.channel_names)}


def multiproc_wrapper(function, args):
    function(*args)


def ipython_cli_flexecute(function: callable, return_results=False, *args, **kwargs):
    """
    Executes function with arguments and keyword arguments in an IPython shell or via command line interface.
    """
    # The following try-except statement will run 'main' without fire.Fire if an interactive shell is in use,
    # otherwise it will run 'main' through fire.Fire so that arguments can easily be passed to 'main' through
    # some non-interactive shell like bash
    try:
        # the following line will return a string if an interactive shell is in use,
        # otherwise raises NameError since get_ipython is not imported from IPython
        # or returns None if get_ipython is present but script is being executed
        # from a non-interactive shell
        if get_ipython().__class__.__name__ != 'NoneType':
            print(f'Using interactive shell {get_ipython().__class__.__name__}.')
            results = function(*args, **kwargs)
        else: raise NameError
    except NameError:
        print('Using non-interactive shell.')
        results = fire.Fire(function)

    return results if return_results else None


def initialize_workflow(dataset_name, SAVE_OUTPUT=True, IS_TEST=False):
    # NOTE: this function is unique to each script
    SCT_NAME = Path(__file__).stem
    PRJ_DIR = Path('../').resolve() if not IS_TEST else Path('../../tests').resolve()
    assert PRJ_DIR.exists()
    out_dir = PRJ_DIR / f'results/{SCT_NAME}' / dataset_name

    # create output directory if it doesn't exist and get image metadata from the input image
    Path.mkdir(out_dir, exist_ok=True, parents=True) if SAVE_OUTPUT else None

    img = BioImage(Path(get_zarr_path(dataset_name)))
    px_res = img.physical_pixel_sizes
    t_res = get_time_interval_in_minutes(dataset_name)
    img_metadata = {'dataset_name': dataset_name,
                    'physical_pixel_sizes': px_res,
                    't_res (min)': t_res,
                    't_res (hr)': t_res / 60
                    }

    return out_dir, img_metadata


def get_density_map_from_thresholds(dataset_name: str, T: int, density_map_sigma, VERBOSE=False) -> np.ndarray:

    DATASET_NAME_LIST = [config_data['name'] for config_data in load_config(config_type='data')]
    assert dataset_name in DATASET_NAME_LIST, f'dataset_name must be one of {DATASET_NAME_LIST}, not {dataset_name}'

    # image_filepath, segmentation_channel, img_metadata, out_dir, gaussian_kernel_sigma, VERBOSE=False
    print(f'T={T} -- loading dataset') if VERBOSE else None
    # get the binning levels of the dataset so we can always use the lowest resolution
    dataset_filepath = Path(get_zarr_path(dataset_name))
    img_bin_level = sorted([level for level in BioImage(dataset_filepath).resolution_levels])[-1]
    # get the name of the cadherin channel
    chan_names = [get_dataset_info(dataset_name)['cdh5_channel_name']]
    # load the raw image data of from the cadherin channel
    raw_arr = load_dataset(dataset_name, channels=chan_names, time_start=T, time_end=T, level=img_bin_level).compute().squeeze()

    print(f'T={T} -- preprocessing image') if VERBOSE else None
    sigma = round(3 * 0.5**img_bin_level) or 1 # 3 was the original value used when processing the highest resolution
    radius = round(20 * 0.5**img_bin_level) or 1 # 20 was the original value used when processing the highest resolution
    processed_img = preprocess(raw_arr, sigma, radius)

    print(f'T={T} -- getting and cleaning image thresholds') if VERBOSE else None
    hyst, hyst_clean, hyst_removed = get_thresholds(processed_img)
    skel = skeletonize(hyst)
    density_map = gaussian(skel, sigma=density_map_sigma)

    return density_map


def get_density_map_from_segmentations(dataset_name: str, T: int, density_map_sigma: float=160, VERBOSE=False) -> np.ndarray:
    DATASET_NAME_LIST = [config_data['name'] for config_data in load_config(config_type='data')]
    assert dataset_name in DATASET_NAME_LIST, f'dataset_name must be one of {DATASET_NAME_LIST}, not {dataset_name}'

    print(f'T={T} -- loading segmentation') if VERBOSE else None
    image_filepaths = get_cdh5_classic_segmentation_paths(dataset_name, sort_paths=True)
    image_filepath = Path(str(*[fp for fp in image_filepaths if extract_T(fp.name) == T]))
    segmentation_channel = get_chan_map(image_filepath)['segmentations_merged']
    seg = BioImage(image_filepath).get_image_data('TCYX', C=segmentation_channel).squeeze()

    print(f'T={T} -- getting density map of image') if VERBOSE else None
    seg_borders = skeletonize(find_boundaries(seg))
    density_map = gaussian(seg_borders, sigma=density_map_sigma)

    return density_map


def multiproc_workflow(args):
    run_density_workflow(*args)


def run_density_workflow(dataset_name, T, img_metadata, out_dir, density_map_sigma, VERBOSE=False):

    print(f'Working on {dataset_name}, T={T}...')
    print(f'- getting density map...') if VERBOSE else None
    density_map = get_density_map_from_thresholds(dataset_name, T, density_map_sigma, VERBOSE)

    data_type = np.uint16
    density_map = (density_map * np.iinfo(data_type).max).astype(data_type)

    out_path = out_dir / f'{dataset_name}_T{T}_density_map.ome.tiff'
    images_out_metadata = {'image_name': f'{dataset_name}_T{T}_sigma{density_map_sigma}' if 'image_name' not in img_metadata else img_metadata['image_name'],
                        'channel_names': ['density_map',],
                        'channel_colors': [(255,255,255),],
                        'physical_pixel_sizes': (1,1,1) if 'physical_pixel_sizes' not in img_metadata else img_metadata['physical_pixel_sizes'],
                        'dim_order': 'YX',
                        'dtype': data_type,
                        }
    print(f'- saving...') if VERBOSE else None
    save_image_output(out_path, [density_map,], images_out_metadata)


def main(N_PROC=1, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=False):

    DATASET_NAME_LIST = [config_data['name'] for config_data in load_config(config_type='data')]

    for dataset_name in DATASET_NAME_LIST:

        # dataset_name = '20240305_T01_001'

        print(f'Initializing workflow for {dataset_name}...')# if VERBOSE else None
        out_dir, img_metadata = initialize_workflow(dataset_name, SAVE_OUTPUT, IS_TEST)

        print(f'Getting timepoints for {dataset_name}...')# if VERBOSE else None
        timepoints = range(get_dataset_duration_in_frames(dataset_name))
        timepoints = timepoints[560:] if IS_TEST else timepoints
        density_map_gaussian_kernel_sigma = 40
        analysis_args_queue = list(zip([dataset_name]*len(timepoints), timepoints, *zip(*[(img_metadata, out_dir, density_map_gaussian_kernel_sigma, VERBOSE)] * len(timepoints))))

        print(f'Running workflow on {dataset_name}...')# if VERBOSE else None
        if N_PROC > 1:
                if __name__ == '__main__':
                    print('Starting multiprocessing...')
                    with Pool(processes=N_PROC) as pool:
                        list(tqdm(pool.imap(multiproc_workflow, analysis_args_queue, chunksize=5), total=len(analysis_args_queue)))
                        pool.close()
                        pool.join()
                    print('Done multiprocessing.')
        else:
            print('Starting single processing...')
            for dataset_name_and_args in analysis_args_queue:
                multiproc_workflow(dataset_name_and_args)
            print('Done single processing.')

    print('\N{microscope} Done analysis.')


if __name__ == '__main__':
    ipython_cli_flexecute(main)
