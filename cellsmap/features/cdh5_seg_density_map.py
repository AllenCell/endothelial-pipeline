import numpy as np
import pandas as pd
from pathlib import Path
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from cellsmap.util.shape_features import numpy_mesh_coords
from cellsmap.util.io import load_dataset, load_config, get_dataset_info, get_zarr_path, get_available_channels, get_time_interval_in_minutes
from cellsmap.util.cdh5_preprocessing import get_cdh5_classic_segmentation, save_image_output
from bioio import BioImage
from cellsmap.util.cdh5_preprocessing import get_cdh5_classic_segmentation_paths, get_dim_map, extract_T
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


def generate_density_map(image_filepath, segmentation_channel, dtype=np.uint8):
    seg = BioImage(image_filepath).get_image_data('TCYX', C=segmentation_channel).squeeze()
    seg_borders = skeletonize(find_boundaries(seg))
    density_map = rescale_intensity(gaussian(seg_borders, sigma=40), out_range=dtype)

    return density_map


def run_workflow(image_filepath, segmentation_channel, img_metadata, out_dir, VERBOSE=False):

    print(f'Working on {Path(image_filepath).name}...')
    print(f'- getting density map...') if VERBOSE else None
    density_map = generate_density_map(image_filepath, segmentation_channel)

    out_path = out_dir / f'{Path(image_filepath).name.split(".")[0]}_density_map{"".join(Path(image_filepath).suffixes)}'
    images_out_metadata = {'image_name': out_path.stem if 'image_name' not in img_metadata else img_metadata['image_name'],
                        'channel_names': ['density_map',],
                        'channel_colors': [(255,255,255),],
                        'physical_pixel_sizes': (1,1,1) if 'physical_pixel_sizes' not in img_metadata else img_metadata['physical_pixel_sizes'],
                        'dim_order': 'YX',
                        'dtype': np.uint8,
                        }
    print(f'- saving...') if VERBOSE else None
    save_image_output(out_path, [density_map,], images_out_metadata)



def main(N_PROC=1, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=False):

    DATASET_NAME_LIST = [config_data['name'] for config_data in load_config(config_type='data')]

    # analysis_args_queue = build_analysis_queue(DATASET_NAME_LIST, SAVE_OUTPUT=SAVE_OUTPUT, IS_TEST=IS_TEST, VERBOSE=VERBOSE)
    for dataset_name in DATASET_NAME_LIST:

        # dataset_name = '20240305_T01_001'

        print(f'Initializing workflow for {dataset_name}...')# if VERBOSE else None
        out_dir, img_metadata = initialize_workflow(dataset_name, SAVE_OUTPUT, IS_TEST)
        # analysis_args_queue = build_tracking_analysis_queue(dataset_name, SAVE_OUTPUT, IS_TEST, VERBOSE)
        # dataset_name, crop, img_bin_level, SAVE_OUTPUT, IS_TEST, VERBOSE = analysis_args_queue[i]

        print(f'Getting segmentation filepaths for {dataset_name}...')# if VERBOSE else None
        image_filepaths = get_cdh5_classic_segmentation_paths(dataset_name, sort_paths=True)
        image_filepaths = image_filepaths[560:] if IS_TEST else image_filepaths
        analysis_args_queue = []
        for img_fp in image_filepaths:
            print(f'Getting channels for filepaths in {img_fp}...')# if VERBOSE else None
            analysis_args_queue.append((Path(img_fp).name, get_chan_map(img_fp)['segmentations_merged'], img_metadata, out_dir, VERBOSE))

        print('Creating multiprocessing wrapper...')# if VERBOSE else None
        multiproc_workflow = lambda args: multiproc_wrapper(run_workflow, args)

        if N_PROC > 1:
                if __name__ == '__main__':
                    print('Starting multiprocessing...')
                    with Pool(processes=N_PROC) as pool:
                        list(tqdm(pool.imap(multiproc_workflow, analysis_args_queue, chunksize=5), total=len(analysis_args_queue)))
                        pool.close()
                        pool.join()
                    print('Done multiprocessing.')
        else:
            for dataset_name_and_args in analysis_args_queue:
                print('Starting single processing...')
                multiproc_workflow(dataset_name_and_args)
                print('Done single processing.')

    print('\N{microscope} Done analysis.')


if __name__ == '__main__':
    ipython_cli_flexecute(main)
