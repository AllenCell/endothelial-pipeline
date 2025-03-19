from skimage.segmentation import find_boundaries
from pathlib import Path
from bioio import BioImage
from multiprocessing import Pool
from tqdm import tqdm
import fire
from cellsmap.util import cdh5_preprocessing as preproc, io
try:
    from IPython import get_ipython
except ModuleNotFoundError:
    pass


def initialize_workflow(dataset_name, SAVE_OUTPUT=True, IS_TEST=False):
    # NOTE: this function is unique to each script
    SCT_NAME = Path(__file__).stem
    PRJ_DIR = Path('../../').resolve() if not IS_TEST else Path('../../../tests').resolve()
    assert PRJ_DIR.exists()
    val_dir = Path(f'//allen/aics/assay-dev/users/Serge/cellsmap_out/{SCT_NAME}')
    out_dir = PRJ_DIR / 'results/cdh5_classic_seg'
    out_dir_list = [out_dir, val_dir]
    if SAVE_OUTPUT:
        [Path.mkdir(out, exist_ok=True, parents=True) for out in out_dir_list]

    img = BioImage(Path(io.get_zarr_path(dataset_name)))
    px_res = img.physical_pixel_sizes
    img_metadata = {'physical_pixel_sizes': px_res,
                    }

    return out_dir_list, img_metadata

def build_classic_seg_analysis_queue(DATASET_NAME_LIST, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=True) -> list:
    """
    Constructs a list of tuples of parameters to pass to generate_results. 
    """
    # done via single processing
    analysis_args_queue = []
    for dataset_name in DATASET_NAME_LIST:

        img_bin_level = 0
        DIM_MAP = io.get_dim_map('TCYX')
        # get the name of the cadherin channel
        chan_names = [chan_name for chan_name in io.get_available_channels(dataset_name) if chan_name in ['CDH5', 'CDH5_Tubulin']]
        # load the raw image data of from the cadherin channel
        raw = io.load_dataset(dataset_name, channels=chan_names, time_start=0, level=img_bin_level)

        if IS_TEST:
            T_list = range(0, 5)
            crop_c = slice(None, None)
            crop_z = slice(None, None)
            crop_y = slice(None, None)
            crop_x = slice(None, None)
            for T in T_list:
                crop = {'T': T, 'C': crop_c,'Z': crop_z, 'Y': crop_y, 'X': crop_x}
                analysis_args_queue.append([dataset_name, crop, img_bin_level, SAVE_OUTPUT, IS_TEST, VERBOSE])
        else:
            # in the line below: replace 'raw.shape[DIM_MAP["T"]]' with an integer
            # to analyze a subset of timepoints in the timelapse
            T_list = range(0, raw.shape[DIM_MAP["T"]])
            crop_c = slice(None, None)
            crop_z = slice(None, None)
            crop_y = slice(None, None)
            crop_x = slice(None, None)
            for T in T_list:
                crop = {'T': T, 'C': crop_c,'Z': crop_z, 'Y': crop_y, 'X': crop_x}
                analysis_args_queue.append([dataset_name, crop, img_bin_level, SAVE_OUTPUT, IS_TEST, VERBOSE])

    return analysis_args_queue

def generate_results_multiproc_wrapper(args):
    dataset_name, crop, img_bin_level, SAVE_OUTPUT, IS_TEST, VERBOSE = args
    generate_results(dataset_name, crop, img_bin_level, SAVE_OUTPUT=SAVE_OUTPUT, IS_TEST=IS_TEST, VERBOSE=VERBOSE)

def generate_results(dataset_name, crop, img_bin_level, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=True):

    T = crop["T"]

    print(f'Working on {dataset_name} -- T={T}...')
    print(f'T={T} -- initializing workflow') if VERBOSE else None
    out_dir_list, img_metadata = initialize_workflow(dataset_name, SAVE_OUTPUT, IS_TEST)
    out_dir, val_dir = out_dir_list

    print(f'T={T} -- loading dataset') if VERBOSE else None
    # get the name of the cadherin channel
    chan_names = [chan_name for chan_name in io.get_available_channels(dataset_name) if chan_name in ['CDH5', 'CDH5_Tubulin']]
    # load the raw image data of from the cadherin channel
    raw_arr = io.load_dataset(dataset_name, channels=chan_names, time_start=T, time_end=T, level=img_bin_level).compute().squeeze()

    print(f'T={T} -- preprocessing image') if VERBOSE else None
    processed_img = preproc.preprocess(raw_arr)

    print(f'T={T} -- getting and cleaning image thresholds') if VERBOSE else None
    hyst, hyst_clean, hyst_removed = preproc.get_thresholds(processed_img)

    print(f'T={T} -- getting and cleaning segmentations') if VERBOSE else None
    seg2_lab_no_mask_merge, seg2_lab = preproc.generate_segmentations(processed_img, hyst, hyst_clean, hyst_removed)
    seg2_lab_no_mask_merge_bounds = find_boundaries(seg2_lab_no_mask_merge)

    if SAVE_OUTPUT:
        # save images for validation
        print(f'T={T} -- saving image input and output overlays') if VERBOSE else None
        val_path = val_dir / dataset_name / f'{dataset_name}_T{T}.ome.tiff'
        Path.mkdir(val_dir / dataset_name, exist_ok=True, parents=True)
        images_out = [raw_arr, processed_img, hyst_clean, seg2_lab, seg2_lab_no_mask_merge, seg2_lab_no_mask_merge_bounds]
        images_out_metadata = {'image_name': dataset_name,
                               'channel_names': ['raw', 'processed', 'hysteresis_threshold', 'segmentations_initial', 'segmentations_merged', 'segmentations_merged_borders'], 
                               'channel_colors': [(255,255,255), (255,255,255), (0,255,255), (255,0,255), (255,0,255), (255,255,0)],
                               'physical_pixel_sizes': img_metadata['physical_pixel_sizes'],
                               'dim_order': 'YX'
                               }
        preproc.save_image_output(val_path, images_out, images_out_metadata)

        # save just the cdh5 segmentations
        out_path = out_dir / dataset_name / f'{dataset_name}_T{T}.ome.tiff'
        Path.mkdir(out_dir / dataset_name, exist_ok=True, parents=True)
        images_out = [seg2_lab_no_mask_merge,]
        images_out_metadata = {'image_name': dataset_name,
                               'channel_names': ['segmentations_merged'], 
                               'channel_colors': [(255,255,255),],
                               'physical_pixel_sizes': img_metadata['physical_pixel_sizes'],
                               'dim_order': 'YX'
                               }
        preproc.save_image_output(out_path, images_out, images_out_metadata)
    else:
        pass



def main(N_PROC=1, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=False):

    DATASET_NAME_LIST = [config_data['name'] for config_data in io.load_config(config_type='data')]

    analysis_args_queue = build_classic_seg_analysis_queue(DATASET_NAME_LIST, SAVE_OUTPUT=SAVE_OUTPUT, IS_TEST=IS_TEST, VERBOSE=VERBOSE)

    if N_PROC > 1:
            if __name__ == '__main__':
                print('Starting multiprocessing...')
                with Pool(processes=N_PROC) as pool:
                    list(tqdm(pool.imap(generate_results_multiproc_wrapper, analysis_args_queue, chunksize=5), total=len(analysis_args_queue)))
                    pool.close()
                    pool.join()
                print('Done multiprocessing.')
    else:
        for dataset_name_and_args in analysis_args_queue:
            generate_results_multiproc_wrapper(dataset_name_and_args)

    print('\N{microscope} Done analysis.')

if __name__ == '__main__':
    # The following try-except statement will run 'main' without fire.Fire if an interactive shell is in use,
    # otherwise it will run 'main' through fire.Fire so that arguments can easily be passed to 'main' through
    # some non-interactive shell like bash
    try:
        # the following line will return a string if an interactive shell is in use,
        # otherwise raises NameError if get_ipython is not imported from IPython
        # or returns None if get_ipython is present but script is being executed
        # from a non-interactive shell
        if get_ipython().__class__.__name__ != 'NoneType':
            print(f'Using interactive shell {get_ipython().__class__.__name__}.')
            main()
        else: raise NameError
    except NameError:
        print('Using non-interactive shell.')
        fire.Fire(main)
