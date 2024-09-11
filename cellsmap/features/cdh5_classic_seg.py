import numpy as np
from skimage.segmentation import find_boundaries
from pathlib import Path
from bioio import BioImage
from multiprocessing import Pool
from tqdm import tqdm
import fire
from cellsmap.util import io, cdh5_preprocessing as preproc



def initialize_workflow(dataset_name, SAVE_OUTPUT=True):
    # NOTE: this function is slightly different than the
    # one found in 'cdh5_nodes_and_edges.py'
    SCT_NAME = Path(__file__).stem

    prj_dir = Path('//allen/aics/assay-dev/users/Serge/')
    assert prj_dir.exists()
    out_dir = prj_dir / f'cellsmap_out/{SCT_NAME}'
    if SAVE_OUTPUT:
        Path.mkdir(out_dir, exist_ok=True, parents=True)

    img = BioImage(Path(io.get_zarr_path(dataset_name)))
    px_res = img.physical_pixel_sizes
    img_metadata = {'physical_pixel_sizes': px_res,
                    }

    return out_dir, img_metadata

def build_classic_seg_analysis_queue(DATASET_NAME_LIST, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=True) -> list:
    """
    Constructs a list of tuples of parameters to pass to generate_results. 
    """
    # done via single processing
    analysis_args_queue = []
    for dataset_name in DATASET_NAME_LIST:

        img_bin = 0
        DIM_MAP = io.get_dim_map('TYX')
        raw = io.load_dataset(dataset_name, time_start=0, resolution=img_bin)

        if IS_TEST:
            # T_list = range(0,1)
            # crop_y = slice(0, raw.shape[DIM_MAP["Y"]])
            # crop_x = slice(0, raw.shape[DIM_MAP["Y"]])
            T_list = range(0, raw.shape[DIM_MAP["T"]], 12)
            crop_y = slice(None, None)
            crop_x = slice(None, None)
            for T in T_list:
                analysis_args_queue.append([dataset_name, T, crop_y, crop_x, img_bin, SAVE_OUTPUT, IS_TEST, VERBOSE])
        else:
            # in the line below: replace 'raw.shape[DIM_MAP["T"]]' with an integer
            # to analyze a subset of timepoints in the timelapse
            T_list = range(0, raw.shape[DIM_MAP["T"]])
            crop_y = slice(None, None)
            crop_x = slice(None, None)
            for T in T_list:
                analysis_args_queue.append([dataset_name, T, crop_y, crop_x, img_bin, SAVE_OUTPUT, IS_TEST, VERBOSE])

    return analysis_args_queue

def generate_results_multiproc_wrapper(args):
    dataset_name, T, crop_y, crop_x, img_bin, SAVE_OUTPUT, IS_TEST, VERBOSE = args
    generate_results(dataset_name, T, crop_y, crop_x, img_bin, SAVE_OUTPUT=SAVE_OUTPUT, IS_TEST=IS_TEST, VERBOSE=VERBOSE)

def generate_results(dataset_name, T, crop_y, crop_x, img_bin, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=True):

    print(f'Working on {dataset_name} -- T={T}...')
    print(f'T={T} -- initializing workflow') if VERBOSE else None
    out_dir, img_metadata = initialize_workflow(dataset_name)

    print(f'T={T} -- loading dataset') if VERBOSE else None
    raw = io.load_dataset(dataset_name, time_start=0, resolution=img_bin)
    img_crop = (slice(T, T+1), crop_y, crop_x)
    raw_arr = raw[img_crop].compute().squeeze()

    print(f'T={T} -- preprocessing image') if VERBOSE else None
    processed_img = preproc.preprocess(raw_arr)

    print(f'T={T} -- getting and cleaning image thresholds') if VERBOSE else None
    hyst, hyst_clean, hyst_removed = preproc.get_thresholds(processed_img)

    print(f'T={T} -- getting and cleaning segmentations') if VERBOSE else None
    seg2_lab_no_mask_merge, seg2_lab = preproc.generate_segmentations(processed_img, hyst, hyst_clean, hyst_removed)
    seg2_lab_no_mask_merge_bounds = find_boundaries(seg2_lab_no_mask_merge)

    if SAVE_OUTPUT:
        print(f'T={T} -- saving image input and output overlays') if VERBOSE else None
        out_path = out_dir/f'{dataset_name}_T{T}.ome.tiff'
        images_out = [raw_arr, processed_img, hyst_clean, seg2_lab, seg2_lab_no_mask_merge, seg2_lab_no_mask_merge_bounds]
        images_out_metadata = {'image_name': dataset_name,
                                'channel_names': [('raw', 'processed', 'hysteresis_threshold', 'segmentations_initial', 'segmentations_merged', 'segmentations_merged_borders')], 
                                'channel_colors': [(255,255,255), (255,255,255), (0,255,255), (255,0,255), (255,0,255), (255,255,0)],
                                'physical_pixel_sizes': img_metadata['physical_pixel_sizes'],
                                'dim_order': 'CYX'
                                }
        preproc.save_image_output(out_path, images_out, images_out_metadata)
    else:
        pass



def main(N_PROC=1, SAVE_OUTPUT=True, IS_TEST=True, VERBOSE=False):

    DATASET_NAME_LIST = ['20240305_T01_001']

    analysis_args_queue = preproc.build_classic_seg_analysis_queue(DATASET_NAME_LIST, SAVE_OUTPUT=SAVE_OUTPUT, IS_TEST=IS_TEST, VERBOSE=VERBOSE)

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
    fire.Fire(main)
