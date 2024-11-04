from pathlib import Path
from bioio import BioImage
from cellsmap.util import io, cdh5_preprocessing as preproc, shape_features as feat
from cellsmap.features.lib_tracking import ipython_cli_flexecute, update_track_table, save_track_labeled_images, run_workflow



def build_tracking_analysis_queue(dataset_name, SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=True):
    """
    Constructs a list of tuples of parameters to pass to generate_results. 
    """
    # done via single processing
    analysis_args_queue = []

    img_bin_level = 0
    DIM_MAP = io.get_dim_map('TCYX')
    # get the name of the cadherin channel
    chan_names = [config_data['cdh5_channel_name'] for config_data in io.load_config(config_type='data') if config_data['name'] == dataset_name]
    # load the raw image data of from the cadherin channel
    raw = io.load_dataset(dataset_name, channels=chan_names, time_start=0, level=img_bin_level)
    timeframe_eval_interval = 1

    if IS_TEST:
        T_list = range(573, raw.shape[DIM_MAP["T"]])
        crop_c = slice(None, None)
        crop_z = slice(None, None)
        crop_y = slice(None, None)
        crop_x = slice(None, None)
        for T in T_list:
            crop = {'T': T, 'C': crop_c, 'Z': crop_z, 'Y': crop_y, 'X': crop_x}
            analysis_args_queue.append([dataset_name, crop, img_bin_level, SAVE_OUTPUT, IS_TEST, VERBOSE])
    else:
        # in the line below: replace 'raw.shape[DIM_MAP["T"]]' with an integer
        # to analyze a subset of timepoints in the timelapse
        T_list = range(0, raw.shape[DIM_MAP["T"]], timeframe_eval_interval)
        crop_c = slice(None, None)
        crop_z = slice(None, None)
        crop_y = slice(None, None)
        crop_x = slice(None, None)
        for T in T_list:
            crop = {'T': T, 'C': crop_c,'Z': crop_z, 'Y': crop_y, 'X': crop_x}
            analysis_args_queue.append([dataset_name, crop, img_bin_level, SAVE_OUTPUT, IS_TEST, VERBOSE])

    return analysis_args_queue

def initialize_workflow(dataset_name, SAVE_OUTPUT=True, IS_TEST=False):
    # NOTE: this function is unique to each script
    SCT_NAME = Path(__file__).stem
    PRJ_DIR = Path('../').resolve() if not IS_TEST else Path('../../tests').resolve()
    assert PRJ_DIR.exists()
    val_dir = Path(f'//allen/aics/assay-dev/users/Serge/cellsmap_out/{SCT_NAME}')
    out_dir = PRJ_DIR / f'results/{SCT_NAME}'
    images_out_dir = val_dir / dataset_name
    tables_out_dir_tracks = out_dir / dataset_name
    out_dir_list = [images_out_dir, tables_out_dir_tracks, out_dir]

    # create output directories if they don't exist and get image metadata from the input image
    if SAVE_OUTPUT:
        [Path.mkdir(out_subdir, exist_ok=True, parents=True) for out_subdir in out_dir_list]

    img = BioImage(Path(io.get_zarr_path(dataset_name)))
    px_res = img.physical_pixel_sizes
    t_res = preproc.get_cdh5_classic_segmentation_time_resolution(dataset_name)
    img_metadata = {'dataset_name': dataset_name,
                    'physical_pixel_sizes': px_res,
                    't_res (min)': t_res,
                    't_res (hr)': t_res / 60
                    }

    return out_dir_list, img_metadata

def generate_results(dataset_name, SAVE_OUTPUT, IS_TEST, VERBOSE):

    # get a list of each timepoint for each dataset to be analyzed along with some corresponding arguments
    analysis_args_queue = build_tracking_analysis_queue(dataset_name, SAVE_OUTPUT=SAVE_OUTPUT, IS_TEST=IS_TEST, VERBOSE=VERBOSE)

    # generate tracks for each dataset using a list of timepoints couple to arguments
    track_table = generate_tracks(analysis_args_queue, tracking_metrics=['region_overlap'])

    # return track_table

def generate_tracks(analysis_args_queue, tracking_metrics=['centroid']):

    # create output directories if they don't exist and get image metadata from the input image
    # run analysis on each timepoint of each dataset
    track_table = []
    for dataset_name_and_args in analysis_args_queue:

        dataset_name, crop, img_bin_level, SAVE_OUTPUT, IS_TEST, VERBOSE = dataset_name_and_args
        print(f'Working on {dataset_name} -- T={crop["T"]}...')

        print(f'Initializing workflow...') if VERBOSE else None
        out_dir_list, img_metadata = initialize_workflow(dataset_name, SAVE_OUTPUT, IS_TEST)
        images_out_dir, tables_out_dir_tracks, out_dir = out_dir_list

        print(f'T={crop["T"]} -- generating results...') if VERBOSE else None
        # NOTE update_track_table is a generator
        track_labeled_image, current_tracks, track_table = update_track_table(dataset_name, crop, track_table, tracking_metrics, VERBOSE)

        print(f'T={crop["T"]} -- saving images...') if VERBOSE and SAVE_OUTPUT else None
        out_path = images_out_dir / f'{dataset_name}_T{crop["T"]}_track_labeled.tif'
        print(f'T={crop["T"]} -- saving to {out_path}') if VERBOSE else None
        chan_names = [config_data['cdh5_channel_name'] for config_data in io.load_config(config_type='data') if config_data['name'] == img_metadata['dataset_name']]
        raw_image = io.load_dataset(img_metadata['dataset_name'], channels=chan_names, time_start=crop["T"], time_end=crop["T"]).compute().squeeze()
        raw_channel = {'image': raw_image, 'name': 'raw_image', 'color': (255,255,255)}

        save_track_labeled_images(out_path, track_labeled_image, img_metadata, extra_channel=raw_channel) if SAVE_OUTPUT else None

    # save the table of the track_ids for the current dataset
    out_path = tables_out_dir_tracks / f'{dataset_name}_cdh5_classic_seg_tracking.tsv'
    track_table.to_csv(out_path, index=False, sep='\t') if SAVE_OUTPUT else None

    # return track_table

def main(SAVE_OUTPUT=True, IS_TEST=False, VERBOSE=False):

    DATASET_NAME_LIST = [config_data['name'] for config_data in io.load_config(config_type='data')]

    for dataset_name in DATASET_NAME_LIST:

        # track_table = generate_results(dataset_name, SAVE_OUTPUT, IS_TEST, VERBOSE)
        track_table = run_workflow(dataset_name, SAVE_OUTPUT, IS_TEST, VERBOSE)
        return track_table

    print('\N{microscope} Done analysis.')


if __name__ == '__main__':
    ipython_cli_flexecute(main)#, SAVE_OUTPUT=True, IS_TEST=True, VERBOSE=True)
