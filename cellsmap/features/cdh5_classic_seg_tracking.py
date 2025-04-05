from pathlib import Path
from bioio import BioImage
from cellsmap.util import dataset_io, cdh5_preprocessing as preproc
from cellsmap.features.lib_tracking import run_tracking
from cellsmap.util.set_output import get_output_path
from cellsmap.util.dataset_io import load_config, ipython_cli_flexecute
from cellsmap.util.general_image_preprocessing import build_analysis_queue, get_chan_map
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd

def initialize_workflow(dataset_name, SAVE_OUTPUT=True, IS_TEST=False):
    # NOTE: this function is unique to each workflow
    SCT_NAME = Path(__file__).stem
    PRJ_DIR = Path('../').resolve() if not IS_TEST else Path('../../tests').resolve()
    assert PRJ_DIR.exists()
    out_dir = PRJ_DIR / f'results/{SCT_NAME}' / dataset_name

    # create output directory if it doesn't exist and get image metadata from the input image
    Path.mkdir(out_dir, exist_ok=True, parents=True) if SAVE_OUTPUT else None

    img = BioImage(Path(dataset_io.get_zarr_path(dataset_name)))
    px_res = img.physical_pixel_sizes
    t_res = dataset_io.get_time_interval_in_minutes(dataset_name)
    img_metadata = {'dataset_name': dataset_name,
                    'physical_pixel_sizes': px_res,
                    't_res (min)': t_res,
                    't_res (hr)': t_res / 60
                    }

    return out_dir, img_metadata


def run_workflow(queue):
    (dataset_name, position), queue_df = queue
    save_output = queue_df['save_output'][0]
    is_test = queue_df['is_test'][0]
    verbose = queue_df['verbose'][0]
    out_dir = queue_df['output_dir'][0] / f'P{position}'

    # out_dir, img_metadata = initialize_workflow(dataset_name, save_output, is_test)

    image_filepaths = preproc.get_cdh5_classic_segmentation_paths(dataset_name, sort_paths=True)
    image_filepaths = image_filepaths[:10] if is_test else image_filepaths
    if image_filepaths:
        segmentation_channel = get_chan_map(image_filepaths[0])['segmentations_merged']

        raw_fps = dataset_io.get_dataset_info(dataset_name)['zarr_path']
        raw_channel = get_chan_map(raw_fps)[str(*[chan for chan in dataset_io.get_available_channels(dataset_name) if chan in ('CDH5', 'CDH5_Tubulin')])]

        run_tracking(in_dir=image_filepaths, out_dir=out_dir, tracking_metrics=['region_overlap'],
                     sorting_key=preproc.extract_T, C=segmentation_channel, extra_in_dir=raw_fps, extra_C=raw_channel, img_metadata=img_metadata,
                     SAVE_OUTPUT=save_output, VERBOSE=verbose)
    else:
        print(f'No segmentation images found for {dataset_name}. Skipping tracking analysis. If this is unexpected check that the IS_TEST argument is set to False.')
        return

def main(n_proc=1, dataset_name=None, save_output=True, overwrite=False, is_test=False, verbose=False):

    if dataset_name == None:
        dataset_name_list = [config_data['name']
                            for config_data in load_config(config_type='data')
                            if (config_data['microscope'] == '3i'
                                and config_data['live_or_fixed_sample'] == 'live')
                                and 'AICS-126' in config_data['cell_lines']]
    else:
        dataset_name_list = [dataset_name]

    analysis_queue = build_analysis_queue(dataset_name_list,
                                          save_output=save_output,
                                          out_dir=get_output_path(Path(__file__).stem, verbose=False),
                                          overwrite=overwrite,
                                          verbose=verbose,
                                          is_test=is_test,
                                          image_validation_frequency=1,
                                          use_original_data=True)

    analysis_queue_df = pd.DataFrame(analysis_queue)
    analysis_queue_per_position = list(analysis_queue_df.groupby(['dataset_name', 'position']))

    if n_proc > 1:
        if __name__ == '__main__':
            print('Starting multiprocessing...')
            with Pool(processes=n_proc) as pool:
                list(tqdm(pool.imap(run_workflow, analysis_queue_per_position, chunksize=1), total=len(analysis_queue_per_position)))
                pool.close()
                pool.join()
            print('Done multiprocessing.')
    else:
        for queue in analysis_queue_per_position:
            run_workflow(queue, save_output, is_test, verbose)

    print('\N{microscope} Done analysis.')


if __name__ == '__main__':
    ipython_cli_flexecute(main)
