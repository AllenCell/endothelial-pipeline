from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cellsmap.util.dataset_io import get_tracking_data_filtered, load_config, get_cdh5_classic_segmentation_path, get_dataset_info, ipython_cli_flexecute
from cellsmap.util.set_output import get_output_path
from cellsmap.util.general_image_preprocessing import get_dim_map, build_analysis_queue
from bioio import BioImage
from skimage import measure
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries
from multiprocessing import Pool
from tqdm import tqdm


def open_image_from_queue(args_dict):
    dataset_name = args_dict['dataset_name']
    scene_index = int(args_dict['scene_index'])
    position = int(args_dict['position'])
    T = int(args_dict['T'])

    dim_order = 'TCZYX'
    dim_map = get_dim_map(dim_order)

    seg_dir = Path(get_cdh5_classic_segmentation_path(dataset_name, position))
    seg_path = seg_dir / f'{dataset_name}_P{position}_T{T}.ome.tiff'
    seg = BioImage(seg_path)
    seg_arr = seg.get_image_dask_data(dim_order, T=0, C=0).squeeze().compute()

    raw_path = Path(get_dataset_info(dataset_name)['original_path'])
    img = BioImage(raw_path)
    img.set_scene(scene_index)
    cdh5_channel = int(get_dataset_info(dataset_name)['egfp_channel_index'])
    img_dask = img.get_image_dask_data(dim_order, T=T, C=cdh5_channel)
    img_arr = img_dask.max(axis=dim_map['Z'], keepdims=True).squeeze().compute()


    # print(f'Image shape: {img_arr.shape}')
    print('raw', T, raw_path.exists(), img_arr.shape)
    print('seg', T, seg_path.exists(), seg_arr.shape)

def main_queue(n_proc=1, dataset_name='20241016_20X', t_final=5, verbose=False):
    """t_final is really only used for testing purposes."""
    out_dir = Path(get_output_path(Path(__file__).stem, verbose=False))

    if dataset_name == None:
        dataset_name_list = [config_data['name']
                            for config_data in load_config(config_type='data')
                            if (config_data['microscope'] == '3i'
                                and config_data['live_or_fixed_sample'] == 'live')
                                and 'AICS-126' in config_data['cell_lines']
                                and config_data['duration'] > 1]
    else:
        dataset_name_list = [dataset_name]

    analysis_queue = build_analysis_queue(dataset_name_list,
                                          t_final=t_final,
                                          use_original_data=True,
                                          out_dir=out_dir,
                                          verbose=verbose)
    for dataset_name in dataset_name_list:
        # base_path = Path('//allen/aics/endothelial/morphological_features/analysis/track_filtering')
        # data_path = base_path / f"{dataset_name}_filtered_tracking_data.tsv"
        # tracking_df = pd.read_csv(data_path, sep='\t')


        tracking_df = get_tracking_data_filtered([dataset_name], as_dask=False)
        # if t_final is not None:
        #     tracking_df = tracking_df.query('T < @t_final')

        # tracking_df = tracking_df[tracking_df['dataset_name'] == dataset_name]
        # analysis_queue_sub = analysis_queue_df[analysis_queue_df['dataset_name'] == dataset_name]
        # position_scene_map = dict(zip(analysis_queue_sub['position'], analysis_queue_sub['scene_index']))
        # tracking_df['scene_index'] = tracking_df['position'].transform(lambda x: position_scene_map[x])
        

        # min_track_duration = 120
        # tracking_df = tracking_df[tracking_df['track_duration'] >= min_track_duration]

        # nm, df_subset_list = zip(*tracking_df.groupby(['dataset_name', 'position', 'T']))
        # record_list_all = [df.to_dict('records') for df in df_subset_list]
        if n_proc > 1:
            if __name__ == '__main__':
                print('Using multiprocessing...')
                with Pool(processes=n_proc) as pool:
                    list(tqdm(pool.imap(open_image_from_queue, analysis_queue), total=len(analysis_queue), desc='Timepoints complete (MP)'))
                    pool.close()
                    pool.join()
                print('Finished multiprocessing.')
        else:
            print('Using single processing...')
            for record in tqdm(analysis_queue, total=len(analysis_queue), desc='Timepoints complete (1P)'):
                open_image_from_queue(record)
            print('Finished single processing.')
    
    print(f'\N{microscope} Done.')




def open_image(arg):
    filepath, T = arg
    img = BioImage(filepath)
    img.set_scene(0)
    img_dask = img.get_image_dask_data('TCZYX', T=T, C=0)
    img_arr = img_dask.compute().squeeze()
    print(img_arr.shape)


def main(n_proc=1):

    filepath = Path('//allen/programs/allencell/data/proj0/fa3/2f4/77f/bde/c1a/d98/749/61e/190/0d1/23/20241016_GE0007381_20X_timelapse_sldy/20241016_GE00007381_20X_timelapse_sldy.sldy')
    t_range = range(20)

    args = list(zip([filepath] * len(t_range), t_range))
    with Pool(processes=n_proc) as pool:
        list(tqdm(pool.imap(open_image, args, chunksize=1), total=len(args)))
        pool.close()
        pool.join()


if __name__ == '__main__':
    ipython_cli_flexecute(main_queue)


# if __name__ == '__main__':
#     ipython_cli_flexecute(main)
