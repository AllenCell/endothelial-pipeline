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

def save_validation_images(roi, img_arr, seg_arr, out_dir, dataset_name, T, padding=50):
    cell_id = roi.label
    track_id = roi.track_id
    validation_subfolder = out_dir / str(track_id)
    Path.mkdir(validation_subfolder, exist_ok=True, parents=True)

    expanded_bbox = tuple([slice(max(0, sl.start - padding), sl.stop + padding) for sl in roi.slice])

    crop_img = img_arr[(..., *expanded_bbox)].squeeze().compute()
    crop_seg = seg_arr[(..., *expanded_bbox)].squeeze().compute()
    crop_seg_outline = find_boundaries(crop_seg)
    track_of_interest = (crop_seg == cell_id * 1) + (crop_seg_outline > 0) * 2
    raw_img_crop = rescale_intensity(np.clip(crop_img, 0, np.percentile(crop_img, 98)), out_range=(0, 1))
    overlay = label2rgb(label=track_of_interest, image=raw_img_crop, bg_label=0, colors=['magenta', 'cyan'])

    fig, ax = plt.subplots()
    ax.imshow(overlay)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(validation_subfolder / f'{dataset_name}_track{track_id}_T{T}.png', bbox_inches='tight', pad_inches=0, dpi=180)
    plt.close(fig)

    return

def generate_and_save_validation_images(group):

    # unpack needed variables
    nm, dframe = group
    dataset_name = dframe['dataset_name'].unique()[0]
    scene_index = int(dframe['scene_index'].unique()[0])
    position = dframe['position'].unique()[0]
    T = dframe['T'].unique()[0]
    out_dir = dframe['out_dir'].unique()[0] / f'{dataset_name}/P{position}'

    print(f'Working on dataset {dataset_name}, P{position} T{T}...')

    raw_path = Path(get_dataset_info(dataset_name)['original_path'])
    seg_path = Path(get_cdh5_classic_segmentation_path(dataset_name, position))
    # NOTE: THE LINE OF CODE BELOW SEEMS TO WORK WITH SINGLE PROCESSING
    #     BUT NOT WITH MULTIPROCESSING. NOT SURE WHY GLOB WOULD DO THIS
    #     MULTIPROCESSING IS ABLE TO GET SEG_PATH CORRECTLY THOUGH
    # seg_path_list = list(seg_path.glob(f'*_T{T}.ome.tiff'))
    # if len(seg_path_list) == 0:
    #     print(f'No segmentation file found for {dataset_name} P{position} at T{T}.')
    #     return
    # elif len(seg_path_list) > 1:
    #     print(f'Multiple segmentation files found for {dataset_name} P{position} at T{T}. Files are: {seg_path}. Skipping.')
    #     return
    # else:
    #     seg_path = Path(seg_path_list[0])
    seg_path = seg_path / f'{dataset_name}_P{position}_T{T}.ome.tiff'
    if not seg_path.exists():
        print(f'No segmentation file found for {dataset_name} P{position} at T{T}.')
        return
    else:
        dim_order = 'TCZYX'
        dim_map = get_dim_map(dim_order)

        print(f'- loading segmentation image {dataset_name} P{position} T{T}...')
        seg = BioImage(seg_path)
        seg_arr = seg.get_image_dask_data(dim_order).squeeze()

        print(f'- loading raw image {dataset_name} P{position} T{T}...')
        img = BioImage(raw_path)
        img.set_scene(scene_index)
        cdh5_channel = get_dataset_info(dataset_name)['egfp_channel_index']
        img_arr = img.get_image_dask_data(dim_order).max(axis=dim_map['Z'], keepdims=True)
        img_arr = img_arr[T, cdh5_channel, :, :, :].squeeze()

        cell_id_to_track_id_map = dict(zip(dframe['label'], dframe['track_id']))

        props = measure.regionprops(seg_arr)
        rois = [reg for reg in props if reg.label in dframe[dframe['T']==T]['label'].unique()]
        for roi in rois:
            roi.track_id = cell_id_to_track_id_map[roi.label]
        padding = 50

        for roi in tqdm(rois, total=len(rois), desc=f'{dataset_name} P{position} T{T} saving track overlays'):
            save_validation_images(roi, img_arr, seg_arr, out_dir, dataset_name, T, padding=padding)
        return


def main(n_proc=1, dataset_name=None, t_final=None):

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
                                          verbose=True)
    analysis_queue_df = pd.DataFrame(analysis_queue)
    for dataset_name in dataset_name_list:
        tracking_df = get_tracking_data_filtered([dataset_name], as_dask=False)
        if t_final is not None:
            tracking_df = tracking_df.query('T < @t_final')

        tracking_df = tracking_df[tracking_df['dataset_name'] == dataset_name]
        analysis_queue_sub = analysis_queue_df[analysis_queue_df['dataset_name'] == dataset_name]
        position_scene_map = dict(zip(analysis_queue_sub['position'], analysis_queue_sub['scene_index']))
        tracking_df['scene_index'] = tracking_df['position'].transform(lambda x: position_scene_map[x])
        
        tracking_df['out_dir'] = out_dir

        min_track_duration = 120
        tracking_df = tracking_df[tracking_df['track_duration'] >= min_track_duration]

        df_subset_list = list(tracking_df.groupby(['dataset_name', 'position', 'T']))
        if n_proc > 1:
            if __name__ == '__main__':
                print('Using multiprocessing...')
                with Pool(processes=n_proc) as pool:
                    list(tqdm(pool.imap(generate_and_save_validation_images, df_subset_list), total=len(df_subset_list), desc='Timepoints complete (MP)'))
                    pool.close()
                    pool.join()
                print('Finished multiprocessing.')
        else:
            print('Using single processing...')
            for df_subset in tqdm(df_subset_list, total=len(df_subset_list), desc='Timepoints complete (1P)'):
                generate_and_save_validation_images(df_subset)
            print('Finished single processing.')
    
    print(f'\N{microscope} Done.')


if __name__ == '__main__':
    ipython_cli_flexecute(main)
