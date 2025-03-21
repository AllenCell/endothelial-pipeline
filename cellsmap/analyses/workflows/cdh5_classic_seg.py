from skimage.segmentation import find_boundaries
from pathlib import Path
from bioio import BioImage
from multiprocessing import Pool
from tqdm import tqdm
import fire
from cellsmap.util import cdh5_preprocessing as preproc, dataset_io, get_sldy_metadata as sldmd
try:
    from IPython import get_ipython
except ModuleNotFoundError:
    pass
from typing import Optional


# def initialize_workflow(dataset_name, save_output=True, is_test=False):
#     # NOTE: this function is unique to each script
#     SCT_NAME = Path(__file__).stem
#     PRJ_DIR = Path('../../').resolve() if not is_test else Path('../../../tests').resolve()
#     assert PRJ_DIR.exists()
#     val_dir = Path(f'//allen/aics/assay-dev/users/Serge/cellsmap_out/{SCT_NAME}')
#     out_dir = PRJ_DIR / 'results/cdh5_classic_seg'
#     out_dir_list = [out_dir, val_dir]
#     if save_output:
#         [Path.mkdir(out, exist_ok=True, parents=True) for out in out_dir_list]

#     img = BioImage(Path(dataset_io.get_zarr_path(dataset_name)))
#     px_res = img.physical_pixel_sizes
#     img_metadata = {'physical_pixel_sizes': px_res,
#                     }

#     return out_dir_list, img_metadata

def build_analysis_queue(dataset_name_list: list,
                         channels: Optional[list[int]]=None,
                         t_start: int=0, t_final: Optional[int]=None, t_step: int=1,
                         overwrite: bool=False, out_dir: Optional[str|Path]=None,
                         img_bin_level: int=0,
                         save_output: bool=True, is_test: bool=False, verbose: bool=False,
                         use_original_data=False) -> list:
    """
    Constructs a list of tuples of parameters to pass to generate_results.
    NOTE THE ARGUMENT 'channels' IS NOT YET IMPLEMENTED.
    """

    analysis_queue: list = []
    prj_dir = Path(__file__).parents[2] if not is_test else Path(__file__).parents[3]
    out_dir = out_dir or prj_dir / 'results' / Path(__file__).stem
    for dataset_name in dataset_name_list:
        img_path = Path(dataset_io.get_zarr_path(dataset_name)) if not use_original_data else Path(dataset_io.get_original_path(dataset_name))
        if img_path.exists():
            if use_original_data:
                img = BioImage(img_path)
            else:
                img_dict = {fp.name: BioImage(fp) for fp in img_path.glob('*.zarr')}
                img_list = list(img_dict.items())
        else:
            print(f"""No image found for dataset {dataset_name} with use_original_data={use_original_data} \n(filepath = {img_path}). \nSkipping...""")
            continue

        num_positions = dataset_io.get_number_of_positions(dataset_name)
        num_scenes = len(img.scenes) if use_original_data else len(img_list)

        assert num_positions % num_scenes == 0, f'Number of positions ({num_positions}) in data_config.yaml must be divisible by number of scenes ({num_scenes}) in the image file for dataset {dataset_name}'
        num_pos_in_T = num_positions // num_scenes
        num_pos_in_S = num_scenes

        positions_in_T, positions_in_S = [], []
        for scene_index in range(num_pos_in_S):
            positions_in_T += list(range(num_pos_in_T))
            positions_in_S += [scene_index] * num_pos_in_T

        # t_start, t_step, t_final = 0, 1, None
        for pos, (pos_in_T, pos_in_S) in enumerate(zip(positions_in_T, positions_in_S)):
            if use_original_data:
                current_img = img
                current_img.set_scene(pos_in_S)
                scene = current_img.current_scene
            else:
                scene, current_img = img_list[pos_in_S]
            # print(pos, pos_in_T, pos_in_S)
            # run the workflow on only the 20X datasets
            # print(pos, pos_in_T, pos_in_S)
            # print(sldmd.get_objective_info(img.metadata)['magnification'])

            print(f'Dataset{dataset_name}: Position {pos} (scene {scene}) -- processing...')

            # if magnification:
            #     if use_original_data:
            #         img_metadata = current_img.metadata
            #     else:
            #         original_path = dataset_io.get_original_path(dataset_name)
            #         img_metadata = sldmd.get_sldy_metadata(original_path)
            #     if sldmd.get_objective_info(img_metadata)['magnification'] != magnification:
            #         print(f'Dataset{dataset_name}: Position{pos} (scene {scene}) -- does not use {magnification}X magnification, skipping...')
            #         continue
            #     else:
            #         print(f'Dataset{dataset_name}: Position {pos} (scene {scene}) -- processing...')

            assert current_img.dims.T % num_pos_in_T == 0, f'Number of timepoints ({current_img.dims.T}) must be divisible by number of positions ({num_pos_in_T}) in the data_config.yaml for dataset {dataset_name} if number of positions does not equal the number of scenes in the image file.'
            # calculate the duration of the positions in frames (they must all have the same duration)
            duration_in_frames = t_final if isinstance(t_final, int) else current_img.dims.T // num_pos_in_T
            # correct the t_start, t_final, and t_step values to account for the intercalation of positions with timeframes
            t_start_adjusted = t_start or pos_in_T
            t_step_adjusted = t_step * num_pos_in_T
            t_final_adjusted = pos_in_T + duration_in_frames * num_pos_in_T
            t_range = range(t_start_adjusted, t_final_adjusted, t_step_adjusted)

            # print(len(t_range), list(t_range))

            # create a new analysis queue entry for each timepoint
            for t in t_range:
                if is_test and (t//num_pos_in_T) >= is_test:
                    break
                else:
                    pass
                crop_t = slice(t, t+1)
                crop_c = slice(None, None)
                crop_z = slice(None, None)
                crop_y = slice(None, None)
                crop_x = slice(None, None)
                crop = {'T': crop_t, 'C': crop_c,'Z': crop_z, 'Y': crop_y, 'X': crop_x}

                # if t >= t_start_adjusted and t < t_final_adjusted:
                analysis_queue.append({'dataset_name': dataset_name,
                                        'scene': scene,
                                        'position': pos,
                                        'crop': crop,
                                        # 'T': t, # replace this with the crop variable from the old method?
                                        'input_path': img_path,
                                        'output_dir': out_dir,
                                        'save_output': save_output,
                                        'overwrite': overwrite,
                                        'img_bin_level': img_bin_level,
                                        'use_original_data': use_original_data,
                                        'is_test': is_test,
                                        'verbose': verbose})
    return analysis_queue

def generate_results_multiproc_wrapper(args):
    dataset_name = args['dataset_name']
    scenes = (args['scene'],)
    crop = args['crop']
    T = crop['T'].start
    img_bin_level = args['img_bin_level']
    save_output = args['save_output']
    out_dir = args['output_dir']
    is_test = args['is_test']
    verbose = args['verbose']
    generate_results(dataset_name, T, scenes, img_bin_level, out_dir=out_dir, save_output=save_output, is_test=is_test, verbose=verbose)

def generate_results(dataset_name, T, scene_list=None, img_bin_level=0, out_dir=None, save_output=True, verbose=True):

    print(f'Working on {dataset_name} -- T={T}...')
    print(f'T={T} -- initializing workflow') if verbose else None
    # out_dir_list, img_metadata = initialize_workflow(dataset_name, save_output, is_test)
    # out_dir, val_dir = out_dir_list
    seg_dir = out_dir / 'segmentations'
    val_dir = out_dir / 'validation'

    print(f'T={T} -- loading dataset') if verbose else None
    # get the name of the cadherin channel
    # chan_names = [chan_name for chan_name in dataset_io.get_available_channels(dataset_name) if chan_name in ['CDH5', 'CDH5_Tubulin']]
    # load the raw image data of from the cadherin channel
    # raw_arr = dataset_io.load_dataset(dataset_name, channels=chan_names, time_start=T, time_end=T, level=img_bin_level).compute().squeeze()

    build_analysis_queue([dataset_name], t_start=T, t_final=T+1, overwrite=False, out_dir=None, img_bin_level=img_bin_level, save_output=False, is_test=False, verbose=False, use_original_data=True)

    original_path = Path(dataset_io.get_original_path(dataset_name))
    zarr_path = Path(dataset_io.get_zarr_path(dataset_name))

    if zarr_path.exists():
        img_path = zarr_path
        img_dict = {fp.name: BioImage(fp) for fp in img_path.glob('*.zarr')}
        scene_list = scene_list if scene_list else img_dict.keys()
    else:
        img_path = original_path
        img = BioImage(img_path)
        scene_list = scene_list or img.scenes

    dim_map = dataset_io.get_dim_map('TCZYX')
    egfp_index = dataset_io.get_dataset_info(dataset_name)['egfp_channel_index']

    for scene in scene_list:
        if zarr_path.exists():
            current_img = img_dict[scene]
        else:
            current_img = img
            current_img.set_scene(scene)
        raw_dask_arr = current_img.get_image_dask_data('TCZYX', T=T, C=egfp_index)
        raw_arr_MIP = raw_dask_arr.max(axis=dim_map['Z'], keepdims=True).compute()


        print(f'T={T} -- preprocessing image') if verbose else None
        processed_img = preproc.preprocess(raw_arr_MIP)

        print(f'T={T} -- getting and cleaning image thresholds') if verbose else None
        hyst, hyst_clean, hyst_removed = preproc.get_thresholds(processed_img)

        print(f'T={T} -- getting and cleaning segmentations') if verbose else None
        seg2_lab_no_mask_merge, seg2_lab = preproc.generate_segmentations(processed_img, hyst, hyst_clean, hyst_removed)
        seg2_lab_no_mask_merge_bounds = find_boundaries(seg2_lab_no_mask_merge)

        if save_output:
            # save images for validation
            print(f'T={T} -- saving image input and output overlays') if verbose else None
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
            out_path = seg_dir / dataset_name / f'{dataset_name}_T{T}.ome.tiff'
            Path.mkdir(seg_dir / dataset_name, exist_ok=True, parents=True)
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



def main(n_proc=1, save_output=True, overwrite=False, is_test=False, verbose=False):

    dataset_name_list = [config_data['name']
                         for config_data in dataset_io.load_config(config_type='data')
                         if (config_data['microscope'] == '3i'
                             and config_data['live_or_fixed_sample'] == 'live')]

    # TODO if possible it would be good to use parallel processing to build analysis_queue
    dataset_name_list = ['20241016_20X']
    analysis_queue = build_analysis_queue(dataset_name_list, t_final=10, overwrite=overwrite, save_output=save_output, is_test=is_test, verbose=verbose, use_original_data=False)
    # analysis_args_queue = build_classic_seg_analysis_queue(dataset_name_list, save_output=save_output, is_test=is_test, verbose=verbose)

    if n_proc > 1:
            if __name__ == '__main__':
                print('Starting multiprocessing...')
                with Pool(processes=n_proc) as pool:
                    list(tqdm(pool.imap(generate_results_multiproc_wrapper, analysis_queue, chunksize=5), total=len(analysis_queue)))
                    pool.close()
                    pool.join()
                print('Done multiprocessing.')
    else:
        for dataset_name_and_args in analysis_queue:
            generate_results_multiproc_wrapper(dataset_name_and_args)

    print('\N{microscope} Done analysis.')

if __name__ == '__main__':
    # ipython_cli_flexecute runs a function via either
    # the command line or an interactive python shell
    dataset_io.ipython_cli_flexecute(main)
