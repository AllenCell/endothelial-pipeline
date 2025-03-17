from pathlib import Path
from bioio import BioImage
import bioio_sldy, bioio_nd2, bioio_ome_zarr
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
# from skimage.segmentation import watershed, find_boundaries
# from skimage.morphology import dilation, disk
from cellsmap.util import io, get_sldy_metadata as sldmd
from cellsmap.util.cdh5_preprocessing import save_image_output
from cellpose import models
from tqdm import tqdm
from scipy.signal import find_peaks
from cellsmap.features.cdh5_classic_seg_tracking import ipython_cli_flexecute
from multiprocessing import Pool

def get_reader(filename_or_filepath: Path|str):
    available_readers = {'.sldy': bioio_sldy.Reader,
                         '.nd2': bioio_nd2.Reader,
                         '.zarr': bioio_ome_zarr.Reader}
    reader = available_readers[Path(filename_or_filepath).suffix]
    return reader

# def build_analysis_queue(dataset_name_list: list, t_to_eval: slice|range|list=slice(None), save_output=False, is_test=False, use_original_data=False) -> list:
def build_analysis_queue(dataset_name_list: list, t_start: int=0, t_final: int=None, t_step: int=1, save_output=True, out_dir: str=None, is_test=False, use_original_data=False) -> list:
    analysis_queue: list = []
    prj_dir = Path(__file__).parents[2] if not is_test else Path(__file__).parents[3]
    out_dir = out_dir or prj_dir / 'results' / Path(__file__).stem
    for dataset_name in dataset_name_list:
        img_path = Path(io.get_zarr_path(dataset_name)) if not use_original_data else Path(io.get_original_path(dataset_name))
        img = BioImage(img_path, reader=get_reader(img_path.name))

        num_positions = io.get_number_of_positions(dataset_name)
        # duration = io.get_dataset_duration_in_frames(dataset_name)

        # assert len(img.scenes) == num_positions or len(img.scenes) == 1, f'Number of positions ({num_positions}) in data_config.yaml does not match number of scenes in the image file for dataset {dataset_name}'
        assert num_positions % len(img.scenes) == 0, f'Number of positions ({num_positions}) in data_config.yaml must be divisible by number of scenes ({len(img.scenes)}) in the image file for dataset {dataset_name}'
        num_pos_in_T = num_positions // len(img.scenes)
        num_pos_in_S = len(img.scenes)

        positions_in_T, positions_in_S = [], []
        for scene_index in range(num_pos_in_S):
            positions_in_T += list(range(num_pos_in_T))
            positions_in_S += [scene_index] * num_pos_in_T

        t_start, t_step, t_final = 0, 1, None
        for pos, (pos_in_T, pos_in_S) in enumerate(zip(positions_in_T, positions_in_S)):
            img.set_scene(pos_in_S)
            # print(pos, pos_in_T, pos_in_S)
            # run the workflow on only the 20X datasets
            # print(pos, pos_in_T, pos_in_S)
            # print(sldmd.get_objective_info(img.metadata)['magnification'])
            if sldmd.get_objective_info(img.metadata)['magnification'] != 20:
                print(f'Position{pos} (scene {img.current_scene}) -- does not use 20X magnification, skipping...')
            else:
                print(f'Position {pos} (scene {img.current_scene}) -- processing...')
                assert img.dims.T % num_pos_in_T == 0, f'Number of timepoints ({img.dims.T}) must be divisible by number of positions ({num_pos_in_T}) in the data_config.yaml for dataset {dataset_name} if number of positions does not equal the number of scenes in the image file.'
                # calculate the duration of the positions in frames (they must all have the same duration)
                duration_in_frames = t_final if isinstance(t_final, int) else img.dims.T // num_pos_in_T
                # correct the t_start, t_final, and t_step values to account for the intercalation of positions with timeframes
                t_start_adjusted = t_start or pos_in_T
                t_step_adjusted = t_step * num_pos_in_T
                t_final_adjusted = pos_in_T + duration_in_frames * num_pos_in_T
                t_range = range(t_start_adjusted, t_final_adjusted, t_step_adjusted)

                # print(list(t_range))

                for i,t in enumerate(t_range):
                    if is_test and i >= 10:
                        break
                    else:
                        pass

                    # if t >= t_start_adjusted and t < t_final_adjusted:
                    analysis_queue.append({'dataset_name': dataset_name,
                                            'scene_index': pos_in_S, #scene,
                                            'position': pos,
                                            'T': t,
                                            'input_path': img_path,
                                            'output_dir': out_dir,
                                            'save_output': save_output,
                                            'use_original_data': use_original_data,
                                            'is_test': is_test})


        # if num_positions == len(img.scenes):
        #     positions, scenes = zip(*enumerate(img.scenes))
        # else:
        #     positions = list(range(num_positions))
        #     scenes = list(img.scenes) * num_positions

        # for pos, scene in zip(positions, scenes):
        #     img.set_scene(scene)
        # # for scene in img.scenes:
        #     # run the workflow on only the 20X datasets
        #     # print(sldmd.get_objective_info(img.metadata)['magnification'])
        #     if sldmd.get_objective_info(img.metadata)['magnification'] != 20:
        #         print(f'{pos}, {scene} does not use 20X magnification, skipping...')
        #         continue
        #     else:
        #         print(f'{pos}, {scene}')
        #         # break
        #         # t_final = img.dims.T
        #         if num_positions == len(img.scenes):
        #             # if the number of positions in the data_config.yaml file matches the number of scenes in the image file
        #             # then assume that the positions are not intercalated with the timeframes
        #             t_start = t_start or 0
        #             t_step = t_step or 1
        #             t_final = t_final or img.dims.T
        #             t_range = range(t_start, t_final, t_step)
        #         else:
        #             # otherwise assume that the positions are mixed in with timeframes
        #             # and thus we will need to separate them out
        #             assert img.dims.T % num_positions == 0, f'Number of timepoints ({img.dims.T}) must be divisible by number of positions ({num_positions}) in the data_config.yaml for dataset {dataset_name} if number of positions does not equal the number of scenes in the image file.'
        #             # calculate the duration of the positions in frames (they must all have the same duration)
        #             duration_in_frames = t_final if isinstance(t_final, int) else img.dims.T // num_positions
        #             # correct the t_start, t_final, and t_step values to account for the intercalation of positions with timeframes
        #             t_start = t_start or pos
        #             t_step = t_step * len(positions)
        #             t_final = pos + duration_in_frames * num_positions# - (num_positions - pos) + 1
        #             t_range = range(t_start, t_final, t_step)


        #         for i,t in enumerate(t_range):
        #             if is_test and i >= 20:
        #                 break
        #             else:
        #                 pass

        #             if t >= 0 and t < t_final:
        #                 analysis_queue.append({'dataset_name': dataset_name,
        #                                        'scene': scene,
        #                                        'position': pos,
        #                                        'T': t,
        #                                        'input_path': img_path,
        #                                        'output_dir': out_dir,
        #                                        'save_output': save_output,
        #                                        'is_test': is_test})
    return analysis_queue

def predict_nuclei_from_brightfield(image: np.ndarray, CellPose_model_path: str) -> np.ndarray:
    nuc_pred_model = models.CellposeModel(gpu=False, pretrained_model=CellPose_model_path)
    predictions = nuc_pred_model.eval(image, channels=[0,0], min_size=50, flow_threshold=0.6, cellprob_threshold=-3.0)
    return predictions





# Predict nuclei from brightfield images using the retrained CellPose model
# for args in tqdm(analysis_queue):

def generate_results(args: dict):
    # break
    print(f'Working on dataset {args["dataset_name"]}, T = {args["T"]}, scene = {args["scene_index"]}...')

    dataset_name = args['dataset_name']
    img_path = args['input_path']
    # img_path = Path(io.get_zarr_path(dataset_name))
    out_dir = Path(args['output_dir']) / dataset_name / f'P{args["position"]}'
    Path.mkdir(out_dir, exist_ok=True, parents=True)

    dim_order = 'TCZYX'
    dim_map = io.get_dim_map(dim_order)

    img = BioImage(img_path)
    img.set_scene(args['scene_index'])
    img_arr = img.get_image_dask_data(dim_order)

    # Load the retrained CellPose label-free nuclear prediction model
    # model_path = Path(r'C:\Users\serge.parent\OneDrive - Allen Institute\Desktop\projects\holistic\cellsmap_labelfree_nuclei_model\bf_std_model_no_preprocess_retrained')
    model_path = Path('//allen/aics/users/serge.parent/cellsmap_labelfree_nuclei_model/bf_std_model_no_preprocess_retrained')
    model_bf_stdproject = models.CellposeModel(gpu=False, pretrained_model=str(model_path))

    # The following is used for when a zarr is available with the BF_center, std, etc. channels available
    if not args['use_original_data']:
        bf_chan = io.get_channel_index(dataset_name, ['BF_Center'])
        bfstd_chan = io.get_channel_index(dataset_name, ['BF_STD'])
        nuc_chan = io.get_channel_index(dataset_name, ['DAPI'])

        # Predict nuclei from brightfield images
        print(' - predicting nuclei from brightfield standard deviation projections...')
        masks_bf_std = model_bf_stdproject.eval(bf_std_dask_arr[args['T'], ...].squeeze(), channels=[0,0], min_size=50, flow_threshold=0.6, cellprob_threshold=0.0)

        # Save the predictions with the brightfield and brightfield standard deviation
        # NOTE: can remove overlay_bf; was here for convience during development
        # overlay_bf = label2rgb(label=masks_bf_std[0], image=rescale_intensity(img_arr[args['T'], bf_chan, ...].squeeze()), bg_label=0)
        if any(nuc_chan):
            print(' - saving overlay of prediction and training image...')
            # val_dir = Path(args['output_dir']) / f'{dataset_name}_cellpose_pred_on_training/P{{args["pos"]}}'
            val_dir = Path(args['output_dir']) / f'validation/{dataset_name}/P{args["position"]}'
            Path.mkdir(val_dir, exist_ok=True, parents=True)

            img_nuc = img_arr[args['T'], nuc_chan, ...].squeeze().compute()
            overlay_nuc = label2rgb(label=masks_bf_std[0], image=rescale_intensity(np.clip(img_nuc, 0, np.percentile(img_nuc, 98))), bg_label=0)

            fig, ax = plt.subplots()
            ax.imshow(overlay_nuc)
            ax.axis('off')
            plt.savefig(val_dir / f'{dataset_name}_T{args["T"]}.png', bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close(fig)

        print(' - saving image...')
        out_path = out_dir / f'{dataset_name}_P{args["position"]}_T{args["T"]}_cellpose.ome.tiff'
        images_out = [img_arr[args['T'], bf_chan, ...].squeeze(), img_arr[args['T'], bfstd_chan, ...].squeeze(), masks_bf_std[0].squeeze()]
        images_out_metadata = {
            'image_name': dataset_name,
            'channel_names': ['BF_Center', 'BF_STD', 'CellPose_prediction'],
            'channel_colors': [(255,255,255), (255,255,255), (0,255,255)],
            'physical_pixel_sizes': img.physical_pixel_sizes,
            'dim_order': 'YX',
            }
        save_image_output(out_path, images_out, images_out_metadata)

    # The following is used for when there is no zarr with those channels available
    else:
        brightfield_index = io.get_dataset_info(dataset_name)['brightfield_channel_index']
        bf_std_dask_arr = img_arr[:,brightfield_index:brightfield_index+1,...].std(axis=dim_map['Z'], keepdims=True)
        bf_std_arr = bf_std_dask_arr[args['T'], ...].squeeze().compute()
        # bf_center_dask_arr = img_arr[:,brightfield_index:brightfield_index+1,...].mean(axis=dim_map['Z'], keepdims=True)
        bf_dask_arr = img_arr[args['T'],brightfield_index:brightfield_index+1,...]
        greatest_std_indices, _ = find_peaks([arr.std().compute() for arr in bf_dask_arr.squeeze()])
        possible_good_contrast_brightfield_plane = np.argmax([bf_dask_arr.squeeze()[i].mean().compute() for i in greatest_std_indices])
        bf_good_contrast_arr = bf_dask_arr.squeeze()[[possible_good_contrast_brightfield_plane]].squeeze().compute()

        # # Load the retrained CellPose label-free nuclear prediction model
        # model_path = Path(r'C:\Users\serge.parent\OneDrive - Allen Institute\Desktop\projects\holistic\cellsmap_labelfree_nuclei_model\bf_std_model_no_preprocess_retrained')
        # model_bf_stdproject = models.CellposeModel(gpu=False, pretrained_model=str(model_path))

        # Predict nuclei from brightfield images
        print(' - predicting nuclei from brightfield standard deviation projections...')
        masks_bf_std = model_bf_stdproject.eval(bf_std_arr, channels=[0,0], min_size=50, flow_threshold=0.6, cellprob_threshold=0.0)



        print(' - saving image...')
        out_path = out_dir / f'{dataset_name}_P{args["position"]}_T{args["T"]}_cellpose.ome.tiff'
        images_out = [bf_good_contrast_arr, bf_std_arr, masks_bf_std[0].squeeze()]
        images_out_metadata = {
            'image_name': dataset_name,
            'channel_names': ['BF_Center', 'BF_STD', 'CellPose_prediction'],
            'channel_colors': [(255,255,255), (255,255,255), (0,255,255)],
            'physical_pixel_sizes': img.physical_pixel_sizes,
            'dim_order': 'YX',
            }
        save_image_output(out_path, images_out, images_out_metadata)

def main(n_proc=1, save_output=True, is_test=False):
    # is_test = True
    out_dir = Path('//allen/aics/users/serge.parent/cellsmap_sandbox') / Path(__file__).stem

    # Build a list of datasets to analyze
    print('All available datasets:')
    dataset_name_list = io.get_available_datasets()
    # NOTE there is a userwarning error popping up when I read .nd2 files in dask
    # so I will only analyze .sldy files for now out of an abundance of caution
    # until .ome.zarr files are available
    nikon_datasets = [dataset_name for dataset_name in dataset_name_list if io.get_dataset_info(dataset_name)['microscope'] == 'Nikon']
    dataset_name_list = [dataset_name for dataset_name in dataset_name_list if io.get_dataset_info(dataset_name)['microscope'] == '3i']
    live_datasets = [dataset_name for dataset_name in dataset_name_list if io.get_dataset_info(dataset_name)['live_or_fixed_sample'] == 'live']
    fixed_datasets = [dataset_name for dataset_name in dataset_name_list if io.get_dataset_info(dataset_name)['live_or_fixed_sample'] == 'fixed']

    # live_datasets = live_datasets[:2]
    # fixed_datasets = fixed_datasets[:2]

    # test_datasets = [
    #     '20240328_T02_001', '20240328_T01_001',
    #     '20250122'] # up to here are fixed datasets
    #     # '20241016_20X', '20241120_20X', '20241203',
    #     # '20241210', '20241217',
    #     # ]
    # dataset_name_list = [name for name in dataset_name_list if name in test_datasets]

    # Get a list of timepoints and associated arguments to process from the list of datasets to analyze
    # analysis_queue = build_analysis_queue(dataset_name_list, t_to_eval=None, use_original_data=False)
    # NOTE the line below was only for prototyping and can be
    # removed when ready to analyze all datasets
    # analysis_queue += build_analysis_queue(['20241120_20X'], t_to_eval=slice(None, None, 25))
    # evaluate every 48 timepoints (ie. 4hrs)
    print('\nBuilding analysis queue...')
    analysis_queue_live = build_analysis_queue(live_datasets, t_step=48, use_original_data=True, out_dir=out_dir, save_output=save_output, is_test=is_test)
    analysis_queue_fixed = build_analysis_queue(fixed_datasets, use_original_data=True, out_dir=out_dir, save_output=save_output, is_test=is_test)
    analysis_queue = analysis_queue_live + analysis_queue_fixed

    if n_proc > 1:
            if __name__ == '__main__':
                print('Starting multiprocessing...')
                with Pool(processes=n_proc) as pool:
                    list(tqdm(pool.imap(generate_results, analysis_queue, chunksize=5), total=len(analysis_queue)))
                    pool.close()
                    pool.join()
                print('Done multiprocessing.')
    else:
        print('Starting single-core processing...')
        for dataset_name_and_args in analysis_queue:
            generate_results(dataset_name_and_args)
        print('Done single-core processing.')

    print('\N{microscope} Done analysis.')

if __name__ == '__main__':
    ipython_cli_flexecute(main)
