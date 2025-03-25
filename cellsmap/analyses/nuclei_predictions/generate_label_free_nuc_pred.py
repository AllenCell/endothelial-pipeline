from pathlib import Path
from bioio import BioImage
import matplotlib.pyplot as plt
import numpy as np
from cellsmap.util import dataset_io, get_sldy_metadata as sldmd
from cellsmap.util.cdh5_preprocessing import save_image_output
from cellpose import models
from tqdm import tqdm
from cellsmap.features.cdh5_classic_seg_tracking import ipython_cli_flexecute
from multiprocessing import Pool

def build_analysis_queue(dataset_name_list: list, t_start: int=0, t_final: int|None=None, t_step: int=1, save_output=True, overwrite=False, out_dir: str|Path|None=None, is_test=False, use_original_data=False) -> list:
    analysis_queue: list = []
    out_dir = dataset_io.get_results_dir(Path(__file__).stem, is_test=is_test)
    for dataset_name in dataset_name_list:
        img_path = Path(dataset_io.get_zarr_path(dataset_name)) if not use_original_data else Path(dataset_io.get_original_path(dataset_name))
        img = BioImage(img_path)

        num_positions = dataset_io.get_number_of_positions(dataset_name)

        assert num_positions % len(img.scenes) == 0, f'Number of positions ({num_positions}) in data_config.yaml must be divisible by number of scenes ({len(img.scenes)}) in the image file for dataset {dataset_name}'
        num_pos_in_T = num_positions // len(img.scenes)
        num_pos_in_S = len(img.scenes)

        positions_in_T, positions_in_S = [], []
        for scene_index in range(num_pos_in_S):
            positions_in_T += list(range(num_pos_in_T))
            positions_in_S += [scene_index] * num_pos_in_T

        for pos, (pos_in_T, pos_in_S) in enumerate(zip(positions_in_T, positions_in_S)):
            img.set_scene(pos_in_S)
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

                for i,t in enumerate(t_range):
                    if is_test and i >= 10:
                        break
                    else:
                        pass

                    if t >= t_start_adjusted and t < t_final_adjusted:
                        analysis_queue.append({'dataset_name': dataset_name,
                                                'scene_index': pos_in_S,
                                                'position': pos,
                                                'T': t,
                                                'input_path': img_path,
                                                'output_dir': out_dir,
                                                'save_output': save_output,
                                                'overwrite': overwrite,
                                                'use_original_data': use_original_data,
                                                'is_test': is_test})
    return analysis_queue

def predict_nuclei_from_brightfield(image: np.ndarray, CellPose_model_path: str) -> np.ndarray:
    nuc_pred_model = models.CellposeModel(gpu=False, pretrained_model=CellPose_model_path)
    predictions = nuc_pred_model.eval(image, channels=[0,0], min_size=50, flow_threshold=0.6, cellprob_threshold=-3.0)
    return predictions


# Predict nuclei from brightfield images using the retrained CellPose model
def generate_results(args: dict):
    print(f'Working on dataset {args["dataset_name"]}, T = {args["T"]}, scene = {args["scene_index"]}...')

    dataset_name = args['dataset_name']
    img_path = args['input_path']
    out_dir = Path(args['output_dir']) / dataset_name / f'P{args["position"]}'
    Path.mkdir(out_dir, exist_ok=True, parents=True)

    out_path = out_dir / f'{dataset_name}_P{args["position"]}_T{args["T"]}_cellpose.ome.tiff'
    if (args['overwrite'] == False) and out_path.exists():
        print(' - output already exists, skipping...')
        return

    else:
        dim_order = 'TCZYX'
        dim_map = dataset_io.get_dim_map(dim_order)

        img = BioImage(img_path)
        img.set_scene(args['scene_index'])
        img_arr = img.get_image_dask_data(dim_order)

        # Load the retrained CellPose label-free nuclear prediction model
        dataset_io.load_config('model')
        model_config = dataset_io.load_config(config_type='model')
        nuclei_models = [model for model in model_config if model['name'] == 'nuc_pred_labelfree']
        assert len(nuclei_models) == 1, f'Expected 1 model path, found {len(nuclei_models)}'
        model_path = Path(nuclei_models[0]['model_path_retrained'])
        model_bf_stdproject = models.CellposeModel(gpu=False, pretrained_model=str(model_path))

        # Calculate the brightfield standard deviation and the brightfield image with the best contrast
        brightfield_index = dataset_io.get_dataset_info(dataset_name)['brightfield_channel_index']
        bf_std_dask_arr = img_arr[:,brightfield_index:brightfield_index+1,...].std(axis=dim_map['Z'], keepdims=True)
        bf_std_arr = bf_std_dask_arr[args['T'], ...].squeeze().compute()

        # Find a brightfield plane with enough contrast to see
        # nuclei by eye
        bf_dask_arr = img_arr[args['T'],brightfield_index:brightfield_index+1,...]
        plane_stdevs = [arr.std().compute() for arr in bf_dask_arr.squeeze()]
        # don't allow the possible good contrast plane to be less than 0 (i.e. the bottom of the Z-stack)
        possible_good_contrast_brightfield_plane = max(0, np.argmin([plane for plane in plane_stdevs]) - 6)
        bf_good_contrast_arr = bf_dask_arr.squeeze()[[possible_good_contrast_brightfield_plane]].squeeze().compute()

        # Predict nuclei from brightfield images
        print(' - predicting nuclei from brightfield standard deviation projections...')
        masks_bf_std = model_bf_stdproject.eval(bf_std_arr, channels=[0,0], min_size=50, flow_threshold=0.6, cellprob_threshold=0.0)

        # Construct and save a multichannel image
        images_out = [bf_good_contrast_arr, bf_std_arr, masks_bf_std[0].squeeze()]
        print(' - saving image...')
        images_out_metadata = {
            'image_name': dataset_name,
            'channel_names': ['BF_Center', 'BF_STD', 'CellPose_prediction'],
            'channel_colors': [(255,255,255), (255,255,255), (0,255,255)],
            'physical_pixel_sizes': img.physical_pixel_sizes,
            'dim_order': 'YX',
            }
        save_image_output(out_path, images_out, images_out_metadata)


def main(n_proc=1, save_output=True, overwrite=False, is_test=False):
    # Set the output directory
    out_dir = dataset_io.get_results_dir(Path(__file__).stem, is_test=is_test)

    # Build a list of datasets to analyze
    print('All available datasets:')
    dataset_name_list = dataset_io.get_available_datasets()
    # NOTE there is a userwarning error popping up when I read .nd2 files in dask
    # so I will only analyze .sldy files for now out of an abundance of caution
    # until .ome.zarr files are available
    nikon_datasets = [dataset_name for dataset_name in dataset_name_list if dataset_io.get_dataset_info(dataset_name)['microscope'] == 'Nikon']
    dataset_name_list = [dataset_name for dataset_name in dataset_name_list if dataset_io.get_dataset_info(dataset_name)['microscope'] == '3i']
    live_datasets = [dataset_name for dataset_name in dataset_name_list if dataset_io.get_dataset_info(dataset_name)['live_or_fixed_sample'] == 'live']
    fixed_datasets = [dataset_name for dataset_name in dataset_name_list if dataset_io.get_dataset_info(dataset_name)['live_or_fixed_sample'] == 'fixed']

    # Get a list of timepoints and associated arguments to process from the list of datasets to analyze
    # evaluate every 48 timepoints (ie. 4hrs)
    print('\nBuilding analysis queue...')
    analysis_queue_live = build_analysis_queue(live_datasets, t_step=1, use_original_data=True, out_dir=out_dir, save_output=save_output, overwrite=overwrite, is_test=is_test)
    analysis_queue_fixed = build_analysis_queue(fixed_datasets, use_original_data=True, out_dir=out_dir, save_output=save_output, overwrite=overwrite, is_test=is_test)
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
