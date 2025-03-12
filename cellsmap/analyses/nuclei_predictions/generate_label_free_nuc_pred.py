from pathlib import Path
from bioio import BioImage
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
# from skimage.segmentation import watershed, find_boundaries
# from skimage.morphology import dilation, disk
from cellsmap.util import io
from cellsmap.util.cdh5_preprocessing import save_image_output
from cellpose import models
from tqdm import tqdm

def build_analysis_queue(dataset_name_list: list, t_to_eval: slice|range|list=None, save_output=False, is_test=False, use_original_data=False) -> list:
    analysis_queue: list = []
    prj_dir = Path(__file__).parents[2] if not is_test else Path(__file__).parents[3]
    out_dir = prj_dir / 'results' / Path(__file__).stem
    for dataset_name in dataset_name_list:
        img_path = Path(io.get_zarr_path(dataset_name)) if not use_original_data else Path(io.get_original_path(dataset_name))
        img = BioImage(img_path)

        t_range = t_to_eval or range(img.dims.T)
        t_range = range(t_range.start or 0, t_range.stop or img.dims.T, t_range.step) if isinstance(t_range, slice) else t_range

        if is_test and len(analysis_queue) >= 5:
            break
        else:
            pass

        for t in t_range:
            if t >= 0 and t < img.dims.T:
                analysis_queue.append({'dataset_name': dataset_name,
                                       'T': t,
                                       'input_path': img_path,
                                       'output_dir': out_dir,
                                       'save_output': save_output,
                                       'is_test': is_test})

    return analysis_queue

def predict_nuclei_from_brightfield(image: np.ndarray, CellPose_model_path: str) -> np.ndarray:
    nuc_pred_model = models.CellposeModel(gpu=False, pretrained_model=CellPose_model_path)
    predictions = nuc_pred_model.eval(image, channels=[0,0], min_size=50, flow_threshold=0.6, cellprob_threshold=-3.0)
    return predictions


# Build a list of datasets to analyze
print('All available datasets:')
dataset_name_list = io.get_available_datasets()

# test_datasets = [
#     '20240328_T02_001', '20240328_T01_001',
#     '20250122'] # up to here are fixed datasets
#     # '20241016_20X', '20241120_20X', '20241203',
#     # '20241210', '20241217',
#     # ]
# dataset_name_list = [name for name in dataset_name_list if name in test_datasets]

# Get a list of timepoints and associated arguments to process from the list of datasets to analyze
analysis_queue = build_analysis_queue(dataset_name_list, t_to_eval=None, use_original_data=True)
# NOTE the line below was only for prototyping and can be
# removed when ready to analyze all datasets
# analysis_queue += build_analysis_queue(['20241120_20X'], t_to_eval=slice(None, None, 25))
analysis_queue = build_analysis_queue(dataset_name_list, t_to_eval=None, use_original_data=True)

# Predict nuclei from brightfield images using the retrained CellPose model
for args in tqdm(analysis_queue):
    print(f'Working on dataset {args["dataset_name"]}, T = {args["T"]}...')

    dataset_name = args['dataset_name']
    img_path = args['input_path']
    # img_path = Path(io.get_zarr_path(dataset_name))
    out_dir = Path(args['output_dir']) / dataset_name
    Path.mkdir(out_dir, exist_ok=True, parents=True)

    dim_order = 'TCZYX'
    dim_map = io.get_dim_map(dim_order)
    bf_chan = io.get_channel_index(dataset_name, ['BF_Center'])
    bfstd_chan = io.get_channel_index(dataset_name, ['BF_STD'])
    nuc_chan = io.get_channel_index(dataset_name, ['DAPI'])

    img = BioImage(img_path)
    img_arr = img.get_image_dask_data(dim_order)

    # Load the retrained CellPose label-free nuclear prediction model
    model_path = Path(r'C:\Users\serge.parent\OneDrive - Allen Institute\Desktop\projects\holistic\cellsmap_labelfree_nuclei_model\bf_std_model_no_preprocess_retrained')
    model_bf_stdproject = models.CellposeModel(gpu=False, pretrained_model=str(model_path))

    # Predict nuclei from brightfield images
    print(' - predicting nuclei from brightfield standard deviation projections...')
    masks_bf_std = model_bf_stdproject.eval(img_arr[args['T'], bfstd_chan, ...].squeeze(), channels=[0,0], min_size=50, flow_threshold=0.6, cellprob_threshold=0.0)

    # Save the predictions with the brightfield and brightfield standard deviation
    # NOTE: can remove overlay_bf; was here for convience during development
    # overlay_bf = label2rgb(label=masks_bf_std[0], image=rescale_intensity(img_arr[args['T'], bf_chan, ...].squeeze()), bg_label=0)
    if any(nuc_chan):
        print(' - saving overlay of prediction and training image...')
        val_dir = Path(args['output_dir']) / f'{dataset_name}_cellpose_pred_on_training'
        Path.mkdir(val_dir, exist_ok=True, parents=True)

        img_nuc = img_arr[args['T'], nuc_chan, ...].squeeze().compute()
        overlay_nuc = label2rgb(label=masks_bf_std[0], image=rescale_intensity(np.clip(img_nuc, 0, np.percentile(img_nuc, 98))), bg_label=0)

        fig, ax = plt.subplots()
        ax.imshow(overlay_nuc)
        ax.axis('off')
        plt.savefig(val_dir / f'{dataset_name}_T{args["T"]}.png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)

    print(' - saving image...')
    out_path = out_dir / f'{dataset_name}_T{args["T"]}_cellpose.ome.tiff'
    images_out = [img_arr[args['T'], bf_chan, ...].squeeze(), img_arr[args['T'], bfstd_chan, ...].squeeze(), masks_bf_std[0].squeeze()]
    images_out_metadata = {
        'image_name': dataset_name,
        'channel_names': ['BF_Center', 'BF_STD', 'CellPose_prediction'],
        'channel_colors': [(255,255,255), (255,255,255), (0,255,255)],
        'physical_pixel_sizes': img.physical_pixel_sizes,
        'dim_order': 'YX',
        }
    save_image_output(out_path, images_out, images_out_metadata)
