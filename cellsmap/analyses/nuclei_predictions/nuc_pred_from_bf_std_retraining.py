from pathlib import Path
from bioio import BioImage
from bioio_base.types import PhysicalPixelSizes
from cellpose import io as cellpose_io, models, train
from cellsmap.util import get_sldy_metadata as sldmd
from cellsmap.util.dataset_io import get_dataset_info, get_original_path, get_zarr_path, load_dataset, load_config
from cellsmap.util.general_image_preprocessing import get_dim_map, build_analysis_queue, save_image_output
from cellsmap.util.set_output import get_output_path
import numpy as np
from skimage.exposure import rescale_intensity
# from skimage.filters import apply_hysteresis_threshold, gaussian, sobel, rank
# from scipy.ndimage import distance_transform_edt
# from skimage.feature import peak_local_max, canny
from skimage.segmentation import watershed, find_boundaries
# from skimage.measure import label, regionprops
# from skimage.morphology import dilation, disk, remove_small_holes, remove_small_objects, binary_closing
from datetime import datetime
import shutil
import re

use_original_data = True
show_watershed_segmentations = True
retrain_Gouthams_model = True
train_from_base_cellpose_nuclei_model = False
if show_watershed_segmentations or train_from_base_cellpose_nuclei_model:
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb


datasets_to_use = ['20240328_T02_001', '20240328_T01_001',
                   '20250415_SlideA_20X',
                   '20250415_SlideE_20X', '20250415_SlideH_20X']
scenes_to_use = {
    '20240328_T02_001': ['20240328_T02_001-1711659785-  8',
                         '20240328_T02_001-1711659785- 24',
                         '20240328_T02_001-1711659785- 39',
                         '20240328_T02_001-1711659785-990',
                         ],
    '20240328_T01_001': ['20240328_T01_001-1711663662-276',
                         '20240328_T01_001-1711663662-293',
                         '20240328_T01_001-1711663662-307',
                         '20240328_T01_001-1711663662-322',
                         '20240328_T01_001-1711663662-337',
                         ],
    '20250415_SlideA_20X': ['20250415_GE0000XXXX_slideA_20X - Position 1 [50]-1744838244-103'],
    '20250415_SlideE_20X': ['20250416_GE0000XXXX_slideE_20X - Position 1 [50]-1744838492-691'],
    '20250415_SlideH_20X': ['20250415_GE0000XXXX_slideH_20X - Position 1 [50]-1744838353-891'],
}

dim_order = 'TCZYX'
dim_map = get_dim_map(dim_order)

# def get_old_cellpose_train_test_losses(model_name_cyto, model_name_live):
def get_old_cellpose_train_test_losses(cellpose_model_dir, model_name_list):
    """
    This function extracts the training and test losses from the run.log file
    produced during the training of a Cellpose model.
    It is only useful for cellpose < 3.1 as newer versions of cellpose
    return the train and test losses along with the model path when
    using the cellpose.train.train_seg function.
    """
    # copy the log file to the model directory and rename it according to model_name
    # print(f'extracting training and test losses from run.log for {model_name}...')
    print(f'extracting training and test losses from run.log...')
    # run_log_filepath = Path.home().joinpath(".cellpose").joinpath("run.log")
    # run_log_filepath_new = Path(cell_cellpose_model_dir)/f'{model_name}_run.log'
    # shutil.move(run_log_filepath, run_log_filepath_new)

    run_log_filepath = Path.home().joinpath(".cellpose").joinpath("run.log")
    assert cellpose_model_dir.exists(), f"Cellpose model directory {cellpose_model_dir} does not exist"
    run_log_filepath_new = Path(cellpose_model_dir) / 'run.log'
    shutil.move(run_log_filepath, run_log_filepath_new)

    pages = {}
    with open(run_log_filepath_new) as run_log:
        pg = {}
        for line in run_log:
            for model_name in model_name_list:
                if model_name in line:
                    if model_name and pg:
                        pages[model_name] = pg
                    model_name = model_name
                    pg = {'train_losses': [], 'test_losses': [], 'time_list': []}
                else:
                    pass

                if 'train_loss' in line:
                    if pg:
                        train_loss = re.findall('train_loss=\d+\.\d+', line)
                        test_loss = re.findall('test_loss=\d+\.\d+', line)
                        time = re.findall('time \d+\.\d+', line)
                        if train_loss:
                            pg['train_losses'].append([float(loss.split('train_loss=')[1]) for loss in train_loss][0])
                        if test_loss:
                            pg['test_losses'].append([float(loss.split('test_loss=')[1]) for loss in test_loss][0])
                        if time:
                            pg['time_list'].append([float(t.split('time ')[1]) for t in time][0])
            if model_name and pg:
                pages[model_name] = pg

    train_losses = {}
    test_losses = {}
    time_list = {}
    for model_name in model_name_list:
        train_losses[model_name] = pages[model_name]['train_losses']
        test_losses[model_name] = pages[model_name]['test_losses']
        time_list[model_name] = pages[model_name]['time_list']

    return train_losses, test_losses, time_list



def get_image_data_from_original(dataset_name, scene, T, verbose=False):
    img_path = Path(get_original_path(dataset_name))
    img = BioImage(img_path)
    # for scene in scenes_to_use:
    img.set_scene(scene)
    img_metadata = img.metadata
    print(dataset_name, img.current_scene) if verbose else None

    channel_names = sldmd.get_channel_name(img.metadata)
    channel_names = [chan.split('/')[0] for chan in channel_names]
    nuc_chan = get_dataset_info(dataset_name)['405_channel_index']
    bf_chan = get_dataset_info(dataset_name)['brightfield_channel_index']
    img_dask_arr_nuc = img.get_image_dask_data(dim_order, C=[nuc_chan], T=T).max(axis=dim_map['Z'], keepdims=True)
    img_dask_arr_bf_std = img.get_image_dask_data(dim_order, C=[bf_chan], T=T).std(axis=dim_map['Z'], keepdims=True)
    return (img_dask_arr_nuc, img_dask_arr_bf_std), img_metadata

def get_image_data_from_zarr(dataset_name):
    for zarr_name in get_zarr_path(dataset_name):
        img_dict_nuc = load_dataset(dataset_name, zarr_name=zarr_name, channels=['DAPI'])
        img_dict_bf = load_dataset(dataset_name, zarr_name=zarr_name, channels=['BF'])
        img_dask_arr_nuc = img_dict_nuc[zarr_name].max(axis=dim_map['Z'], keepdims=True)
        img_dask_arr_bf_std = img_dict_bf[zarr_name].std(axis=dim_map['Z'], keepdims=True)
        yield (zarr_name, img_dask_arr_nuc, img_dask_arr_bf_std)

# def get_segmentation(normd_nuc, thresh_low=None, thresh_high=None):

#     # detect nuclei in the foreground by adding a hysteresis
#     # threshold on a local leveling of the image and canny edges
#     # on the non-leveled image and then filling in holes
#     normd_nuc_al = rank.autolevel(normd_nuc, footprint=disk(100))
#     # normd_nuc_al = normd_nuc
#     low, high = np.percentile(normd_nuc, (thresh_low, thresh_high))
#     edges_canny = canny(normd_nuc, low_threshold=low, high_threshold=high)
#     normd_nuc_al[edges_canny] = 0
#     low_al, high_al = np.percentile(normd_nuc_al, (thresh_low, thresh_high))
#     thresh = apply_hysteresis_threshold(normd_nuc_al, low_al, high_al)
#     # thresh = binary_closing(thresh, disk(3))
#     thresh = remove_small_holes(thresh+edges_canny, area_threshold=2000)

#     # catch missed nuclei by doing a watershed on the background too
#     # this can happen if a nuclei is too dim and has a fuzzy edge
#     bg_thresh = ~thresh
#     bg_dist = distance_transform_edt(bg_thresh)
#     bg_peaks = np.zeros(bg_dist.shape, dtype=bool)
#     bg_peaks[tuple(zip(*peak_local_max(bg_dist, min_distance=15)))] = True
#     bg_peaks = label(dilation(bg_peaks, footprint=disk(5)))
#     bg_ws = watershed(rescale_intensity(bg_dist, out_range=(1,0)), markers=bg_peaks, mask=bg_thresh)
#     props = regionprops(bg_ws, intensity_image=normd_nuc)
#     missed_labels = [prop.label for prop in props if prop.intensity_mean > low]
#     missed_nuclei = np.isin(bg_ws, missed_labels)
#     thresh = thresh + missed_nuclei

#     # label the nuclei by segmenting themseg
#     # get the seeds for watershed
#     dist = distance_transform_edt(thresh)
#     peaks_img = np.zeros(dist.shape, dtype=bool)
#     peaks_img[tuple(zip(*peak_local_max(dist, min_distance=15)))] = True
#     peaks_img = label(dilation(peaks_img, footprint=disk(5)))
#     # get the basins for watershed
#     edges_sobel = sobel(normd_nuc_al)
#     basins = rescale_intensity(dist, out_range=(1,0)) * rescale_intensity(edges_sobel, out_range=(0,1))
#     # do the watershed
#     ws = watershed(basins, markers=peaks_img, mask=thresh)

#     return ws


def save_overlay(labels, bg_img, out_name, outlines=True, face=True):
    seg_outlines = find_boundaries(labels)
    if outlines and face:
        labels[seg_outlines] = 0
    elif outlines and not face:
        labels = seg_outlines
    else:
        pass
    overlay = label2rgb(label=labels, image=bg_img, bg_label=0)

    fig, ax = plt.subplots()
    ax.imshow(overlay)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(out_name, bbox_inches='tight', pad_inches=0, dpi=180)


analysis_queue = build_analysis_queue(datasets_to_use,
                                      use_original_data=use_original_data,
                                      overwrite=True,
                                      out_dir=get_output_path(Path(__file__).stem, verbose=False),)
# analysis_queue = analysis_queue[:1]
# Generate ground truths from nuclei labeled with DAPI
# using the Cellpose base nuclei model
nuc_model = models.CellposeModel(gpu=False, model_type='nuclei')
# images = []
# labels = []

for analysis_args in analysis_queue:
    dataset_name = analysis_args['dataset_name']
    scene_name = analysis_args['scene_name']
    position = analysis_args['position']
    T = analysis_args['T']
    out_dir_val = analysis_args['output_dir'] / f'validation_overlays/{dataset_name}/'
    out_dir = analysis_args['output_dir'] / f'cellpose_base_nuclei_model_segmentations/{dataset_name}/'

    if scene_name in scenes_to_use[dataset_name]:
        print(f'Working on {dataset_name} P{position} {scene_name}...')
        pass
    else:
        print(f'{dataset_name} P{position} {scene_name} not in scenes_to_use. Skipping.')
        continue

    if use_original_data:
        img_dask_arrs, image_metadata = get_image_data_from_original(dataset_name, scene_name, T)
        voxel_size = sldmd.get_voxel_size(image_metadata)
    else:
        print(f'Zarrs not yet implemented. Skipping {dataset_name} P{position}.')
        continue # zarrs not yet implemented
        # img_dask_arrs = get_image_data_from_zarr(dataset_name)

    nuc_max, bf_std = img_dask_arrs
    nuc_max = nuc_max.compute().squeeze()
    bf_std = bf_std.compute().squeeze()
    normd_nuc = rescale_intensity(nuc_max, out_range=np.uint16)
    # normd_nuc = rescale_intensity(np.clip(normd_nuc, 0, np.percentile(normd_nuc,99)), out_range=(0,1))
    # thresh_low, thresh_high = threshold_vals[dataset_name]
    # seg = get_segmentation(normd_nuc, thresh_low, thresh_high)
    normd_nuc_clipped = rescale_intensity(np.clip(normd_nuc, 0, np.percentile(normd_nuc,99)), out_range=(0,1))
    seg, flows, styles = nuc_model.eval(normd_nuc_clipped, channels=[0,0], min_size=500, flow_threshold=0.6, cellprob_threshold=-3.0)

    out_dir_val.mkdir(exist_ok=True, parents=True)
    out_name_val = out_dir_val / f'{dataset_name}_P{position}_classic_seg.png'
    save_overlay(seg, normd_nuc_clipped, out_name_val, outlines=True, face=True)
    # save_overlay(seg, normd_nuc, out_name, outlines=True, face=True)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_name = out_dir / f'{dataset_name}_P{position}_classic_seg.png'
    images_out = [seg]
    images_out_metadata = {
        'image_name': dataset_name,
        'channel_names': ['cellpose_nuclei_prediction'], 
        'channel_colors': [(255,255,255)],
        'physical_pixel_sizes': PhysicalPixelSizes(**voxel_size),
        'dim_order': 'YX',
        'dtype': None,
        }
    save_image_output(out_path=out_name,
                      images=images_out,
                      images_metadata=images_out_metadata)
    # images.append(bf_std)
    # labels.append(ws)


cellpose_io.logger_setup()
# retrain Goutham's Cellpose model
if retrain_Gouthams_model:
    model_config = load_config(config_type='model')
    nuclei_models = [model for model in model_config if model['name'] == 'nuc_pred_labelfree']
    assert len(nuclei_models) == 1, f'Expected 1 model path, found {len(nuclei_models)}'
    model_path = Path(nuclei_models[0]['model_path'])
    model_dir = model_path.parent / datetime.today().date().strftime('%Y%m%d')
    model_dir.mkdir(exist_ok=True, parents=True)

    model_bf_stdproject = models.CellposeModel(gpu=False, pretrained_model=str(model_path))

    model_path = train.train_seg(model_bf_stdproject.net,
                                train_data=images, train_labels=labels,
                                channels=[0,0], normalize=True,
                                weight_decay=1e-4, SGD=True, learning_rate=0.1,
                                n_epochs=100,
                                save_path=model_dir, model_name="bf_std_model_no_preprocess_retrained")

if train_from_base_cellpose_nuclei_model:
    # fine-tune the basic CellPose nuclei model
    model_dir_from_default = model_dir / 'from_cellpose_nuclei_model_default'
    model_dir_from_default.mkdir(exist_ok=True)
    model_nuclei_original = models.CellposeModel(gpu=False, model_type='nuclei')
    model_path = train.train_seg(model_nuclei_original.net,
                                train_data=images, train_labels=labels,
                                channels=[0,0], normalize=True,
                                weight_decay=1e-4, SGD=True, learning_rate=0.1,
                                n_epochs=100,
                                save_path=model_dir_from_default, model_name="bf_std_model_no_preprocess_retrained")

    model_nuclei_original_finetuned = models.CellposeModel(gpu=False, pretrained_model=str(model_path))

    test_img_path = Path(get_original_path('20241120_20X'))
    test_img_bf_chan = get_dataset_info('20241120_20X')['brightfield_channel_index']
    test_img = BioImage(test_img_path)
    test_img_dask_arr = test_img.get_image_dask_data(dim_order, T=[0], C=test_img_bf_chan).std(axis=dim_map['Z'], keepdims=True)
    test_img_arr = test_img_dask_arr.compute().squeeze()
    test_prediction, flows, probs = model_nuclei_original_finetuned.eval(test_img_arr, channels=[0,0], min_size=50, flow_threshold=0.6, cellprob_threshold=-3.0)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    image_rescaled = rescale_intensity(np.clip(test_img_arr,
                                            a_min=np.percentile(test_img_arr, 1),
                                            a_max=np.percentile(test_img_arr, 99)),
                                    out_range=(0,1))
    ax[0].imshow(image_rescaled, cmap='gray')
    ax[0].set_title('BF STD')
    ax[1].imshow(label2rgb(test_prediction))
    ax[1].set_title('Watershed')
    overlay = label2rgb(label=test_prediction, image=image_rescaled, bg_label=0)
    ax[2].imshow(overlay)
    [ax.set_axis_off() for ax in ax]
    plt.tight_layout()
    plt.show()

if show_watershed_segmentations:
    for i in range(len(images)):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        image_rescaled = rescale_intensity(np.clip(images[i],
                                                a_min=np.percentile(images[i], 1),
                                                a_max=np.percentile(images[i], 99)),
                                        out_range=(0,1))
        ax[0].imshow(image_rescaled, cmap='gray')
        ax[0].set_title('BF STD')
        ax[1].imshow(label2rgb(labels[i]))
        ax[1].set_title('Watershed')
        overlay = label2rgb(label=labels[i], image=image_rescaled, bg_label=0)
        ax[2].imshow(overlay)
        [ax.set_axis_off() for ax in ax]
        plt.tight_layout()
        plt.show()
