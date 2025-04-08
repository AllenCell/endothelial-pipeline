from pathlib import Path
from bioio import BioImage
from cellpose import io as cellpose_io, models, train
from cellsmap.util import dataset_io, get_sldy_metadata as sldmd
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.filters import apply_hysteresis_threshold
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.morphology import dilation, disk

# NOTE
# because we don't have zarr files for the datasets in the
# datasets_to_use list, the loading of fixed data used for
# retraining the model would need to use the original data
# which could be compplicated or inconsistent. Therefore I
# will check this script after the zarrs are available.
# Until then, this script is not functional.
use_original_data = True
show_watershed_segmentations = True
train_from_base_cellpose_nuclei_model = True
if show_watershed_segmentations or train_from_base_cellpose_nuclei_model:
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb


datasets_to_use = ['20240328_T02_001', '20240328_T01_001',]
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
                         ]
}

dim_order = 'TCZYX'
dim_map = dataset_io.get_dim_map(dim_order)


def get_image_data_from_original(dataset_name, scenes_to_use):
    img_path = Path(dataset_io.get_original_path(dataset_name))
    img = BioImage(img_path)
    for scene in scenes_to_use:
        img.set_scene(scene)
        print(dataset_name, img.current_scene)

        channel_names = sldmd.get_channel_name(img.metadata)
        channel_names = [chan.split('/')[0] for chan in channel_names]
        nuc_chan = channel_names.index('405')
        bf_chan = channel_names.index('TL')
        img_dask_arr_nuc = img.get_image_dask_data(dim_order, C=[nuc_chan]).max(axis=dim_map['Z'], keepdims=True)
        img_dask_arr_bf_std = img.get_image_dask_data(dim_order, C=[bf_chan]).std(axis=dim_map['Z'], keepdims=True)
        yield (img_dask_arr_nuc, img_dask_arr_bf_std)

def get_image_data_from_zarr(dataset_name):
    nuc_chan = int(*dataset_io.get_channel_index(dataset_name, ['DAPI']))
    bf_chan = int(*dataset_io.get_channel_index(dataset_name, ['Brightfield']))
    img_dict_nuc = dataset_io.load_dataset(dataset_name, channels=[nuc_chan,])
    img_dict_bf = dataset_io.load_dataset(dataset_name, channels=[bf_chan,])
    for filename in img_dict_nuc:
        img_dask_arr_nuc = img_dict_nuc[filename].max(axis=dim_map['Z'], keepdims=True)
        img_dask_arr_bf_std = img_dict_bf[filename].std(axis=dim_map['Z'], keepdims=True)
        yield (img_dask_arr_nuc, img_dask_arr_bf_std)

# Generate ground truths from nuclei labeled with DAPI using
# a classic segmentation approach (watershed)
images = []
labels = []
for dataset_name in datasets_to_use:
    if use_original_data:
        imgs_to_eval = get_image_data_from_original(dataset_name, scenes_to_use[dataset_name])
    else:
        imgs_to_eval = get_image_data_from_zarr(dataset_name)

    # print(f'Processing dataset={dataset_name}, T={t}...')
    for img_dask_arr in imgs_to_eval:
        nuc_max, bf_std = img_dask_arr
        nuc_max = nuc_max.compute().squeeze()
        bf_std = bf_std.compute().squeeze()
        normd_nuc = rescale_intensity(nuc_max, out_range=(0,1))
        thresh = apply_hysteresis_threshold(normd_nuc, np.percentile(normd_nuc, 80), np.percentile(normd_nuc, 85))
        dist = distance_transform_edt(thresh)
        peaks_img = np.zeros(dist.shape, dtype=bool)
        peaks_img[tuple(zip(*peak_local_max(dist, min_distance=15)))] = True
        peaks_img = label(dilation(peaks_img, footprint=disk(5)))
        ws = watershed(rescale_intensity(dist, out_range=(1,0)), markers=peaks_img, mask=thresh)

        images.append(bf_std)
        labels.append(ws)


cellpose_io.logger_setup()
# retrain Goutham's Cellpose model
model_config = dataset_io.load_config(config_type='model')
nuclei_models = [model for model in model_config if model['name'] == 'nuc_pred_labelfree']
assert len(nuclei_models) == 1, f'Expected 1 model path, found {len(nuclei_models)}'
model_path = Path(nuclei_models[0]['model_path'])
model_dir = model_path.parent

model_bf_stdproject = models.CellposeModel(gpu=False, pretrained_model=str(model_path))

model_path, train_losses, test_losses = train.train_seg(model_bf_stdproject.net,
                            train_data=images, train_labels=labels,
                            channels=[0,0], normalize=True,
                            # test_data=test_images, test_labels=test_labels,
                            weight_decay=1e-4, SGD=True, learning_rate=0.1,
                            n_epochs=100,
                            save_path=model_dir, model_name="bf_std_model_no_preprocess_retrained")

if train_from_base_cellpose_nuclei_model:
    # fine-tune the basic CellPose nuclei model
    model_dir_from_default = model_dir / 'from_cellpose_nuclei_model_default'
    Path.mkdir(model_dir_from_default, exist_ok=True)
    model_nuclei_original = models.CellposeModel(gpu=False, model_type='nuclei')
    model_path, train_losses, test_losses = train.train_seg(model_nuclei_original.net,
                                train_data=images, train_labels=labels,
                                channels=[0,0], normalize=True,
                                # test_data=test_images, test_labels=test_labels,
                                weight_decay=1e-4, SGD=True, learning_rate=0.1,
                                n_epochs=100,
                                save_path=model_dir_from_default, model_name="bf_std_model_no_preprocess_retrained")

    model_nuclei_original_finetuned = models.CellposeModel(gpu=False, pretrained_model=str(model_path))

    test_img_path = Path(dataset_io.get_original_path('20241120_20X'))
    test_img_bf_chan = dataset_io.get_dataset_info('20241120_20X')['brightfield_channel_index']
    test_img = BioImage(test_img_path)
    test_img_dask_arr = test_img.get_image_dask_data(dim_order, T=[0], C=test_img_bf_chan).std(axis=dim_map['Z'], keepdims=True)
    test_img_arr = test_img_dask_arr.compute().squeeze()
    # test = model_nuclei_original_finetuned.eval(images[0])
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

