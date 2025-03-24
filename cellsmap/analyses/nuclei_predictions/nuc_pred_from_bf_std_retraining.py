from pathlib import Path
from bioio import BioImage
from cellpose import io as cellpose_io, models, train
from cellsmap.util import dataset_io
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.filters import apply_hysteresis_threshold
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.morphology import dilation, disk

datasets_to_use = ['20240328_T02_001', '20240328_T01_001',]


dim_order = 'TCZYX'
dim_map = dataset_io.get_dim_map(dim_order)


# Generate ground truths from nuclei labeled with DAPI using
# a classic segmentation approach (watershed)
images = []
labels = []
for dataset_name in datasets_to_use:
    bfstd_chan = dataset_io.get_channel_index(dataset_name, ['BF_STD'])
    nuc_chan = dataset_io.get_channel_index(dataset_name, ['DAPI'])

    img_path = Path(dataset_io.get_zarr_path(dataset_name))

    img = BioImage(img_path)

    for t in range(img.dims.T):
        print(f'Processing dataset={dataset_name}, T={t}...')
        normd_nuc = rescale_intensity(img.get_image_dask_data(dim_order, T=t, C=nuc_chan).compute().squeeze(), out_range=(0,1))
        thresh = apply_hysteresis_threshold(normd_nuc, np.percentile(normd_nuc, 80), np.percentile(normd_nuc, 85))
        dist = distance_transform_edt(thresh)
        peaks_img = np.zeros(dist.shape, dtype=bool)
        peaks_img[tuple(zip(*peak_local_max(dist, min_distance=15)))] = True
        peaks_img = label(dilation(peaks_img, footprint=disk(5)))
        ws = watershed(rescale_intensity(dist, out_range=(1,0)), markers=peaks_img, mask=thresh)

        images.append(img.get_image_dask_data(dim_order, T=t, C=bfstd_chan).compute().squeeze())
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

