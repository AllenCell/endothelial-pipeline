from pathlib import Path
from bioio import BioImage
from cellpose import io as cellpose_io, models, train
from cellsmap.util import io
from matplotlib import pyplot as plt
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.filters import apply_hysteresis_threshold
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.morphology import dilation, disk
from skimage.color import label2rgb
# output = io.load_train_test_data(train_dir, test_dir, image_filter="_img",
#                                 mask_filter="_masks", look_one_level_down=False)
# images, labels, image_names, test_images, test_labels, image_names_test = output

datasets_to_use = ['20231122_T02_001', '20240328_T02_001', '20240328_T01_001',]

out_dir = Path('../../').resolve() / 'results' / Path(__file__).stem
# Path.mkdir(out_dir, exist_ok=True, parents=True)

dim_order = 'TCZYX'
dim_map = io.get_dim_map(dim_order)


# img_path = Path(io.get_zarr_path(dataset_name))
# img = BioImage(img_path)
images = []
labels = []
for dataset_name in datasets_to_use:
    bf_chan = io.get_channel_index(dataset_name, ['BF_Center'])
    bfstd_chan = io.get_channel_index(dataset_name, ['BF_STD'])
    nuc_chan = io.get_channel_index(dataset_name, ['DAPI'])

    img_path = Path(io.get_zarr_path(dataset_name))

    img = BioImage(img_path)

    for t in range(img.dims.T):
        print(f'Processing dataset={dataset_name}, T={t}...')
        # img_at_T = img.get_image_dask_data(dim_order, T=t)
        normd_nuc = rescale_intensity(img.get_image_dask_data(dim_order, T=t, C=nuc_chan).compute().squeeze(), out_range=(0,1))
        thresh = apply_hysteresis_threshold(normd_nuc, np.percentile(normd_nuc, 80), np.percentile(normd_nuc, 85))
        dist = distance_transform_edt(thresh)
        peaks_img = np.zeros(dist.shape, dtype=bool)
        peaks_img[tuple(zip(*peak_local_max(dist, min_distance=15)))] = True
        peaks_img = label(dilation(peaks_img, footprint=disk(5)))
        ws = watershed(rescale_intensity(dist, out_range=(1,0)), markers=peaks_img, mask=thresh)

        images.append(img.get_image_dask_data(dim_order, T=t, C=bfstd_chan).compute().squeeze())
        labels.append(ws)


# img_list = [BioImage(Path(io.get_zarr_path(dataset_name))) for dataset_name in datasets_to_use]

# images = [img.get_image_dask_data(dim_order, T=t, C=4) for img in img_list for t in range(img.dims.T)]

# plt.imshow(images[0].squeeze())

cellpose_io.logger_setup()
# e.g. retrain a Cellpose model
model_path = r"C:\Users\serge.parent\OneDrive - Allen Institute\Desktop\projects\holistic\cellsmap_labelfree_nuclei_model\bf_std_model_no_preprocess"
model_bf_stdproject = models.CellposeModel(gpu=False, pretrained_model=model_path)

model_path, train_losses, test_losses = train.train_seg(model_bf_stdproject.net,
                            train_data=images, train_labels=labels,
                            channels=[0,0], normalize=True,
                            # test_data=test_images, test_labels=test_labels,
                            weight_decay=1e-4, SGD=True, learning_rate=0.1,
                            n_epochs=100, model_name="bf_std_model_no_preprocess_retrained")


test_dataset_name = '20241016_20X'
# test_dataset_name = '20231122_T02_001'
test_image = BioImage(Path(io.get_zarr_path(test_dataset_name)))
bfstd_chan = io.get_channel_index(test_dataset_name, ['BF_STD'])
bf_chan = io.get_channel_index(test_dataset_name, ['BF_Center'])
model_path = Path(r'C:\Users\serge.parent\OneDrive - Allen Institute\Desktop\projects\holistic\cellsmap\cellsmap\analyses\nuclei_predictions\models\bf_std_model_no_preprocess_retrained')
model_path.exists()
model_bf_stdproject_new = models.CellposeModel(gpu=False, pretrained_model=str(model_path))
test = test_image.get_image_dask_data(dim_order, T=0, C=bfstd_chan)
test_bf = test_image.get_image_dask_data(dim_order, T=0, C=bf_chan)

test_crop = (slice(0, 1000), slice(0, 1000))
test = test.compute().squeeze()[test_crop]
test_bf = test_bf.compute().squeeze()[test_crop]

try:
    nuc_chan = io.get_channel_index(test_dataset_name, ['DAPI'])
    test_nuc = test_image.get_image_dask_data(dim_order, T=0, C=nuc_chan) if nuc_chan else None
    test_nuc = test_nuc.compute().squeeze()[test_crop] if nuc_chan else None
except ValueError:
    print('No DAPI channel found.')
    test_nuc = None


masks_bf_std = model_bf_stdproject_new.eval(test, channels=[0,0], min_size=50, flow_threshold=0.6, cellprob_threshold=0)#-3.0)

overlay_bf = label2rgb(label=masks_bf_std[0], image=rescale_intensity(test_bf, out_range=(0,1)), bg_label=0)
plt.imshow(overlay_bf[test_crop])

overlay_bf = label2rgb(label=masks_bf_std[0], image=rescale_intensity(np.clip(test, a_min=0, a_max=np.percentile(test, 95)), out_range=(0,1)), bg_label=0)
plt.imshow(overlay_bf[test_crop])

if test_nuc:
    overlay_nuc = label2rgb(label=masks_bf_std[0], image=rescale_intensity(np.clip(test_nuc, a_min=0, a_max=np.percentile(test, 95)), out_range=(0,1)), bg_label=0)
    plt.imshow(overlay_nuc[test_crop])

plt.imshow(rescale_intensity(np.clip(test, a_min=0, a_max=np.percentile(test, 90)), out_range=(0,1))[test_crop])
# plt.imshow(rescale_intensity(test, out_range=(0,1))[test_crop])

# compare 10 or 20 timepoints throughout the timelapse
# and estimate the false positive and false negative rates

