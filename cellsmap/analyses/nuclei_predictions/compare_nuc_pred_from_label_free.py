from pathlib import Path
from bioio import BioImage
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.filters import apply_hysteresis_threshold
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.morphology import dilation, disk
from cellsmap.util import dataset_io
from cellpose import models
import re

# NOTE
# because we don't have zarr files for the datasets in the
# test_datasets list, the function dataset_io.get_channel_index
# does not work. Therefore this script is currently broken.
use_original_data = False

def plot_and_save_overlays(overlay_bf, overlay_nuc, out_dir, dataset_name, timepoint, filename_suffix=''):
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.imshow(overlay_bf)
    ax2.imshow(overlay_nuc)
    ax1.axis('off')
    ax2.axis('off')
    ax1.set_title('Brightfield Std Dev Overlay')
    ax2.set_title('DAPI Overlay')
    plt.tight_layout()
    fig.savefig(out_dir / dataset_name / f'{dataset_name}_T{timepoint}_bf_std_nuc_pred{filename_suffix}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


print('All available datasets:')
dataset_names_all = dataset_io.get_available_datasets()

test_datasets = ['20240328_T02_001', '20240328_T01_001',]

dataset_names_list = [name for name in dataset_names_all if name in test_datasets]

out_dir = dataset_io.get_results_dir(Path(__file__).stem, is_test=False)
Path.mkdir(out_dir, exist_ok=True, parents=True)

# CellPose label-free nuclear prediction model that Goutham trained:
model_config = dataset_io.load_config(config_type='model')
nuclei_models = [model for model in model_config if model['name'] == 'nuc_pred_labelfree']
assert len(nuclei_models) == 1, f'Expected 1 model path, found {len(nuclei_models)}'
model_path = Path(nuclei_models[0]['model_path_retrained'])
model_bf_stdproject = models.CellposeModel(gpu=False, pretrained_model=str(model_path))

# CytoDL nuclei predictions from Benji:
cytodl_nuc_pred_dir = list(Path(out_dir / 'raw_seg').glob('*.tif*'))

# create empty list to store nuclei count data:
nuclei_count_data = []

for dataset_name in dataset_names_list:

    Path.mkdir(out_dir / dataset_name, exist_ok=True, parents=True)

    if use_original_data:
        img_path = Path(dataset_io.get_original_path(dataset_name))
    else:
        img_path = Path(dataset_io.get_zarr_path(dataset_name))
    img = BioImage(img_path)
    dim_order = 'TCZYX'
    dim_map = dataset_io.get_dim_map(dim_order)
    bf_chan = dataset_io.get_channel_index(dataset_name, ['BF_Center'])
    bfstd_chan = dataset_io.get_channel_index(dataset_name, ['BF_STD'])
    nuc_chan = dataset_io.get_channel_index(dataset_name, ['DAPI'])
    img_arr = img.get_image_dask_data(dim_order)

    # function to extract the timepoint from the CytoDL output files:
    get_T_from_path = lambda x: int(re.findall('T_[0-9]+', x.stem)[-1].split('T_')[-1])

    for timepoint in range(len(img_arr)):
        print(f'Working on dataset {dataset_name}, T = {timepoint}...')

        img_at_T = img_arr[timepoint].compute()

        # Use the CellPose model to predict nuclei from the brightfield std dev channel:
        masks_bf_std = model_bf_stdproject.eval(img_at_T[bfstd_chan].squeeze(), channels=[0,0], min_size=50, flow_threshold=0.6, cellprob_threshold=0)

        overlay_bf = label2rgb(label=masks_bf_std[0], image=rescale_intensity(img_at_T[bf_chan].squeeze()), bg_label=0)
        overlay_nuc = label2rgb(label=masks_bf_std[0], image=rescale_intensity(np.clip(img_at_T[nuc_chan].squeeze(), 0, np.percentile(img_at_T[nuc_chan].squeeze(), 98))), bg_label=0)
        plot_and_save_overlays(overlay_bf, overlay_nuc, out_dir, dataset_name, timepoint, filename_suffix='_cellpose')

        cytodl_nuc_pred_path = [fp for fp in cytodl_nuc_pred_dir if dataset_name in str(fp.stem) and get_T_from_path(fp) == timepoint]
        assert len(cytodl_nuc_pred_path) == 1, f'Expected 1 file for {dataset_name} T{timepoint}, found {len(cytodl_nuc_pred_path)}'
        cytodl_nuc_pred = BioImage(cytodl_nuc_pred_path[0]).get_image_data().squeeze()

        overlay_bf = label2rgb(label=cytodl_nuc_pred, image=rescale_intensity(img_at_T[bf_chan].squeeze()), bg_label=0)
        overlay_nuc = label2rgb(label=cytodl_nuc_pred, image=rescale_intensity(np.clip(img_at_T[nuc_chan].squeeze(), 0, np.percentile(img_at_T[nuc_chan].squeeze(), 98))), bg_label=0)
        plot_and_save_overlays(overlay_bf, overlay_nuc, out_dir, dataset_name, timepoint, filename_suffix='_cytodl')

        overlay_bf = label2rgb(label=masks_bf_std[0].astype(bool)*1 + cytodl_nuc_pred.astype(bool)*2, image=rescale_intensity(img_at_T[bf_chan].squeeze()), bg_label=0, colors=['red', 'cyan', 'yellow'])
        overlay_nuc = label2rgb(label=masks_bf_std[0].astype(bool)*1 + cytodl_nuc_pred.astype(bool)*2, image=rescale_intensity(np.clip(img_at_T[nuc_chan].squeeze(), 0, np.percentile(img_at_T[nuc_chan].squeeze(), 98))), bg_label=0, colors=['red', 'cyan', 'yellow'])
        plot_and_save_overlays(overlay_bf, overlay_nuc, out_dir, dataset_name, timepoint, filename_suffix='_cellpose_vs_cytodl')

        # do a classic watershed segmentation on the DAPI channel for comparison:
        normd_nuc = rescale_intensity(img_at_T[nuc_chan].squeeze(), out_range=(0,1))
        thresh = apply_hysteresis_threshold(normd_nuc, np.percentile(normd_nuc, 80), np.percentile(normd_nuc, 85))
        dist = distance_transform_edt(thresh)
        peaks_img = np.zeros(dist.shape, dtype=bool)
        peaks_img[tuple(zip(*peak_local_max(dist, min_distance=15)))] = True
        peaks_img = label(dilation(peaks_img, footprint=disk(5)))
        ws = watershed(rescale_intensity(dist, out_range=(1,0)), markers=peaks_img, mask=thresh)
        overlay_bf3 = label2rgb(label=ws, image=rescale_intensity(img_at_T[bf_chan].squeeze()), bg_label=0)#, colors=['orange'])
        overlay_nuc3 = label2rgb(label=ws, image=rescale_intensity(np.clip(normd_nuc, 0, 0.1)), bg_label=0)#, colors=['orange'])
        plot_and_save_overlays(overlay_bf3, overlay_nuc3, out_dir, dataset_name, timepoint, filename_suffix='_classic')


        nuclei_count_data.append({
            'dataset_name': dataset_name,
            'T': timepoint,
            'image_id': '_'.join([dataset_name, str(timepoint)]),
            'nuclei_count': np.count_nonzero(np.unique(cytodl_nuc_pred)),
            'method': 'CytoDL',
            'fraction_wrt_classic': np.count_nonzero(np.unique(cytodl_nuc_pred)) / np.count_nonzero(np.unique(ws)),
        })
        nuclei_count_data.append({
            'dataset_name': dataset_name,
            'T': timepoint,
            'image_id': '_'.join([dataset_name, str(timepoint)]),
            'nuclei_count': np.count_nonzero(np.unique(masks_bf_std[0])),
            'method': 'CellPose',
            'fraction_wrt_classic': np.count_nonzero(np.unique(masks_bf_std[0])) / np.count_nonzero(np.unique(ws)),
        })
        nuclei_count_data.append({
            'dataset_name': dataset_name,
            'T': timepoint,
            'image_id': '_'.join([dataset_name, str(timepoint)]),
            'nuclei_count': np.count_nonzero(np.unique(ws)),
            'method': 'classic',
            'fraction_wrt_classic': np.count_nonzero(np.unique(ws)) / np.count_nonzero(np.unique(ws)),
        })


nuclei_count_df = pd.DataFrame(nuclei_count_data)
fig, ax = plt.subplots()
sns.barplot(data=nuclei_count_data,
            x='dataset_name',
            y='nuclei_count',
            hue='method',
            ax=ax)
plt.tight_layout()
fig.savefig(out_dir / 'nuclei_counts.png', bbox_inches='tight', dpi=180)

for nm, grp in nuclei_count_df.groupby('dataset_name'):
    fig, ax = plt.subplots()
    sns.barplot(data=grp,
                x='image_id',
                y='nuclei_count',
                hue='method',
                ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_title(nm)
    plt.tight_layout()
    fig.savefig(out_dir / f'{nm}_nuclei_counts.png', bbox_inches='tight', dpi=180)

for nm, grp in nuclei_count_df.groupby('dataset_name'):
    fig, ax = plt.subplots()
    sns.barplot(data=grp,
                x='image_id',
                y='fraction_wrt_classic',
                hue='method',
                ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_title(nm)
    plt.tight_layout()
    fig.savefig(out_dir / f'{nm}_nuclei_counts_fractions.png', bbox_inches='tight', dpi=180)
