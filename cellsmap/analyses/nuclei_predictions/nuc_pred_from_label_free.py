from pathlib import Path
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from bioio_base.types import PhysicalPixelSizes
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from cellsmap.util import io
from cellpose import models

print('All available datasets:')
dataset_names_all = io.get_available_datasets()

test_datasets = ['20231122_T02_001',]
                #  '20240328_T02_001', '20240328_T01_001',
                #  '20241016_20X', '20241105_20X', '20241120_20X']

dataset_names_list = [name for name in dataset_names_all if name in test_datasets]

# the label-free nuclear prediction model that Goutham trained:
GN_nuc_model_path = '//allen/aics/assay-dev/computational/data/endothileal_cell_data/Timelapse_20x_dataset/montage_data_trainsets/models_weights/BF_STD_patch_model/models/bf_std_model_no_preprocess'

out_dir = Path('../../').resolve() / 'results' / Path(__file__).stem
Path.mkdir(out_dir, exist_ok=True, parents=True)

for dataset_name in dataset_names_list:
    img_path = Path(io.get_zarr_path(dataset_name))
    img = BioImage(img_path)
    dim_order = 'TCZYX'
    dim_map = io.get_dim_map(dim_order)
    bf_chan = io.get_channel_index(dataset_name, ['BF_Center'])
    bfstd_chan = io.get_channel_index(dataset_name, ['BF_STD'])
    nuc_chan = io.get_channel_index(dataset_name, ['DAPI'])
    img_arr = img.get_image_dask_data(dim_order)

    for timepoint in range(len(img_arr)):
        # plt.imshow(img_at_T.squeeze())
        img_at_T = img_arr[timepoint].compute()

        model_bf_stdproject = models.CellposeModel(gpu=False, pretrained_model=GN_nuc_model_path)

        # bfield_std = io.load_dataset(dataset_name, time_start=0, time_end=0, level=0, channels=['BF_STD']).compute().squeeze()

        masks_bf_std = model_bf_stdproject.eval(img_at_T[bfstd_chan].squeeze(), channels=[0,0], min_size=50, flow_threshold=0.6, cellprob_threshold=0)
        overlay_nuc = label2rgb(label=masks_bf_std[0], image=rescale_intensity(np.clip(img_at_T[nuc_chan].squeeze(), 0, np.percentile(img_at_T[nuc_chan].squeeze(), 98))), bg_label=0, colors=['red'])
        plt.imshow(overlay_nuc)

        # break

        overlay_bf = label2rgb(label=masks_bf_std[0], image=rescale_intensity(img_at_T[bf_chan].squeeze()), bg_label=0)
        overlay_nuc = label2rgb(label=masks_bf_std[0], image=rescale_intensity(np.clip(img_at_T[nuc_chan].squeeze(), 0, np.percentile(img_at_T[nuc_chan].squeeze(), 98))), bg_label=0, colors=['red'])

        fig, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.imshow(overlay_bf)
        ax2.imshow(overlay_nuc)
        ax1.axis('off')
        ax2.axis('off')
        ax1.set_title('Brightfield Std Dev Overlay')
        ax2.set_title('DAPI Overlay')
        plt.tight_layout()
        fig.savefig(out_dir / f'{dataset_name}_T{timepoint}_bf_std_nuc_pred.png', bbox_inches='tight', dpi=600)

        # CytoDL nuclei prediction from Benji
        cytodl_nuc_pred_dir = list(Path(out_dir / 'raw_seg').glob('*.tif*'))
        # cytodl_nuc_pred_path = [fp for fp in cytodl_nuc_pred_dir if str(img_path.stem).split('.')[0] in str(fp.stem)]
        cytodl_nuc_pred_path = cytodl_nuc_pred_dir[0]
        cytodl_nuc_pred = BioImage(cytodl_nuc_pred_path).get_image_data().squeeze()

        overlay_bf2 = label2rgb(label=masks_bf_std[0].astype(bool)*1 + cytodl_nuc_pred.astype(bool)*2, image=rescale_intensity(img_at_T[bf_chan].squeeze()), bg_label=0, colors=['red', 'cyan', 'yellow'])
        overlay_nuc2 = label2rgb(label=masks_bf_std[0].astype(bool)*1 + cytodl_nuc_pred.astype(bool)*2, image=rescale_intensity(np.clip(img_at_T[nuc_chan].squeeze(), 0, np.percentile(img_at_T[nuc_chan].squeeze(), 98))), bg_label=0, colors=['red', 'cyan', 'yellow'])

        fig, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.imshow(overlay_bf2)
        ax2.imshow(overlay_nuc2)
        ax1.axis('off')
        ax2.axis('off')
        ax1.set_title('Brightfield Std Dev Overlay')
        ax2.set_title('DAPI Overlay')
        plt.tight_layout()
        fig.savefig(out_dir / f'{dataset_name}_T{timepoint}_bf_std_nuc_pred2.png', bbox_inches='tight', dpi=600)


        break
