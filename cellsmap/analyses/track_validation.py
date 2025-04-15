from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.ndimage import gaussian_filter1d
# from cellsmap.analyses.cdh5_nodes_and_edges_analysis import stringified_floatlist_to_floatlist
# from matplotlib.colors import TwoSlopeNorm
# from cellsmap.util import io
from cellsmap.util.cdh5_preprocessing import extract_T
from bioio import BioImage
from skimage import measure
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from tqdm import tqdm

def save_validation_images(roi, img_arr, C_labels, C_outlines, C_raws, out_dir, dataset_name, T, padding=50):
    track_id = roi.label
    validation_subfolder = out_dir / dataset_name / str(track_id)
    Path.mkdir(validation_subfolder, exist_ok=True, parents=True)

    expanded_bbox = tuple([slice(max(0, sl.start - padding), sl.stop + padding) for sl in roi.slice])

    crop = img_arr[(..., *expanded_bbox)].squeeze().compute()
    track_of_interest = (crop[C_labels] == track_id * 1) + (crop[C_outlines] > 0) * 2
    raw_img_crop = rescale_intensity(np.clip(crop[C_raws], 0, np.percentile(crop[C_raws], 98)), out_range=(0, 1))
    overlay = label2rgb(label=track_of_interest, image=raw_img_crop, bg_label=0, colors=['magenta', 'cyan'])

    fig, ax = plt.subplots()
    ax.imshow(overlay)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(validation_subfolder / f'{dataset_name}_track{track_id}_T{T}.png', bbox_inches='tight', pad_inches=0, dpi=180)
    plt.close(fig)

    return


def generate_and_save_validation_images(dframe, dataset_name, T, out_dir):
    print(f'Working on dataset {dataset_name}, T = {T}...')
    # T = 0
    # C = 0
    C_labels, C_outlines, C_raws = 0, 1 ,2

    img = BioImage(tracking_img_paths[dataset_name][T])
    img_arr = img.get_image_dask_data('TCZYX')

    props = measure.regionprops(img_arr[:, C_labels, :, :, :].squeeze())
    rois = [reg for reg in props if reg.label in dframe.query('T==@T')['track_id'].unique()]
    # roi[0].slice
    # ymin, xmin, ymax, xmax = 0, 0, img.dims.Y+1, img.dims.X+1
    # test = list(zip((ymin, xmin, ymax, xmax), roi[0].bbox))
    # [max(x) for x in test]
    # ypad, xpad = 50, 50
    # padding = np.array([-ypad, -xpad, ypad, xpad])
    padding = 50

    # np.array(roi[0].bbox)
    # np.array(roi[0].bbox) + padding

    for roi in tqdm(rois):
        save_validation_images(roi, img_arr, C_labels, C_outlines, C_raws, out_dir, dataset_name, T, padding=padding)
    # list(tqdm(rois, total=len(rois), desc='Saving validation images...').map(lambda roi: save_validation_images(roi, img_arr, out_dir, dataset_name, T, padding=padding))

    # for roi in rois:
    #     track_id = roi.label
    #     validation_subfolder = out_dir / dataset_name / str(track_id)
    #     Path.mkdir(validation_subfolder, exist_ok=True, parents=True)

    #     # for sl in roi.slice:
    #         # print(sl.start)
    #         # break
    #         # slice(max(0, sl.start - padding), sl.stop + padding)

    #     expanded_bbox = tuple([slice(max(0, sl.start - padding), sl.stop + padding) for sl in roi.slice])

    #     crop = img_arr[(..., *expanded_bbox)].squeeze().compute()
    #     track_of_interest = (crop[C_labels] == track_id * 1) + (crop[C_outlines] > 0) * 2
    #     raw_img_crop = rescale_intensity(np.clip(crop[C_raws], 0, np.percentile(crop[C_raws], 98)), out_range=(0, 1))
    #     overlay = label2rgb(label=track_of_interest, image=raw_img_crop, bg_label=0, colors=['magenta', 'cyan'])

    #     fig, ax = plt.subplots()
    #     ax.imshow(overlay)
    #     ax.axis('off')
    #     plt.tight_layout()
    #     fig.savefig(validation_subfolder / f'{dataset_name}_track{track_id}_T{T}.png', bbox_inches='tight', pad_inches=0, dpi=180)
    #     plt.close(fig)

    # if T > 0: break
    return


prj_dir = Path(__file__).parents[1]
out_dir = prj_dir / f'results/{Path(__file__).stem}'
data_dir = Path(r'C:\Users\serge.parent\OneDrive - Allen Institute\Desktop\projects\holistic\cellsmap\cellsmap\results\test_tracking_output_exploration')
assert data_dir.exists(), f'Data directory {data_dir} not found.'

tracking_img_dir = Path('//allen/aics/assay-dev/users/Serge/cellsmap_out/cdh5_classic_seg_tracking')
assert tracking_img_dir.exists(), f'Data directory {tracking_img_dir} not found.'

tracking_img_paths = {dataset_path.name: {extract_T(fp): fp for fp in sorted(dataset_path.glob('tracked_images/*.tif*'), key=extract_T)} for dataset_path in tracking_img_dir.glob('*')}
tracking_df = pd.read_csv(data_dir / 'filtered_tracking_results.tsv', sep='\t')

tracking_df = tracking_df.query('track_duration >= 168')

# track_id 65 from 20241016_20X is pretty long and might be good to check
dataset_name = '20241016_20X'
# track_id = 65
df_sub = tracking_df.query('dataset_name==@dataset_name')

# for T in df_sub['T'].unique():
for (dataset_name, T), df_subset in df_sub.groupby(['dataset_name', 'T']):

    print(dataset_name, T)
    if T >= 0:
        generate_and_save_validation_images(df_subset, dataset_name, T, out_dir)


# NOTE a plotting test for distance travelled by the centroid
# plt.plot(df_sub['T'], np.cumsum(df_sub['centroid_displacement']))

