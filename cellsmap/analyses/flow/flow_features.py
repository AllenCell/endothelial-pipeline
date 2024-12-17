# %% Import libraries
from pathlib import Path
import importlib
import flow_calculator
import concurrent
from pandas import DataFrame
importlib.reload(flow_calculator)

# %% Make list of datasets to analzye
dataset_list = ['20240305_T01_001', '20240917_20X_48hr', '20241016_20X']
debug = True
ncores = 1#30
delta_t = 1
out_dir = Path('../../').resolve() / 'results' / Path(__file__).stem
out_path_list = []

# %%
if __name__ == '__main__':
    # run using multiple cores
    if ncores > 1:
        with concurrent.futures.ProcessPoolExecutor(ncores) as executor:
            for dataset_name in dataset_list:
                out_path = flow_calculator.compute_and_save_flow_field(out_dir, dataset_name, delta_t, executor, ncores, debug)
                out_path_list.append(out_path)

    else:
        for dataset_name in dataset_list:
            executor = None
            out_path = flow_calculator.compute_and_save_flow_field(out_dir, dataset_name, delta_t, executor, ncores, debug)
            out_path_list.append(out_path)

# %% Save the locations of the outputs
DataFrame({'dataset_name':dataset_list, 'vector_field_image_paths': out_path_list}).to_csv(out_dir / 'vector_field_image_paths.csv', index=False)


# %% Example of usage:
from bioio import BioImage
from cellsmap.util.io import load_dataset
import numpy as np
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

# %%
# Get the paths to the vector field images
dataset_list = ['20241016_20X']

for dataset_name in dataset_list:
    # Load vector field image for a dataset
    img = flow_calculator.load_vector_field_img(out_dir, dataset_name)

    # Get the channel names and their indices from the outputted image
    # (the vector field images have metadata saved with them)
    image_paths = flow_calculator.get_vector_field_image_paths(out_dir)
    im_path = image_paths[dataset_name]
    chan_map = {img.channel_names: i for i, img.channel_names in enumerate(BioImage(im_path).channel_names)}
    print(f'The channel names and their indices are: {chan_map}')

    # Get some random crops of the vector fields (this is randomly sampling in time too)
    roi_shape = (1, 4, 64, 64) # ROI: region of interest
    num_rois = 10
    rois = flow_calculator.get_random_roi(img.shape, roi_shape, num_rois=num_rois, random_seed=42)
    # apply these crops to the vector field images
    crops = [img[r] for r in rois]
    # load the roi crops into memory
    crops_in_memory = [c.compute() for c in crops]

    # Get a couple of basic descriptive statistics of the norm and angle channels
    norm_chan_idx = chan_map['norm']
    mag_means = [c[0, norm_chan_idx, ...].mean() for c in crops_in_memory]
    mag_stds = [c[0, norm_chan_idx, ...].std() for c in crops_in_memory]

    angle_chan_idx = chan_map['theta']
    ang_means = [c[0, angle_chan_idx, ...].mean() for c in crops_in_memory]
    ang_stds = [c[0, angle_chan_idx, ...].std() for c in crops_in_memory]

    # Show the vector magnitude channel from the random crops
    nrows, ncols, figsize = 2, 5, (15, 7)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    [ax.imshow(crops_in_memory[i][0, norm_chan_idx, ...], cmap='gray') for i,ax in enumerate(axs.flatten())]
    [ax.axis('off') for i,ax in enumerate(axs.flatten())]
    [ax.set_title(f'mean: {mag_means[i]:.2f}, std: {mag_stds[i]:.2f}') for i,ax in enumerate(axs.flatten())]
    fig.suptitle('Vector norms from random crops', size=20)
    plt.tight_layout()
    plt.show()

    # Show the vector angles channel from the random crops
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    [ax.imshow(crops_in_memory[i][0, angle_chan_idx, ...], cmap='gray') for i,ax in enumerate(axs.flatten())]
    [ax.axis('off') for i,ax in enumerate(axs.flatten())]
    [ax.set_title(f'mean: {mag_means[i]:.2f}, std: {mag_stds[i]:.2f}') for i,ax in enumerate(axs.flatten())]
    fig.suptitle('Vector angles from random crops', size=20)
    plt.tight_layout()
    plt.show()

    # Show the image crops from the random roi regions vs the same
    # image at the next timepoint too
    img = load_dataset(dataset_name, channels=['CDH5'], level=2)
    rois_future = [(slice(r[0].start+delta_t, r[0].stop+delta_t), *r[1:]) for r in rois]
    crops = [np.stack([img[rois[i]], img[rois_future[i]], np.zeros(img[rois[i]].shape)], axis=-1) for i in range(len(rois))]
    crops_in_memory = [c.compute() for c in crops]
    crops_in_memory = [rescale_intensity(c, out_range=np.uint8) for c in crops_in_memory]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    [ax.imshow(crops_in_memory[i][0, 0, ...], cmap='gray') for i,ax in enumerate(axs.flatten())]
    [ax.axis('off') for i,ax in enumerate(axs.flatten())]
    fig.suptitle('CDH5 from random crops (red = t, green = t + delta_t)', size=20)
    plt.tight_layout()
    plt.show()

    # Use the loaded vector information to create the flow field
    vfield = flow_calculator.load_vector_field_img(out_dir, dataset_name)
    vfield_crops = [vfield[r] for r in rois]
    vfield_crops = [c.compute() for c in vfield_crops]
    vx_crops, vy_crops = zip(*[(c[0, chan_map['vx'], ...].squeeze(), c[0, chan_map['vy'], ...].squeeze()) for c in vfield_crops])

    img = load_dataset(dataset_name, channels=['CDH5'], level=2)
    img_crops = [img[r] for r in rois]
    img_crops = [c.compute().squeeze() for c in img_crops]

    flow_graphs = [flow_calculator.FlowCalculator.make_vector_field_map(img, vx, vy, display=False, return_map=True) for img, vx, vy in zip(img_crops, vx_crops, vy_crops)]
    # trim white-space from the flow graphs
    keep_me_indices = [np.where(~np.all(g==255, axis=-1)) for g in flow_graphs]
    keep_me_slices = [(slice(np.min(i), np.max(i)), slice(np.min(j), np.max(j)), slice(None)) for i,j in keep_me_indices]
    flow_graphs = [flow_graphs[i][arr_slice] for i,arr_slice in enumerate(keep_me_slices)]

    rois_as_titles = [list(zip(*[(slc.start, slc.stop) for slc in r])) for r in rois]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[d*1.5 for d in figsize])
    [ax.imshow(flow_graphs[i], cmap='gray') for i, ax in enumerate(axs.flatten())]
    [ax.axis('off') for i, ax in enumerate(axs.flatten())]
    [ax.set_title(f'roi start: {rois_as_titles[i][0]}\nroi stop:{rois_as_titles[i][1]}') for i,ax in enumerate(axs.flatten())]
    fig.suptitle('Flow fields', size=20)
    plt.tight_layout()
    plt.show()


# %% Create figures for validation:
# The figure should be of PC1 vs PC2 for different crops with select crops shown
# and linked back to their corresponding data points in the PC1 vs PC2 plot

# %% 0. imports, function definitions, and dataset loading
from skimage import filters
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import pandas as pd
from matplotlib import gridspec as gs

def compute_PCA_on_features(features: list[np.ndarray], n_components: int=10, return_as_dataframe=False) -> PCA:
    feat_arr = np.asarray([feature.ravel() for feature in features])
    pca = PCA(n_components=n_components)
    # normalize features
    pca.fit((feat_arr - feat_arr.mean()) / feat_arr.std())
    feats_proj = pca.transform(feat_arr).reshape(len(features),-1)
    if return_as_dataframe:
        feats_proj = pd.DataFrame(data=feats_proj)
    else:
        pass
    return pca, feats_proj

def discrete_divergence_like(vx, vy):
    vx_dx = np.gradient(vx, axis=1)
    vy_dy = np.gradient(vy, axis=0)
    return vx_dx + vy_dy

def discrete_curl_like(vx, vy):
    vy_dx = np.gradient(vy, axis=1)
    vx_dy = np.gradient(vx, axis=0)
    return vy_dx - vx_dy

def source_vector_field_example(show_vector_field=False):
    xx, yy = np.meshgrid(np.arange(-10, 11), np.arange(-10, 11))
    vx = xx
    vy = yy
    vfield = (vx, vy)
    if show_vector_field:
        fig, ax = plt.subplots()
        ax.quiver(xx, yy, *vfield)
        ax.set_aspect('equal')
        plt.show()
    return vfield

def sink_vector_field_example(show_vector_field=False):
    xx, yy = np.meshgrid(np.arange(-10, 11), np.arange(-10, 11))
    vx = -1 * xx
    vy = -1 * yy
    vfield = (vx, vy)
    if show_vector_field:
        fig, ax = plt.subplots()
        ax.quiver(xx, yy, *vfield)
        ax.set_aspect('equal')
        plt.show()
    return vfield

def saddle_vector_field_example(show_vector_field=False):
    xx, yy = np.meshgrid(np.arange(-10, 11), np.arange(-10, 11))
    vx = xx
    vy = -1 * yy
    vfield = (vx, vy)
    if show_vector_field:
        fig, ax = plt.subplots()
        ax.quiver(xx, yy, *vfield)
        ax.set_aspect('equal')
        plt.show()
    return vfield

def ridge_vector_field_example(show_vector_field=False):
    xx, yy = np.meshgrid(np.arange(-10, 11), np.arange(-10, 11))
    vx = xx
    vy = 0 * yy
    vfield = (vx, vy)
    if show_vector_field:
        fig, ax = plt.subplots()
        ax.quiver(xx, yy, *vfield)
        ax.set_aspect('equal')
        plt.show()
    return vfield

def valley_vector_field_example(show_vector_field=False):
    xx, yy = np.meshgrid(np.arange(-10, 11), np.arange(-10, 11))
    vx = -1 * xx
    vy = 0 * yy
    vfield = (vx, vy)
    if show_vector_field:
        fig, ax = plt.subplots()
        ax.quiver(xx, yy, *vfield)
        ax.set_aspect('equal')
        plt.show()
    return vfield

def solenoidal_vector_field_example(show_vector_field=False):
    xx, yy = np.meshgrid(np.arange(-10, 11), np.arange(-10, 11))
    vx = -1 * yy
    vy = xx
    vfield = (vx, vy)
    if show_vector_field:
        fig, ax = plt.subplots()
        ax.quiver(xx, yy, *vfield)
        ax.set_aspect('equal')
        plt.show()
    return vfield

def get_divergence_curl_example(vfield: str='solenoidal', show_vector_field=False) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    '''Returns an example vector field, its divergence, and its curl'''
    example_vfields = {'source': source_vector_field_example,
                       'sink': sink_vector_field_example,
                       'saddle': saddle_vector_field_example,
                       'ridge': ridge_vector_field_example,
                       'valley': valley_vector_field_example,
                       'solenoidal': solenoidal_vector_field_example}
    divergence = discrete_divergence_like(*example_vfields[vfield]())
    curl = discrete_curl_like(*example_vfields[vfield]())
    return {'vector_field':example_vfields[vfield](show_vector_field), 'divergence':divergence, 'curl':curl}

test = get_divergence_curl_example()

def get_point_closest_to_reference_point(points: np.ndarray, reference_point: tuple[float, float]) -> tuple[tuple[float, float], int]:
    distances = np.linalg.norm(points - np.array(reference_point), axis=1)
    assert len(distances) == len(points)
    return points[np.argmin(distances)], np.argmin(distances)

def get_quadrant_means(points: np.ndarray, origin: tuple[float, float]=(0,0)) -> list[np.ndarray]:
    origin = np.array(origin)

    top_right = points[(points[:,0] > origin[0]) & (points[:,1] > origin[1])]
    top_left = points[(points[:,0] < origin[0]) & (points[:,1] > origin[1])]
    bottom_left = points[(points[:,0] < origin[0]) & (points[:,1] < origin[1])]
    bottom_right = points[(points[:,0] > origin[0]) & (points[:,1] < origin[1])]

    quadrant_means = [quad_points.mean(axis=0) for quad_points in [top_right, top_left, bottom_left, bottom_right]]

    return quadrant_means

# create an output directory for the plots
out_dir_val = out_dir / 'validation_plots'
out_dir_plots = out_dir_val / f'{dataset_name}'
Path.mkdir(out_dir_plots, exist_ok=True, parents=True)

# Pick a dataset
dataset_name = dataset_list[0]

# Load vector field image for a dataset
flow_feat_img = flow_calculator.load_vector_field_img(out_dir, dataset_name)

# Get the channel names and their indices from the outputted image
# (the vector field images have metadata saved with them)
image_paths = flow_calculator.get_vector_field_image_paths(out_dir)
im_path = image_paths[dataset_name]
chan_map = {flow_feat_img.channel_names: i for i, flow_feat_img.channel_names in enumerate(BioImage(im_path).channel_names)}
print(f'The channel names and their indices are: {chan_map}')


# %% 1. Get a bunch of crops:
# Get some random crops of the vector fields (this is randomly sampling in time too)
# and load the roi crops into memory
roi_shape = (1, 4, 64, 64)
num_rois = 200
rois = flow_calculator.get_random_roi(flow_feat_img.shape, roi_shape, num_rois=num_rois, random_seed=42)


# %% 2. Get flow features of the crops
crops = [flow_feat_img[r] for r in rois]
crops_in_memory = [c.compute() for c in crops]


# %% 3. Compute PCA on the crops
# use the angle mean and std as features

features_vx = [c[:,chan_map['vx'],...].squeeze() for c in crops_in_memory]
features_vy = [c[:,chan_map['vy'],...].squeeze() for c in crops_in_memory]
features_angles = [c[:,chan_map['theta'],...].squeeze() for c in crops_in_memory]
features_angles_std = [np.abs(feat_ang).std() for feat_ang in features_angles]
features_mags = [c[:,chan_map['norm'],...].squeeze() for c in crops_in_memory]
# features_mags = [minmax_scale(feat_mag, feature_range=(0, 1)) for feat_mag in features_mags]
features_mags_std = [feat_mag.std() for feat_mag in features_mags]
features_vx_mean = [feat.mean() for feat in features_vx]
features_vy_mean = [feat.mean() for feat in features_vy]
features_vx_std = [feat.std() for feat in features_vx]
features_vy_std = [feat.std() for feat in features_vy]
angles_of_vector_mean = flow_calculator.FlowCalculator.compute_angles(features_vx_mean, features_vy_mean)
mag_of_vector_mean = flow_calculator.FlowCalculator.compute_magnitudes(features_vx_mean, features_vy_mean)
# features_mags_mean = [feat.mean() for feat in features_mags]
# features_mags_stdevs = [feat.std() for feat in features_mags]


# {'angle_of_vector_mean': angles_of_vector_mean, 'magnitude_of_vector_mean': mag_of_vector_mean}
# features_summary = dict(zip(['angle_mean', 'angle_std'], zip(*[(feat.mean(), feat.std()) for feat in features_angles])))
# features_summary = dict(zip(['angle_mean', 'angle_std'], zip(*[get_angle_mean_and_std(angles_arr) for angles_arr in features_angles])))

divergence = [discrete_divergence_like(crop[:,chan_map['vx'],...].squeeze(), crop[:,chan_map['vy'],...].squeeze()) for crop in crops_in_memory]
curl = [discrete_divergence_like(crop[:,chan_map['vx'],...].squeeze(), crop[:,chan_map['vy'],...].squeeze()) for crop in crops_in_memory]
features_diverg = [(d.mean(), d.std()) for d in divergence]
features_curl = [(c.mean(), c.std()) for c in curl]
# features = [np.array([(crop[:,i,...].mean(), crop[:,i,...].std()) for i in range(0, crop.shape[1])]).ravel() for crop in crops_in_memory]
features = [np.array([(crop[:,chan_map[chan],...].mean(), crop[:,chan_map[chan],...].std()) for chan in ['vx', 'vy', 'theta', 'norm']]).ravel() for crop in crops_in_memory]
# features = [np.array([(crop[:,chan_map[chan],...].mean(), crop[:,chan_map[chan],...].std()) for chan in ['vx', 'vy']]).ravel() for crop in crops_in_memory]
# features = [np.array([(crop[:,chan_map[chan],...].mean(), crop[:,chan_map[chan],...].std()) for chan in ['theta']]).ravel() for crop in crops_in_memory]
# features = list(zip(
#                     *list(zip(*[feat for feat in features])),
#                     angles_of_vector_mean.tolist(),
#                     mag_of_vector_mean.tolist(),
#                     *list(zip(*[feat for feat in features_diverg])),
#                     *list(zip(*[feat for feat in features_curl]))
#                     ))
features = list(zip(
                    # *list(zip(*[feat for feat in features])),
                    # angles_of_vector_mean.tolist(),
                    # mag_of_vector_mean.tolist(),
                    *list(zip(*[feat for feat in features_diverg])),
                    # *list(zip(*[feat for feat in features_curl]))
                    ))
features = list(zip(
                    # *list(zip(*[feat for feat in features])),
                    # angles_of_vector_mean.tolist(),
                    # mag_of_vector_mean.tolist(),
                    # *list(zip(*[feat for feat in features_diverg])),
                    # *list(zip(*[feat for feat in features_curl]))
                    np.linalg.norm([features_vx_mean, features_vy_mean], axis=0),
                    np.linalg.norm([features_vx_std, features_vy_std], axis=0)
                    ))
features = list(zip(
                    # *list(zip(*[feat for feat in features])),
                    # angles_of_vector_mean.tolist(),
                    # mag_of_vector_mean.tolist(),
                    # features_angles_std,
                    features_mags_std,
                    list(zip(*[feat for feat in features_diverg]))[1],
                    # *list(zip(*[feat for feat in features_curl]))
                    ))
# features = list(zip(*list(zip(*[(arr.mean(),
#                                  arr.std()) for arr in features_vx])),
#                     *list(zip(*[(arr.mean(),
#                                  arr.std()) for arr in features_vy])),
#                     angles_of_vector_mean.tolist(),
#                     mag_of_vector_mean.tolist()))
# features = list(zip(angles_of_vector_mean, mag_of_vector_mean))
features = [np.array(feat) for feat in features]
pca, feats_proj = compute_PCA_on_features(features, n_components=2, return_as_dataframe=True)

# rescale the pca features to be between -1 and 1
feats_proj[0] = minmax_scale(feats_proj[0], feature_range=(-1, 1))
feats_proj[1] = minmax_scale(feats_proj[1], feature_range=(-1, 1))

# create a dataframe of the features and PCs
angles_and_pcs = pd.concat([feats_proj.reset_index(inplace=False, names='crop_id'),
                            # pd.DataFrame(features_summary),
                            pd.DataFrame({'angle_of_vector_mean': angles_of_vector_mean}),
                            pd.DataFrame({'features_angles_std': features_angles_std}),
                            pd.DataFrame({'magnitude_of_vector_mean': mag_of_vector_mean}),
                            pd.DataFrame({'features_mags_std': features_mags_std}),
                            pd.DataFrame({'divergence_mean': [d[0] for d in features_diverg]}),
                            pd.DataFrame({'divergence_std': [d[1] for d in features_diverg]}),
                            pd.DataFrame({'curl_mean': [c[0] for c in features_curl]}),
                            pd.DataFrame([{'T':t.start, 'start_y':y.start, 'start_x':x.start} for t,c,y,x in rois])
                            ], axis='columns')


# %% 4, 5. Plot the first two PCs and a selected crop with the flow field for several crops
# load the cdh5 channel of the dataset in the crop region
img = load_dataset(dataset_name, channels=['CDH5'], level=2)
img_crops = [img[r] for r in rois]
img_crops = [c.compute().squeeze() for c in img_crops]

# Use the loaded vector information to create the flow field
# cmap_norm = np.asarray([(crop[:,chan_map['norm'],...].min(), crop[:,chan_map['norm'],...].max()) for crop in crops_in_memory])
cmap_norm = None#cmap_norm.min(), cmap_norm.max()
cmap = 'summer'
flow_graphs = [flow_calculator.FlowCalculator.make_vector_field_map(img, crop[:,chan_map['vx'], ...].squeeze(), crop[:,chan_map['vy'], ...].squeeze(), resolution=15, cmap_norm=cmap_norm, cmap=cmap, display=False, return_map=True) for img, crop in zip(img_crops, crops_in_memory)]
# trim white-space from the flow graphs
keep_me_indices = [np.where(~np.all(g==255, axis=-1)) for g in flow_graphs]
keep_me_slices = [(slice(np.min(i), np.max(i)), slice(np.min(j), np.max(j)), slice(None)) for i,j in keep_me_indices]
flow_graphs = [flow_graphs[i][arr_slice] for i,arr_slice in enumerate(keep_me_slices)]

rois_as_titles = [list(zip(*[(slc.start, slc.stop) for slc in r])) for r in rois]

num_crops_to_plot = 20
for i in range(num_crops_to_plot):
    print(f'Plotting crop {i+1} / {num_crops_to_plot}')
    ang, mag = crops_in_memory[i][:,chan_map['theta'],...].squeeze(), crops_in_memory[i][:,chan_map['norm'],...].squeeze()

    fig = plt.figure(figsize=(15, 3))
    axs = gs.GridSpec(ncols=5, nrows=1, figure=fig)

    ax1 = fig.add_subplot(axs[0, 0])
    ax1.scatter(angles_and_pcs[0], angles_and_pcs[1], marker='.', alpha=0.7)
    ax1.scatter(angles_and_pcs.query('crop_id == @i')[0], angles_and_pcs.query('crop_id == @i')[1], marker='.', color='r', zorder=10)
    ax1.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('PC1 vs PC2 for crops')

    ax2 = fig.add_subplot(axs[0, 1])
    ax2.scatter(np.rad2deg(angles_and_pcs['angle_of_vector_mean']), angles_and_pcs['magnitude_of_vector_mean'], marker='.', alpha=0.7)
    ax2.scatter(np.rad2deg(angles_and_pcs.query('crop_id == @i')['angle_of_vector_mean']), angles_and_pcs.query('crop_id == @i')['magnitude_of_vector_mean'], marker='.', color='r', zorder=10)
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(30))
    ax2.xaxis.set_major_locator(plt.MultipleLocator(90))
    ax2.tick_params(axis='x', which='minor')
    ax2.set_xlim(-180, 180)
    ax2.set_xlabel('Mean vector angle (degrees)')
    ax2.set_ylabel('Mean vector magnitude (px)')
    ax2.set_title('Angle vs magnitude for crops')

    with plt.rc_context({'axes.edgecolor': 'red', 'xtick.color':'red', 'ytick.color':'red', 'xtick.labelcolor':'red', 'ytick.labelcolor':'red'}):
        ax3 = fig.add_subplot(axs[0, 2])
        ax3.imshow(flow_graphs[i], cmap='gray')
        ax3.set_title(f'Flow Field (crop {i})', color='r')
        ax3.text(x=0.5, y=-0.14, ha='center', va='top', transform=ax3.transAxes,
                 s=f'roi start: {rois_as_titles[i][0]}\nroi stop:{rois_as_titles[i][1]})')

        ax4 = fig.add_subplot(axs[0, 3], projection='polar')
        ax4.hist(ang.ravel(), bins=72, color='k')
        y_min, y_max = ax4.get_ylim()
        ax4.arrow(x=float(angles_and_pcs.query('crop_id == @i')['angle_of_vector_mean'].iloc[0]), y=0, dx=0, dy=0.9*y_max, head_width=0.1, head_length=0.15*y_max, length_includes_head=True, lw=2, color='red', alpha=0.5)
        # create minor ticks on the polar plot
        [ax4.plot((theta, theta), (0.95 * y_max, y_max), c='r', lw=0.5, zorder=0) for theta in np.linspace(0, 2*np.pi, 24+1)]
        ax4.set_ylim(y_min, y_max)
        ax4.set_theta_direction(-1)
        ax4.set_xlim(-np.pi, np.pi)
        ax4.yaxis.set_visible(False)
        ax4.set_title('Angle distribution', color='r')

        ax5 = fig.add_subplot(axs[0, 4])
        ax5.hist(mag.ravel(), bins=72, color='k')
        ax5.axvline(angles_and_pcs.query('crop_id == @i')['magnitude_of_vector_mean'].iloc[0], color='r', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Magnitude (px)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Magnitude distribution', color='r')

    plt.tight_layout()
    fig.savefig(out_dir_plots / f'{dataset_name}_crop_{i}.png')
    plt.close(fig)


# %% 6. Get the most "average" point in each quadrant of PC-space as an example
# get example crops from each quadrant

pc_points = angles_and_pcs[[0, 1]].to_numpy()
quadrants_origin = np.mean(pc_points, axis=0)
quadrant_means = get_quadrant_means(pc_points, origin=quadrants_origin)
quadrant_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
example_points = {}
for i, quad_mean in enumerate(quadrant_means):
    example_point, example_index = get_point_closest_to_reference_point(pc_points, reference_point=quad_mean)
    example_crop = angles_and_pcs.iloc[example_index]
    # ensure that the example crop is using the correct points from example_pt
    assert all(example_crop[[0, 1]].to_numpy() == example_point)

    example_points[i] = {'color':quadrant_colors[i], 'record': example_crop}

    # points = np.ma.masked_array(data=angles_and_pcs[[0, 1]], mask=np.all(np.sign(angles_and_pcs[[0, 1]]) == np.sign(quad_mean), axis=1))
    # example_pt = get_point_closest_to_center(points.to_numpy(), quad_mean)
    # example_crop = angles_and_pcs.iloc[example_pt[1]]
    # # ensure that the example crop is using the correct points from example_pt
    # assert all(example_crop[[0, 1]].to_numpy() == example_pt[0])

    # example_points[i] = {'color':quadrant_colors[i], 'record': example_crop}

# generate a flow fields with the colormap normalized to the vector magnitudes of the
# crops being used as examples
cmap_norm = np.asarray([(crops_in_memory[example_points[i]['record']['crop_id'].astype(int)][:,chan_map['norm'],...].min(), crops_in_memory[example_points[i]['record']['crop_id'].astype(int)][:,chan_map['norm'],...].max()) for i in example_points])
cmap_norm = cmap_norm.min(), cmap_norm.max()
flow_graphs = [flow_calculator.FlowCalculator.make_vector_field_map(img, crop[:,chan_map['vx'], ...].squeeze(), crop[:,chan_map['vy'], ...].squeeze(), resolution=15, cmap_norm=cmap_norm, cmap=cmap, display=False, return_map=True) for img, crop in zip(img_crops, crops_in_memory)]
# trim white-space from the flow graphs
keep_me_indices = [np.where(~np.all(g==255, axis=-1)) for g in flow_graphs]
keep_me_slices = [(slice(np.min(i), np.max(i)), slice(np.min(j), np.max(j)), slice(None)) for i,j in keep_me_indices]
flow_graphs = [flow_graphs[i][arr_slice] for i,arr_slice in enumerate(keep_me_slices)]


# %% 7. Plot two PCs and the example crops from each of the 4 quadrants

fig = plt.figure(figsize=(15, 6))
axs = gs.GridSpec(ncols=5, nrows=2, figure=fig)
ax1 = fig.add_subplot(axs[0, 0])
ax1.scatter(angles_and_pcs[0], angles_and_pcs[1], marker='.', c='grey', alpha=0.7)
ax1.axvline(quadrants_origin[0], color='k', linestyle='--', alpha=0.5)
ax1.axhline(quadrants_origin[1], color='k', linestyle='--', alpha=0.5)
ax1.set_xlabel('PC1 (normalized)')
ax1.set_ylabel('PC2 (normalized)')
ax1.set_title('PC1 vs PC2 for crops')

# ax2 = fig.add_subplot(axs[1, 0])
# ax2.scatter(np.rad2deg(angles_and_pcs['angle_of_vector_mean']), angles_and_pcs['magnitude_of_vector_mean'], marker='.', c='grey', alpha=0.7)
# ax2.xaxis.set_minor_locator(plt.MultipleLocator(30))
# ax2.xaxis.set_major_locator(plt.MultipleLocator(90))
# ax2.tick_params(axis='x', which='minor')
# ax2.set_xlim(-180, 180)
# ax2.set_xlabel('Mean vector angle (degrees)')
# ax2.set_ylabel('Mean vector magnitude (px)')
# ax2.set_title('Angle vs magnitude for crops')

# ax2 = fig.add_subplot(axs[1, 0])
# ax2.scatter(np.rad2deg(angles_and_pcs['angle_of_vector_mean']), angles_and_pcs['divergence_mean'], marker='.', c='grey', alpha=0.7)
# ax2.xaxis.set_minor_locator(plt.MultipleLocator(30))
# ax2.xaxis.set_major_locator(plt.MultipleLocator(90))
# ax2.tick_params(axis='x', which='minor')
# ax2.set_xlim(-180, 180)
# ax2.set_xlabel('Mean vector angle (degrees)')
# ax2.set_ylabel('Mean divergence')
# ax2.set_title('Angle vs divergence for crops')

# ax2 = fig.add_subplot(axs[1, 0])
# ax2.scatter(angles_and_pcs['divergence_std'], angles_and_pcs['divergence_mean'], marker='.', c='grey', alpha=0.7)
# # ax2.xaxis.set_minor_locator(plt.MultipleLocator(30))
# # ax2.xaxis.set_major_locator(plt.MultipleLocator(90))
# # ax2.tick_params(axis='x', which='minor')
# # ax2.set_xlim(-180, 180)
# ax2.set_xlabel('Divergence standard deviation')
# ax2.set_ylabel('Divergence mean')
# ax2.set_title('Diverge mean vs\nstandard deviation for crops')

# ax2 = fig.add_subplot(axs[1, 0])
# ax2.scatter(np.rad2deg(angles_and_pcs['features_angles_std']), angles_and_pcs['divergence_mean'], marker='.', c='grey', alpha=0.7)
# # ax2.xaxis.set_minor_locator(plt.MultipleLocator(30))
# # ax2.xaxis.set_major_locator(plt.MultipleLocator(90))
# # ax2.tick_params(axis='x', which='minor')
# # ax2.set_xlim(-180, 180)
# ax2.set_xlabel('Angles St. Dev.')
# ax2.set_ylabel('Divergence mean')
# ax2.set_title('Diverge mean vs\nstandard deviation for crops')

ax2 = fig.add_subplot(axs[1, 0])
ax2.scatter(np.rad2deg(angles_and_pcs['features_mags_std']), angles_and_pcs['divergence_std'], marker='.', c='grey', alpha=0.7)
# ax2.xaxis.set_minor_locator(plt.MultipleLocator(30))
# ax2.xaxis.set_major_locator(plt.MultipleLocator(90))
# ax2.tick_params(axis='x', which='minor')
# ax2.set_xlim(-180, 180)
ax2.set_xlabel('Magnitude St. Dev.')
ax2.set_ylabel('Divergence St. Dev.')
ax2.set_title('Divergence std vs.\n Magnitude std for crops')

for i, example_pt in example_points.items():
    quad_color, quad_record, crop_id = example_pt['color'], example_pt['record'], example_pt['record']['crop_id'].astype(int)
    ang, mag = crops_in_memory[crop_id][:,chan_map['theta'],...].squeeze(), crops_in_memory[crop_id][:,chan_map['norm'],...].squeeze()
    divergence_at_crop, curl_at_crop = divergence[crop_id], curl[crop_id]

    ax1.scatter(quad_record[0], quad_record[1], marker='.', color=quad_color, zorder=10)
    # ax1.scatter(*quadrant_means[i], marker='x', color=quad_color, alpha=0.5)
    ax1.scatter(*quadrant_means[i], marker='x', color=quad_color, alpha=0.5)

    # ax2.scatter(np.rad2deg(quad_record['angle_of_vector_mean']), quad_record['magnitude_of_vector_mean'], marker='.', color=quad_color, zorder=10)
    # ax2.scatter(np.rad2deg(quad_record['angle_of_vector_mean']), quad_record['divergence_mean'], marker='.', color=quad_color, zorder=10)
    # ax2.scatter(quad_record['divergence_std'], quad_record['divergence_mean'], marker='.', color=quad_color, zorder=10)
    # ax2.scatter(np.rad2deg(quad_record['features_angles_std']), quad_record['divergence_mean'], marker='.', color=quad_color, zorder=10)
    ax2.scatter(np.rad2deg(quad_record['features_mags_std']), quad_record['divergence_std'], marker='.', color=quad_color, zorder=10)

    with plt.rc_context({key: quad_color for key in ['axes.edgecolor', 'xtick.color', 'ytick.color', 'xtick.labelcolor', 'ytick.labelcolor']}):
        ax3 = fig.add_subplot(axs[0, i+1])
        ax3.imshow(flow_graphs[crop_id], cmap='gray')
        ax3.axis('off')
        ax3.set_title(f'Flow Field (crop {crop_id})', color=quad_color)
        ax3.text(x=0.5, y=-0.14, ha='center', va='top', transform=ax3.transAxes,
                 s=f'roi start: {rois_as_titles[crop_id][0]}\nroi stop:{rois_as_titles[crop_id][1]})')

        ax4 = fig.add_subplot(axs[1, i+1], projection='polar')
        ax4.hist(ang.ravel(), bins=72, color=quad_color, alpha=1)
        y_min, y_max = ax4.get_ylim()
        ax4.arrow(x=float(quad_record['angle_of_vector_mean']), y=0, dx=0, dy=0.9*y_max, head_width=0.1, head_length=0.15*y_max, length_includes_head=True, lw=1, ls='-', facecolor=quad_color, edgecolor='k', alpha=0.5)
        # create minor ticks on the polar plot
        [ax4.plot((theta, theta), (0.95 * y_max, y_max), c=quad_color, lw=0.5, zorder=0) for theta in np.linspace(0, 2*np.pi, 24+1)]
        ax4.set_ylim(y_min, y_max)
        ax4.set_theta_direction(-1)
        ax4.set_xlim(-np.pi, np.pi)
        ax4.yaxis.set_visible(False)
        ax4.set_title('Angle distribution', color=quad_color)

        # ax5 = fig.add_subplot(axs[2, i+1])
        # ax5.hist(mag.ravel(), bins=72, color=quad_color, alpha=1)
        # ax5.axvline(quad_record['magnitude_of_vector_mean'], color='k', linestyle='--', alpha=0.5)
        # ax5.set_xlabel('Magnitude (px)')
        # ax5.set_ylabel('Frequency')
        # ax5.set_title('Magnitude distribution', color=quad_color)

        # ax5 = fig.add_subplot(axs[2, i+1])
        # ax5.hist(divergence_at_crop.ravel(), bins=72, color=quad_color, alpha=1)
        # ax5.axvline(quad_record['divergence_mean'], color='k', linestyle='--', alpha=0.5)
        # ax5.set_xlabel('Divergence')
        # ax5.set_ylabel('Frequency')
        # ax5.set_title('Divergence distribution', color=quad_color)

plt.tight_layout()
fig.savefig(out_dir_val / f'{dataset_name}_multicrop.png')
plt.close(fig)


# %% Create some synthetic data to test the above vector field plotting:
from skimage.draw import circle_perimeter
from skimage.filters import gaussian
# create empty synthetic data with shape (time, channel, y, x)
synth_shape_y, synth_shape_x = 512, 512
num_circles_per_axis = 10
circle_radii = 20
synth_img = np.zeros((5, 1, synth_shape_y, synth_shape_x), dtype=np.uint8)
# add a bunch of circles throughout the image that move down 1 pixel and to the left 1 pixel
# after each timepoint (total travel distance is sqrt(2) pixels per timepoint)
for i in range(len(synth_img)):
    circle_centers = np.meshgrid(range(0, synth_shape_y, synth_shape_y//num_circles_per_axis), range(0, synth_shape_x, synth_shape_x//num_circles_per_axis))
    circle_centers = list(zip(*[c_arr.ravel().tolist() for c_arr in circle_centers]))
    circle_indices = list(zip(*[circle_perimeter(y+i, x-i, circle_radii) for y,x in circle_centers]))
    circle_indices = np.asarray([np.concatenate(indices) for indices in circle_indices])
    indices_too_low = np.any(circle_indices < np.array([[0],[0]], ndmin=2), axis=0, keepdims=True)
    indices_too_high = np.any(circle_indices >= np.array([[synth_shape_y],[synth_shape_x]], ndmin=2), axis=0, keepdims=True)
    circle_indices = circle_indices[:, ~np.any(indices_too_low | indices_too_high, axis=0)]
    ts, cs = [i] * len(circle_indices[0]), [0] * len(circle_indices[0])
    synth_img[(ts, cs, *circle_indices)] = 255
    synth_img[i, 0, ...] = gaussian(synth_img[i, 0, ...], sigma=2, preserve_range=True)

# compute flow vectors from the synthetic data
flow_graphs = []
for i in range(len(synth_img)-1):
    print(f'Computing flow for frame {i} to {i+1}')
    flow = flow_calculator.FlowCalculator.compute_flow(synth_img[i].squeeze(), synth_img[i+1].squeeze(), radius=30)
    flow_graphs.append(flow)


# %% compute angles and magnitudes of flow vectors on synthetic data
vx, vy = flow_graphs[0]
theta = flow_calculator.FlowCalculator.compute_angles(vx, vy)
mag = flow_calculator.FlowCalculator.compute_magnitudes(vx, vy)

angles_deg = np.rad2deg(theta)
# mean_angle_deg = angles_deg[angles_deg > 0].mean()
mean_angle_deg = np.rad2deg(flow_calculator.FlowCalculator.compute_angles(vx.mean(), vy.mean()))
mean_mag = flow_calculator.FlowCalculator.compute_magnitudes(vx.mean(), vy.mean())

# mag_mean = mag[mag > 0].mean()
print(f'Flow angle mean: {mean_angle_deg} \nFlow magnitude mean: {mean_mag}')
# %% Generate flow fields for synthetic data
synth_data_example = flow_calculator.FlowCalculator.make_vector_field_map(synth_img[0].squeeze(), vx, vy, resolution=10, display=True, return_map=True)
keep_me_indices = np.where(~np.all(synth_data_example==255, axis=-1))
i, j = keep_me_indices
keep_me_slices = (slice(np.min(i), np.max(i)), slice(np.min(j), np.max(j)), slice(None))
flow_graphs = synth_data_example[keep_me_slices]


# %% Plot the synthetic data and the flow field
fig = plt.figure(figsize=(6, 3))
axs = gs.GridSpec(ncols=2, nrows=1, figure=fig)

ax1 = fig.add_subplot(axs[0, 0])
ax1.imshow(flow_graphs, cmap='gray')
ax1.set_title(f'''Flow Field (synthetic data)\nMean (True) angle: {mean_angle_deg:.2f} deg (135 deg)\nMean (True) mag: {mean_mag:.2f} px (sqrt(2) px)''',
              fontsize=10)

ax2 = fig.add_subplot(axs[0, 1], projection='polar')
ax2.arrow(x=np.deg2rad(mean_angle_deg), y=0, dx=0, dy=0.9, head_width=0.1, head_length=0.15, length_includes_head = True, lw=2, color='r', zorder=10)
# create minor ticks on the polar plot
[ax2.plot((theta, theta), (0.95, 1), c='k', lw=0.5, zorder=0) for theta in np.linspace(0, 2*np.pi, 36+1)]
ax2.set_ylim(0, 1)
ax2.set_theta_direction(-1)
ax2.set_xlim(-np.pi, np.pi)
ax2.yaxis.set_visible(False)
ax2.set_title('Mean angle orientation', fontsize=10)

plt.tight_layout()
fig.savefig(out_dir_val / f'{dataset_name}_synth_data.png')
plt.close(fig)

# %%
