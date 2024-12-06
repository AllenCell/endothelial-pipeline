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
    # and load the roi crops into memory
    roi_shape = (1, 4, 64, 64)
    rois = flow_calculator.get_random_roi(img.shape, roi_shape, num_rois=10, random_seed=42)
    crops = [img[r] for r in rois]
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

# %% Create a figure for validation
# The figure should be of PC1 vs PC2 for different crops with select crops shown
# and linked back to their corresponding data points in the PC1 vs PC2 plot


# %% Pick and load the flow features of a dataset:
# Pick a dataset
dataset_name = '20241016_20X'

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
rois = flow_calculator.get_random_roi(flow_feat_img.shape, roi_shape, num_rois=10, random_seed=42)
crops = [flow_feat_img[r] for r in rois]
crops_in_memory = [c.compute() for c in crops]


# %% 2. Get flow features of the crops



# %% 3. Compute PCA on the crops


# %% 4. Plot the first two PCs of each crop


# %% 5. Display what some of the crops look like from the PC1 vs PC2 plot

# Get a couple of basic descriptive statistics of the norm and angle channels
norm_chan_idx = chan_map['norm']
mag_means = [c[0, norm_chan_idx, ...].mean() for c in crops_in_memory]
mag_stds = [c[0, norm_chan_idx, ...].std() for c in crops_in_memory]

angle_chan_idx = chan_map['theta']
ang_means = [c[0, angle_chan_idx, ...].mean() for c in crops_in_memory]
ang_stds = [c[0, angle_chan_idx, ...].std() for c in crops_in_memory]

# %%
