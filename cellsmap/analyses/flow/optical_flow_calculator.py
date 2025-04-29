import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec as gs
from skimage import registration as skreg
from skimage.draw import circle_perimeter
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from sklearn.decomposition import PCA
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pathlib import Path
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from bioio_base.types import PhysicalPixelSizes
from cellsmap.util import dataset_io
from matplotlib.projections.polar import PolarAxes

def expand_crop_region(crop_region, padding):
    crop_region = tuple(crop if isinstance(crop, slice) else slice(*crop) for crop in crop_region)
    crop_region = tuple((max(0, crop.start - padding if crop.start else 0) or None, crop.stop + padding if crop.stop else None) for crop in crop_region)
    return tuple((slice(*crop)) for crop in crop_region)

class FlowCalculator():
    """
    Class to implement functionalities related to flow field. We want
    to use the flow field as a proxy for the collective motion of
    endothelial cells. Velocities in here are computed in units of
    pixels/timeframe.

    Example using multiprocessing:
    --------
    from cellsmap.analyses.flow import flow_calculator
    import concurrent.futures
    from pathlib import Path

    # Define a location to save the output to:
    out_dir = Path(__file__).resolve().parent / 'results'
    # Define how many cores you want to use:
    ncores = 20
    # Define how many timeframes into the future you want to use to calculate a vector of the flow field:
    delta_t = 1 # (calculate the vector for the current timeframe to the next one)

    # Set up multiprocessing:
    with concurrent.futures.ProcessPoolExecutor(ncores) as executor:
        # The line below saves the flow field vectors as an image and also returns the path to the saved image:
        out_path = flow_calculator.compute_and_save_flow_field(out_dir, dataset_name, delta_t, executor, ncores)

    Example using single processing:
    --------
    from cellsmap.analyses.flow import flow_calculator
    from pathlib import Path

    # Define a location to save the output to:
    out_dir = Path(__file__).resolve().parent / 'results'
    # Define how many timeframes into the future you want to use to calculate a vector of the flow field:
    delta_t = 1 # (calculate the vector for the current timeframe to the next one)

    # The line below saves the flow field vectors as an image and also returns the path to the saved image:
    out_path = flow_calculator.compute_and_save_flow_field(out_dir, dataset_name, delta_t)
    """
    debug = False # Run in debug mode or not
    ncores = 1 # Number of cores to be used in the calculation
    radius = 30 # Radius of the neighborhood after downscaling

    def __init__(self, dataset, position_or_scene_index, debug=False):
        self.dataset = dataset
        self.position = position_or_scene_index
        if debug:
            self.set_debug_mode_on()

    def initialize(self, channel: list[str], delta_t: int=1, level: int=1, load_from_file=None, ncores: int=20):
        """
        channel: list[str]
            A list of the names of the channels to be loaded from the dataset.
        delta_t: int
            The difference between the start and end timeframes when calculating the flow field vectors.
        """
        self.channel = channel
        zarr_name = dataset_io.get_zarr_name(self.dataset, self.position)
        self.channel_index = dataset_io.get_channel_index(self.dataset, channel)[zarr_name]
        assert len(self.channel) == 1 and len(self.channel) == 1, f"Only one channel is implemented for flow calculation. Channels provided were {self.channel} corresponding to channel indices {self.channel_index}."
        self.delta_t = delta_t
        self.load_dataset(level)
        # self.dims = dict(zip('TCYX', self.data.shape))
        # self.dim_order = dict(zip('TCYX', range(len(self.data.shape))))
        if load_from_file:
            self.load_flow_from_file(load_from_file)
        self.set_number_of_cores(ncores)

    def load_dataset(self, level):
        # self.data = dataset_io.load_dataset(self.dataset, channels=self.channel, level=2)
        img_data = dataset_io.load_dataset_position_as_dask_array(self.dataset, self.position, self.channel, level=level) # level=2 not present in ZARRs anymore
        self.data = img_data.max(axis=dataset_io.get_dim_map('TCZYX')['Z'])

    def load_flow_from_file(self, fname):
        with open(fname, "rb") as fpk:
            result = pickle.load(fpk)
        assert result["dataset"] == self.dataset
        self.vx = np.array(result["vx"]).reshape(result["shape"])
        self.vy = np.array(result["vy"]).reshape(result["shape"])

    def set_debug_mode_on(self):
        self.debug = True

    def set_flow_radius(self, radius):
        self.radius = radius

    def set_number_of_cores(self, ncores):
        self.ncores = ncores

    def compute_flow_field(self, executor=None, save=None):
        step = 1
        duration = dataset_io.get_dataset_duration_in_frames(self.dataset)
        if self.debug:
            step = int(duration/10)
            print(f"Running debug mode. Using step of {step} frames.")

        tps = np.arange(self.data.shape[0]-self.delta_t)[::step]

        # NOTE trying to instantiate executor in this function results in the script hanging
        # after completing the first dataset in the for-loop in flow_features.py;
        # instantiating executor in the main function and passing it as an argument fixes this issue
        if not executor:
            print('starting single processing')
            flow = list(tqdm(map(self.compute_flow_at_timepoint, tps), total=len(tps)))
            print('done single processing')
        else:
            print('starting multiprocessing')
            flow = list(tqdm(executor.map(self.compute_flow_at_timepoint, tps), total=len(tps)))
            print('done multiprocessing')

        self.im = np.array([im for (im, _, _) in flow])
        self.vx = np.array([fx for (_, fx, _) in flow])
        self.vy = np.array([fy for (_, _ ,fy) in flow])
        self.tps = tps

    def compute_flow_at_timepoint(self, time):
        # note that the compute_flow will return an empty array for vx and vy
        # the image intensity is not bright enough, therefore we rescale it
        raw0 = rescale_intensity(self.data[time, 0], out_range=self.data.dtype.type)
        raw1 = rescale_intensity(self.data[time+self.delta_t, 0], out_range=self.data.dtype.type)
        (vx, vy) = self.compute_flow(raw0, raw1, radius=self.radius)
        return (raw1, vx, vy)

    def save_vector_field_as_img(self, out_dir):
        Path.mkdir(out_dir, exist_ok=True, parents=True)
        out_path = out_dir / f"{self.dataset}_vector_field.ome.tiff"
        vector_data = np.asarray([np.asarray(FlowCalculator.compute_flow_features_as_img(im, vx, vy, keepdims=True)) for im, vx, vy in zip(self.im, self.vx, self.vy)])
        # keep only the vx and vy components and the norms when saving as an
        # image since the t, x, and y are trivial to determine from the image
        vector_data = vector_data[:, 2:6, ...]

        image_name = self.dataset
        ch_colors = [(0,255,255), (255,0,255), (255,255,0), (255,255,255)]
        ch_names = ['vx', 'vy', 'norm', 'theta']
        img_dim_order = 'YX'
        px_res = PhysicalPixelSizes(*[1, *[dataset_io.get_xy_pixel_size_in_um(self.dataset)] * len(img_dim_order)])
        dim_order_out = 'TCYX'

        OmeTiffWriter.save(vector_data,
                           out_path,
                           physical_pixel_sizes=px_res,
                           dim_order=dim_order_out,
                           image_name=image_name,
                           channel_names=ch_names,
                           channel_colors=ch_colors)
        return out_path

    @staticmethod
    def compute_angles(vx, vy):
        """
        Angles are in the correct quadrants (see docs for numpy.arctan2).
        """
        return np.arctan2(vy, vx)

    @staticmethod
    def compute_magnitudes(vx, vy):
        return np.linalg.norm([vx, vy], axis=0)

    @staticmethod
    def get_features_from_vector_field_image(vector_field_image: np.ndarray, chan_map: dict={'vx':0, 'vy':1, 'norm':2, 'theta':3}):

        features_vx = vector_field_image[:,chan_map['vx'],...].squeeze()
        features_vy = vector_field_image[:,chan_map['vy'],...].squeeze()
        features_mags = vector_field_image[:,chan_map['norm'],...].squeeze()
        features_vx_mean, features_vx_std = features_vx.mean(), features_vx.std()
        features_vy_mean, features_vy_std = features_vy.mean(), features_vy.std()
        features_mags_mean, features_mags_std = features_mags.mean(), features_mags.std()

        angle_of_mean_vector = FlowCalculator.compute_angles(features_vx_mean, features_vy_mean)
        mag_of_mean_vector = FlowCalculator.compute_magnitudes(features_vx_mean, features_vy_mean)

        divergence = discrete_divergence_like(vector_field_image[:,chan_map['vx'],...].squeeze(), vector_field_image[:,chan_map['vy'],...].squeeze())
        curl = discrete_divergence_like(vector_field_image[:,chan_map['vx'],...].squeeze(), vector_field_image[:,chan_map['vy'],...].squeeze())
        features_diverg_mean, feature_diverg_std = divergence.mean(), divergence.std()
        features_curl_mean, features_curl_std = curl.mean(), curl.std()

        features = {'angle_of_mean_vector': angle_of_mean_vector, 'magnitude_of_mean_vector': mag_of_mean_vector,
                    'divergence_mean': features_diverg_mean, 'divergence_std': feature_diverg_std,
                    'curl_mean': features_curl_mean, 'curl_std': features_curl_std,
                    'vector_magnitudes_mean': features_mags_mean, 'vector_magnitudes_std': features_mags_std,
                    'vector_x_mean': features_vx_mean, 'vector_x_std': features_vx_std,
                    'vector_y_mean': features_vy_mean, 'vector_y_std': features_vy_std,
                    }
        return features

    @staticmethod
    def compute_flow(image0, image1, radius=radius, display=False, return_map=False):

        vy, vx = skreg.optical_flow_ilk(image0, image1, radius=radius)

        if return_map or display:

            plot = FlowCalculator.make_vector_field_map(image1, vx, vy, display=display)

            return (vx, vy), plot

        return (vx, vy)

    @staticmethod
    def make_vector_field_map(image, vx, vy, resolution=20, cmap_norm: tuple[float, float] | None = None, cmap: str="inferno", display=True, return_map=False, hide_axes=True):

        norm = np.sqrt(vx ** 2 + vy ** 2)

        nl, nc = image.shape
        step = max(nl//resolution, nc//resolution)

        y, x = np.mgrid[:nl:step, :nc:step]
        vx_ = vx[::step, ::step]
        vy_ = vy[::step, ::step]
        n_ = norm[::step, ::step]

        fig, ax = plt.subplots(1, 1, dpi=150)
        canvas = FigureCanvas(fig)
        ax.imshow(image, cmap='gray')
        ax.set_axis_off() if hide_axes else None
        ax.quiver(x, y, vx_, vy_, n_, units='dots', angles='xy', scale_units='xy', width=4, cmap=cmap, norm=plt.Normalize(*cmap_norm if cmap_norm else (n_.min(), n_.max())))
        plt.tight_layout()

        canvas.draw()
        plot = np.frombuffer(canvas.buffer_rgba(), dtype="uint8").copy()
        # the .copy() at the end above is required so that the plot is
        # not cleared when calling plt.show(); this way you can both show
        # the plot with display=True and return it with return_map=True
        plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        if display:
            plt.show()

        plt.close("all")

        if return_map:
            return plot

    def save_vector_field_as_pickle(self, out_dir):
        out_dir = out_dir / Path(__file__).stem
        Path.mkdir(out_dir, exist_ok=True, parents=True)
        out_path = out_dir / f"{self.dataset}_vector_field.flow"
        vector_data = dict(zip(('timeframe', 'x', 'y', 'vx', 'vy', 'norm', 'shape'), zip(*[(tp, *FlowCalculator.get_vector_field_as_img(im, vx, vy), vx.shape) for tp, im, vx, vy in zip(self.tps, self.im, self.vx, self.vy)])))

        with open(f"{out_path}", "wb") as fpk:
            pickle.dump(vector_data, fpk)

    @staticmethod
    def compute_flow_features_as_img(image, vx, vy, keepdims=True):
        norm = FlowCalculator.compute_magnitudes(vx, vy)
        theta = FlowCalculator.compute_angles(vx, vy)
        nl, nc = image.shape
        y, x = np.mgrid[:nl, :nc]
        return [x, y, vx, vy, norm, theta] if keepdims else [np.ravel(arr) for arr in (x, y, vx, vy, norm, theta)]

    def save_flow_field(self, path):
        result = {
            "dataset": self.dataset,
            "vx": self.vx.flatten().tolist(),
            "vy": self.vy.flatten().tolist(),
            "shape": np.array(self.vx.shape).tolist()
        }
        with open(f"{path}/{self.dataset}.flow", "wb") as fpk:
            pickle.dump(result, fpk)

    def run_flow_field_analysis(self):
        self.compute_flow_field()
        self.calculate_flow_velocity()
        self.calculate_instantaneous_velocity()

def compute_and_save_flow_field(out_dir, dataset_name, position, delta_t=1, level=1, executor=None, ncores=1, debug=False):
    print(f'Analyzing dataset: {dataset_name}')

    # Initialize the flow field calculator
    flowc = FlowCalculator(dataset=dataset_name, position_or_scene_index=position, debug=debug)
    channel_name = ['EGFP']
    print('initializing...')
    flowc.initialize(channel=channel_name, delta_t=delta_t, level=level, ncores=ncores)

    # Compute the flow field
    print('computing flow field...')
    flowc.compute_flow_field(executor=executor)

    # Save the flow field results
    print('saving results...')
    out_path = flowc.save_vector_field_as_img(out_dir)
    print(f'Vector field image saved to {out_path}')

    return out_path

def get_vector_field_image_paths(out_dir):
    out_dir = Path(out_dir)
    df = pd.read_csv(out_dir / 'vector_field_image_paths.csv')
    return dict(zip(df['dataset_name'], df['vector_field_image_paths']))

def load_vector_field_img(img_dir, dataset_name):
    # Get the paths to the vector field images
    image_paths = get_vector_field_image_paths(img_dir)
    im_path = image_paths[dataset_name]

    # Lazy-load the vector field images
    img = BioImage(im_path)
    vector_field = img.get_image_dask_data("TCYX")

    return vector_field

def get_random_roi(image_shape: tuple[int, ...], roi_shape: tuple[int, ...], num_rois: int=1, random_seed: int | None = None):
    """
    Returns a random region of interest (roi) within an image.

    Parameters
    ----------
    image_shape : np.ndarray.shape
        Shape of the image from which to extract the roi.
    roi_shape : tuple[int, ...]
        Shape of the roi to extract from the image.
    num_rois : int, optional
        Number of rois to extract from the image. Default is 1.
    random_seed : int, optional
        Seed for the random number generator. Default is None.

    Returns
    -------
    roi: list[tuple[slice, ...]]
        A list of tuples of slices that can be used in the numpy array
        that image_shape is based on to get a slice of that array.

    Example usage:
    Given some ndarray 'image' with shape (12, 2048, 2048) representing
    the size of the dimensions 'TYX' in that order:

    >>> image_shape = image.shape
    >>> roi_shape = (3, 64, 64)
    >>> rand_roi = get_random_roi(image_shape, roi_shape, num_rois=1, random_seed=42)
    >>> rand_crop = [image[roi] for roi in rand_roi]
    >>> print(rand_crop.shape)
    (3, 64, 64)

    """
    assert isinstance(num_rois, int) and num_rois > 0, f"num_rois must be a positive integer, not {num_rois}."
    assert len(image_shape) == len(roi_shape), f"image_shape and roi_shape must have the same number of dimensions, but image_shape has {len(image_shape)} dimensions and roi_shape has {len(roi_shape)} dimensions."
    image_shape_array = np.asarray(image_shape)
    roi_shape_array = np.array(roi_shape)
    rand_gen = np.random.default_rng(seed=random_seed)
    rand_coord = np.asarray([np.array(rand_gen.integers(0, random_space_limit + 1, size=num_rois), ndmin=1) for random_space_limit in image_shape_array - roi_shape_array])
    roi = [tuple([slice(start, stop) for start, stop in zip(*coord_pair)]) for coord_pair in zip(rand_coord.T, rand_coord.T + np.array(roi_shape_array, ndmin=rand_coord.ndim))]
    return roi



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

def get_point_closest_to_reference_point(points: np.ndarray, reference_point: tuple[float, float]):
    distances = np.linalg.norm(points - np.array(reference_point), axis=1)
    assert len(distances) == len(points)
    return points[np.argmin(distances)], np.argmin(distances)

def get_quadrant_means(points: np.ndarray, origin: tuple[float, float]=(0,0)) -> list[np.ndarray]:
    top_right = points[(points[:,0] > origin[0]) & (points[:,1] > origin[1])]
    top_left = points[(points[:,0] < origin[0]) & (points[:,1] > origin[1])]
    bottom_left = points[(points[:,0] < origin[0]) & (points[:,1] < origin[1])]
    bottom_right = points[(points[:,0] > origin[0]) & (points[:,1] < origin[1])]

    quadrant_means = [quad_points.mean(axis=0) for quad_points in [top_right, top_left, bottom_left, bottom_right]]

    return quadrant_means



def generate_synthetic_data():
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

    return synth_img

def compute_synthetic_image_flow_vectors_and_summarize(synth_img: np.ndarray, delta_t: int=1, radius: int=30):
    flow_graphs = []
    for i in range(0, len(synth_img)-1, delta_t):
        print(f'Computing flow for frame {i} to {i+delta_t}...')
        flow = FlowCalculator.compute_flow(synth_img[i].squeeze(), synth_img[i+delta_t].squeeze(), radius=radius)
        flow_graphs.append(flow)


    # compute angles and magnitudes of flow vectors on synthetic data
    vx, vy = flow_graphs[0]
    mean_angle_deg = np.rad2deg(FlowCalculator.compute_angles(vx.mean(), vy.mean()))
    mean_mag = FlowCalculator.compute_magnitudes(vx.mean(), vy.mean())

    print(f'Flow angle mean: {mean_angle_deg} \nFlow magnitude mean: {mean_mag}')

    return flow_graphs, vx, vy, mean_angle_deg, mean_mag

def generate_validation_plot(out_dir: Path, raw_image, vector_image, features_and_pcs: pd.DataFrame, quadrants_origin, example_points: dict, vector_field_channel_map: dict={'vx':0, 'vy':1, 'norm':2, 'theta':3}):
    rois = []
    for i, example_pt in example_points.items():
        roi = [slice(start, stop) for start, stop in zip(example_pt['record'][['T', 'start_c', 'start_y', 'start_x']].values.ravel(),
                                                         (example_pt['record'][['T', 'start_c', 'start_y', 'start_x']].values + example_pt['record'][['delta_t', 'size_c', 'size_y', 'size_x']].values).ravel())]
        rois.append(tuple(roi))

    # generate a flow fields with the colormap normalized to the vector magnitudes of the
    # crops being used as examples
    # find the extreme values of the example crops to use as the colormap normalization:
    cmap = 'summer'
    cmap_norm_all = np.asarray([(vector_image[roi][:,vector_field_channel_map['norm'],...].min(), vector_image[roi][:,vector_field_channel_map['norm'],...].max()) for roi in rois])
    cmap_norm_min_max = cmap_norm_all.min(), cmap_norm_all.max()
    num_examples = len(example_points)

    fig = plt.figure(figsize=((1+num_examples) * 3, 6))
    axs = gs.GridSpec(ncols=(1+num_examples), nrows=2, figure=fig, hspace=0.6)
    ax1 = fig.add_subplot(axs[0, 0])
    ax1.scatter(features_and_pcs[0], features_and_pcs[1], marker='.', c='grey', alpha=0.7)
    ax1.axvline(quadrants_origin[0], color='k', linestyle='--', alpha=0.5)
    ax1.axhline(quadrants_origin[1], color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('PC1 (normalized)')
    ax1.set_ylabel('PC2 (normalized)')
    ax1.set_title('PC1 vs PC2 for crops')

    ax2 = fig.add_subplot(axs[1, 0])
    ax2.scatter(features_and_pcs['vector_magnitudes_std'], features_and_pcs['divergence_std'], marker='.', c='grey', alpha=0.7)
    ax2.set_xlabel('Magnitude St. Dev.')
    ax2.set_ylabel('Divergence St. Dev.')
    ax2.set_title('Divergence std vs.\n Magnitude std for crops')

    for i, example_pt in example_points.items():
        roi = rois[i]
        quad_color, quad_record, crop_id = example_pt['color'], example_pt['record'], example_pt['record']['crop_id'].astype(int)
        feats_and_pcs_at_roi = features_and_pcs.query('crop_id == @crop_id')
        dataset_name = feats_and_pcs_at_roi['dataset_name'].iloc[0]
        cropped_raw_img, cropped_vector_img = raw_image[roi].compute(), vector_image[roi].compute()
        flow_graph_at_roi = get_trimmed_vector_field_map(cropped_raw_img.squeeze(), 
                                                         cropped_vector_img[:,feats_and_pcs_at_roi['vx_chan_index'], ...].squeeze(), 
                                                         cropped_vector_img[:,feats_and_pcs_at_roi['vy_chan_index'], ...].squeeze(), 
                                                         resolution=15, 
                                                         cmap_norm=cmap_norm_min_max, 
                                                         cmap=cmap, 
                                                         display=False, 
                                                         return_map=True)
        ang = cropped_vector_img[:,feats_and_pcs_at_roi['theta_chan_index'],...].squeeze()
        roi_as_title = list(zip(*[(slc.start, slc.stop) for slc in roi]))

        ax1.scatter(quad_record[0], quad_record[1], marker='.', color=quad_color, zorder=10)

        if example_pt['quadrant_mean'] is not None:
            ax1.scatter(*example_pt['quadrant_mean'], marker='x', color=quad_color, alpha=0.5)

        ax2.scatter(quad_record['vector_magnitudes_std'], quad_record['divergence_std'], marker='.', color=quad_color, zorder=10)

        with plt.rc_context({key: quad_color for key in ['axes.edgecolor', 'xtick.color', 'ytick.color', 'xtick.labelcolor', 'ytick.labelcolor']}):
            ax3 = fig.add_subplot(axs[0, i+1])
            ax3.imshow(flow_graph_at_roi, cmap='gray')
            ax3.axis('off')
            ax3.set_title(f'Flow Field (crop {crop_id})', color=quad_color)
            ax3.text(x=0.5, y=-0.05, ha='center', va='top', transform=ax3.transAxes,
                    s=f'roi start: {roi_as_title[0]}\nroi stop:{roi_as_title[1]})')

            ax4 = fig.add_subplot(axs[1, i+1], projection='polar')
            assert isinstance(ax4, PolarAxes)
            ax4.hist(ang.ravel(), bins=72, color=quad_color, alpha=1)
            y_min, y_max = ax4.get_ylim()
            ax4.arrow(x=float(quad_record['angle_of_mean_vector']), y=0, dx=0, dy=0.9*y_max, head_width=0.1, head_length=0.15*y_max, length_includes_head=True, lw=1, ls='-', facecolor=quad_color, edgecolor='k', alpha=0.5)
            # create minor ticks on the polar plot
            [ax4.plot((theta, theta), (0.95 * y_max, y_max), c=quad_color, lw=0.5, zorder=0) for theta in np.linspace(0, 2*np.pi, 24+1)]
            ax4.set_ylim(y_min, y_max)
            ax4.set_theta_direction(-1)
            ax4.set_xlim(-np.pi, np.pi)
            ax4.yaxis.set_visible(False)
            ax4.set_title('Angle distribution', color=quad_color, y=1.15)

    plt.tight_layout()
    crop_ids_for_filename = '-'.join([str(example_pt['record']['crop_id']) for example_pt in example_points.values()])
    fig.savefig(out_dir / f'{dataset_name}_crop_{crop_ids_for_filename}.png')
    plt.close(fig)

def get_trimmed_vector_field_map(image, vx, vy, resolution=20, cmap_norm: tuple[float, float] | None=None, cmap: str="inferno", display=True, return_map=False, hide_axes=True):
    # get the vector field map:
    vecfield_map = FlowCalculator.make_vector_field_map(image.squeeze(), vx, vy, resolution=resolution, cmap_norm=cmap_norm, cmap=cmap, display=display, return_map=return_map, hide_axes=hide_axes)
    # keep anything that isn't white-space:
    keep_me_indices = np.where(~np.all(vecfield_map==255, axis=-1))
    i, j = keep_me_indices
    keep_me_slices = (slice(np.min(i), np.max(i)), slice(np.min(j), np.max(j)), slice(None))
    vecfield_map_trimmed = vecfield_map[keep_me_slices]

    return vecfield_map_trimmed

def discrete_divergence_like(vx, vy):
    vx_dx = np.gradient(vx, axis=1)
    vy_dy = np.gradient(vy, axis=0)
    return vx_dx + vy_dy

def discrete_curl_like(vx, vy):
    vy_dx = np.gradient(vy, axis=1)
    vx_dy = np.gradient(vx, axis=0)
    return vy_dx - vx_dy

# vector_field_examples:
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

def get_divergence_curl_example(vfield='solenoidal', show_vector_field=False):
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


# The following checks that the last image is the same when loaded as it was when being saved:
# import numpy as np
# test_path = Path(r'C:\Users\serge.parent\OneDrive - Allen Institute\Desktop\projects\holistic_state\cellsmap\cellsmap\results\flow_calculator\20241016_20X_vector_field.ome.tiff')
# test_img = BioImage(test_path)
# img = test_img.get_image_data("TCYX")
# np.all(flowc.vx == img[:,0,...])