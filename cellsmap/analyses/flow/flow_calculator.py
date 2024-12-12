import pickle
import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as statm
import matplotlib.pyplot as plt
from skimage import registration as skreg
from matplotlib.backends.backend_agg import FigureCanvas
from pathlib import Path
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from bioio_base.types import PhysicalPixelSizes

from cellsmap.util import io

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

    def __init__(self, dataset, debug=False):
        self.dataset = dataset
        if debug:
            self.set_debug_mode_on()

    def initialize(self, channel: list[str], delta_t: int=1, load_from_file=None, ncores: int=20):
        """
        channel: list[str]
            A list of the names of the channels to be loaded from the dataset.
        delta_t: int
            The difference between the start and end timeframes when calculating the flow field vectors.
        """
        self.channel = channel
        self.channel_index = io.get_channel_index(self.dataset, channel)
        assert len(self.channel) == 1 and len(self.channel) == 1, f"Only one channel is implemented for flow calculation. Channels provided were {self.channel} corresponding to channel indices {self.channel_index}."
        self.delta_t = delta_t
        self.load_dataset()
        # self.dims = dict(zip('TCYX', self.data.shape))
        # self.dim_order = dict(zip('TCYX', range(len(self.data.shape))))
        if load_from_file:
            self.load_flow_from_file(load_from_file)
        self.set_number_of_cores(ncores)

    def load_dataset(self):
        self.data = io.load_dataset(self.dataset, channels=self.channel, level=2)
        # self.df = load_data.load_dataset(self.dataset)
        # self.reader = load_data.get_dataset_original_file_reader(self.dataset)
        # self.reader = AICSImage("/allen/aics/microscopy/Data/RnD_Sandbox/3500006247_20240227_Deliverable_ZSD1/3500006247_20240227_20X_Timelapse-03.czi")
        # self.reader.set_scene("P22-C7")

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
        duration = io.get_dataset_duration_in_frames(self.dataset)
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
        raw0 = self.data[time, 0]
        raw1 = self.data[time+self.delta_t, 0]
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
        px_res = PhysicalPixelSizes(*[1, *[io.get_xy_pixel_size_in_um(self.dataset)] * len(img_dim_order)])
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
    def summarize_flow_features(vx, vy, axis=None):
        magnitudes = FlowCalculator.compute_angles(vx, vy)
        angles = FlowCalculator.compute_angles(vx, vy)
        features = [magnitudes, angles]
        feature_means = [np.mean(ft, axis=axis) for ft in features]
        feature_stds = [np.std(magnitudes, axis=axis) for ft in features]
        return feature_means, feature_stds

    @staticmethod
    def compute_flow(image0, image1, radius=radius, display=False, return_map=False):

        vy, vx = skreg.optical_flow_ilk(image0, image1, radius=radius)

        if return_map or display:

            plot = FlowCalculator.make_vector_field_map(image1, vx, vy, display=display)

            return (vx, vy), plot

        return (vx, vy)

    @staticmethod
    def make_vector_field_map(image, vx, vy, resolution=20, display=True, return_map=False, hide_axes=True):

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
        ax.quiver(x, y, vx_, vy_, n_, units='dots', angles='xy', scale_units='xy', width=4, cmap="inferno")
        plt.tight_layout()

        canvas.draw()
        plot = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

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
        # pd.DataFrame(vector_data).to_csv(out_path, sep='\t')
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

def compute_and_save_flow_field(out_dir, dataset_name, delta_t=1, executor=None, ncores=1, debug=False):
    print(f'Analyzing dataset: {dataset_name}')

    # Initialize the flow field calculator
    flowc = FlowCalculator(dataset=dataset_name, debug=debug)
    channel_name = [chan for chan in io.get_available_channels(dataset_name) if chan in ('CDH5_Tubulin', 'CDH5')]
    print('initializing...')
    flowc.initialize(channel=channel_name, delta_t=delta_t, ncores=ncores)

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

def get_random_roi(image_shape: np.ndarray.shape, roi_shape: tuple[int, ...], num_rois: int=1, random_seed: int=None) -> tuple[slice, ...]:
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
    image_shape = np.asarray(image_shape)
    roi_shape = np.array(roi_shape)
    rand_gen = np.random.default_rng(seed=random_seed)
    rand_coord = np.asarray([np.array(rand_gen.integers(0, random_space_limit + 1, size=num_rois), ndmin=1) for random_space_limit in image_shape - roi_shape])
    roi = [tuple([slice(start, stop) for start, stop in zip(*coord_pair)]) for coord_pair in zip(rand_coord.T, rand_coord.T + np.array(roi_shape, ndmin=rand_coord.ndim))]
    return roi

# Check that the last image is the same when loaded as it was when being saved:
# import numpy as np
# test_path = Path(r'C:\Users\serge.parent\OneDrive - Allen Institute\Desktop\projects\holistic_state\cellsmap\cellsmap\results\flow_calculator\20241016_20X_vector_field.ome.tiff')
# test_img = BioImage(test_path)
# img = test_img.get_image_data("TCYX")
# np.all(flowc.vx == img[:,0,...])