import logging
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tqdm
from bioio import BioImage
from cyto_dl.utils.arg_checking import get_dtype
from monai.data import MetaTensor, SmartCacheDataset
from monai.transforms import Transform
from numpy.typing import DTypeLike

from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_annotated_positions,
    get_annotated_timepoints_for_position,
    get_available_zarr_files,
    get_position_integer_from_zarr_file_path,
)
from endo_pipeline.library.process.z_stack_selection import get_plane_indices
from endo_pipeline.settings.image_data import LOG_EPSILON

logger = logging.getLogger(__name__)

MIN_Z_BOUND = 0
MAX_Z_BOUND = 24


class LogImaged(Transform):
    """
    Apply logarithmic transformation to image data in a dictionary.

    This transform takes an input dictionary containing image data under a specified key,
    applies a logarithmic transformation to the image data, and stores the transformed
    image back in the dictionary under a specified output key. The transformation is
    performed using the formula: `log_image = log(image + 1e-12)`.

    Parameters
    ----------
    keys : str
        Key in the input dictionary where the original image data is stored.
    """

    def __init__(self, keys: str = "image") -> None:
        """
        Initialize the LogImage transform.

        Parameters
        ----------
        keys : str
            Key in the input dictionary where the original image data is stored.
        """
        super().__init__()
        self.keys = keys

    def __call__(self, data: dict) -> dict:
        """
        Apply logarithmic transformation to the image data.

        Parameters
        ----------
        data : dict
            Input dictionary containing image data under `keys`.

        Returns
        -------
        dict
            Output dictionary with transformed image data under `keys`, overwriting data in place.
        """
        if self.keys not in data:
            logger.error("Input key '%s' not found in data dictionary.", self.keys)
            raise KeyError(f"Input key '{self.keys}' not found in data dictionary.")

        img = data[self.keys]

        # Apply logarithmic transformation
        log_img = np.log(img + LOG_EPSILON)

        # convert to MetaTensor to preserve metadata if available
        log_image_tensor = MetaTensor(log_img, meta=getattr(img, "meta", None))

        # Store transformed image in output dictionary
        data[self.keys] = log_image_tensor

        return data


class BioIOImageLoaderd(Transform):
    """
    Enumerates scenes and timepoints for dictionary with format.

    .. code-block:: python
        {
            path_key: path,
            channel_key: channel,
            scene_key: scene,
            timepoint_key: timepoint
        }

    Differs from ``monai_bio_reader`` in that reading ``kwargs`` are passed in the dictionary,
    instead of being fixed at initialization. The filepath will be saved in the dictionary
    as ``filename_or_obj`` (with or without metadata depending on ``include_meta_in_filename``).
    """

    def __init__(
        self,
        path_key: str = "path",
        scene_key: str = "scene",
        resolution_key: str = "resolution",
        kwargs_keys: list[str] | None = None,
        out_key: str = "raw",
        allow_missing_keys: bool = False,
        dtype: np.dtype | DTypeLike = np.float16,
        dask_load: bool = True,
        include_meta_in_filename: bool = False,
    ) -> None:
        """
        Initialize the ``BioIOImageLoaderd`` transform.

        Parameters
        ----------
        path_key
            Key for the path to the image.
        scene_key
            Key for the scene number.
        resolution_key
            Key for the resolution level.
        kwargs_keys
            Keys for the kwargs to pass to BioImage.get_image_dask_data.
        out_key
            Key for the output image.
        allow_missing_keys
            Whether to allow missing keys in the data dictionary.
        dtype
            Data type to cast the image to after loading.
        dask_load
            Load images using Dask if True, else load them directly into memory.
        include_meta_in_filename
            Include metadata in the filename of the output image if True, else use only the path.
        """
        super().__init__()
        self.path_key = path_key
        if kwargs_keys is None:
            kwargs_keys = ["dimension_order_out", "C", "T", "Z"]
        self.kwargs_keys = kwargs_keys
        self.allow_missing_keys = allow_missing_keys
        self.out_key = out_key
        self.resolution_key = resolution_key
        self.scene_key = scene_key
        self.dtype = get_dtype(dtype)
        self.dask_load = dask_load
        self.include_meta_in_filename = include_meta_in_filename

    def split_args(self, arg: str) -> list[int] | str:
        """Split arguments that are comma-separated strings into lists of integers."""
        if isinstance(arg, str) and "," in arg:
            return list(map(int, arg.split(",")))
        return arg

    def _get_filename(self, path: str, kwargs: dict) -> str:
        if self.include_meta_in_filename:
            logger.debug("Including metadata in filename")
            path = path.split(".")[0] + "_" + "_".join([f"{k}_{v}" for k, v in kwargs.items()])
        # remove illegal characters from filename
        path = re.sub(r'[<>:"|?*]', "", path)
        logger.debug("Generated filename: [ %s ]", path)
        return path

    def __call__(self, data: dict) -> dict:
        """
        Load image data from the path specified in the data dictionary
        using the arguments specified in the data dictionary.
        """
        # copying prevents the dataset from being modified inplace
        # important when using partially cached datasets so that the
        # memory use doesn't increase over time
        data = data.copy()
        if self.path_key not in data and not self.allow_missing_keys:
            logger.error("Missing key in data dictionary: [ %s ]", self.path_key)
            raise KeyError(f"Missing key {self.path_key} in data dictionary")
        path = data[self.path_key]
        logger.debug("Loading image from path: [ %s ]", path)
        img = BioImage(path)
        if self.scene_key in data:
            img.set_scene(data[self.scene_key])
        if self.resolution_key in data:
            logger.debug("Setting resolution level to: [ %s ]", data[self.resolution_key])
            img.set_resolution_level(data[self.resolution_key])
        kwargs = {k: self.split_args(data[k]) for k in self.kwargs_keys if k in data}
        logger.debug("Using kwargs for image loading: [ %s ]", kwargs)

        if self.dask_load:
            logger.debug("Loading image data using Dask")
            img_as_array = img.get_image_dask_data(**kwargs).compute()  # type: ignore[arg-type]
        else:
            logger.debug("Loading image data directly into memory")
            img_as_array = img.get_image_data(**kwargs)  # type: ignore[arg-type]
        logger.debug("Image data loaded with shape: [ %s ]", img_as_array.shape)
        logger.debug("Casting image data to dtype: [ %s ]", self.dtype)
        img_as_array = img_as_array.astype(self.dtype)
        if self.scene_key in data:
            kwargs["scene"] = data[self.scene_key]
        logger.debug("Updating kwargs with filename or object")
        kwargs.update({"filename_or_obj": self._get_filename(path, kwargs)})

        logger.debug("Adding image data to dictionary under key: [ %s ]", self.out_key)
        data[self.out_key] = MetaTensor(img_as_array, meta=kwargs)
        return data


class MultiDimImageDataset(SmartCacheDataset):
    """
    Dataset converting a `.csv` file listing multi dimensional (timelapse or
    multi-scene) files and some metadata into batches of metadata intended for the
    BioIOImageLoaderd class.
    """

    def __init__(
        self,
        dataframe_path: Path | str,
        img_path_column: str = "path",
        channel_column: str = "channel",
        spatial_dims: int = 3,
        scene_column: str = "scene",
        resolution_column: str = "resolution",
        time_start_column: str = "frame_start",
        time_stop_column: str = "frame_stop",
        time_step_column: str = "frame_step",
        timepoints_to_exclude_column: str = "exclude_frames",
        z_start_column: str = "z_start",
        z_stop_column: str = "z_stop",
        z_step_column: str = "z_step",
        extra_columns: Sequence[str] = [],
        transform: Callable | Sequence[Callable] | None = None,
        **cache_kwargs: Any,
    ) -> None:
        """
        Initialize a dataset that reads multi-dimensional images using metadata from a dataframe
        (loaded from a .parquet file) and prepares them for processing.

        **Multi-channel images**
        The ``channel_column`` parameter should be specified to indicate which channel(s)
        to extract from the image. To load multiple channels, the entries of this column
        should be a list of the channel indices (e.g. ``[0,1,2]``). Else, this
        column should contain a single channel index (e.g. ``0`` or ``1``).

        **Image spatial dimensions**
        The output image will be in the format ``CZYX`` or ``CYX`` depending on the
        ``spatial_dims`` parameter. This is to ensure compatibility with dictionary-based
        MONAI-style transforms. The ``spatial_dims`` parameter specifies the number of spatial
        dimensions in the output image, which can be either 2 (for ``YX``) or 3 (for ``ZYX``).

        **Multi-scene images**
        If the input images are multi-scene images, the ``scene_column`` parameter should be
        specified. This column should contain the names of the scenes to extract from the
        multi-scene image. If not specified, all scenes will be extracted. If multiple scenes
        are specified, the column entry should be a list (e.g. ``[scene1,scene2]``).

        **Multi-resolution images**
        If the there are multiple resolution level available for the input images, the
        ``resolution_column`` parameter should be specified. This column should contain the
        resolution level at which to load the image. If not specified, the resolution level
        is assumed to be 0 (full resolution).

        **Timelapse images**
        If there are multiple timepoints available for the input images, the ``time_start_column``,
        ``time_stop_column``, and ``time_step_column`` parameters should be specified. These columns
        should contain the start timepoint, stop timepoint, and step between timepoints (step
        defaults to 1) respectively. If not specified, all timepoints are extracted. To specify the
        last timepoint, you can use -1 in the ``time_stop_column``, which will be interpreted as the
        last timepoint available in the image. The timepoints are zero-indexed, so the first
        timepoint is 0.

        **Excluding timepoints**
        If you want to exclude specific timepoints from the timelapse image, you can specify
        the ``timepoints_to_exclude_column`` parameter. This column should contain a
        list of timepoints to exclude (e.g. ``[1,3,5]``).

        **Z slices *
        If the input images are 3D and you want to extract specific Z slices, the
        ``z_start_column``, ``z_stop_column``, and ``z_step_column`` parameters should be
        specified. These columns should contain the start Z slice, stop Z slice, and step between Z
        slices (step defaults to 1) respectively. If not specified, all Z slices are extracted.

        **Extra columns**
        The ``extra_columns`` parameter allows you to specify additional columns from the dataframe
        that you want to include in the output dictionary. These columns will be added to the
        output dictionary as-is if found, otherwise their values will be set to None.

        **Transforms**
        The ``transform`` parameter allows you to specify a list or composition of MONAI-style
        transforms to apply to the image metadata. The first transform in the list should be an
        instance of ``BioIOImageLoaderd``, which will load the image data from the path specified in
        the metadata dictionary. The subsequent transforms can be any MONAI-style transforms that
        operate on the metadata dictionary. If no transforms are specified, the dataset will
        default to an empty list, meaning no transforms will be applied.

        ** CacheDataset **
        The dataset is a subclass of ``monai.data.CacheDataset``, which means it can be used with
        MONAI's caching mechanism to speed up data loading and processing. The ``cache_kwargs``
        parameter allows you to specify additional keyword arguments to pass to the
        ``CacheDataset``. To skip the caching mechanism, set ``cache_num`` to 0.

        Parameters
        ----------
        dataframe_path
            Path to .parquet file containing metadata table for loading the images.
        img_path_column
            Column in metadata table that contains path to timelapse or multi-scene file
        channel_column
            Column in metadata table that contains which channel to extract from the file.
        spatial_dims
            Spatial dimension of output image.
        scene_column
            Column in metadata table that contains scenes to extract from a multi-scene file.
        resolution_column
            Column in metadata table that contains resolution to extract from multi-resolution file.
        time_start_column
            Column in metadata table specifying the first timepoint in timelapse image to load.
        time_stop_column
            Column in metadata table specifying the last timepoint in timelapse image to load.
        time_step_column
            Column in metadata table specifying step size between timepoints.
        timepoints_to_exclude_column
            Column in metadata table specifying timepoints to exclude from the timelapse image.
        z_start_column
            Column in metadata table specifying the lowest Z slice to extract.
        z_stop_column
            Column in metadata table specifying the highest Z slice to extract.
        z_step_column
            Column in metadata table specifying step between Z slices.
        extra_columns
            List of extra columns to include in the output dictionary.
        transform
            List (or ``Compose`` object) of Monai-style transforms to apply to the image metadata.
        cache_kwargs:
            Additional keyword arguments to pass to ``CacheDataset``.
        """
        df = pd.read_parquet(dataframe_path)

        # Get distributed training info
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Store dataset parameters
        self.img_path_column = img_path_column
        self.channel_column = channel_column
        self.scene_column = scene_column
        self.resolution_column = resolution_column
        self.time_start_column = time_start_column
        self.time_stop_column = time_stop_column
        self.time_step_column = time_step_column
        self.timepoints_to_exclude_column = timepoints_to_exclude_column
        self.z_start_column = z_start_column
        self.z_stop_column = z_stop_column
        self.z_step_column = z_step_column
        self.extra_columns = list(extra_columns)
        if spatial_dims not in (2, 3):
            raise ValueError(f"`spatial_dims` must be 2 or 3, got {spatial_dims}")
        self.spatial_dims = spatial_dims

        # First expand all data to get total sample count
        # Debug
        logger.info(f"[Rank {rank}] Expanding dataframe rows to individual samples...")
        data = self.get_per_file_args(df.reset_index(drop=True))

        # Now distribute samples evenly across ranks
        if world_size > 1:
            # Debug
            logger.info(f"[Rank {rank}] Total samples before distribution: {len(data)}")

            # Calculate balanced distribution
            total_samples = len(data)
            samples_per_rank = total_samples // world_size
            extra_samples = total_samples % world_size

            # Each rank gets either samples_per_rank or samples_per_rank + 1 samples
            start_idx = rank * samples_per_rank + min(rank, extra_samples)
            end_idx = start_idx + samples_per_rank + (1 if rank < extra_samples else 0)

            # Slice the data for this rank
            data = data[start_idx:end_idx]

            # Debug
            logger.info(
                f"[Rank {rank}] Samples assigned to this rank: {len(data)} (indices {start_idx}:{end_idx})"
            )

        # Debug:
        else:
            logger.info(f"Single process training with {len(data)} samples")

        if transform is None:
            transform = []

        super().__init__(data, transform, **cache_kwargs)

    def _get_scenes(self, row: dict, img: BioImage) -> tuple:
        """Get scenes from the row data."""
        scenes = row.get(self.scene_column, -1)
        if scenes != -1:
            if not isinstance(scenes, list | tuple):
                logger.error("Scenes should be a list or tuple, got type: [ %s ]", type(scenes))
                raise TypeError(f"Scenes should be a list or tuple, got type: {type(scenes)}")
            for scene in scenes:
                if scene not in img.scenes:
                    logger.error(
                        "Scene [ %s ] not found in image [ %s ]. Available scenes: [ %s ]",
                        scene,
                        row[self.img_path_column],
                        img.scenes,
                    )
                    raise ValueError(
                        f"For image {row[self.img_path_column]} unable to find scene `{scene}`."
                    )
        else:
            scenes = img.scenes
        return scenes

    def _get_timepoints(self, row: dict, img: BioImage) -> list:
        """Get timepoints from the row data."""
        start = row.get(self.time_start_column, 0)
        stop = row.get(self.time_stop_column, -1)
        # can use -1 to indicate the last timepoint
        if stop == -1:
            stop = img.dims.T - 1
        step = row.get(self.time_step_column, 1)
        timepoints = range(start, stop + 1, step)
        timepoints_as_list = list(timepoints)
        timepoints_to_exclude = row.get(self.timepoints_to_exclude_column, None)
        if timepoints_to_exclude is not None:
            logger.debug(
                "Excluding timepoints: [ %s ] from available timepoints: [ %s ]",
                timepoints_to_exclude,
                timepoints_as_list,
            )
            timepoints_as_list = list(set(timepoints) - set(timepoints_to_exclude))
        logger.debug("Loading image with timepoints: [ %s ]", timepoints_as_list)
        return sorted(timepoints_as_list)

    def _get_z_slices(self, row: dict, img: BioImage) -> list:
        """Get Z slices from the row data."""
        z_start = row.get(self.z_start_column, 0)
        z_stop = row.get(self.z_stop_column, -1)
        # can use -1 to indicate the last Z slice
        if z_stop == -1:
            z_stop = img.dims.Z - 1
        z_step = row.get(self.z_step_column, 1)
        z_slices = range(z_start, z_stop + 1, z_step)
        logger.debug("Loading image with Z slices: [ %s ]", list(z_slices))
        return list(z_slices)

    def _get_channel(self, row: dict) -> int | list[int]:
        """Get channel(s) from the row data."""
        channel = row[self.channel_column]
        if isinstance(channel, list | tuple):
            logger.debug("Loading image with channels: [ %s ]", channel)
        else:
            logger.debug("Loading image with channel: [ %s ]", channel)
        if isinstance(channel, np.ndarray):
            # convert numpy array to list
            # otherwise this results in UserWarnings
            # from monai.data.MetaTensor
            channel = channel.tolist()
        return channel

    def get_per_file_args(self, df: pd.DataFrame) -> list[dict]:
        """Get image loading arguments for each file in the dataframe."""
        img_data = []
        for row in tqdm.tqdm(df.itertuples()):
            row_data = []
            row_dict: dict = row._asdict()  # type: ignore[operator]
            img = BioImage(row_dict[self.img_path_column])
            scenes = self._get_scenes(row_dict, img)
            channel = self._get_channel(row_dict)
            for scene in scenes:
                img.set_scene(scene)
                timepoints = self._get_timepoints(row_dict, img)
                for timepoint in timepoints:
                    image_loading_args = {
                        "dimension_order_out": "C" + "ZYX"[-self.spatial_dims :],
                        "C": channel,
                        "scene": scene,
                        "T": timepoint,
                        "original_path": row_dict[self.img_path_column],
                        "resolution": row_dict.get(self.resolution_column, 0),
                    }
                    # only get Z slices if spatial_dims is 3
                    if self.spatial_dims == 3:
                        z_slices = self._get_z_slices(row_dict, img)
                        image_loading_args["Z"] = z_slices
                    # get and add extra columns if specified
                    extra_columns = {col: row_dict.get(col) for col in self.extra_columns}
                    extra_columns.update(image_loading_args)
                    row_data.append(extra_columns)
            img_data.extend(row_data)
        return img_data


def get_z_slice_bounds_per_position(
    dataset_config: DatasetConfig,
    z_slice_offsets: tuple[int, int] | None,
) -> dict[int, dict[str, int]]:
    """
    Parse dataset annotations to get lower and upper z-slice
    bounds per position for image loading and processing.

    **Z-stack offsets**

    The ``z_slice_offsets`` parameter allows for flexible control over the z-slice loading.
    If ``z_slice_offsets`` is provided, it limits the number of z-slices to load
    by slicing about a global center (annotated in the datset config). If it
    is ``None``, all z-slices are loaded from the raw brightfield images.

    Parameters
    ----------
    dataset_config
        Dataset configuration object.
    z_slice_offsets
        Lower and upper bounds for z-slicing.

    Returns
    -------
    :
        Dictionary with z-slice start and stop indices per position.
    """
    # get z-slice offsets per position if specified
    if z_slice_offsets is not None:
        logger.debug(
            "Using z-stack offsets: [ %s ]",
            z_slice_offsets,
        )
    else:
        # if no z-stack offsets are provided, pass in None
        # to the dataframe builder
        logger.debug("No z-stack offsets provided, using full range in Z.")

    # if z_slice_offsets is not None, get z-slice ranges
    # for each position in the dataset (i.e., zarr file)
    # else, fixed full range is 0 to 24
    available_zarr_files = get_available_zarr_files(dataset_config)
    z_slice_bounds_per_position = {}
    for zarr_file_path in available_zarr_files:
        # get position from zarr path as an integer (e.g., 'P0' -> 0)
        position_as_int = get_position_integer_from_zarr_file_path(zarr_file_path)
        # get z-slice indices for the given position
        if z_slice_offsets is not None:
            z_slices = get_plane_indices(
                dataset_config,
                position_as_int,
                lower_offset=z_slice_offsets[0],
                upper_offset=z_slice_offsets[1],
            )
        else:
            z_slices = [MIN_Z_BOUND, MAX_Z_BOUND]
        z_slice_bounds_per_position[position_as_int] = {
            "z_start": z_slices[0],
            "z_stop": z_slices[-1],
        }

    return z_slice_bounds_per_position


def get_include_positions(dataset_config: DatasetConfig) -> list[int]:
    """Get list of positions to include based on annotations."""
    exclude_positions = get_annotated_positions(dataset_config)
    only_include_positions = list(set(dataset_config.zarr_positions) - set(exclude_positions))
    return only_include_positions


def get_exclude_frames(
    dataset_config: DatasetConfig, exclude_cell_piling: bool = False
) -> dict[int, list[int]]:
    """Get dict of frames to exclude per position based on annotations."""
    # if exclude_cell_piling is True, then get all annotated timepoints
    # else, get timepoints for all annotations except CELL_PILING
    annotations = None  # default to all annotations
    if not exclude_cell_piling:
        annotations = [ann for ann in TimepointAnnotation if "PILING" not in ann.name]

    # parse dataset annotations to get timepoints to exclude per position
    exclude_frames = {
        pos: get_annotated_timepoints_for_position(dataset_config, pos, annotations=annotations)
        for pos in dataset_config.zarr_positions
    }

    return exclude_frames


def build_zarr_image_loading_dataframe(
    dataset_config: DatasetConfig,
    resolution_level: int = 1,
    channel: int | list[int] = 0,
    frame_start: int | None = None,
    frame_stop: int | None = None,
    frame_step: int | None = None,
    z_slice_bounds_per_position: dict[int, dict[str, int]] | None = None,
    only_include_positions: list[int] | None = None,
    exclude_frames: dict[int, list[int]] | None = None,
) -> pd.DataFrame:
    """Build a DataFrame with metadata for loading Zarr images as a ``MultiDimImageDataset``."""
    # generate csv with paths to zarr files for each position in the dataset
    available_zarr_files = get_available_zarr_files(dataset_config)
    zarr_file_paths = [str(zarr_file) for zarr_file in available_zarr_files]  # convert Path to str

    df = pd.DataFrame({"path": zarr_file_paths})
    df["resolution"] = resolution_level
    if isinstance(channel, int):
        df["channel"] = channel
    else:
        # need to make sure list is not split
        # across multiple rows
        df["channel"] = df["path"].apply(lambda x: channel)

    # add temporary column with position index for filtering
    df["position_index"] = df["path"].apply(lambda x: get_position_integer_from_zarr_file_path(x))

    # only load images for specified position indices
    if only_include_positions is not None:
        logger.debug(
            "Filtering Zarr files to only include positions: [ %s ]", only_include_positions
        )

        df = df[df["position_index"].isin(only_include_positions)]

    # if start and stop for loading timepoints are specified, add to dataframe
    if (frame_start is not None) and (frame_stop is not None):
        df["frame_start"] = frame_start
        df["frame_stop"] = frame_stop
    # frame step defaults in loader to 1, but can be specified
    if frame_step is not None:
        df["frame_step"] = frame_step

    # add column for excluding frames, if specified
    if exclude_frames is not None:
        # if position has no frames to exclude, set to None
        df["exclude_frames"] = df["position_index"].apply(lambda x: exclude_frames.get(x, None))

    # if start and stop for loading z slices are specified, add to dataframe
    if z_slice_bounds_per_position is not None:
        # get z info dict for each position index
        # unpack the start, stop, and step values from those dictionaries
        df["z_start"] = df["position_index"].apply(
            lambda x: z_slice_bounds_per_position.get(x, {}).get("z_start", 0)
        )
        df["z_stop"] = df["position_index"].apply(
            lambda x: z_slice_bounds_per_position.get(x, {}).get("z_stop", -1)
        )
        df["z_step"] = df["position_index"].apply(
            lambda x: z_slice_bounds_per_position.get(x, {}).get("z_step", 1)
        )

    # remove temporary column with position index
    df = df.drop(columns=["position_index"])

    return df
