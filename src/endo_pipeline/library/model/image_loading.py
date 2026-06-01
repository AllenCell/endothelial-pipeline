"""Methods and classes for loading images for model training and inference."""

import logging
import os
import re
import typing
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from cyto_dl.utils.arg_checking import get_dtype
from monai.data import MetaTensor, SmartCacheDataset
from monai.transforms import Transform
from numpy.typing import DTypeLike

if typing.TYPE_CHECKING:
    from bioio import BioImage
    from omegaconf import ListConfig

from endo_pipeline.configs import DatasetConfig, get_position_integer_from_zarr_file_path
from endo_pipeline.io import load_dataframe, load_image
from endo_pipeline.manifests import (
    build_dataframe_location_from_string,
    build_image_location_from_string,
    get_available_zarr_locations,
)
from endo_pipeline.settings.diffae_feature_dataframes import CytoDLLoadDataKeys
from endo_pipeline.settings.image_data import DIFFAE_ZARR_RESOLUTION_LEVEL, LOG_EPSILON, NUM_ZSLICES

logger = logging.getLogger(__name__)


class LogImaged(Transform):
    """Apply logarithmic transformation to image data in a dictionary.

    This transform takes an input dictionary containing image data under a specified key,
    applies a logarithmic transformation to the image data, and stores the transformed
    image back in the dictionary under a specified output key. The transformation is
    performed using the formula: `log_image = log(image + 1e-12)`.
    """

    def __init__(self, keys: "list | ListConfig | str" = "image") -> None:
        """Initialize the LogImage transform.

        Parameters
        ----------
        keys
            Key in the input dictionary where the original image data is stored.

        """
        super().__init__()
        self.keys = [keys] if isinstance(keys, str) else keys

    def __call__(self, data: dict) -> dict:
        """Apply logarithmic transformation to the image data.

        Parameters
        ----------
        data
            Input dictionary containing image data under `keys`.

        Returns
        -------
        :
            Output dictionary with transformed image data under
            `keys`, overwriting data in place.

        """
        for key in self.keys:
            if key not in data:
                logger.error("Input key '%s' not found in data dictionary.", key)
                raise KeyError(f"Input key '{key}' not found in data dictionary.")

            img = data[key]

            # Apply logarithmic transformation
            log_img = np.log(img + LOG_EPSILON)

            # Convert to MetaTensor to preserve metadata if available
            log_image_tensor = MetaTensor(log_img, meta=getattr(img, "meta", None))

            # Store transformed image in output dictionary
            data[key] = log_image_tensor

        return data


class BioIOImageLoaderd(Transform):
    """Enumerates scenes and timepoints for dictionary with format.

    .. code-block:: python

        {
            path_key: path,
            channel_key: channel,
            scene_key: scene,
            timepoint_key: timepoint
        }

    Differs from `monai_bio_reader` in that reading `kwargs` are passed in the dictionary,
    instead of being fixed at initialization. The filepath will be saved in the dictionary
    as `filename_or_obj` (with or without metadata depending on `include_meta_in_filename`).
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
        """Initialize the `BioIOImageLoaderd` transform.

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
        """Generate a filename for the output image based on the input path and kwargs."""
        if self.include_meta_in_filename:
            logger.debug("Including metadata in filename")
            path = path.split(".")[0] + "_" + "_".join([f"{k}_{v}" for k, v in kwargs.items()])
        # remove illegal characters from filename
        path = re.sub(r'[<>:"|?*]', "", path)
        logger.debug("Generated filename: [ %s ]", path)
        return path

    def __call__(self, data: dict) -> dict:
        """Load image data as specified in the input dictionary.

        Parameters
        ----------
        data
            Input dictionary containing keys for the image path, scene,
            resolution, and any additional kwargs for loading.

        Returns
        -------
        :
            Output dictionary with loaded image data under `out_key` and
            metadata in the filename if `include_meta_in_filename` is True.

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

        # Convert "path" (which may be local path or an S3 URI) into a location
        # object so we can use built-in image loading.
        img_loc = build_image_location_from_string(path)
        img = load_image(img_loc, read=False)

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
    """Converts a `.csv` file of image paths and metadata into a batches of metadata for loading.

    Intended as input to the BioIOImageLoaderd transform, which will read the
    images based on the metadata in the dictionary
    """

    def __init__(
        self,
        dataframe_path: str,
        img_path_column: str = CytoDLLoadDataKeys.FILE_PATH,
        channel_column: str = CytoDLLoadDataKeys.CHANNELS,
        spatial_dims: int = 3,
        scene_column: str = CytoDLLoadDataKeys.SCENE,
        resolution_column: str = CytoDLLoadDataKeys.RESOLUTION,
        time_start_column: str = CytoDLLoadDataKeys.TIME_START,
        time_end_column: str = CytoDLLoadDataKeys.TIME_END,
        time_step_column: str = CytoDLLoadDataKeys.TIME_STEP,
        timepoints_to_include_column: str = CytoDLLoadDataKeys.INCLUDE_TIMEPOINTS,
        z_start_column: str = CytoDLLoadDataKeys.Z_START,
        z_end_column: str = CytoDLLoadDataKeys.Z_END,
        z_step_column: str = CytoDLLoadDataKeys.Z_STEP,
        num_devices: int = 1,
        extra_columns: Sequence[str] = [],
        transform: Callable | Sequence[Callable] | None = None,
        **cache_kwargs: typing.Any,
    ) -> None:
        """Initialize the MultiDimImageDataset.

        **Multi-channel images**

        The ``channel_column`` parameter should be specified to indicate which
        channel(s) to extract from the image. To load multiple channels, the
        entries of this column should be a list of the channel indices (e.g.
        ``[0,1,2]``). Else, this column should contain a single channel index
        (e.g. ``0`` or ``1``).

        **Image spatial dimensions**

        The output image will be in the format ``CZYX`` or ``CYX`` depending on
        the ``spatial_dims`` parameter. This is to ensure compatibility with
        dictionary-based MONAI-style transforms. The ``spatial_dims`` parameter
        specifies the number of spatial dimensions in the output image, which
        can be either 2 (for ``YX``) or 3 (for ``ZYX``).

        **Multi-scene images**

        If the input images are multi-scene images, the ``scene_column``
        parameter should be specified. This column should contain the names of
        the scenes to extract from the multi-scene image. If not specified, all
        scenes will be extracted. If multiple scenes are specified, the column
        entry should be a list (e.g. ``[scene1,scene2]``).

        **Multi-resolution images**

        If the there are multiple resolution level available for the input
        images, the ``resolution_column`` parameter should be specified. This
        column should contain the resolution level at which to load the image.
        If not specified, the resolution level is assumed to be 0 (full
        resolution).

        **Timelapse images**

        If there are multiple timepoints available for the input images, the
        ``time_start_column``, ``time_stop_column``, and ``time_step_column``
        parameters should be specified. These columns should contain the start
        timepoint, stop timepoint, and step between timepoints (step defaults to
        1) respectively. If not specified, all timepoints are extracted. To
        specify the last timepoint, you can use -1 in the ``time_stop_column``,
        which will be interpreted as the last timepoint available in the image.
        The timepoints are zero-indexed, so the first timepoint is 0.

        **Excluding timepoints**

        If you want to exclude specific timepoints from the timelapse image, you
        can specify the ``timepoints_to_exclude_column`` parameter. This column
        should contain a list of timepoints to exclude (e.g. ``[1,3,5]``).

        **Z slices**

        If the input images are 3D and you want to extract specific Z slices,
        the ``z_start_column``, ``z_stop_column``, and ``z_step_column``
        parameters should be specified. These columns should contain the start Z
        slice, stop Z slice, and step between Z slices (step defaults to 1)
        respectively. If not specified, all Z slices are extracted.

        **Extra columns**

        The ``extra_columns`` parameter allows you to specify additional columns
        from the dataframe that you want to include in the output dictionary.
        These columns will be added to the output dictionary as-is if found,
        otherwise their values will be set to None.

        **Transforms**

        The ``transform`` parameter allows you to specify a list or composition
        of MONAI-style transforms to apply to the image metadata. The first
        transform in the list should be an instance of ``BioIOImageLoaderd``,
        which will load the image data from the path specified in the metadata
        dictionary. The subsequent transforms can be any MONAI-style transforms
        that operate on the metadata dictionary. If no transforms are specified,
        the dataset will default to an empty list, meaning no transforms will be
        applied.

        **SmartCacheDataset**

        The dataset is a subclass of ``monai.data.SmartCacheDataset``, which
        means it can be used with MONAI's caching mechanism to speed up data
        loading and processing. The ``cache_kwargs`` parameter allows you to
        specify additional keyword arguments to pass to the
        ``SmartCacheDataset``.

        **Distributed training**

        If you are using distributed training with multiple devices/processes,
        the dataset will automatically distribute the samples evenly across the
        devices/processes based on the ``LOCAL_RANK`` and ``WORLD_SIZE``
        environment variables. If these variables are not set, the dataset will
        use the ``num_devices`` parameter to determine the number of
        devices/processes.

        Parameters
        ----------
        dataframe_path
            Path to .parquet file containing metadata table for loading the
            images.
        img_path_column
            Column in metadata table that contains path to timelapse or
            multi-scene file
        channel_column
            Column in metadata table that contains which channel to extract from
            the file.
        spatial_dims
            Spatial dimension of output image.
        scene_column
            Column in metadata table that contains scenes to extract from a
            multi-scene file.
        resolution_column
            Column in metadata table that contains resolution to extract from
            multi-resolution file.
        time_start_column
            Column in metadata table specifying the first timepoint in timelapse
            image to load.
        time_end_column
            Column in metadata table specifying the last timepoint in timelapse
            image to load.
        time_step_column
            Column in metadata table specifying step size between timepoints.
        timepoints_to_include_column
            Column in metadata table specifying which timepoints to include from
            the image.
        z_start_column
            Column in metadata table specifying the lowest Z slice to extract.
        z_end_column
            Column in metadata table specifying the highest Z slice to extract.
        z_step_column
            Column in metadata table specifying step between Z slices.
        extra_columns
            List of extra columns to include in the output dictionary.
        transform
            List (or ``Compose`` object) of Monai-style transforms to apply to
            the image metadata.
        num_devices
            Optional, number of devices/processes for distributed training.
        cache_kwargs
            Additional keyword arguments to pass to ``CacheDataset``.

        """

        # Convert dataframe "path" (which may be local path or an S3 URI) into a
        # location object so we can use built-in dataframe loading.
        df_loc = build_dataframe_location_from_string(dataframe_path)
        df = load_dataframe(df_loc)
        rank = int(os.environ.get("LOCAL_RANK", 0))
        # Use WORLD_SIZE from environment, fallback to num_devices, then default to 1
        world_size = int(os.environ.get("WORLD_SIZE", num_devices or 1))

        # Store dataset parameters
        self.img_path_column = img_path_column
        self.channel_column = channel_column
        self.scene_column = scene_column
        self.resolution_column = resolution_column
        self.time_start_column = time_start_column
        self.time_end_column = time_end_column
        self.time_step_column = time_step_column
        self.timepoints_to_include_column = timepoints_to_include_column
        self.z_start_column = z_start_column
        self.z_end_column = z_end_column
        self.z_step_column = z_step_column
        self.extra_columns = list(extra_columns)
        if spatial_dims not in (2, 3):
            raise ValueError(f"`spatial_dims` must be 2 or 3, got {spatial_dims}")
        self.spatial_dims = spatial_dims

        # First expand all data to get total sample count
        logger.info("[Rank %d] Expanding dataframe rows to individual samples", rank)
        data = self.get_per_file_args(df.reset_index(drop=True))

        # Now distribute samples evenly across ranks
        if world_size > 1:
            logger.info("[Rank %d] Total samples before distribution: %d", rank, len(data))

            # Calculate balanced distribution
            total_samples = len(data)
            samples_per_rank = total_samples // world_size
            extra_samples = total_samples % world_size

            # Each rank gets either samples_per_rank or samples_per_rank + 1 samples
            start_idx = rank * samples_per_rank + min(rank, extra_samples)
            end_idx = start_idx + samples_per_rank + (1 if rank < extra_samples else 0)

            # Slice the data for this rank
            data = data[start_idx:end_idx]

            logger.info(
                "[Rank %d] Samples assigned to this rank: %d (indices %d:%d)",
                rank,
                len(data),
                start_idx,
                end_idx,
            )

        else:
            logger.info(f"Single process training with {len(data)} samples")

        if transform is None:
            transform = []

        super().__init__(data, transform, **cache_kwargs)

    def _get_scenes(self, row: dict, img: "BioImage") -> tuple:
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

    def _get_timepoints(self, row: dict, img: "BioImage") -> list:
        """Get timepoints from the row data."""
        start = row.get(self.time_start_column, 0)
        stop = row.get(self.time_end_column, -1)
        # can use -1 to indicate the last timepoint
        if stop == -1:
            stop = img.dims.T - 1
        step = row.get(self.time_step_column, 1)
        timepoints = range(start, stop + 1, step)
        timepoints_as_list = list(timepoints)
        timepoints_to_include = row.get(self.timepoints_to_include_column, None)
        if timepoints_to_include is not None:
            logger.debug(
                "Only including timepoints: [ %s ] from available timepoints: [ %s ]",
                timepoints_to_include,
                timepoints_as_list,
            )
            timepoints_as_list = list(set(timepoints_as_list).intersection(timepoints_to_include))
        logger.debug("Loading image with timepoints: [ %s ]", timepoints_as_list)
        return sorted(timepoints_as_list)

    def _get_z_slices(self, row: dict, img: "BioImage") -> list:
        """Get Z slices from the row data."""
        z_start = row.get(self.z_start_column, 0)
        z_stop = row.get(self.z_end_column, -1)
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

        for img_path, group in df.groupby(self.img_path_column):
            # Convert image "path" (which may be local path or an S3 URI) into a
            # location object so we can use built-in image loading.
            img_loc = build_image_location_from_string(str(img_path))
            img = load_image(img_loc, read=False)

            # We expect that input images are not multiscene. This check makes
            # sure that if scenes are specified in the dataframe, they are all
            # the same value.
            if self.scene_column in group.columns and group[self.scene_column].nunique() > 1:
                logger.error("Loading does not support different scenes from the same image.")
                raise ValueError("Dataset loading does not support multiscene images.")

            # Get the list of scenes for this image using the first entry. If
            # there are multiple scenes in the image, use only the first.
            scene = self._get_scenes(group.iloc[0].to_dict(), img)[0]
            img.set_scene(scene)

            row_data = []

            for row in group.itertuples():
                row_dict: dict = row._asdict()  # type: ignore[operator]
                channel = self._get_channel(row_dict)
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


def get_plane_indices(
    dataset_config: DatasetConfig,
    position: int,
    lower_offset: int,
    upper_offset: int,
) -> list[int]:
    """
    Get a list of plane indices based on the provided outputs about the global center.

    The indices are constrained between 0 and 24.

    Parameters
    ----------
    dataset_config
        Configuration object containing dataset-specific information.
    position
        The position index for which the plane indices are calculated.
    lower_offset
        The number of planes below the center plane to include.
    upper_offset
        The number of planes above the center plane to include.

    Returns
    -------
    list
        A list of plane indices within the specified range, constrained between 0 and 24.
    """
    if dataset_config.center_z_plane is None:
        logger.error(
            "Center z-plane information is missing for dataset [ %s ].", dataset_config.name
        )
        raise ValueError("Center z-plane information is missing in the dataset configuration.")
    global_center_plane = dataset_config.center_z_plane[position]
    lower_bound = max(0, global_center_plane - lower_offset)
    upper_bound = min(24, global_center_plane + upper_offset)

    return list(range(lower_bound, upper_bound + 1))


def get_z_slice_bounds_per_position(
    dataset_config: DatasetConfig,
    z_slice_offsets: tuple[int, int] | None,
) -> dict[int, dict[CytoDLLoadDataKeys, int]]:
    """Get lower and upper z-slice bounds per position for image loading and processing.

    **Z-slice offsets**

    The ``z_slice_offsets`` parameter allows for flexible control over the
    z-slice loading. If ``z_slice_offsets`` is provided, it limits the number of
    z-slices to load by slicing about a global center as annotated in the
    dataset config. If it is ``None``, all z-slices are loaded from the raw
    brightfield images.

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
            "Using z-slice offsets: [ %s ] ",
            z_slice_offsets,
        )
    else:
        # if no z-slice offsets are provided, pass in None
        # to the dataframe builder
        logger.debug("No z-slice offsets provided, using full range in Z.")

    # if z_slice_offsets is not None, get z-slice ranges
    # for each position in the dataset (i.e., zarr file)
    # else, fixed full range is 0 to 24
    z_slice_bounds_per_position = {}
    for position_as_int in dataset_config.zarr_positions:
        # get z-slice indices for the given position
        if z_slice_offsets is not None:
            z_slices = get_plane_indices(
                dataset_config,
                position_as_int,
                lower_offset=z_slice_offsets[0],
                upper_offset=z_slice_offsets[1],
            )
        else:
            z_slices = [0, NUM_ZSLICES - 1]
        z_slice_bounds_per_position[position_as_int] = {
            CytoDLLoadDataKeys.Z_START: z_slices[0],
            CytoDLLoadDataKeys.Z_END: z_slices[-1],
        }

    return z_slice_bounds_per_position


def build_zarr_image_loading_dataframe(
    dataset_config: DatasetConfig,
    resolution_level: int = DIFFAE_ZARR_RESOLUTION_LEVEL,
    channel: int | list[int] = 0,
    frame_start: int | None = None,
    frame_stop: int | None = None,
    frame_step: int | None = None,
    z_slice_bounds_per_position: dict[int, dict[CytoDLLoadDataKeys, int]] | None = None,
    only_include_positions: list[int] | None = None,
    only_include_frames: dict[int, list[int]] | None = None,
) -> pd.DataFrame:
    """Build a DataFrame with image loading metadata for `.zarr` files.

    **Timepoint and position filtering**

    The `only_include_positions` and `only_include_frames` parameters allow for
    flexible filtering of which positions and timepoints to include in the
    DataFrame.

    If `only_include_positions` is provided, only the specified position indices
    will be included in the DataFrame.

    If `only_include_frames` is provided, only the specified timepoints for each
    position will be included in the DataFrame.

    If these parameters are not provided, all positions and timepoints will be
    included.

    Parameters
    ----------
    dataset_config
        Dataset configuration object containing information about the dataset
        and its structure.
    resolution_level
        Resolution level to load from the Zarr files.
    channel
        Channel(s) to load from the Zarr files.
    frame_start
        First timepoint to load from the Zarr files. If None, loading starts
        from the first timepoint.
    frame_stop
        Last timepoint to load from the Zarr files. If None, loading goes until
        the last timepoint.
    frame_step
        Step size between timepoints to load. If None, all timepoints between
        start and stop are loaded.
    z_slice_bounds_per_position
        Dictionary with z-slice start and stop indices per position for loading
        from the Zarr files. If None, all z-slices are loaded.
    only_include_positions
        List of position indices to include in the DataFrame.
    only_include_frames
        Optional, dictionary of timepoints to include for each position.

    Returns
    -------
    :
        DataFrame with metadata for loading images from Zarr files,
        which can be used to create a `MultiDimImageDataset`.

    """

    # Build list of zarr locations for each position in the dataset. Prefer
    # grabbing local paths first, which is faster to load, if it exists.
    # Otherwise, try to grab the S3 URI.
    available_zarr_locs = get_available_zarr_locations(dataset_config)
    zarr_file_locs = []
    for loc in available_zarr_locs:
        if loc.path is not None and loc.path.exists():
            zarr_file_locs.append(loc.path.as_posix())
        elif loc.s3uri is not None:
            zarr_file_locs.append(loc.s3uri)

    df = pd.DataFrame({CytoDLLoadDataKeys.FILE_PATH: zarr_file_locs})
    df[CytoDLLoadDataKeys.RESOLUTION] = resolution_level
    if isinstance(channel, int):
        df[CytoDLLoadDataKeys.CHANNELS] = channel
    else:
        # need to make sure list is not split
        # across multiple rows
        df[CytoDLLoadDataKeys.CHANNELS] = df[CytoDLLoadDataKeys.FILE_PATH].apply(lambda x: channel)

    # add temporary column with position index for filtering
    df["position_index"] = df[CytoDLLoadDataKeys.FILE_PATH].apply(
        lambda x: get_position_integer_from_zarr_file_path(x)
    )

    # only load images for specified position indices
    if only_include_positions is not None:
        logger.debug(
            "Filtering Zarr files to only include positions: [ %s ]", only_include_positions
        )

        df = df[df["position_index"].isin(only_include_positions)]

    # if start and stop for loading timepoints are specified, add to dataframe
    if (frame_start is not None) and (frame_stop is not None):
        df[CytoDLLoadDataKeys.TIME_START] = frame_start
        df[CytoDLLoadDataKeys.TIME_END] = frame_stop
    # frame step defaults in loader to 1, but can be specified
    if frame_step is not None:
        df[CytoDLLoadDataKeys.TIME_STEP] = frame_step

    # add column for excluding frames, if specified
    if only_include_frames is not None:
        # if position has no frames to exclude, set to None
        df[CytoDLLoadDataKeys.INCLUDE_TIMEPOINTS] = df["position_index"].apply(
            lambda x: only_include_frames.get(x, None)
        )

    # if start and stop for loading z slices are specified, add to dataframe
    if z_slice_bounds_per_position is not None:
        # get z info dict for each position index
        # unpack the start, stop, and step values from those dictionaries
        df[CytoDLLoadDataKeys.Z_START] = df["position_index"].apply(
            lambda x: z_slice_bounds_per_position.get(x, {}).get(CytoDLLoadDataKeys.Z_START, 0)
        )
        df[CytoDLLoadDataKeys.Z_END] = df["position_index"].apply(
            lambda x: z_slice_bounds_per_position.get(x, {}).get(CytoDLLoadDataKeys.Z_END, -1)
        )
        df[CytoDLLoadDataKeys.Z_STEP] = df["position_index"].apply(
            lambda x: z_slice_bounds_per_position.get(x, {}).get(CytoDLLoadDataKeys.Z_STEP, 1)
        )

    # remove temporary column with position index
    df = df.drop(columns=["position_index"])

    return df
