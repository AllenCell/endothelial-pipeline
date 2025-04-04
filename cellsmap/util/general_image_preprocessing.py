import yaml
import dask
import numpy as np
import pandas as pd
import re
from pathlib import Path
from bioio import BioImage
from bioio.writers import OmeTiffWriter
import dask.array
from cellsmap.util.dataset_io import get_zarr_path, get_original_path, get_total_number_of_positions
from cellsmap.util.set_output import get_output_path
from cellsmap.util.get_sldy_metadata import get_objective_info
from typing import List, Dict, Any, Union, Tuple, Callable, Optional

def extract_T(fp_as_string: Union[str, Path], int_only=True, use_last_match=True):
    """
    Extract the timepoint value from a string or Path.name.
    Searches for the pattern "T[0-9]+" to find the timepoint.
    If use_last_match is True then the last match will be used,
    otherwise the first one will be used.

    Parameters
    ----------
    fp_as_string: str or Path
        A string or Path.name to get the timepoint from.
    int_only: bool
        Whether to return just the timepoint as an integer or
        an entire string (i.e. 10 vs 'T010')
        Default is True.
    use_last_match: bool
        Whether to use the last match (in the event that multiple possible
        timepoint values were found in the string).
        If False then the first match will be used.
        E.g. image_name_T1_etc_T57.tif can return either T1 or T57, but
        will return 57 by default. Ideally the timepoint in fp_as_string
        would be unambiguous.
        Default is True.

    Returns
    -------
    t: int or str
        The timepoint represented as an integer if int_only is True, otherwise
        the timepoint represented as a string including the T before.
    """

    if isinstance(fp_as_string, Path):
        fp_as_string = str(fp_as_string)

    index = -1 if use_last_match else 0
    t = re.findall('T[0-9]+', fp_as_string)
    if t:
        t_value = int(t[index].split('T')[-1])
    else:
        t_value = 0 
        print("""No 'T[0-9]+' found in filename. Assuming only 
              1 timepoint and assuming T = 0.""")

    return t_value if int_only else f'T{t_value}'

def get_dim_map(dim_order: str) -> dict:

    dims = [a for a in dim_order]
    dim_nums = tuple(range(len(dims)))
    dim_map = dict(zip(dims, dim_nums))

    return dim_map

def get_chan_map(filepath: Path) -> dict:
    img = BioImage(filepath)
    return {name:index for index, name in enumerate(img.channel_names)}

def build_analysis_queue(dataset_name_list: list,
                         t_start: int=0, t_final: int|None=None, t_step: int=1,
                         img_bin_level: int=0,
                         save_output=True, overwrite=False, out_dir: str|Path|None=None,
                         magnification: int|None=None,
                         verbose=None, is_test=False, use_original_data=False) -> list:
    analysis_queue: list = []
    out_dir = Path(out_dir) if out_dir != None else Path(get_output_path('analysis_queue_output_temp', verbose=False))
    for dataset_name in dataset_name_list:
        img_path = Path(get_zarr_path(dataset_name)) if not use_original_data else Path(get_original_path(dataset_name))
        img = BioImage(img_path)

        num_positions = get_total_number_of_positions(dataset_name)

        assert num_positions % len(img.scenes) == 0, f'Number of positions ({num_positions}) in data_config.yaml must be divisible by number of scenes ({len(img.scenes)}) in the image file for dataset {dataset_name}'
        num_pos_in_T = num_positions // len(img.scenes)
        num_pos_in_S = len(img.scenes)

        positions_in_T, positions_in_S = [], []
        for scene_index in range(num_pos_in_S):
            positions_in_T += list(range(num_pos_in_T))
            positions_in_S += [scene_index] * num_pos_in_T

        for pos, (pos_in_T, pos_in_S) in enumerate(zip(positions_in_T, positions_in_S)):
            img.set_scene(pos_in_S)
            if magnification !=None and get_objective_info(img.metadata)['magnification'] != magnification:
                print(f'Position{pos} (scene {img.current_scene}) -- does not use 20X magnification, skipping...') if verbose else None
            else:
                print(f'Position {pos} (scene {img.current_scene}) -- processing...') if verbose else None
                assert img.dims.T % num_pos_in_T == 0, f'Number of timepoints ({img.dims.T}) must be divisible by number of positions ({num_pos_in_T}) in the data_config.yaml for dataset {dataset_name} if number of positions does not equal the number of scenes in the image file.'
                # calculate the duration of the positions in frames (they must all have the same duration)
                duration_in_frames = t_final if isinstance(t_final, int) else img.dims.T // num_pos_in_T
                # correct the t_start, t_final, and t_step values to account for the intercalation of positions with timeframes
                t_start_adjusted = t_start or pos_in_T
                t_step_adjusted = t_step * num_pos_in_T
                t_final_adjusted = pos_in_T + duration_in_frames * num_pos_in_T
                t_range = range(t_start_adjusted, t_final_adjusted, t_step_adjusted)

                for i,t in enumerate(t_range):
                    if is_test and i >= 10:
                        break
                    else:
                        pass

                    if t >= t_start_adjusted and t < t_final_adjusted:
                        analysis_queue.append({
                            'dataset_name': dataset_name,
                            'image_bin_level': img_bin_level,
                            'scene_index': pos_in_S,
                            'position': pos,
                            'T': t,
                            'input_path': img_path,
                            'output_dir': out_dir,
                            'save_output': save_output,
                            'overwrite': overwrite,
                            'use_original_data': use_original_data,
                            'is_test': is_test,
                            'verbose': verbose,
                            })
    return analysis_queue

def restore_full_dims(image: np.ndarray, current_dims: str, full_dims: str='TCZYX') -> np.ndarray:
    """
    Takes an array with specified image dims and restores dimensions with size 1
    that are present in full_dims.
    Useful for saving images using BIOIO such that they have the full TCZYX
    dimensionality so that they will load properly when opened with ImageJ/FIJI.

    NOTE: the letters in current_dims and full_dims are case sensitive and their
    order matters.

    Parameters
    ----------
    image: np.array
        The image to restore the dimensions of.

    current_dims: str
        The dimensions of 'image' as a string. Possible dimensions are:
            S: scene / position
            T: time
            C: channel
            Z: z position
            Y: y position
            X: x position

    full_dims: str
        The dimensions to restore the image to.

    Returns
    -------
    image: np.ndarray
        The image with its dimensions expanded.
    """

    assert all([dim in list(full_dims) for dim in list(current_dims)]), "All dimensions in current_dims must be in full_dims."
    dim_map = get_dim_map(full_dims)
    for dim in full_dims:
        if dim not in list(current_dims):
            image = np.expand_dims(image, axis=dim_map[dim])

    return image

def save_image_output(out_path: Union[str, Path], images: List[np.ndarray], images_metadata: dict, dtype: Optional[Any] = None) -> None:
    """
    Combines a list of images into a single image and saves it as an OME-TIFF
    along with metadata using bioio.OmeTiffWriter.save().

    Parameters
    ----------
    out_path: Path

    images: list
        A list of numpy arrays

    images_metadata: dict
        The metadata to pass along to OmeTiffWrite.save().
        Requires the following keys:
            'image_name': string
            'channel_colors': [(int, int, int) , (int, int, int), ... )]
                where each tuple is a color defined in RGB and the length of channel_colors
                is the same length as the length of images
            'channel_names': [string, string, ...]
                where each string is a channel name and channel_names is the same length as
                the length of images
            'physical_pixel_sizes': (Z, Y, X)
                the physical pixel sizes in the order Z, Y, X.
            'dim_order': string
                the order of the dimensions of the arrays in images (e.g. 'CYX')

    Returns
    -------
    Nothing (saves an image to out_path).
    """

    assert all([img.shape == images[-1].shape for img in images]), "All images must have the same shape."
    # if a data type is not specified then use the smallest uint type that can hold the max value
    # among all images being saved
    if not dtype:
        img_max = max([img.max() for img in images])
        dtypes = {np.iinfo(dtype).max: dtype for dtype in (np.uint8, np.uint16, np.uint32) if img_max <= np.iinfo(dtype).max}

        assert dtypes, \
        '''
        Max pixel value in one of the channels to be saved exceeds uint32 data type, unable to save OME-TIFFs with dtype uint64 of greater. 
        Please find a way to reduce the max value in the culprit channel or save the image in a different format.
        '''

        dtype = dtypes[min(dtypes)]
    else:
        pass

    image_name = images_metadata['image_name']
    ch_colors = images_metadata['channel_colors']
    ch_names = images_metadata['channel_names']
    px_res = images_metadata['physical_pixel_sizes']
    img_dim_order = images_metadata['dim_order']
    dim_order_out = 'TCZYX'

    dim_map = get_dim_map(dim_order_out)

    merged_img = np.concatenate([
        restore_full_dims(img, img_dim_order, full_dims=dim_order_out) for img in images], 
        axis=dim_map['C']).astype(dtype)

    OmeTiffWriter.save(merged_img, 
                       out_path,
                       physical_pixel_sizes=px_res,
                       dim_order=dim_order_out,
                       image_name=image_name,
                       channel_names=ch_names,
                       channel_colors=ch_colors)