from cellsmap.util.io import (
    get_original_path,
    get_channel_order,
)
import dask.delayed
import dask.array as da
import numpy as np


def get_slidebook_image_path(dataset_name: str, channel: int, timepoint: int):
    sld_path = get_original_path(dataset_name)
    sld_path = sld_path / f"ImageData_Ch{channel}_TP{timepoint:07d}.npy"
    return sld_path


def get_image_format(dataset_name: str, channel: int, timepoint: int):
    sld_path = get_slidebook_image_path(dataset_name, channel, timepoint)
    array = np.load(sld_path)
    return array.shape, array.dtype


@dask.delayed
def delayed_np_load(filename):
    return np.load(filename)


def load_original_slidebook_image(
    dataset_name: str, channel: int, timepoint: int
) -> da.Array:
    sld_path = get_slidebook_image_path(dataset_name, channel, timepoint)
    delayed_array = delayed_np_load(sld_path)
    return delayed_array


def process_timepoint(tp: int, dataset: str, shape, dtype):
    gfp_index, bf_index = get_channel_order(dataset)
    gfp = load_original_slidebook_image(dataset, channel=gfp_index, timepoint=tp)
    bf = load_original_slidebook_image(dataset, channel=bf_index, timepoint=tp)
    gfp_da = da.from_delayed(gfp, shape=shape, dtype=dtype)
    bf_da = da.from_delayed(bf, shape=shape, dtype=dtype)
    stack = da.stack([gfp_da, bf_da], axis=0)
    return stack


def get_timepoints(pos: int, dataset: str, number_positions: int = 6):
    # t_final = get_dataset_duration_in_frames(dataset) # if testing set t_final = 10
    t_final = 10  # testing with just 10 timepoints per scene for now
    timepoints = range(pos, t_final * number_positions, number_positions)
    return timepoints


def process_position(pos: int, dataset: str, number_positions: int = 6) -> da.Array:
    timepoints = get_timepoints(pos, dataset, number_positions)
    shape, dtype = get_image_format(dataset, 0, 0)
    results = [process_timepoint(tp, dataset, shape, dtype) for tp in timepoints]
    scene = da.stack(results, axis=0)
    print(f"finished processing {len(timepoints)} timepoints")
    return scene
