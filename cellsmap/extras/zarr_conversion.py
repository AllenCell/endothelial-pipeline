#%%
import numpy as np
from pathlib import Path
from cellsmap.util import io
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from bioio.writers import OmeTiffWriter, OmeZarrWriter

#%%
CHANNEL_NAMES = ["gfp", "bf_mean", "bf_max", "bf_std", "bf_center"]
#%%
def two_D_image_processing(gfp, brf):
    center_slice = brf.std(axis=(1,2)).argmin()

    stack = np.concatenate([
        gfp.max(axis=0, keepdims=True),# MIP of GFP channel
        brf.mean(axis=0, keepdims=True),# Mean of BF channel
        brf.max(axis=0, keepdims=True),# MIP of BF channel
        brf.std(axis=0, keepdims=True),# STD of BF channel
        brf[center_slice:center_slice+1]# Center slice of BF channel
        ], axis=0) # T, Y, X
    
    return stack
    
def process_this_timepoint(tp: int, dataset: str):
    gfp = io.load_original_slidebook_image(dataset, channel=0, timepoint=tp)
    brf = io.load_original_slidebook_image(dataset, channel=1, timepoint=tp)
    
    stack = two_D_image_processing(gfp, brf)
    
    return stack

def get_timepoints(pos: int, dataset: str, number_positions: int = 6):
    # t_final = io.get_dataset_duration_in_frames(dataset)
    t_final = 10
    timepoints = range(pos, t_final * number_positions, number_positions)
    return timepoints

def process_this_position(pos: int, dataset: str, number_positions: int = 6):
    timepoints = get_timepoints(pos, dataset, number_positions)
    args = [(tp, dataset) for tp in timepoints]
    with Pool() as pool:
        results = list(
            tqdm(
                pool.starmap(process_this_timepoint, args),
                total=len(args),
                desc=f"Processing timepoints for position {pos}"
            )
        )
    
    scene = np.stack(results, axis=0)
    
    return scene

def save_to_tiff(image_stacks: list[np.array], output_folder: Path, fname: str, channel_names: list):
    channel_names = [CHANNEL_NAMES for _ in range(len(image_stacks))]
    OmeTiffWriter.save(image_stacks, 
                       output_folder / f"{fname}.ome.tif", 
                       dim_order="TCYX", 
                       channel_names=channel_names)

# Currently does not work    
def save_to_zarr(image_stacks: list[np.array], output_folder: Path, fname: str, channel_names: list):
    channel_names = [CHANNEL_NAMES for _ in range(len(image_stacks))]

    writer = OmeZarrWriter(output_folder)
    writer.write_image(image_data=image_stacks,
                       image_name=f"{fname}.ome.zarr",
                       physical_pixel_sizes=(1.0, 0.5, 0.5),
                       channel_names=channel_names,
                       channel_colors=None)    
    
def convert(dataset: str, output_folder: Path, fname: str, n_positions: int = 6, save_to_zarr: bool = False):
    all_scenes = [process_this_position(i, dataset, n_positions) for i in range(n_positions)]
    if save_to_zarr:
        save_to_zarr(all_scenes, output_folder, fname, CHANNEL_NAMES)
    else:
        save_to_tiff(all_scenes, output_folder, fname, CHANNEL_NAMES)