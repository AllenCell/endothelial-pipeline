#%%
import numpy as np
from pathlib import Path
from cellsmap.util import io
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from bioio.writers import OmeTiffWriter
from bioio.writers import OmeZarrWriter2 as OmeZarrWriter

#%%
CHANNEL_NAMES = ["egfp", "bf"]
    
def process_this_timepoint(tp: int, dataset: str):
    gfp_index, bf_index = io.get_channel_order(dataset)
    gfp = io.load_original_slidebook_image(dataset, channel=gfp_index, timepoint=tp)
    brf = io.load_original_slidebook_image(dataset, channel=bf_index, timepoint=tp)
    
    # Add a new axis to represent the channel dimension
    gfp = np.expand_dims(gfp, axis=0)  # Shape becomes (1, Z, Y, X)
    brf = np.expand_dims(brf, axis=0)  # Shape becomes (1, Z, Y, X)
    
    # Concatenate along the new channel axis
    stack = np.concatenate([gfp, brf], axis=0)  # Shape becomes (2, Z, Y, X)
    stack = np.concatenate([gfp, brf], axis=0) # C, Z, Y, X,
    return stack

def get_timepoints(pos: int, dataset: str, number_positions: int = 6):
    # t_final = io.get_dataset_duration_in_frames(dataset) 
    t_final = 10 # testing with just 10 timepoints per scene for now
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

def process_timelapse(dataset: str, n_positions: int = 6):
    all_scenes = [process_this_position(i, dataset, n_positions) for i in range(n_positions)]
    return all_scenes

def save_to_tiff(image_stacks: list[np.array], output_folder: Path, fname: str, channel_names: list):
    channel_names = [CHANNEL_NAMES for _ in range(len(image_stacks))]
    OmeTiffWriter.save(image_stacks, 
                       output_folder, 
                       dim_order="TCZYX", 
                       channel_names=channel_names)

# def save_to_zarr(images: list[np.array], output_folder: Path, channel_names: list):    

#%%
# Example usage
if __name__ == "__main__":
    dataset = '20240305_T01_001'
    images = process_timelapse(dataset)    
    output = Path(f'/allen/aics/assay-dev/users/Chantelle/outputs/temp_tiffs/{dataset}v2.tif')
    save_to_tiff(images, output, dataset, CHANNEL_NAMES)
# %%
