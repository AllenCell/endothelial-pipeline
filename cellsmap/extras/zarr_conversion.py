#%%
import numpy as np
from pathlib import Path
from cellsmap.util import io
from multiprocessing import Pool
from bioio import BioImage
from bioio_base.types import PhysicalPixelSizes
from bioio.writers import ome_zarr_writer_2 as ome_zarr_writer

#%%
CHANNEL_NAMES = ["egfp", "bf"]

def get_sldy_metadata(dataset):
    original_path = str(io.get_original_path(dataset))
    dataset_path = original_path.rsplit('/', 1)[0]
    im = BioImage(dataset_path)
    xy_pixel_size_in_um = im.physical_pixel_sizes
    metadata = im.metadata
    z_step_um = metadata['channel_record']['CExposureRecord70'][0]['mInterplaneSpacing']
    
    physical_pixel_sizes = PhysicalPixelSizes(
        Z=z_step_um,
        Y=xy_pixel_size_in_um.Y,
        X=xy_pixel_size_in_um.X,
    )
    return physical_pixel_sizes

def process_timepoint(tp: int, dataset: str):
    gfp_index, bf_index = io.get_channel_order(dataset)
    gfp = io.load_original_slidebook_image(dataset, channel=gfp_index, timepoint=tp)
    bf = io.load_original_slidebook_image(dataset, channel=bf_index, timepoint=tp)
    stack = np.stack([gfp, bf], axis=0) # C, Z, Y, X # convert this to a dask stack
    return stack

def get_timepoints(pos: int, dataset: str, number_positions: int = 6):
    # t_final = io.get_dataset_duration_in_frames(dataset)
    t_final = 10 # testing with just 10 timepoints per scene for now
    timepoints = range(pos, t_final * number_positions, number_positions)
    return timepoints

def process_position(pos: int, dataset: str, number_positions: int = 6):
    timepoints = get_timepoints(pos, dataset, number_positions)
    args = [(tp, dataset) for tp in timepoints]
    with Pool() as pool:
        results = pool.starmap(process_timepoint, args)
    scene = np.stack(results, axis=0)
    print(f"finished processing {timepoints} timepoints")
    return scene

def get_channel_indices(channels: list[str]) -> list[int]:
    channel_map = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}
    return [channel_map[channel] for channel in channels]

def _get_level_shapes(_shape,):
    """
    Uses XY and Z scaling parameters to determine the
    image data shape at different resolutions.
    """
    _xy_scaling = [0.5]
    _z_scaling = [1.0]
    
    if len(_xy_scaling) != len(_z_scaling):
        raise ValueError(f"Found XY and Z scaling with different length: XY={_xy_scaling}, Z={_z_scaling}.")
    source_shape = _shape
    level_shapes = [source_shape]
    nchannels = source_shape[1]

    for sid, _ in enumerate(_xy_scaling):
        z_scaling = np.prod(_z_scaling[:sid+1])
        xy_scaling = np.prod(_xy_scaling[:sid+1])
        level_shape = (
            source_shape[0],
            nchannels,
            int(source_shape[2]*z_scaling),
            int(source_shape[3]*xy_scaling),
            int(source_shape[4]*xy_scaling)
        )
        level_shapes.append(level_shape)
    return level_shapes

def _get_zarr_chunk_dims(im_shape):
    chunk_dims = []
    level_shapes = _get_level_shapes(im_shape)
    for i, dim in enumerate(level_shapes):
        z = np.min([dim[-3], 4**i])
        chunk_dims.append((1, 1, z, dim[-2], dim[-1]))
    print(f"Level shapes: {level_shapes}")
    print(f"ZARR chunk dims: {chunk_dims}")
    return chunk_dims
    
def write_scene(im: np.array, 
                    channels: list[str], 
                    full_zarr_path: Path, 
                    dataset: str,
                    position: int,
                    physical_pixel_sizes):

    zarr_chunk_dims_tuples = _get_zarr_chunk_dims(im.shape)

    writer = ome_zarr_writer.OmeZarrWriter()
    writer.init_store(
        output_path = full_zarr_path,
        shapes = _get_level_shapes(im.shape),
        chunk_sizes = zarr_chunk_dims_tuples, 
        dtype = im.dtype
    )

    # Use all channels, if channels are not specific by user
    _channels = [c for c in range(im.shape[1])]

    writer.write_t_batches_array(im, channels=_channels, tbatch=1)

    physical_scale = {
        "c": 1.0, # default value for channel
        "t": 1.0,
        "z": physical_pixel_sizes.Z,
        "y": physical_pixel_sizes.Y,
        "x": physical_pixel_sizes.X,
    }
    physical_units = {
        "x": "micrometer",
        "y": "micrometer",
        "z": "micrometer",
        "t": "minute",
    }
    meta = writer.generate_metadata(
        image_name = f"{dataset}_{position}",
        channel_names = channels,
        physical_dims = physical_scale,
        physical_units = physical_units,
        channel_colors = [0xFFFFFF for i in range(im.shape[1])]
    )
    writer.write_metadata(meta)
    return


#%%
dataset = '20240305_T01_001'
output = f'/allen/aics/assay-dev/users/Chantelle/outputs/temp_tiffs/{dataset}_0.ome.zarr'
# timepoints = get_timepoints(0, dataset)
# stack = process_timepoint(timepoints[0], dataset)
scene = process_position(0, dataset)
print('get metadata')
physical_pixel_sizes = get_sldy_metadata(dataset)
print("saving zarr")
write_scene(scene, CHANNEL_NAMES, output, dataset, 0, physical_pixel_sizes)




# # Example usage
# if __name__ == "__main__":
#     dataset = '20240305_T01_001'
#     output = Path('/allen/aics/assay-dev/users/Chantelle/outputs/temp_tiffs/')
#     scene = process_position(0, dataset, output)
#     write_scene(scene, CHANNEL_NAMES, output, dataset, 0)
    
#     del scene
#     gc.collect()


#%%    
# def process_timelapse(dataset: str, n_positions: int = 6):
#     all_scenes = [process_position(i, dataset, n_positions) for i in range(n_positions)]
#     return all_scenes


# def save_to_tiff(image_stacks: list[np.array], output_folder: Path, dataset: str, channel_names: list):
#     output_folder.mkdir(parents=True, exist_ok=True)
#     channel_names = [CHANNEL_NAMES for _ in range(len(image_stacks))]
#     OmeTiffWriter.save(image_stacks, 
#                        f"{output_folder}/{dataset}.ome.tiff", 
#                        dim_order="TCZYX", 
#                        channel_names=channel_names)

# def write_scene(im: np.array, 
#                     channels: list[str], 
#                     output: Path, 
#                     dataset: str,
#                     position: int):
    
#     save_path = output / f"{dataset}_{position}.ome.zarr"
#     writer = OmeZarrWriter(save_path)
#     print(im.shape)
#     pps = get_sldy_metadata(dataset)
    
#     print("saving ome.zarr")
#     writer.write_image(im,
#                        image_name=f"{dataset}_{position}",
#                        physical_pixel_sizes=pps, 
#                        channel_names=channels,
#                        channel_colors=None,
#                        scale_factor=2, 
#                        scale_num_levels=2,
#                        dimension_order="TCZYX",
#                        )
#     return