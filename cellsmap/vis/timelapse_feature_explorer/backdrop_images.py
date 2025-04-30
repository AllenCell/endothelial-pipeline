from cellsmap.util import dataset_io
from pathlib import Path
from bioio import BioImage
import numpy as np
from bioio import BioImage
from skimage import exposure
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import imageio.v3 as iio

def bf_slice(img, frame):
    bf_stack = img.get_image_dask_data('ZYX', C=1, T=frame)
    stdevs = [plane.std().compute() for plane in bf_stack.squeeze()]
    best_plane = max(0, np.argmin(stdevs) - 5) # move 5 planes down to have contrast
    bf_slice = img.get_image_dask_data('YX', Z=best_plane, C=1, T=0)
    return bf_slice.compute()
    
def bf_std_dev(img, frame):
    bf_img = img.get_image_dask_data('ZYX', C=1, T=frame)
    bf_std_dev = bf_img.std(axis=0)
    return bf_std_dev.compute()
    
def gfp_max_proj(img, frame):
    gfp = img.get_image_dask_data('ZYX', C=0, T=frame)
    gfp_max_proj = gfp.max(axis=0)
    return gfp_max_proj.compute()

def contrast_stretching(image, method='percentile', low_percentile=1, high_percentile=99):
    """
    Apply contrast stretching to an image.

    Parameters:
    image (ndarray): The input image.
    method (str): The method of contrast stretching ('min-max' or 'percentile').
    low_percentile (int): The low percentile for percentile contrast stretching.
    high_percentile (int): The high percentile for percentile contrast stretching.

    Returns:
    ndarray: The contrast-stretched image.
    """
    if method == 'min-max':
        low = image.min()
        high = image.max()
    elif method == 'percentile':
        low, high = np.percentile(image, (low_percentile, high_percentile))
    
    stretched_image = exposure.rescale_intensity(image, in_range=(low, high), out_range=(0, 255))
    return stretched_image

def process_frame(func, img, frame, dataset, position, backdrop, output_dir):
    # Run the specific image processing function
    image_to_save = func(img, frame)

    # Contrast stretch to 0–255 range
    image_contrasted = contrast_stretching(image_to_save, method='percentile')

    # Convert to 8-bit unsigned int
    image_contrasted = np.clip(image_contrasted, 0, 255).astype(np.uint8)

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save image
    fname = f"{dataset}_P{position}_{backdrop}_{frame}.png"
    output_path = output_dir / fname
    iio.imwrite(output_path, image_contrasted)

def generate_backdrops(dataset: str, position: int, backdrops: list, output_dir: Path):
    zarr_name = dataset_io.get_zarr_name(dataset, position)
    zarr_path = dataset_io.get_zarr_dir(dataset)
    filepath = Path(zarr_path) / zarr_name
    img = BioImage(filepath)
    img.set_resolution_level(1)

    backdrop_functions = {
        "bf_slice": bf_slice,
        "bf_std_dev": bf_std_dev,
        "gfp_max_proj": gfp_max_proj,
    }

    for backdrop, func in backdrop_functions.items():
        if backdrop in backdrops:
            print(f"Generating {backdrop} for dataset {dataset}, position {position}...")

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        process_frame,
                        func, img, frame, dataset, position, backdrop, output_dir
                    )
                    for frame in range(img.shape[0])
                ]

                for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Processing frames for {backdrop}"):
                    _.result()  # Catch exceptions

            
def add_backdrop_fname_to_manifest(df, dataset: str, position: int, backdrops: list, output_dir: Path):
    """
    Add the backdrop file name to the manifest DataFrame.
    """
    for backdrop in backdrops:
        df[f"{backdrop}_backdrop"] = df.apply(
            lambda row: output_dir / f"{dataset}_P{position}_{backdrop}_{row['image_index']}.png", axis=1
        )
    return df