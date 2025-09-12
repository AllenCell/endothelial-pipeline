# %%
from typing import Any

import numpy as np

from endo_pipeline.library.process.image_processing import (
    clip_image,
    max_proj,
    scale_intensity_range_percentiles,
    std_dev,
    z_score_normalize_intensity,
)
from endo_pipeline.settings.image_data import LOG_EPSILON


def process_brightfield(bf_stack: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a brightfield image stack following the same steps as the diffae model DataLoader:
    type conversion, standard deviation projection, clipping, and normalization.

    Args:
        bf_stack (Any): Input brightfield image stack as a Dask array.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - bf_stack_float32_computed: The bf stack as a computed np.ndarray in float32 format.
            - standard_dev_proj: The standard deviation projection along the Z-axis.
            - standard_dev_proj_log: The log of the standard deviation projection.
            - clipped_im: The image clipped by specified percentiles.
            - normalized_im: The Z-score normalized image.
    """
    # STEP 1: Load the brightfield stack as a Dask array, convert to float32
    bf_stack_float32 = bf_stack.astype("float32")
    bf_stack_float32_computed = bf_stack_float32.compute()  # compute for visualization

    # STEP 2: Standard deviation projection along the Z-axis
    standard_dev_proj = std_dev(bf_stack_float32, axis=0, unbiased=False).astype("float32")

    # STEP 3: Log transform the standard deviation projection
    standard_dev_proj_log = np.log(
        standard_dev_proj + LOG_EPSILON
    )  # Add small constant to avoid log(0)

    # STEP 3: Clip image by percentiles
    clipped_im = clip_image(standard_dev_proj_log, low_pct=0.1, high_pct=99.9)

    # STEP 4: Z-score normalize
    normalized_im = z_score_normalize_intensity(clipped_im)

    return (
        bf_stack_float32_computed,
        standard_dev_proj,
        standard_dev_proj_log,
        clipped_im,
        normalized_im,
    )


def process_cdh5(cdh5_stack: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a CDH5 image stack following the same steps as the diffae model DataLoader:
    type conversion, maximum projection, and intensity scaling.

    Args:
        cdh5_stack (Any): Input CDH5 image stack as a Dask array.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - cdh5_stack_float32_computed: The CDH5 stack as a computed np.ndarray in float32 format.
            - max_proj_im: The maximum projection along the Z-axis.
            - scaled_im: The image scaled to a specified intensity range.
    """
    # STEP 1: Load the CDH5 stack as a Dask array, convert to float32
    cdh5_stack_float32 = cdh5_stack.astype("float32")
    cdh5_stack_float32_computed = cdh5_stack_float32.compute()  # compute for visualization

    # STEP 2: Maximum projection along the Z-axis
    max_proj_im = max_proj(cdh5_stack_float32, axis=0).astype("float32")

    # STEP 3: Clip image by percentiles, Map linearly -1 to 1
    scaled_im = scale_intensity_range_percentiles(max_proj_im, 10, 98, -1.0, 1.0, clip=True)

    return cdh5_stack_float32_computed, max_proj_im, scaled_im
