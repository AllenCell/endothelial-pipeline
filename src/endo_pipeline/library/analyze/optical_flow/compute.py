import logging

import numpy as np
from scipy import stats
from skimage.registration import optical_flow_tvl1

from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.optical_flow import OPTICAL_FLOW_BASE_FEATURES
from typing import  NamedTuple

logger = logging.getLogger(__name__)

class OpticalFlowImagePair(NamedTuple):
    """Structure for optical flow image pair."""

    t0: int
    t1: int
    dt: int

class OpticalFlowImagePairCrop(NamedTuple):
    """Structure for image pair crop."""

    start_x: np.ndarray
    start_y: np.ndarray
    crop_indices: np.ndarray
    crop_size: int

# ---------------------------------------------------------------------------
# Flow statistics
# ---------------------------------------------------------------------------
def compute_flow_statistics(
    u: np.ndarray,
    v: np.ndarray,
    crop0: np.ndarray,
    crop1: np.ndarray,
    crop_idx: int,
    timepoint: int,
    dt: int,
    thresh: float,
    speed_threshold: float = 1.0,
) -> dict:
    """Compute summary statistics from a 2-D optical-flow field (u, v).

    Pixels are included only when the intensity in *either* ``crop0``
    or ``crop1`` exceeds ``thresh``.

    Parameters
    ----------
    u, v
        Horizontal/vertical flow components, shape ``(H, W)``.
    crop0, crop1
        Intensity images at successive timepoints, shape ``(H, W)``.
    crop_idx
        Spatial crop identifier.
    timepoint
        Frame index of ``crop0``.
    dt
        Temporal stride between the two frames.
    thresh
        Intensity threshold for foreground masking.
    speed_threshold
        Minimum pixel speed for the "fast" coherence features.

    Returns
    -------
    :
        Flat dictionary of scalar statistics with identifying fields.
    """
    base: dict[str, int | float] = {
        ColumnName.CROP_INDEX: crop_idx,
        ColumnName.TIMEPOINT: timepoint,
        "dt": dt,
    }
    mask = (crop0 > thresh) | (crop1 > thresh)

    # Build the NaN key set dynamically based on enabled features.
    nan_keys = OPTICAL_FLOW_BASE_FEATURES

    if not mask.any():
        logger.debug(
            "No foreground pixels above thresh=%.3g for crop_idx=%d, timepoint=%d; returning NaNs.",
            thresh,
            crop_idx,
            timepoint,
        )
        base.update(dict.fromkeys(nan_keys, np.nan))
        return base

    sp = np.sqrt(u[mask] ** 2 + v[mask] ** 2)
    ang = np.arctan2(v[mask], u[mask])
    um, vm = u[mask], v[mask]

    nz = sp > 0
    muv = (
        float(np.sqrt(np.mean(um[nz] / sp[nz]) ** 2 + np.mean(vm[nz] / sp[nz]) ** 2))
        if nz.any()
        else 0.0
    )

    # --- Thresholded coherence (speed > threshold) ---
    fast = sp > speed_threshold
    n_fast = int(fast.sum())
    if fast.any():
        muv_fast = float(
            np.sqrt(np.mean(um[fast] / sp[fast]) ** 2 + np.mean(vm[fast] / sp[fast]) ** 2)
        )
    else:
        muv_fast = np.nan

    # --- Radial coherence ---
    H, W = u.shape
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.mgrid[:H, :W]
    ry = (yy - cy).astype(np.float32)
    rx = (xx - cx).astype(np.float32)
    r_mag = np.sqrt(rx**2 + ry**2)
    sp_full = np.sqrt(u**2 + v**2)
    radial_mask = mask & (r_mag > 0) & (sp_full > 0)
    if radial_mask.any():
        rm = r_mag[radial_mask]
        rx_hat = rx[radial_mask] / rm
        ry_hat = ry[radial_mask] / rm
        sp_rm = sp_full[radial_mask]
        ux_hat = u[radial_mask] / sp_rm
        uy_hat = v[radial_mask] / sp_rm
        dot_products = ux_hat * rx_hat + uy_hat * ry_hat
        radial_coh = float(dot_products.mean())
        radial_coh_w = float(np.average(dot_products, weights=rm))
    else:
        radial_coh = np.nan
        radial_coh_w = np.nan

    base.update(
        {
            ColumnName.OpticalFlow.SPEED_MEAN_BASE: float(sp.mean()),
            ColumnName.OpticalFlow.UNIT_VECTOR_MEAN_BASE: muv,
            ColumnName.OpticalFlow.SPEED_STD_BASE: float(sp.std()),
            ColumnName.OpticalFlow.ANGLE_MEAN_BASE: float(
                np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
            ),
            ColumnName.OpticalFlow.ANGLE_STD_BASE: float(stats.circstd(ang)),
            ColumnName.OpticalFlow.U_MEAN_BASE: float(um.mean()),
            ColumnName.OpticalFlow.V_MEAN_BASE: float(vm.mean()),
            ColumnName.OpticalFlow.U_STD_BASE: float(um.std()),
            ColumnName.OpticalFlow.V_STD_BASE: float(vm.std()),
        }
    )

    base[ColumnName.OpticalFlow.SPEED_ABOVE_1_COUNT_BASE] = n_fast
    base[ColumnName.OpticalFlow.UNIT_VECTOR_MEAN_FAST_BASE] = muv_fast

    base[ColumnName.OpticalFlow.RADIAL_COHERENCE_BASE] = radial_coh
    base[ColumnName.OpticalFlow.RADIAL_COHERENCE_WEIGHTED_BASE] = radial_coh_w

    return base


# ---------------------------------------------------------------------------
# TVL1 Optical Flow wrappers
# ---------------------------------------------------------------------------
def compute_tvl1(
    f0: np.ndarray,
    f1: np.ndarray,
    attachment: float = 7.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run TVL1 optical flow on two 2-D frames.

    Wraps :func:`skimage.registration.optical_flow_tvl1` and swaps
    the returned ``(v, u)`` order to ``(u, v)`` for consistency with
    the ``(x, y)`` / ``(col, row)`` convention.

    Parameters
    ----------
    f0
        Reference frame.
    f1
        Subsequent frame.
    attachment
        TVL1 data-fidelity weight (λ).  Lower values yield smoother
        flow fields.

    Returns
    -------
    u
        Horizontal (column-direction) flow component.
    v
        Vertical (row-direction) flow component.
    """
    v, u = optical_flow_tvl1(f0, f1, attachment=attachment)
    return u, v


def compute_image_pair_flow(
    f0: np.ndarray,
    f1: np.ndarray,
    sy: np.ndarray,
    ey: np.ndarray,
    sx: np.ndarray,
    ex: np.ndarray,
    crop_indices: np.ndarray,
    t0: int,
    dt: int,
    thresh: float,
    attachment: float = 7.5,
    speed_threshold: float = 1.0,
) -> list[dict]:
    """Run TVL1 on a full-resolution frame pair, then compute per-crop stats.

    This is the *image-scope* strategy: TVL1 runs once on the full image and the
    resulting flow field is sliced per crop.  Compared to the crop-scope
    approach, this avoids boundary artifacts and is faster when many crops share
    one image.

    Parameters
    ----------
    f0
        Full-resolution reference frame.
    f1
        Full-resolution subsequent frame.
    sy
        1-D array of crop start-row indices.
    ey
        1-D array of crop end-row indices.
    sx
        1-D array of crop start-column indices.
    ex
        1-D array of crop end-column indices.
    crop_indices
        1-D array of integer crop identifiers.
    t0
        Timepoint index of the reference frame.
    dt
        Temporal stride between the two frames.
    thresh
        Intensity threshold for foreground masking.
    attachment
        TVL1 data-fidelity weight (λ).
    speed_threshold
        Speed threshold for fast-coherence features.

    Returns
    -------
    list[dict]
        One dictionary per crop containing scalar flow statistics
        (see :func:`compute_flow_statistics`).
    """
    u, v = compute_tvl1(f0, f1, attachment=attachment)
    n_crops = len(crop_indices)
    return [
        compute_flow_statistics(
            u[sy[i] : ey[i], sx[i] : ex[i]],
            v[sy[i] : ey[i], sx[i] : ex[i]],
            f0[sy[i] : ey[i], sx[i] : ex[i]],
            f1[sy[i] : ey[i], sx[i] : ex[i]],
            int(crop_indices[i]),
            t0,
            dt,
            thresh,
            speed_threshold,
        )
        for i in range(n_crops)
    ]


def calculate_optical_flow_intensity_threshold(intensity_percentile: float, images: list[np.ndarray]) -> float:

    if intensity_percentile <= 0:
        return -float("inf")

    return float(
        np.percentile(
            np.concatenate([image.ravel()[::10] for image in images]),
            intensity_percentile,
        )
    )
