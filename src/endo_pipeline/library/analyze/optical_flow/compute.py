import logging

import numpy as np
from scipy import stats
from skimage.registration import optical_flow_tvl1

from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.optical_flow import (
    COHERENCE_BOX_SIZES,
    OPTICAL_FLOW_COMPUTE_FEATURES,
    OPTICAL_FLOW_FAST_FEATURES,
    OPTICAL_FLOW_RADIAL_FEATURES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _block_average_flow(
    u: np.ndarray,
    v: np.ndarray,
    box: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average (u, v) flow vectors within non-overlapping box*box blocks.

    Parameters
    ----------
    u, v
        Flow components, shape ``(H, W)``.
    box
        Side length of each square block.  When ``box == 1`` the arrays
        are returned unchanged.

    Returns
    -------
    :
        Block-averaged ``(u, v)`` arrays, shape ``(H // box, W // box)``.
    """
    if box == 1:
        return u, v
    H, W = u.shape
    Ht = (H // box) * box
    Wt = (W // box) * box
    u_b = u[:Ht, :Wt].reshape(Ht // box, box, Wt // box, box).mean(axis=(1, 3))
    v_b = v[:Ht, :Wt].reshape(Ht // box, box, Wt // box, box).mean(axis=(1, 3))
    return u_b, v_b


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
    compute_block_coherence: bool = False,
    compute_fast_coherence: bool = False,
    compute_radial_coherence: bool = False,
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
    compute_block_coherence
        If True, compute multi-scale block-averaged coherence
        (``optical_flow_angle_std_box{N}``) for each box size.
    compute_fast_coherence
        If True, compute coherence metrics only over pixels whose
        speed exceeds *speed_threshold*.
    compute_radial_coherence
        If True, compute radial coherence metrics (dot product of
        unit flow with unit radial vector from crop centre).
    speed_threshold
        Minimum pixel speed for the "fast" coherence features.
        Only used when *compute_fast_coherence* is True.

    Returns
    -------
    :
        Flat dictionary of scalar statistics with identifying fields.
    """
    base: dict[str, int | float] = {
        ColumnName.CROP_INDEX: crop_idx,
        "timepoint": timepoint,
        "dt": dt,
    }
    mask = (crop0 > thresh) | (crop1 > thresh)

    # Build the NaN key set dynamically based on enabled features.
    nan_keys: list[str] = list(OPTICAL_FLOW_COMPUTE_FEATURES)
    if compute_fast_coherence:
        nan_keys += OPTICAL_FLOW_FAST_FEATURES
    if compute_radial_coherence:
        nan_keys += OPTICAL_FLOW_RADIAL_FEATURES

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
    if compute_fast_coherence:
        fast = sp > speed_threshold
        n_fast = int(fast.sum())
        if fast.any():
            muv_fast = float(
                np.sqrt(np.mean(um[fast] / sp[fast]) ** 2 + np.mean(vm[fast] / sp[fast]) ** 2)
            )
        else:
            muv_fast = np.nan

    # --- Radial coherence ---
    if compute_radial_coherence:
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
            ColumnName.OpticalFlowCompute.SPEED_MEAN: float(sp.mean()),
            ColumnName.OpticalFlowCompute.UNIT_VECTOR_MEAN: muv,
            ColumnName.OpticalFlowCompute.SPEED_STD: float(sp.std()),
            ColumnName.OpticalFlowCompute.ANGLE_MEAN: float(
                np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
            ),
            ColumnName.OpticalFlowCompute.ANGLE_STD: float(stats.circstd(ang)),
            ColumnName.OpticalFlowCompute.U_MEAN: float(um.mean()),
            ColumnName.OpticalFlowCompute.V_MEAN: float(vm.mean()),
            ColumnName.OpticalFlowCompute.U_STD: float(um.std()),
            ColumnName.OpticalFlowCompute.V_STD: float(vm.std()),
        }
    )

    if compute_fast_coherence:
        base[ColumnName.OpticalFlowCompute.SPEED_ABOVE_1_COUNT] = n_fast
        base[ColumnName.OpticalFlowCompute.UNIT_VECTOR_MEAN_FAST] = muv_fast

    if compute_radial_coherence:
        base[ColumnName.OpticalFlowCompute.RADIAL_COHERENCE] = radial_coh
        base[ColumnName.OpticalFlowCompute.RADIAL_COHERENCE_WEIGHTED] = radial_coh_w

    # --- Optional Multi-scale coherence ---
    if compute_block_coherence:
        u_2d = u.copy()
        v_2d = v.copy()
        u_2d[~mask] = 0.0
        v_2d[~mask] = 0.0

        for box in COHERENCE_BOX_SIZES:
            ub, vb = _block_average_flow(u_2d, v_2d, box)
            sp_b = np.sqrt(ub**2 + vb**2)
            ang_b = np.arctan2(vb, ub)
            nz_b = sp_b > 0
            key = f"optical_flow_angle_std_box{box}"
            if nz_b.any():
                base[key] = float(stats.circstd(ang_b[nz_b]))
            else:
                base[key] = np.nan

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


def compute_crop_flow(
    c0: np.ndarray,
    c1: np.ndarray,
    crop_idx: int,
    tp: int,
    dt: int,
    thresh: float = 0.0,
    attachment: float = 7.5,
    compute_block_coherence: bool = False,
    compute_fast_coherence: bool = False,
    compute_radial_coherence: bool = False,
    speed_threshold: float = 1.0,
) -> dict:
    """Run TVL1 on a single crop pair and return summary statistics.

    Convenience wrapper that calls :func:`compute_tvl1` followed by
    :func:`compute_flow_statistics` for the *crop-scope* strategy
    (TVL1 is run independently on each crop).

    Parameters
    ----------
    c0
        Crop from the reference frame.
    c1
        Crop from the subsequent frame.
    crop_idx
        Spatial crop identifier.
    tp
        Timepoint index of the reference frame.
    dt
        Temporal stride between the two frames.
    thresh
        Intensity threshold for foreground masking.
    attachment
        TVL1 data-fidelity weight (λ).
    compute_block_coherence
        If True, compute multi-scale block-averaged coherence
        statistics.
    compute_fast_coherence
        If True, compute speed-thresholded coherence.
    compute_radial_coherence
        If True, compute radial coherence.
    speed_threshold
        Speed threshold for fast-coherence features.

    Returns
    -------
    :
        Flat dictionary of scalar flow statistics keyed by feature
        name, including ``crop_index``, ``timepoint``, and ``dt``.
    """
    u, v = compute_tvl1(c0, c1, attachment=attachment)
    return compute_flow_statistics(
        u,
        v,
        c0,
        c1,
        crop_idx,
        tp,
        dt,
        thresh,
        compute_block_coherence,
        compute_fast_coherence,
        compute_radial_coherence,
        speed_threshold,
    )


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
    compute_block_coherence: bool = False,
    compute_fast_coherence: bool = False,
    compute_radial_coherence: bool = False,
    speed_threshold: float = 1.0,
) -> list[dict]:
    """Run TVL1 on a full-resolution frame pair, then compute per-crop stats.

    This is the *image-scope* strategy: TVL1 runs once on the full image
    and the resulting flow field is sliced per crop.  Compared to the
    crop-scope approach (:func:`compute_crop_flow`), this avoids
    boundary artifacts and is faster when many crops share one image.

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
    compute_block_coherence
        If True, compute multi-scale block-averaged coherence
        statistics for each crop.
    compute_fast_coherence
        If True, compute speed-thresholded coherence.
    compute_radial_coherence
        If True, compute radial coherence.
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
            compute_block_coherence,
            compute_fast_coherence,
            compute_radial_coherence,
            speed_threshold,
        )
        for i in range(n_crops)
    ]
