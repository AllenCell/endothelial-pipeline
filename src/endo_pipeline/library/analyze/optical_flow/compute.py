import logging

import numpy as np
from scipy import stats
from skimage.registration import optical_flow_tvl1

from endo_pipeline.settings.optical_flow import COHERENCE_BOX_SIZES, OPTICAL_FLOW_BASE_FEATURES

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
    tuple[np.ndarray, np.ndarray]
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
        Default False.

    Returns
    -------
        Flat dictionary of scalar statistics with identifying fields.
    """
    base: dict[str, int | float] = {"crop_index": crop_idx, "timepoint": timepoint, "dt": dt}
    mask = (crop0 > thresh) | (crop1 > thresh)
    if not mask.any():
        logger.debug(
            "No foreground pixels above thresh=%.3g for crop_idx=%d, timepoint=%d; returning NaNs.",
            thresh,
            crop_idx,
            timepoint,
        )
        base.update(dict.fromkeys(OPTICAL_FLOW_BASE_FEATURES, np.nan))
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
    base.update(
        {
            "optical_flow_mean_speed": float(sp.mean()),
            "optical_flow_mean_unit_vector": muv,
            "optical_flow_std_speed": float(sp.std()),
            "optical_flow_mean_angle": float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())),
            "optical_flow_angle_std": float(stats.circstd(ang)),
            "optical_flow_mean_u": float(um.mean()),
            "optical_flow_mean_v": float(vm.mean()),
            "optical_flow_std_u": float(um.std()),
            "optical_flow_std_v": float(vm.std()),
        }
    )

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
        flow fields.  Default ``7.5``.

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
        Intensity threshold for foreground masking.  Default ``0.0``.
    attachment
        TVL1 data-fidelity weight (λ).  Default ``7.5``.
    compute_block_coherence
        If True, compute multi-scale block-averaged coherence
        statistics.  Default False.

    Returns
    -------
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
        TVL1 data-fidelity weight (λ).  Default ``7.5``.
    compute_block_coherence
        If True, compute multi-scale block-averaged coherence
        statistics for each crop.  Default False.

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
        )
        for i in range(n_crops)
    ]
