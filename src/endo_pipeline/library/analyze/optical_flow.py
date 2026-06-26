import logging
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from skimage.registration import optical_flow_tvl1

from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
from endo_pipeline.settings.image_data import DIFFAE_DEFAULT_CROP_SIZE
from endo_pipeline.settings.optical_flow import DEFAULT_EMA_ALPHA

logger = logging.getLogger(__name__)


OPTICAL_FLOW_INDEX_COLUMNS = [ColumnName.CROP_INDEX, ColumnName.TIMEPOINT]
"""Optical flow dataframe index columns."""


class OpticalFlowImagePair(NamedTuple):
    """Structure for optical flow image pair."""

    t0: int
    t1: int
    dt: int


class OpticalFlowImagePairCrops(NamedTuple):
    """Structure for image pair crops."""

    start_x: np.ndarray
    start_y: np.ndarray
    crop_indices: np.ndarray
    crop_size: int


def compute_flow_statistics(
    u: np.ndarray,
    v: np.ndarray,
    crop0: np.ndarray,
    crop1: np.ndarray,
    crop_idx: int,
    timepoint: int,
    dt: int,
    intensity_threshold: float,
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
    intensity_threshold
        Intensity threshold for foreground masking.

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
    mask = (crop0 > intensity_threshold) | (crop1 > intensity_threshold)

    if not mask.any():
        logger.debug(
            "No foreground pixels above thresh=%.3g for crop_idx=%d, timepoint=%d; returning NaNs.",
            intensity_threshold,
            crop_idx,
            timepoint,
        )
        base.update(
            {
                ColumnTemplate.OPTICAL_FLOW_SPEED_MEAN: np.nan,
                ColumnTemplate.OPTICAL_FLOW_MEAN_UNIT_VECTOR: np.nan,
                ColumnTemplate.OPTICAL_FLOW_SPEED_STD: np.nan,
                ColumnTemplate.OPTICAL_FLOW_ANGLE_MEAN: np.nan,
                ColumnTemplate.OPTICAL_FLOW_ANGLE_STD: np.nan,
                ColumnTemplate.OPTICAL_FLOW_U_MEAN: np.nan,
                ColumnTemplate.OPTICAL_FLOW_V_MEAN: np.nan,
                ColumnTemplate.OPTICAL_FLOW_U_STD: np.nan,
                ColumnTemplate.OPTICAL_FLOW_V_STD: np.nan,
            }
        )
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
            ColumnTemplate.OPTICAL_FLOW_SPEED_MEAN: float(sp.mean()),
            ColumnTemplate.OPTICAL_FLOW_MEAN_UNIT_VECTOR: muv,
            ColumnTemplate.OPTICAL_FLOW_SPEED_STD: float(sp.std()),
            ColumnTemplate.OPTICAL_FLOW_ANGLE_MEAN: float(
                np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
            ),
            ColumnTemplate.OPTICAL_FLOW_ANGLE_STD: float(stats.circstd(ang)),
            ColumnTemplate.OPTICAL_FLOW_U_MEAN: float(um.mean()),
            ColumnTemplate.OPTICAL_FLOW_V_MEAN: float(vm.mean()),
            ColumnTemplate.OPTICAL_FLOW_U_STD: float(um.std()),
            ColumnTemplate.OPTICAL_FLOW_V_STD: float(vm.std()),
        }
    )

    return base


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
    image_pair: OpticalFlowImagePair,
    crops: OpticalFlowImagePairCrops,
    intensity_threshold: float,
    attachment: float = 7.5,
) -> list[dict]:
    """
    Run TVL1 on a full-resolution frame pair, then compute per-crop stats.

    TVL1 runs once on the full image and the resulting flow field is sliced per
    crop, which avoids boundary artifacts and is faster when many crops share
    one image.

    Parameters
    ----------
    f0
        Full-resolution reference frame.
    f1
        Full-resolution subsequent frame.
    image_pair
        Timepoints and temporal stride for image pair.
    crops
        Crop indices and coordinates for image pair.
    intensity_threshold
        Intensity threshold for foreground masking.
    attachment
        TVL1 data-fidelity weight (λ).

    Returns
    -------
    list[dict]
        One dictionary per crop containing scalar flow statistics
        (see :func:`compute_flow_statistics`).
    """

    sx = crops.start_x
    sy = crops.start_y
    ex = crops.start_x + crops.crop_size
    ey = crops.start_y + crops.crop_size
    crop_indices = crops.crop_indices

    u, v = compute_tvl1(f0, f1, attachment=attachment)
    n_crops = len(crop_indices)

    return [
        compute_flow_statistics(
            u[sy[i] : ey[i], sx[i] : ex[i]],
            v[sy[i] : ey[i], sx[i] : ex[i]],
            f0[sy[i] : ey[i], sx[i] : ex[i]],
            f1[sy[i] : ey[i], sx[i] : ex[i]],
            crop_indices[i],
            image_pair.t0,
            image_pair.dt,
            intensity_threshold,
        )
        for i in range(n_crops)
    ]


def calculate_optical_flow_intensity_threshold(
    intensity_percentile: float, images: list[np.ndarray]
) -> float:
    """
    Calculate intensity threshold for across images for given percentile.

    Parameters
    ----------
    intensity_percentile
        Intensity percentile thresholds for optical flow masking.
    images
        List of image arrays.
    """

    if intensity_percentile <= 0:
        return -float("inf")

    return float(
        np.percentile(
            np.concatenate([image.ravel()[::10] for image in images]),
            intensity_percentile,
        )
    )


def build_optical_flow_feature_cols(
    max_dt: int,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
) -> list[str]:
    """Return all optical-flow column names for dt = 1..max_dt.

    Generates the Cartesian product of base features and temporal stride given
    by `max_dt`, yielding names like ``optical_flow_mean_speed_dt1``.

    Parameters
    ----------
    max_dt
        Maximum temporal gap (inclusive).
    ema_alpha
        EMA smoothing alpha value.

    Returns
    -------
    :
        List of optical flow column names.
    """

    raw_feature_templates = [
        ColumnTemplate.OPTICAL_FLOW_SPEED_MEAN,
        ColumnTemplate.OPTICAL_FLOW_SPEED_STD,
        ColumnTemplate.OPTICAL_FLOW_MEAN_UNIT_VECTOR,
        ColumnTemplate.OPTICAL_FLOW_ANGLE_MEAN,
        ColumnTemplate.OPTICAL_FLOW_ANGLE_STD,
        ColumnTemplate.OPTICAL_FLOW_U_MEAN,
        ColumnTemplate.OPTICAL_FLOW_V_MEAN,
        ColumnTemplate.OPTICAL_FLOW_U_STD,
        ColumnTemplate.OPTICAL_FLOW_V_STD,
    ]
    raw_features = [
        template % dt for dt in range(1, max_dt + 1) for template in raw_feature_templates
    ]

    ema_feature_templates = [
        ColumnTemplate.OPTICAL_FLOW_EMA_MEAN_UNIT_VECTOR,
    ]
    ema_str = str(ema_alpha).replace(".", "")
    ema_features = [
        template % (ema_str, dt)
        for dt in range(1, max_dt + 1)
        for template in ema_feature_templates
    ]

    return raw_features + ema_features


def build_image_pair_crops_for_cell_centered(
    df: pd.DataFrame,
) -> Callable[[int], OpticalFlowImagePairCrops]:
    """
    Build image pair crop for cell-centered patches as a function of timepoint.

    For tracked crops, crop coordinates change between timepoints. Return
    callable is a wrapper around the dictionary mapping timepoint to crops.

    Parameters
    ----------
    df
        Dataframe containing grid-based features.

    Returns
    -------
    :
        Callable for image pair crop tuple.
    """

    crop_size = (
        int(df[ColumnName.DiffAEData.CROP_SIZE_X].iloc[0])
        if ColumnName.DiffAEData.CROP_SIZE_X in df.columns
        else DIFFAE_DEFAULT_CROP_SIZE
    )

    tracked_crops: dict[int, OpticalFlowImagePairCrops] = {}

    for t, grp in df.groupby(ColumnName.TIMEPOINT):
        tracked_crops[int(t)] = OpticalFlowImagePairCrops(
            start_x=grp[ColumnName.DiffAEData.START_X].values.astype(int),
            start_y=grp[ColumnName.DiffAEData.START_Y].values.astype(int),
            crop_indices=grp[ColumnName.CROP_INDEX].values.astype(int),
            crop_size=crop_size,
        )

    return lambda tp: tracked_crops[tp]


def build_image_pair_crops_for_grid_based(
    df: pd.DataFrame,
) -> Callable[[int], OpticalFlowImagePairCrops]:
    """
    Build image pair crop for grid-based patches as a function of timepoint.

    For grid crops, crop coordinates and indices are the same across timepoints
    so returned callable is just a wrapper around a single object. Dataframe
    must include `CROP_INDEX`, `START_X`, `START_Y` columns. A `CROP_SIZE_X`
    column is optional.

    Parameters
    ----------
    df
        Dataframe containing grid-based features.

    Returns
    -------
    :
        Callable for image pair crop tuple.
    """

    crop_size = (
        int(df[ColumnName.DiffAEData.CROP_SIZE_X].iloc[0])
        if ColumnName.DiffAEData.CROP_SIZE_X in df.columns
        else DIFFAE_DEFAULT_CROP_SIZE
    )

    crop_df = (
        df[
            [
                ColumnName.DiffAEData.START_X,
                ColumnName.DiffAEData.START_Y,
                ColumnName.CROP_INDEX,
            ]
        ]
        .drop_duplicates(subset=[ColumnName.CROP_INDEX])
        .sort_values(by=[ColumnName.DiffAEData.START_Y, ColumnName.DiffAEData.START_X])
        .reset_index(drop=True)
    )

    return lambda _: OpticalFlowImagePairCrops(
        start_x=crop_df[ColumnName.DiffAEData.START_X].values.astype(int),
        start_y=crop_df[ColumnName.DiffAEData.START_Y].values.astype(int),
        crop_indices=crop_df[ColumnName.CROP_INDEX].values.astype(int),
        crop_size=crop_size,
    )


def build_merged_optical_flow_dataframe(
    df_base: pd.DataFrame, records: list[dict], max_dt: int, ema_alpha: float
) -> pd.DataFrame:
    """
    Build merged dataframe from optical flow records with EMA smoothing.

    Parameters
    ----------
    df_base
        Base dataframe to merge features into.
    records
        List of records of optical flow features.
    max_dt
        Maximum temporal gap (inclusive).
    ema_alpha
        EMA smoothing alpha value.

    Returns
    -------
    :
        Merged optical flow dataframe.
    """

    # Merge records into a dataframe
    df_records = pd.DataFrame(records)
    feature_names = [c for c in df_records.columns if c not in (*OPTICAL_FLOW_INDEX_COLUMNS, "dt")]

    # Pivot dataframe on the dt field
    parts = []
    for feat in feature_names:
        pv = df_records.pivot_table(
            index=OPTICAL_FLOW_INDEX_COLUMNS,
            columns="dt",
            values=feat,
            aggfunc="first",
        )
        pv.columns = pd.Index([f"{feat}_dt{int(c)}" for c in pv.columns])
        parts.append(pv)

    df_pivoted = pd.concat(parts, axis=1).reset_index()

    # Merge pivoted records with original dataframe
    df_base = df_base.merge(
        df_pivoted,
        left_on=OPTICAL_FLOW_INDEX_COLUMNS,
        right_on=OPTICAL_FLOW_INDEX_COLUMNS,
        how="left",
    )

    # Fill in any missing columns with NaN
    all_feature_column_names = build_optical_flow_feature_cols(max_dt=max_dt, ema_alpha=ema_alpha)
    for col in all_feature_column_names:
        if col not in df_base.columns:
            df_base[col] = np.nan

    # Sort data by index columns
    df_base = df_base.sort_values(OPTICAL_FLOW_INDEX_COLUMNS)

    # Apply EMA smoothing
    alpha_tag = str(ema_alpha).replace(".", "")
    for dt in range(1, max_dt + 1):
        raw_col = ColumnTemplate.OPTICAL_FLOW_MEAN_UNIT_VECTOR % dt
        ema_col = ColumnTemplate.OPTICAL_FLOW_EMA_MEAN_UNIT_VECTOR % (alpha_tag, dt)

        if raw_col not in df_base.columns:
            continue

        df_base[ema_col] = df_base.groupby(ColumnName.CROP_INDEX)[raw_col].transform(
            lambda s, a=ema_alpha: s.ewm(alpha=a, adjust=False).mean()
        )

    return df_base
