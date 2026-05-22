"""DataFrame wrangling — crop grids, pivoting, column names."""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from endo_pipeline.library.analyze.optical_flow.compute import OpticalFlowImagePairCrops
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.image_data import DIFFAE_DEFAULT_CROP_SIZE
from endo_pipeline.settings.optical_flow import (
    DEFAULT_EMA_ALPHAS,
    OPTICAL_FLOW_BASE_FEATURES,
    OPTICAL_FLOW_EMA_STEMS,
)

OPTICAL_FLOW_INDEX_COLUMNS = [ColumnName.CROP_INDEX, ColumnName.TIMEPOINT]


def build_optical_flow_feature_cols(
    max_dt: int,
    ema_alphas: Sequence[float] = DEFAULT_EMA_ALPHAS,
) -> list[str]:
    """Return all optical-flow column names for dt = 1..max_dt.

    Generates the Cartesian product of base features and temporal stride given
    by `max_dt`, yielding names like ``optical_flow_mean_speed_dt1``.

    Parameters
    ----------
    max_dt
        Maximum temporal gap (inclusive).
    ema_alphas
        EMA smoothing alpha values.  Defaults to
        :data:`~endo_pipeline.settings.optical_flow.DEFAULT_EMA_ALPHAS`.

    Returns
    -------
    :
        List of ``{feature}_dt{d}`` column names.
    """
    # --- raw (non-EMA) features ---
    features = OPTICAL_FLOW_BASE_FEATURES

    # --- EMA-smoothed coherence columns ---
    ema_stems = OPTICAL_FLOW_EMA_STEMS

    ema_features: list[str] = []
    for alpha in ema_alphas:
        tag = str(alpha).replace(".", "")
        ema_features += [f"ema{tag}_{stem}" for stem in ema_stems]

    all_features = features + ema_features
    return [f"{f}_dt{d}" for d in range(1, max_dt + 1) for f in all_features]


def build_image_pair_crops_for_tracked(
    df: pd.DataFrame,
) -> Callable[[int], OpticalFlowImagePairCrops]:
    """
    Build image pair crop for tracked crops as a function of timepoint.

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

    crop_size = int(
        df[ColumnName.DiffAEData.CROP_SIZE_X].iloc[0]
        if ColumnName.DiffAEData.CROP_SIZE_X in df.columns
        else DIFFAE_DEFAULT_CROP_SIZE
    )

    tracked_crops: dict[int, OpticalFlowImagePairCrops] = {}

    for t, grp in df.groupby(ColumnName.TIMEPOINT):
        ci_ = grp[ColumnName.CROP_INDEX].values.astype(int)
        tracked_crops[int(t)] = OpticalFlowImagePairCrops(
            start_x=grp[ColumnName.DiffAEData.START_X].values.astype(int),
            start_y=grp[ColumnName.DiffAEData.START_Y].values.astype(int),
            crop_indices=ci_,
            crop_size=crop_size,
        )

    return lambda tp: tracked_crops[tp]


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------
def build_image_pair_crops_for_grid(df: pd.DataFrame) -> Callable[[int], OpticalFlowImagePairCrops]:
    """
    Build image pair crop for grid crops as a function of timepoint.

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

    crop_size = (
        int(df[ColumnName.DiffAEData.CROP_SIZE_X].iloc[0])
        if ColumnName.DiffAEData.CROP_SIZE_X in df.columns
        else DIFFAE_DEFAULT_CROP_SIZE
    )

    return lambda _: OpticalFlowImagePairCrops(
        start_x=crop_df[ColumnName.DiffAEData.START_X].values.astype(int),
        start_y=crop_df[ColumnName.DiffAEData.START_Y].values.astype(int),
        crop_indices=crop_df[ColumnName.CROP_INDEX].values.astype(int),
        crop_size=crop_size,
    )


def build_merged_optical_flow_dataframe(
    df_base: pd.DataFrame, records: list[dict], max_dt: int, ema_alphas: Sequence[float]
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
    ema_alphas
        EMA smoothing alpha values.

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
    all_feature_column_names = build_optical_flow_feature_cols(max_dt=max_dt, ema_alphas=ema_alphas)
    for col in all_feature_column_names:
        if col not in df_base.columns:
            df_base[col] = np.nan

    # Sort data by index columns
    df_base = df_base.sort_values(OPTICAL_FLOW_INDEX_COLUMNS)

    # Apply EMA smoothing
    for alpha in ema_alphas:
        alpha_tag = str(alpha).replace(".", "")
        for dt in range(1, max_dt + 1):
            for stem in OPTICAL_FLOW_EMA_STEMS:
                raw_col = f"{stem}_dt{dt}"
                ema_col = f"ema{alpha_tag}_{stem}_dt{dt}"

                if raw_col not in df_base.columns:
                    continue

                df_base[ema_col] = df_base.groupby(ColumnName.CROP_INDEX)[raw_col].transform(
                    lambda s, a=alpha: s.ewm(alpha=a, adjust=False).mean()
                )

    return df_base
