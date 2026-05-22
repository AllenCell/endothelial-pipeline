"""DataFrame wrangling — crop grids, pivoting, column names."""

from collections.abc import Sequence

import pandas as pd

from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.image_data import DIFFAE_DEFAULT_CROP_SIZE
from endo_pipeline.settings.optical_flow import (
    DEFAULT_EMA_ALPHAS,
    OPTICAL_FLOW_BASE_FEATURES,
    OPTICAL_FLOW_EMA_STEMS,
)


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


def build_tracked_crop_lookup_table(df: pd.DataFrame) -> dict[int, tuple]:
    """
    Build a per-timepoint crop lookup table mapping timepoint to coordinates.

    Coordinates are stored as: (start_y, end_y, start_x, end_x, crop_ids). This
    mapping is necessary because crop coordinates changes between timepoints.

    Parameters
    ----------
    df
        Dataframe containing crop locations for each timepoint.

    Returns
    -------
    :
        Map of timepoint to coordinate.
    """

    crop_size = int(
        df[ColumnName.DiffAEData.CROP_SIZE_X].iloc[0]
        if ColumnName.DiffAEData.CROP_SIZE_X in df.columns
        else DIFFAE_DEFAULT_CROP_SIZE
    )

    tracked_crops: dict[int, tuple] = {}

    for t, grp in df.groupby(ColumnName.TIMEPOINT):
        sx_ = grp[ColumnName.DiffAEData.START_X].values.astype(int)
        sy_ = grp[ColumnName.DiffAEData.START_Y].values.astype(int)
        ex_ = sx_ + crop_size
        ey_ = sy_ + crop_size
        ci_ = grp[ColumnName.CROP_INDEX].values
        tracked_crops[int(t)] = (sy_, ey_, sx_, ex_, ci_)

    return tracked_crops


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------
def build_crop_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unique crop index with spatial bounds.

    Extracts distinct crops from the input DataFrame, sorted by spatial
    position (top-to-bottom, left-to-right), and appends ``end_x`` /
    ``end_y`` columns computed from the crop start coordinates plus
    crop size.

    Parameters
    ----------
    df
        Feature DataFrame containing at least ``CROP_INDEX``,
        ``START_X``, ``START_Y``, and (optionally) ``CROP_SIZE_X``
        columns.

    Returns
    -------
    :
        One row per crop with columns ``START_X``, ``START_Y``,
        ``CROP_INDEX``, ``end_x``, and ``end_y``.
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
    sz = (
        int(df[ColumnName.DiffAEData.CROP_SIZE_X].iloc[0])
        if ColumnName.DiffAEData.CROP_SIZE_X in df.columns
        else 128
    )
    crop_df[ColumnName.DiffAEData.END_X] = crop_df[ColumnName.DiffAEData.START_X] + sz
    crop_df[ColumnName.DiffAEData.END_Y] = crop_df[ColumnName.DiffAEData.START_Y] + sz
    return crop_df


# ---------------------------------------------------------------------------
# Pivot helper
# ---------------------------------------------------------------------------
def pivot_flow_records(records: list[dict]) -> pd.DataFrame:
    """Pivot a list of flow-stat dicts into a wide DataFrame.

    Each input dict has keys ``crop_index``, ``frame_number``, ``dt``, and
    one entry per base feature.  The function pivots on ``dt`` so that
    the output has one row per ``(crop_index, frame_number)`` and columns
    named ``{feature}_dt{n}``.

    Block-coherence columns (``optical_flow_angle_std_box{N}``) are
    also pivoted when present in the records.

    Parameters
    ----------
    records
        List of dictionaries returned by
        :func:`~optical_flow.compute.compute_flow_statistics`.

    Returns
    -------
    :
        Wide-format DataFrame indexed by ``crop_index`` and
        ``frame_number``, with one column per feature-dt combination.
    """
    df = pd.DataFrame(records)
    index_cols = [ColumnName.CROP_INDEX, ColumnName.TIMEPOINT]
    # Discover all feature columns (everything except index + dt)
    feature_names = [c for c in df.columns if c not in (*index_cols, "dt")]
    parts = []
    for feat in feature_names:
        pv = df.pivot_table(
            index=index_cols,
            columns="dt",
            values=feat,
            aggfunc="first",
        )
        pv.columns = pd.Index([f"{feat}_dt{int(c)}" for c in pv.columns])
        parts.append(pv)
    return pd.concat(parts, axis=1).reset_index()
