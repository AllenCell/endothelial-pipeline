"""DataFrame wrangling — crop grids, pivoting, column names."""

from collections.abc import Sequence

import pandas as pd

from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.optical_flow import COHERENCE_BOX_SIZES, OPTICAL_FLOW_BASE_FEATURES


# ---------------------------------------------------------------------------
# Feature column helpers
# ---------------------------------------------------------------------------
def build_optical_flow_feature_cols(
    max_dt: int,
    compute_block_coherence: bool = False,
    base_features: Sequence[str] = OPTICAL_FLOW_BASE_FEATURES,
    coherence_box_sizes: Sequence[int] = COHERENCE_BOX_SIZES,
) -> list[str]:
    """Return all optical-flow column names for dt = 1..max_dt.

    Generates the Cartesian product of *base_features* and temporal
    strides 1..max_dt, yielding names like
    ``optical_flow_mean_speed_dt1``.  When *compute_block_coherence* is
    True, also includes ``optical_flow_angle_std_box{N}_dt{d}`` columns.

    Parameters
    ----------
    max_dt
        Maximum temporal gap (inclusive).
    compute_block_coherence
        If True, include block-averaged coherence column names.
    base_features
        Base feature names.  Defaults to
        :data:`~endo_pipeline.settings.optical_flow.OPTICAL_FLOW_BASE_FEATURES`.
    coherence_box_sizes
        Box sizes for multi-scale coherence columns.  Defaults to
        :data:`~endo_pipeline.settings.optical_flow.COHERENCE_BOX_SIZES`.

    Returns
    -------
        List of ``{feature}_dt{d}`` column names.
    """
    features = list(base_features)
    if compute_block_coherence:
        features += [f"optical_flow_angle_std_box{box}" for box in coherence_box_sizes]
    return [f"{f}_dt{d}" for d in range(1, max_dt + 1) for f in features]


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
        One row per crop with columns ``START_X``, ``START_Y``,
        ``CROP_INDEX``, ``end_x``, and ``end_y``.
    """
    crop_df = (
        df[
            [
                ColumnName.DiffAEData.START_X,
                ColumnName.DiffAEData.START_Y,
                ColumnName.DiffAEData.CROP_INDEX,
            ]
        ]
        .drop_duplicates(subset=[ColumnName.DiffAEData.CROP_INDEX])
        .sort_values(by=[ColumnName.DiffAEData.START_Y, ColumnName.DiffAEData.START_X])
        .reset_index(drop=True)
    )
    sz = (
        int(df[ColumnName.DiffAEData.CROP_SIZE_X].iloc[0])
        if ColumnName.DiffAEData.CROP_SIZE_X in df.columns
        else 128
    )
    crop_df["end_x"] = crop_df[ColumnName.DiffAEData.START_X] + sz
    crop_df["end_y"] = crop_df[ColumnName.DiffAEData.START_Y] + sz
    return crop_df


# ---------------------------------------------------------------------------
# Pivot helper
# ---------------------------------------------------------------------------
def pivot_flow_records(records: list[dict]) -> pd.DataFrame:
    """Pivot a list of flow-stat dicts into a wide DataFrame.

    Each input dict has keys ``crop_index``, ``timepoint``, ``dt``, and
    one entry per base feature.  The function pivots on ``dt`` so that
    the output has one row per ``(crop_index, timepoint)`` and columns
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
    pd.DataFrame
        Wide-format DataFrame indexed by ``crop_index`` and
        ``timepoint``, with one column per feature-dt combination.
    """
    df = pd.DataFrame(records)
    index_cols = ["crop_index", "timepoint"]
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
