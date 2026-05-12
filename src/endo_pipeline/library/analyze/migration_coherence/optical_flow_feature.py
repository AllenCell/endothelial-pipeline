import logging

import numpy as np
import pandas as pd

from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.optical_flow import build_optical_flow_feature_cols
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.migration_coherence import (
    MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
    OPTICAL_FLOW_DATAFRAME_MERGE_COLUMNS,
)
from endo_pipeline.settings.optical_flow import DEFAULT_OPTICAL_FLOW_MANIFEST_NAME

logger = logging.getLogger(__name__)


def add_optical_flow_features(
    df: pd.DataFrame,
    datasets: list[str] | None = None,
    optical_flow_manifest_name: str = f"{DEFAULT_OPTICAL_FLOW_MANIFEST_NAME}_grid",
    optical_flow_feature_columns: list[ColumnName.OpticalFlow] | None = None,
    merge_columns: list[str | ColumnName.DiffAEData] | None = None,
) -> pd.DataFrame:
    """
    Load optical-flow features and merge them with an existing dataframe.

    **Dataframe column requirements**

    The input dataframe must contain the columns specified in `merge_columns`
    and the optical flow dataframe(s) must contain the columns specified in both
    `merge_columns` and `optical_flow_feature_columns`. If not provided, these
    parameters will default to `OPTICAL_FLOW_DATAFRAME_MERGE_COLUMNS` and
    `OPTICAL_FLOW_BASE_FEATURES`, respectively.


    Parameters
    ----------
    df
        Input dataframe containing rows to enrich with optical-flow features.
    datasets
        Dataset names to process.
    optical_flow_manifest_name
        Name of the dataframe manifest containing optical-flow feature tables.
    optical_flow_feature_columns
        Optional, list of optical flow feature column names to merge into the
        input dataframe.
    merge_columns
        Optional, list of column names to merge on.

    Returns
    -------
    :
        Concatenated dataframe with optical-flow features merged in.
    """
    if datasets is None:
        datasets = df[ColumnName.DATASET].unique().tolist()

    merge_columns_ = merge_columns or list(OPTICAL_FLOW_DATAFRAME_MERGE_COLUMNS)
    optical_flow_feature_columns_ = optical_flow_feature_columns or build_optical_flow_feature_cols(
        max_dt=1,
        compute_fast_coherence=True,
        compute_radial_coherence=True,
    )
    required_columns = merge_columns_ + optical_flow_feature_columns_
    check_required_columns_in_dataframe(df, merge_columns_)
    dataframe_manifest_optical_flow = load_dataframe_manifest(optical_flow_manifest_name)

    merged_dfs = []
    for dataset_name in datasets:
        logger.info("Adding optical flow features for dataset: %s", dataset_name)

        df_dataset = df[df[ColumnName.DATASET] == dataset_name]

        optical_flow_location = get_dataframe_location_for_dataset(
            dataframe_manifest_optical_flow, dataset_name
        )
        df_optical_flow = load_dataframe(optical_flow_location)
        check_required_columns_in_dataframe(df_optical_flow, required_columns)
        df_optical_flow = df_optical_flow[required_columns]

        # TODO: fix duplicate rows at the source (optical flow workflow)
        df_optical_flow = df_optical_flow.drop_duplicates(subset=merge_columns_)

        df_merged = df_dataset.merge(df_optical_flow, on=merge_columns_, how="left")
        merged_dfs.append(df_merged)

    return pd.concat(merged_dfs, ignore_index=True)


def add_binned_mean_to_fixed_points(
    df_fp: pd.DataFrame,
    df_of: pd.DataFrame,
    fp_x_col: str,
    fp_y_col: str,
    fp_z_col: str,
    binned_col: str,
    of_x_col: str | None = None,
    of_y_col: str | None = None,
    of_z_col: str | None = None,
    bin_size_xyz: tuple[float, float, float] = (MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,) * 3,
    n_bootstrap: int | None = 500,
    ci_lower_percentile: float = 5,
    ci_upper_percentile: float = 95,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Compute the mean of *binned_col* in the 3D bin surrounding each fixed point.

    For each row in *df_fp*, the function finds all rows in *df_of* that fall
    within a bin of size *bin_size_xyz* centered on the fixed point's
    coordinates and computes the mean of *binned_col* over those rows.

    When *n_bootstrap* is provided, bootstrap confidence intervals for the
    binned mean are also computed and added as additional columns.

    The fixed-points dataframe and optical-flow dataframe may use different
    column names for the same spatial axes. When the ``of_*`` column
    parameters are ``None`` they fall back to the corresponding ``fp_*``
    column.

    Parameters
    ----------
    df_fp
        Fixed-points dataframe with columns *fp_x_col*, *fp_y_col*,
        *fp_z_col*.
    df_of
        Optical-flow dataframe with columns *of_x_col*, *of_y_col*,
        *of_z_col*, and *binned_col*.
    fp_x_col, fp_y_col, fp_z_col
        Column names for the three spatial axes in *df_fp*.
    binned_col
        Column name whose bin-mean is computed.
    of_x_col, of_y_col, of_z_col
        Column names for the three spatial axes in *df_of*.  When ``None``,
        defaults to the corresponding ``fp_*`` column name.
    bin_size_xyz
        Half-widths ``(dx, dy, dz)`` defining the bin extent around each
        fixed point.
    n_bootstrap
        Number of bootstrap resamples for confidence intervals.  When
        ``None`` (default), no CI columns are added.
    ci_lower_percentile
        Lower percentile for the confidence interval.
    ci_upper_percentile
        Upper percentile for the confidence interval.
    seed
        Random seed for reproducibility of bootstrap resampling.

    Returns
    -------
    :
        A copy of *df_fp* with an additional column ``mean_{binned_col}``
        containing the mean value of *binned_col* in each fixed point's bin.
        When *n_bootstrap* is set, also adds
        ``mean_{binned_col}_ci_lower`` and ``mean_{binned_col}_ci_upper``.
    """
    of_x = of_x_col if of_x_col is not None else fp_x_col
    of_y = of_y_col if of_y_col is not None else fp_y_col
    of_z = of_z_col if of_z_col is not None else fp_z_col

    compute_ci = n_bootstrap is not None
    rng = np.random.default_rng(seed) if compute_ci else None

    dx, dy, dz = bin_size_xyz
    means: list[float] = []
    ci_lows: list[float] = []
    ci_highs: list[float] = []

    for _, row in df_fp.iterrows():
        mask = (
            (df_of[of_x] >= row[fp_x_col] - dx / 2)
            & (df_of[of_x] < row[fp_x_col] + dx / 2)
            & (df_of[of_y] >= row[fp_y_col] - dy / 2)
            & (df_of[of_y] < row[fp_y_col] + dy / 2)
            & (df_of[of_z] >= row[fp_z_col] - dz / 2)
            & (df_of[of_z] < row[fp_z_col] + dz / 2)
        )
        bin_values = df_of.loc[mask, binned_col].dropna().values
        means.append(float(bin_values.mean()) if len(bin_values) > 0 else np.nan)

        if compute_ci:
            assert rng is not None
            assert n_bootstrap is not None
            if len(bin_values) < 2:
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)
            else:
                boot_means = np.array(
                    [
                        rng.choice(bin_values, size=len(bin_values), replace=True).mean()
                        for _ in range(n_bootstrap)
                    ]
                )
                ci_lows.append(np.percentile(boot_means, ci_lower_percentile))
                ci_highs.append(np.percentile(boot_means, ci_upper_percentile))

    mean_col = f"mean_{binned_col}"
    result = df_fp.copy()
    result[mean_col] = means
    if compute_ci:
        result[f"{mean_col}_{ColumnName.BootstrapAnalysis.CI_LOWER}"] = ci_lows
        result[f"{mean_col}_{ColumnName.BootstrapAnalysis.CI_UPPER}"] = ci_highs
    return result
