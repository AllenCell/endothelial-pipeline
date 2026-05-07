"""Assemble per-fixed-point tables for migration-coherence regression analysis."""

from __future__ import annotations

import logging

import pandas as pd

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_by_shear_stress,
    filter_dataframe_to_flow_condition_by_timepoint,
    filter_dataframe_to_steady_state,
)
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_optical_flow_features,
)
from endo_pipeline.library.visualize.summary_plot import (
    _process_bootstrap_dataframe_for_plot,
)
from endo_pipeline.manifests import DataframeManifest
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    METADATA_COLUMNS_TO_KEEP,
)

logger = logging.getLogger(__name__)


def assemble_fixed_points_dataframe(
    dataset_names: list[str],
    feature_dataframe_manifest: DataframeManifest,
    fixed_points_bootstrap_dataframe_manifest: DataframeManifest,
    bootstrap_threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Assemble per-fixed-point dataframe across datasets and flow conditions.

    For each ``(dataset, flow_condition)`` tuple this loads the per-dataset
    feature dataframe, adds optical-flow features, loads the corresponding
    bootstrapped fixed points, filters them by ``bootstrap_threshold``, and
    enriches each fixed point with binned-mean optical-flow values (migration
    coherence and speed) and the nematic-order column. The resulting rows are
    concatenated across all datasets/conditions.

    Replicates the data-assembly portion of
    :func:`endo_pipeline.library.visualize.summary_plot.plot_cross_dataset_summaries`
    so the same per-fixed-point table can be used for downstream regression.

    Parameters
    ----------
    dataset_names
        Datasets to include.
    feature_dataframe_manifest
        Manifest of per-dataset feature dataframes (PCA-filtered).
    fixed_points_bootstrap_dataframe_manifest
        Manifest of per-dataset bootstrapped-fixed-points dataframes.
    bootstrap_threshold
        Minimum bootstrap detection rate to retain a fixed point.

    Returns
    -------
    pandas.DataFrame
        One row per high-confidence fixed point per flow condition, with
        cluster-mean structural columns (``polar_angle_cluster_mean``,
        ``polar_radius_cluster_mean``, ``pc3_flipped_cluster_mean``,
        ``nematic_order_cluster_mean``), confidence-interval columns, and
        binned-mean optical-flow features (``mean_optical_flow_*``).
    """
    column_names = [
        ColumnName.DiffAEData.POLAR_ANGLE,
        ColumnName.DiffAEData.POLAR_RADIUS,
        ColumnName.DiffAEData.PC3_FLIPPED,
    ]
    optical_flow_features = [
        ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
        ColumnName.OpticalFlow.SPEED_MEAN,
    ]

    df_fp_all_list: list[pd.DataFrame] = []

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning("No feature dataframe for [ %s ]. Skipping.", dataset_name)
            continue
        if dataset_name not in fixed_points_bootstrap_dataframe_manifest.locations:
            logger.warning("No fixed-point bootstrap dataframe for [ %s ]. Skipping.", dataset_name)
            continue

        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        columns_to_compute = [*METADATA_COLUMNS_TO_KEEP["grid"], *DYNAMICS_COLUMN_NAMES]
        df = df_[columns_to_compute].compute()
        dataset_config = load_dataset_config(dataset_name)
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)
        df_of = add_optical_flow_features(df_steady_state, datasets=[dataset_name])

        df_bootstrap = load_dataframe(
            fixed_points_bootstrap_dataframe_manifest.locations[dataset_name], delay=False
        )

        for flow_condition in dataset_config.flow_conditions:
            df_flow = filter_dataframe_to_flow_condition_by_timepoint(
                df_of, dataset_config, flow_condition
            )
            df_bootstrap_flow = filter_dataframe_by_shear_stress(
                df_bootstrap, flow_condition.shear_stress
            )
            df_fp = _process_bootstrap_dataframe_for_plot(
                df_bootstrap_flow,
                df_flow,
                bootstrap_threshold,
                dataset_name,
                flow_condition,
                optical_flow_features,
                convert_angle_to_nematic=True,
                column_names=column_names,
                x_axis_mode="dataset",
                dataset_config=dataset_config,
            )
            if not df_fp.empty:
                df_fp_all_list.append(df_fp)

    if not df_fp_all_list:
        return pd.DataFrame()
    return pd.concat(df_fp_all_list, ignore_index=True)
