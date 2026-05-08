from typing import Literal

from endo_pipeline.cli import tags

TAGS = ["analysis", "regression", tags.CPU_ONLY]


def main(
    dataset_summary_list: Literal["low_high", "intermediate", "perturbation"] = "intermediate",
) -> None:
    """
    Regress migration coherence on structural stable-point features.

    Fits leave-one-dataset-out cross-validated regressions of migration
    coherence on structural features (polar_r, polar_theta, rho, and
    nematic_order = cos(2 * polar_theta)) at stable points, to assess
    whether migration coherence is determined by morphology.

    Parameters
    ----------
    dataset_summary_list
        Dataset collection used to assemble the fixed-point table. One of
        "low_high", "intermediate", or "perturbation".
    """
    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.migration_regression import (
        assemble_fixed_points_dataframe,
        estimate_noise_floor_mse,
        leave_one_dataset_out_regression,
        plot_predictions_scatter,
        plot_single_feature_scatter,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.flow_field_dataframes import (
        DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING,
        StabilityLabel,
    )
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
    from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    base_name = (
        f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{MIGRATION_COHERENCE_CROP_PATTERN}"
    )
    feature_dataframe_manifest = load_dataframe_manifest(f"{base_name}_pca_filtered")
    fixed_points_bootstrap_dataframe_manifest = load_dataframe_manifest(
        f"{DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING}_{base_name}"
    )

    datasets = SUMMARY_PLOT_DATASETS[dataset_summary_list]
    if DEMO_MODE:
        datasets = datasets[:1]
        logger.info("DEMO MODE, only processing first dataset [ %s ]", datasets[0])

    output_dir = get_output_path(__file__, dataset_summary_list)

    df_fp = assemble_fixed_points_dataframe(
        dataset_names=datasets,
        feature_dataframe_manifest=feature_dataframe_manifest,
        fixed_points_bootstrap_dataframe_manifest=fixed_points_bootstrap_dataframe_manifest,
    )
    logger.info("Assembled %d fixed points across %d datasets", len(df_fp), len(datasets))

    out_csv = output_dir / "stable_points_with_migration_coherence.csv"
    df_fp.to_csv(out_csv, index=False)
    logger.info("Wrote stable-point table to %s", out_csv)

    # Restrict to stable fixed points with a defined coherence target.
    from endo_pipeline.library.analyze.migration_regression import COHERENCE_COLUMN

    df_reg = df_fp[
        (df_fp[ColumnName.VectorField.STABILITY] == StabilityLabel.STABLE)
        & df_fp[COHERENCE_COLUMN].notna()
    ].reset_index(drop=True)
    n_datasets = df_reg[ColumnName.DATASET].nunique()
    if n_datasets < 2:
        logger.warning(
            "Skipping LOO-CV regression: only %d datasets with stable points "
            "(need >= 2). Likely running in demo mode.",
            n_datasets,
        )
        return

    benchmark, predictions = leave_one_dataset_out_regression(df_reg)
    noise_floor = estimate_noise_floor_mse(df_reg)
    benchmark["noise_floor_mse"] = noise_floor

    benchmark_csv = output_dir / "regression_benchmark.csv"
    predictions_csv = output_dir / "regression_predictions.csv"
    benchmark.to_csv(benchmark_csv, index=False)
    predictions.to_csv(predictions_csv, index=False)
    logger.info("Wrote regression benchmark to %s", benchmark_csv)
    logger.info("Wrote held-out predictions to %s", predictions_csv)
    logger.info(
        "Estimated coherence noise-floor MSE = %.5g (sqrt = %.4f)",
        noise_floor,
        float(noise_floor**0.5),
    )

    # save scatterplot of all features
    fig = plot_predictions_scatter(predictions, benchmark)
    save_plot_to_path(fig, output_dir, "regression_true_vs_predicted", tight_layout=False)

    # save scatterplots for select features
    for feature_set in benchmark["feature_set"].unique():
        fig = plot_single_feature_scatter(
            predictions[predictions["feature_set"] == feature_set],
            benchmark[benchmark["feature_set"] == feature_set],
            feature_set=feature_set,
        )
        save_plot_to_path(
            fig,
            output_dir,
            f"regression_true_vs_predicted_{feature_set}",
            tight_layout=False,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
