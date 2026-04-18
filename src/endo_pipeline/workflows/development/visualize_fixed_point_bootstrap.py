from endo_pipeline.cli import CropPattern, Datasets


def main(
    crop_pattern: CropPattern = "grid",
    datasets: Datasets | None = None,
    bootstrap_threshold: float = 0.5,
) -> None:
    """
    Visualize bootstrap-validated fixed points from `bootstrap-fixed-points`.

    #dynamical-systems #diffae-feature-analysis #bootstrap

    **Overview**

    This workflow loads the bootstrap confidence interval (CI) dataframes
    produced by `generate-3d-flow-field-bootstrap`, filters to fixed points
    whose *bootstrap detection rate* meets or exceeds `bootstrap_threshold`, and
    plots their locations with per-coordinate confidence interval error bars.

    **Filtering**

    Only fixed points with `bootstrap_detection_rate >= bootstrap_threshold` are
    retained. The `bootstrap_detection_rate` is the fraction of bootstrap
    iterations in which a fixed point was detected within the specified
    `bootstrap_match_radius` of a baseline fixed point.

    **Visualizations**

    This workflow produces a two-panel plot for each dataset, showing the
    high-confidence fixed points in two 2D projections:

    - Top panel: PC1 (polar angle) vs PC2 (polar radius)
    - Bottom panel: PC1 (polar angle) vs PC3 (PC3-flipped)

    Fixed points are drawn as scatter markers coloured by stability
    classification (stable = blue circle, saddle = grey triangle, unstable = red
    square, indeterminate = yellow plus); markers are placed at the bootstrap
    cluster mean coordinate (mean of all matched bootstrap fixed point
    coordinates) and error bars show the per-coordinate bootstrap CIs at the
    percentiles used during the bootstrap run (`FP_CI_LOWER_PERCENTILE` and
    `FP_CI_UPPER_PERCENTILE`).

    If fewer than two bootstrap hits were obtained for a fixed point the CI is
    `nan` and no error bar is drawn. If high-confidence fixed points are found
    for two or more datasets a combined comparison figure coloured by dataset is
    also saved.

    Parameters
    ----------
    crop_pattern
        The crop pattern to load bootstrap fixed point dataframes for.
    datasets
        Optional list of specific datasets to visualize.
    bootstrap_threshold
        Minimum bootstrap detection rate for a fixed point to be included in the plots.

    """
    import logging

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.visualize.diffae_features.feature_viz import (
        get_dataset_color,
        get_label_for_column,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        BIN_LIMITS_THETA_RESCALED,
        DYNAMICS_COLUMN_NAMES,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_3d import (
        DATASET_COLLECTION_FOR_3D_DYNAMICS,
        FIGSIZE_2D_FLOW_FIELD,
        NROWS_2D_FLOW_FIELD,
    )
    from endo_pipeline.settings.flow_field_dataframes import (
        DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING,
        STABILITY_COLOR_DICT,
        STABILITY_MARKER_DICT,
        StabilityLabel,
        StabilityLegendHandle,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    column_names: list[Column.DiffAEData] = list(DYNAMICS_COLUMN_NAMES)

    base_name = f"{model_manifest_name}_{run_name}_{crop_pattern}"
    bootstrap_fp_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING}_{base_name}"

    # Flexible DEMO_MODE loading pattern: try without demo suffix first so this
    # workflow can visualise a full production run even when DEMO_MODE is set.
    # Fall back to the demo-suffixed manifest only in DEMO_MODE.
    try:
        bootstrap_fp_manifest = load_dataframe_manifest(bootstrap_fp_manifest_name)
    except FileNotFoundError:
        if DEMO_MODE:
            fallback_name = f"{DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING}_{base_name}_demo"
            logger.warning(
                "Bootstrap fixed point manifest [ %s ] not found; trying [ %s ].",
                bootstrap_fp_manifest_name,
                fallback_name,
            )
            bootstrap_fp_manifest = load_dataframe_manifest(fallback_name)
        else:
            raise

    n_bootstrap = bootstrap_fp_manifest.parameters.get("n_bootstrap_samples")
    if n_bootstrap is None:
        logger.warning(
            "Number of bootstrap samples not found in manifest parameters; "
            "bootstrap detection rates will be included in the plots but "
            "not the number of bootstrap samples."
        )

    dataset_names = datasets or get_datasets_in_collection(DATASET_COLLECTION_FOR_3D_DYNAMICS)
    if DEMO_MODE:
        logger.warning("DEMO MODE: Processing no more than two datasets for quick visualization.")
        dataset_names = dataset_names[: min(len(dataset_names), 2)]

    fig_savedir = get_output_path(__file__, crop_pattern)

    # Axis bounds from global bin limits, one tuple (min, max) per column
    bounds_for_plots = BIN_LIMITS_DYNAMICS.copy()
    if RESCALE_THETA:
        bounds_for_plots[Column.DiffAEData.POLAR_ANGLE] = BIN_LIMITS_THETA_RESCALED

    all_high_confidence_dfs: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # Per-dataset loop
    # ------------------------------------------------------------------
    for dataset_name in dataset_names:
        if dataset_name not in bootstrap_fp_manifest.locations:
            logger.warning(
                "No bootstrap fixed point dataframe found in manifest [ %s ] for dataset "
                "[ %s ]. Skipping.",
                bootstrap_fp_manifest.name,
                dataset_name,
            )
            continue

        logger.info("Loading bootstrap fixed point dataframe for dataset [ %s ].", dataset_name)
        location = get_dataframe_location_for_dataset(bootstrap_fp_manifest, dataset_name)
        df = load_dataframe(location, delay=False)

        n_total = len(df)
        high_confidence_df = df[
            df[Column.BootstrapAnalysis.DETECTION_RATE] >= bootstrap_threshold
        ].copy()

        if high_confidence_df.empty:
            logger.warning(
                "No fixed points with bootstrap_detection_rate >= %.2f for dataset "
                "[ %s ] (%d fixed points in total). Skipping plot for this dataset.",
                bootstrap_threshold,
                dataset_name,
                n_total,
            )
            continue

        logger.info(
            "Dataset [ %s ]: %d / %d fixed point(s) pass bootstrap threshold (>= %.2f).",
            dataset_name,
            len(high_confidence_df),
            n_total,
            bootstrap_threshold,
        )

        high_confidence_df[Column.DATASET] = dataset_name
        all_high_confidence_dfs.append(high_confidence_df)

        # Per-dataset figures
        fig, axes = plt.subplots(NROWS_2D_FLOW_FIELD, 1, figsize=FIGSIZE_2D_FLOW_FIELD)
        suptitle_str = f"{dataset_name} — Bootstrap-Validated Fixed Points\n(Detection Rate \u2265 {bootstrap_threshold:.2%}"
        suptitle_suffix = ")" if n_bootstrap is None else f", n bootstrap = {n_bootstrap})"
        fig.suptitle(f"{suptitle_str}{suptitle_suffix}")

        for ax, column_x, column_y in [
            (axes[0], column_names[0], column_names[1]),  # PC1 vs PC2
            (axes[1], column_names[0], column_names[2]),  # PC1 vs PC3
        ]:
            for _, row in high_confidence_df.iterrows():
                stability = row[Column.VectorField.STABILITY]
                detection_rate = row[Column.BootstrapAnalysis.DETECTION_RATE]
                print(
                    f"Processing fixed point with stability {stability} and detection rate {detection_rate:.2f}"
                )
                color = STABILITY_COLOR_DICT.get(stability, "gray")
                marker = STABILITY_MARKER_DICT.get(stability, "o")

                x = row[f"{column_x}_{Column.BootstrapAnalysis.CLUSTER_MEAN}"]
                y = row[f"{column_y}_{Column.BootstrapAnalysis.CLUSTER_MEAN}"]
                xlabel = get_label_for_column(column_x)
                ylabel = get_label_for_column(column_y)

                x_lo = row[f"{column_x}_{Column.BootstrapAnalysis.CI_LOWER}"]
                x_hi = row[f"{column_x}_{Column.BootstrapAnalysis.CI_UPPER}"]
                y_lo = row[f"{column_y}_{Column.BootstrapAnalysis.CI_LOWER}"]
                y_hi = row[f"{column_y}_{Column.BootstrapAnalysis.CI_UPPER}"]
                print(x_lo, x_hi, y_lo, y_hi)

                xerr = (
                    [[max(0.0, x - x_lo)], [max(0.0, x_hi - x)]]
                    if not (np.isnan(x_lo) or np.isnan(x_hi))
                    else None
                )
                yerr = (
                    [[max(0.0, y - y_lo)], [max(0.0, y_hi - y)]]
                    if not (np.isnan(y_lo) or np.isnan(y_hi))
                    else None
                )

                ax.errorbar(
                    x,
                    y,
                    xerr=xerr,
                    yerr=yerr,
                    fmt=marker,
                    color=color,
                    markeredgecolor="black",
                    markersize=8,
                    capsize=4,
                    elinewidth=1.2,
                    ecolor=color,
                    zorder=3,
                )

            ax.set_xlim(bounds_for_plots[column_x])
            ax.set_ylim(bounds_for_plots[column_y])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{xlabel} vs {ylabel}")

        # Legend from the stability labels present in this dataset
        present_stabilities = set(high_confidence_df[Column.VectorField.STABILITY].unique())
        legend_handles = [
            StabilityLegendHandle(stability_label=s)
            for s in StabilityLabel
            if s in present_stabilities
        ]
        if legend_handles:
            axes[-1].legend(handles=legend_handles, title="Stability", loc="best")

        plt.tight_layout()

        save_plot_to_path(fig, fig_savedir, f"bootstrap_fixed_points_ci_{dataset_name}")
        plt.close(fig)

    # Combined cross-dataset figure
    if len(all_high_confidence_dfs) < 2:
        logger.warning(
            "High-confidence fixed points found for fewer than two datasets; "
            "skipping combined comparison figure."
        )
        return

    combined_df = pd.concat(all_high_confidence_dfs, ignore_index=True)
    dataset_list = combined_df[Column.DATASET].unique().tolist()

    fig_combined, axes_combined = plt.subplots(
        NROWS_2D_FLOW_FIELD, 1, figsize=FIGSIZE_2D_FLOW_FIELD
    )
    suptitle_str = f"Bootstrap-Validated Stable Fixed Points — All Datasets\n(Detection Rate \u2265 {bootstrap_threshold:.2%}"
    fig_combined.suptitle(f"{suptitle_str}{suptitle_suffix}")

    for ax, column_x, column_y in [
        (axes_combined[0], column_names[0], column_names[1]),
        (axes_combined[1], column_names[0], column_names[2]),
    ]:
        for ds_name, ds_df in combined_df.groupby(Column.DATASET):
            ds_color = get_dataset_color(ds_name)
            for _, row in ds_df.iterrows():
                stability = row[Column.VectorField.STABILITY]
                # only plot stable fixed points in the combined figure for clearer comparison
                if stability != StabilityLabel.STABLE:
                    continue

                x = row[f"{column_x}_{Column.BootstrapAnalysis.CLUSTER_MEAN}"]
                y = row[f"{column_y}_{Column.BootstrapAnalysis.CLUSTER_MEAN}"]
                xlabel = get_label_for_column(column_x)
                ylabel = get_label_for_column(column_y)

                x_lo = row[f"{column_x}_{Column.BootstrapAnalysis.CI_LOWER}"]
                x_hi = row[f"{column_x}_{Column.BootstrapAnalysis.CI_UPPER}"]
                y_lo = row[f"{column_y}_{Column.BootstrapAnalysis.CI_LOWER}"]
                y_hi = row[f"{column_y}_{Column.BootstrapAnalysis.CI_UPPER}"]

                xerr = (
                    [[max(0.0, x - x_lo)], [max(0.0, x_hi - x)]]
                    if not (np.isnan(x_lo) or np.isnan(x_hi))
                    else None
                )
                yerr = (
                    [[max(0.0, y - y_lo)], [max(0.0, y_hi - y)]]
                    if not (np.isnan(y_lo) or np.isnan(y_hi))
                    else None
                )

                ax.errorbar(
                    x,
                    y,
                    xerr=xerr,
                    yerr=yerr,
                    fmt=marker,
                    color=ds_color,
                    markeredgecolor="black",
                    markersize=8,
                    capsize=4,
                    elinewidth=1.2,
                    ecolor=ds_color,
                    zorder=3,
                )

        ax.set_xlim(bounds_for_plots[column_x])
        ax.set_ylim(bounds_for_plots[column_y])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xlabel} vs {ylabel}")

    dataset_legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=get_dataset_color(ds),
            linestyle="",
            markeredgecolor="black",
            markersize=8,
            label=ds,
        )
        for ds in dataset_list
    ]
    axes_combined[-1].legend(
        handles=dataset_legend_handles, title="Dataset", loc="best", fontsize=6
    )

    save_plot_to_path(fig_combined, fig_savedir, "bootstrap_stable_fixed_points_ci_combined")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
