from typing import Literal

from endo_pipeline.cli import Datasets, StrList


def main(
    crop_pattern: Literal["grid", "tracked"] = "grid",
    datasets: Datasets | None = None,
    columns: StrList | None = None,
) -> None:
    """
    Load and plot the output of the compute-autocorrelation workflow.

    #diffae #correlation-analysis #visualization

    **Workflow defaults**
        - model_manifest_name: DEFAULT_MODEL_MANIFEST_NAME
        - run_name: DEFAULT_MODEL_RUN_NAME
        - crop_pattern: "grid"
        - datasets: all datasets present in the autocorrelation manifest
        - columns: "dynamics analyses" features (DYNAMICS_COLUMN_NAMES)

    **Workflow output**

    For each dataset in the autocorrelation manifest, saves one figure per
    feature showing:
        - The mean autocorrelation function (ACF) across crops as a function of
          lag (in hours).
        - A shaded band showing the per-crop percentile interval
          (lower/upper percentile) around the mean.

    Additionally saves one summary figure showing the mean ACF for all datasets
    overlaid on a single axes per feature, colored by shear stress condition.

    Parameters
    ----------
    crop_pattern
        Crop pattern of the autocorrelation results to visualize.
    datasets
        Optional, specific list of datasets or dataset collections to visualize.
        If not provided, all datasets in the autocorrelation manifest are used.
    columns
        Optional, specific list of feature column names to visualize. If not
        provided, will use all three "dynamics analyses" features: polar theta,
        polar r, and rho.

    """
    import logging
    from typing import cast

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_shear_stress_label_for_dataset, load_dataset_config
    from endo_pipeline.io import (
        get_output_path,
        join_sorted_strings,
        load_dataframe,
        save_plot_to_path,
    )
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_shear_stress
    from endo_pipeline.library.analyze.numerics.correlations import (
        exponential_decay,
        fit_exp_decay_and_get_relaxation_timescale,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.library.visualize.summary_plot import _build_jitter_map
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.autocorrelations import AUTOCORRELATION_DATAFRAME_MANIFEST_PREFIX
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.summary_plot import COLOR_PALETTE, DATASET_COLOR_MAP
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    plt.style.use("endo_pipeline.figure")
    logger = logging.getLogger(__name__)

    # Reconstruct the manifest name produced by compute-autocorrelation.
    column_names = columns or list(DYNAMICS_COLUMN_NAMES)
    columns_str = join_sorted_strings(cast(list[str], column_names))
    base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}"
    demo_suffix = "_demo" if DEMO_MODE else ""
    manifest_name = (
        f"{AUTOCORRELATION_DATAFRAME_MANIFEST_PREFIX}_{columns_str}_{base_name}{demo_suffix}"
    )

    logger.info("Loading autocorrelation manifest [ %s ]", manifest_name)
    autocorrelation_manifest = load_dataframe_manifest(manifest_name)

    dataset_names = datasets or list(autocorrelation_manifest.locations.keys())
    if DEMO_MODE:
        logger.warning("DEMO MODE: Visualizing no more than two datasets.")
        dataset_names = dataset_names[:2]

    output_path = get_output_path(__file__, crop_pattern)

    global_xlim = (0.0, 8.5)
    global_ylim = (-0.25, 1.1)

    # Accumulate relaxation times and per-dataset metadata per feature.
    relaxation_rows: list[dict] = []

    dataset_color_map = {
        ds: DATASET_COLOR_MAP.get(ds, COLOR_PALETTE[i % len(COLOR_PALETTE)])
        for i, ds in enumerate(dataset_names)
    }

    for dataset_name in dataset_names:
        if dataset_name not in autocorrelation_manifest.locations:
            logger.warning(
                "Dataset [ %s ] not found in manifest [ %s ], skipping.",
                dataset_name,
                manifest_name,
            )
            continue

        df = load_dataframe(autocorrelation_manifest.locations[dataset_name])

        # Keep only positive lags for plotting (ACF is symmetric around zero).
        df_positive = df[df[Column.AutoCorrelation.LAG] > 0]

        features = df_positive[Column.AutoCorrelation.FEATURE].unique().tolist()
        dataset_config = load_dataset_config(dataset_name)

        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress
            fig_title = get_shear_stress_label_for_dataset(dataset_config, flow_condition)
            filename = f"acf_{dataset_name}_shear_{flow_condition.shear_stress_bin}"
            df_flow = filter_dataframe_by_shear_stress(df_positive, shear_stress)

            n_cols = min(3, len(features))
            n_rows = int(np.ceil(len(features) / n_cols))
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
            )

            for feat_idx, feature in enumerate(features):
                ax: plt.Axes = axes[feat_idx // n_cols][feat_idx % n_cols]
                df_feat = df_flow[df_flow[Column.AutoCorrelation.FEATURE] == feature]

                df_feat = df_feat.sort_values(Column.AutoCorrelation.LAG)
                lags_hours = 5.0 * df_feat[Column.AutoCorrelation.LAG].to_numpy() / 60.0
                acf_mean = df_feat[Column.AutoCorrelation.ACF_MEAN].to_numpy()
                acf_lb = df_feat[Column.AutoCorrelation.ACF_LOWER_PERCENTILE].to_numpy()
                acf_ub = df_feat[Column.AutoCorrelation.ACF_UPPER_PERCENTILE].to_numpy()

                ax.plot(lags_hours, acf_mean, "k-", linewidth=2)
                ax.fill_between(lags_hours, acf_lb, acf_ub, color="gray", alpha=0.2)
                ax.set_xlabel("Lag (hours)")
                ax.set_ylabel("ACF")

                # Fit single exponential decay and annotate relaxation timescale + R².
                exp_fit, relaxation_time, r_squared = fit_exp_decay_and_get_relaxation_timescale(
                    acf_mean, lags_hours, exp_decay_func="exponential_decay"
                )
                ax.plot(
                    lags_hours,
                    exponential_decay(lags_hours, *exp_fit),
                    color="darkturquoise",
                    linestyle="--",
                    linewidth=1.25,
                    label="exp fit",
                )
                relaxation_rows.append(
                    {
                        "feature": feature,
                        "relaxation_time": relaxation_time,
                        "r_squared": r_squared,
                        "shear_stress_numeric": flow_condition.shear_stress_bin,
                        "label": fig_title,
                        "color": dataset_color_map[dataset_name],
                        "dataset": dataset_name,
                    }
                )

                ax.set_title(
                    f"{get_label_for_column(feature)}\n"
                    rf"$\tau$ = {relaxation_time:.2f} hr, $R^2$ = {r_squared:.2f}",
                    fontsize=10,
                )
                ax.set_xlim(global_xlim)
                ax.set_ylim(global_ylim)

            fig.suptitle(
                f"Autocorrelation - {fig_title}",
                fontsize=12,
                y=1.01,
            )
            save_plot_to_path(fig, output_path, filename, show_and_close=False)
            plt.close(fig)

            logger.info(
                "Saved ACF figure for dataset [ %s ] flow condition [ %s ].",
                dataset_name,
                fig_title,
            )

    df_relaxation = pd.DataFrame(
        relaxation_rows,
        columns=[
            "feature",
            "relaxation_time",
            "r_squared",
            "shear_stress_numeric",
            "label",
            "color",
            "dataset",
        ],
    )

    # Plot a histogram of relaxation times for each feature across all datasets.
    if not df_relaxation.empty:
        features_all = df_relaxation["feature"].unique().tolist()
        n_cols = min(3, len(features_all))
        n_rows = int(np.ceil(len(features_all) / n_cols))
        fig_hist, axes_hist = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
        )
        for feat_idx, feature in enumerate(features_all):
            ax_hist: plt.Axes = axes_hist[feat_idx // n_cols][feat_idx % n_cols]
            tau_values = df_relaxation.loc[df_relaxation["feature"] == feature, "relaxation_time"]
            ax_hist.hist(tau_values, bins="auto", color="steelblue", edgecolor="white")
            ax_hist.set_xlabel(r"Relaxation time $\tau$ (hours)")
            ax_hist.set_ylabel("Count")
            ax_hist.set_title(get_label_for_column(feature), fontsize=10)

        fig_hist.suptitle("Relaxation Time Distributions", fontsize=12, y=1.01)
        save_plot_to_path(
            fig_hist, output_path, "acf_relaxation_time_histograms", show_and_close=False
        )
        plt.close(fig_hist)
        logger.info("Saved relaxation time histogram figure.")

    # Cross-dataset summary: scatter of relaxation times vs shear stress, one panel per feature.
    if not df_relaxation.empty:
        features_summary = df_relaxation["feature"].unique().tolist()
        n_cols = min(3, len(features_summary))
        n_rows = int(np.ceil(len(features_summary) / n_cols))
        fig_summary, axes_summary = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
        )
        for feat_idx, feature in enumerate(features_summary):
            ax_s: plt.Axes = axes_summary[feat_idx // n_cols][feat_idx % n_cols]
            df_feat = df_relaxation[df_relaxation["feature"] == feature].copy()
            jitter_map = _build_jitter_map(df_feat, jitter_width=0.4)
            for _, row in df_feat.iterrows():
                x_jittered = row["shear_stress_numeric"] + jitter_map.get(
                    (row["dataset"], row["shear_stress_numeric"]), 0.0
                )
                ax_s.scatter(
                    x_jittered,
                    row["relaxation_time"],
                    color=row["color"],
                    s=60,
                    alpha=0.85,
                    zorder=3,
                )
            shear_ticks = sorted(df_feat["shear_stress_numeric"].unique())
            ax_s.set_xticks(shear_ticks)
            ax_s.set_xticklabels([str(v) for v in shear_ticks])
            ax_s.set_xlabel(r"Shear stress (dyn/cm$^2$)")
            ax_s.set_ylabel(r"Relaxation time $\tau$ (hours)")
            ax_s.set_title(get_label_for_column(feature), fontsize=10)

        fig_summary.suptitle(r"Relaxation Times $\tau$ — All Datasets", fontsize=12, y=1.01)
        save_plot_to_path(
            fig_summary, output_path, "acf_relaxation_time_cross_dataset", show_and_close=False
        )
        plt.close(fig_summary)
        logger.info("Saved cross-dataset relaxation time summary figure.")

    # Scatter plot of R² values across all datasets, one column per feature with x-axis jitter.
    if not df_relaxation.empty:
        features_r2 = df_relaxation["feature"].unique().tolist()
        fig_r2, ax_r2 = plt.subplots(figsize=(max(6, 2.5 * len(features_r2)), 5))
        rng = np.random.default_rng(seed=0)
        for feat_idx, feature in enumerate(features_r2):
            r2_values = df_relaxation.loc[
                df_relaxation["feature"] == feature, "r_squared"
            ].to_numpy()
            jitter = rng.uniform(-0.075, 0.075, size=len(r2_values))
            ax_r2.scatter(
                feat_idx + jitter,
                r2_values,
                c="black",
                s=40,
                alpha=0.3,
                zorder=3,
            )
        ax_r2.set_xticks(range(len(features_r2)))
        ax_r2.set_xticklabels(
            [get_label_for_column(f) for f in features_r2], rotation=30, ha="right"
        )
        ax_r2.set_ylabel(r"$R^2$")
        ax_r2.set_title(r"Exponential Fit $R^2$ — All Datasets", fontsize=12)
        ax_r2.set_ylim(0.98, 1.005)
        fig_r2.tight_layout()
        save_plot_to_path(fig_r2, output_path, "acf_r2_scatter", show_and_close=False)
        plt.close(fig_r2)
        logger.info("Saved R² scatter plot.")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
