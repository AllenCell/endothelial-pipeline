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
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.autocorrelations import AUTOCORRELATION_DATAFRAME_MANIFEST_PREFIX
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

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

                ax.plot(lags_hours, acf_mean, linewidth=2)
                ax.fill_between(lags_hours, acf_lb, acf_ub, alpha=0.2)
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                ax.set_xlabel("Lag (hours)")
                ax.set_ylabel("ACF")

                # Fit single exponential decay and annotate relaxation timescale + R².
                exp_fit, relaxation_time, r_squared = fit_exp_decay_and_get_relaxation_timescale(
                    acf_mean, lags_hours, exp_decay_func="exponential_decay"
                )
                ax.plot(
                    lags_hours,
                    exponential_decay(lags_hours, *exp_fit),
                    linestyle="--",
                    linewidth=1.5,
                    label="exp fit",
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


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
