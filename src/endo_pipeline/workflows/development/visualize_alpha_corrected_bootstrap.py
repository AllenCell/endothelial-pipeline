from endo_pipeline.cli import Datasets, PatchType
from endo_pipeline.settings.bootstrap_fixed_points import BOOTSTRAP_THRESHOLD


def main(
    patch_type: PatchType = "grid_based",
    datasets: Datasets | None = None,
    bootstrap_threshold: float = BOOTSTRAP_THRESHOLD,
) -> None:
    """
    Compare bootstrapped fixed points for the alpha-corrected and Ito drift.

    #dynamical-systems #fixed-points #grid-based #cell-centered

    This workflow loads the alpha-corrected bootstrap confidence interval (CI)
    dataframes produced by `bootstrap-alpha-corrected-fixed-points` and overlays
    them on the precomputed Ito-interpretation results from
    `bootstrap-fixed-points`.

    For each dataset and flow condition, fixed points whose bootstrap detection
    rate meets or exceeds `bootstrap_threshold` are plotted at their bootstrap
    cluster mean with per-coordinate confidence interval error bars. Markers are
    colored and shaped by stability classification, and distinguished between
    interpretations by marker edge color (black = Ito, red = alpha-corrected).

    A shift between the two sets of fixed points indicates that the fixed point
    location is sensitive to the choice of stochastic interpretation, which
    occurs where the noise amplitude varies in feature space. Where the
    confidence intervals of the two interpretations overlap, the correction is
    not resolvable at the current bootstrap sample size.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe visualize-alpha-corrected-bootstrap -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe visualize-alpha-corrected-bootstrap --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `diffae_model_training` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will visualize the
    comparison for at most two datasets.

    Parameters
    ----------
    patch_type
        Patch type used to calculate the bootstrapped fixed points.
    datasets
        List of datasets or dataset collections to visualize.
    bootstrap_threshold
        Minimum bootstrap detection rate for including fixed points.
    """

    import logging

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.bootstrap_fixed_points import (
        load_high_confidence_bootstrap_results,
    )
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_shear_stress
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.library.visualize.fixed_points import (
        get_stability_legend_handles,
        plot_bootstrap_fixed_points_on_axis,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.bootstrap_fixed_points import (
        ALPHA_CORRECTED_EDGE_COLOR,
        ALPHA_CORRECTED_MANIFEST_SUFFIX,
        ITO_EDGE_COLOR,
    )
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        DYNAMICS_COLUMN_NAMES,
    )
    from endo_pipeline.settings.flow_field_3d import FIGSIZE_2D_FLOW_FIELD, NROWS_2D_FLOW_FIELD
    from endo_pipeline.settings.manifest_names import BOOTSTRAPPING_MANIFEST_NAMES
    from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    column_names = list(DYNAMICS_COLUMN_NAMES)

    # Load bootstrap manifests for each interpretation
    ito_manifest_name = BOOTSTRAPPING_MANIFEST_NAMES[patch_type]
    alpha_manifest_name = f"{ito_manifest_name}_{ALPHA_CORRECTED_MANIFEST_SUFFIX}"
    ito_manifest = load_dataframe_manifest(ito_manifest_name)
    alpha_manifest = load_dataframe_manifest(alpha_manifest_name)

    n_bootstrap_ito = ito_manifest.parameters.get("num_bootstrap_iterations")
    n_bootstrap_alpha = alpha_manifest.parameters.get("num_bootstrap_iterations")
    sde_alpha = alpha_manifest.parameters.get("sde_alpha")

    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to at most two datasets")
        dataset_names = dataset_names[: min(len(dataset_names), 2)]

    # Axis bounds from global bin limits, one tuple (min, max) per column
    bounds_for_plots = BIN_LIMITS_DYNAMICS.copy()

    # Legend handles distinguishing the two interpretations by marker edge color
    ito_label = f"It{Unicode.O_WITH_CIRCUMFLEX} (n = {n_bootstrap_ito})"
    alpha_label = f"{Unicode.ALPHA} = {sde_alpha} (n = {n_bootstrap_alpha})"
    interpretation_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=ITO_EDGE_COLOR,
            markersize=9,
            label=ito_label,
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=ALPHA_CORRECTED_EDGE_COLOR,
            markersize=9,
            label=alpha_label,
        ),
    ]

    for dataset_name in dataset_names:
        ito_df = load_high_confidence_bootstrap_results(
            ito_manifest, dataset_name, bootstrap_threshold
        )
        alpha_df = load_high_confidence_bootstrap_results(
            alpha_manifest, dataset_name, bootstrap_threshold
        )

        if ito_df.empty and alpha_df.empty:
            logger.warning(
                "No high-confidence fixed points (detection rate >= %.2f) for dataset "
                "[ %s ] in either interpretation. Skipping.",
                bootstrap_threshold,
                dataset_name,
            )
            continue

        logger.info(
            "Dataset [ %s ]: %d Ito and %d alpha-corrected fixed point(s) pass threshold.",
            dataset_name,
            len(ito_df),
            len(alpha_df),
        )

        dataset_config = load_dataset_config(dataset_name)

        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress

            ito_df_flow = filter_dataframe_by_shear_stress(ito_df, shear_stress)
            alpha_df_flow = filter_dataframe_by_shear_stress(alpha_df, shear_stress)

            if ito_df_flow.empty and alpha_df_flow.empty:
                continue

            fig, axes = plt.subplots(NROWS_2D_FLOW_FIELD, 1, figsize=FIGSIZE_2D_FLOW_FIELD)
            fig.suptitle(
                f"{dataset_name} - Fixed Points: {ito_label} vs {alpha_label}\n"
                f"(Shear Stress {shear_stress}, "
                f"Detection Rate {Unicode.GEQ} {bootstrap_threshold:.2%})"
            )

            for ax, column_x, column_y in [
                (axes[0], column_names[0], column_names[1]),
                (axes[1], column_names[0], column_names[2]),
            ]:
                plot_bootstrap_fixed_points_on_axis(
                    ax, ito_df_flow, column_x, column_y, edge_color=ITO_EDGE_COLOR
                )
                plot_bootstrap_fixed_points_on_axis(
                    ax,
                    alpha_df_flow,
                    column_x,
                    column_y,
                    edge_color=ALPHA_CORRECTED_EDGE_COLOR,
                )

                xlabel = get_label_for_column(column_x)
                ylabel = get_label_for_column(column_y)
                ax.set_xlim(bounds_for_plots[column_x])
                ax.set_ylim(bounds_for_plots[column_y])
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f"{xlabel} vs {ylabel}")

            stability_handles = get_stability_legend_handles([ito_df_flow, alpha_df_flow])
            axes[-1].legend(
                handles=[*interpretation_handles, *stability_handles],
                title="Interpretation / Stability",
                loc="best",
            )

            plt.tight_layout()

            save_plot_to_path(
                fig,
                output_path,
                f"alpha_corrected_fixed_points_comparison_{dataset_name}"
                f"_shear_{flow_condition.shear_stress_bin}",
            )
            plt.close(fig)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
