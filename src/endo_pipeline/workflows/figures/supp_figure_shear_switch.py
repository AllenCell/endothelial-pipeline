from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    **Figure 5**. Analysis of the flow switch experiments

    #main-figure #perturbation

    | Panel | Description                                                                               | Notes                |
    | ----- | ----------------------------------------------------------------------------------------- | -------------------- |

    ## Example usage

    To run the figure workflow:

    ```bash
    uv run endopipe figure-5
    ```

    ## Figure panels

    All panels in this workflow can be run without GPU.

    Parameters
    ----------
    include_panels
        List of panels to include in figure. Leave empty to include all panels.

    """
    import logging
    from collections import defaultdict
    from pathlib import Path

    import matplotlib.pyplot as plt

    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.figures import (
        FigurePanel,
        build_figure_from_panels,
        parse_placeholder_panels,
    )
    from endo_pipeline.library.visualize.summary_plot import (
        build_dataframe_for_fixed_point_dataset_summary,
    )
    from endo_pipeline.library.visualize.summary_plot_flow_switch import plot_flow_switch_summaries
    from endo_pipeline.manifests import check_dataset_has_dataframes, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.manifest_names import (
        BOOTSTRAPPING_MANIFEST_NAMES,
        OPTICAL_FLOW_MANIFEST_NAMES,
    )
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_PATCH_TYPE
    from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

    plt.style.use("endo_pipeline.figure")

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    # Panel letters, in order, for each regime's rho and speed plots
    regime_order = ["min_to_max", "max_to_min", "min_to_high", "max_to_medium"]
    rho_panel_letters = dict(zip(regime_order, ["A", "B", "C", "D"], strict=True))
    speed_panel_letters = dict(zip(regime_order, ["E", "F", "G", "H"], strict=True))
    placeholders = parse_placeholder_panels(
        include_panels, [*rho_panel_letters.values(), *speed_panel_letters.values()]
    )

    feature_manifest = load_dataframe_manifest(
        FEATURES_FILTERED_MANIFEST_NAMES[MIGRATION_COHERENCE_PATCH_TYPE]
    )
    bootstrap_manifest = load_dataframe_manifest(
        BOOTSTRAPPING_MANIFEST_NAMES[MIGRATION_COHERENCE_PATCH_TYPE]
    )
    optical_flow_manifest = load_dataframe_manifest(
        OPTICAL_FLOW_MANIFEST_NAMES[MIGRATION_COHERENCE_PATCH_TYPE]
    )
    required_dataframe_manifests = [feature_manifest, bootstrap_manifest, optical_flow_manifest]

    # Group flow switch datasets by shear stress regime (e.g. "min_to_max")
    datasets_by_regime: dict[str, list[str]] = defaultdict(list)
    for dataset_name in get_datasets_in_collection("flow_switch"):
        if not check_dataset_has_dataframes(required_dataframe_manifests, dataset_name):
            continue
        regime = load_dataset_config(dataset_name).shear_stress_regime
        regime_str = "_to_".join(r.value for r in regime)
        datasets_by_regime[regime_str].append(dataset_name)

    if not datasets_by_regime:
        logger.error("No flow switch datasets with the required data were found.")
        return

    background_datasets = [
        dataset_name
        for dataset_name in get_datasets_in_collection("shear_stress")
        if check_dataset_has_dataframes(required_dataframe_manifests, dataset_name)
    ]

    summary_kwargs = {
        "feature_dataframe_manifest": feature_manifest,
        "bootstrap_dataframe_manifest": bootstrap_manifest,
        "column_names": [Column.DiffAEData.PC3_FLIPPED],
        "stable_only": True,
    }

    df_background = build_dataframe_for_fixed_point_dataset_summary(
        dataset_names=background_datasets, **summary_kwargs
    )

    rho_plots_dict: dict[str, Path] = {}
    speed_plots_dict: dict[str, Path] = {}
    for regime, dataset_names in datasets_by_regime.items():
        df_flow_switch = build_dataframe_for_fixed_point_dataset_summary(
            dataset_names=dataset_names, all_flow_conditions=True, **summary_kwargs
        )

        rho_plots_dict[regime] = plot_flow_switch_summaries(
            df_flow_switch,
            df_background,
            output_path=output_path,
            regime_label=regime,
            figure_size=(3.25, 1.25),
            column_names=[Column.DiffAEData.PC3_FLIPPED],
            color_by_column=Column.OpticalFlow.UNIT_VECTOR_MEAN,
            **placeholders[rho_panel_letters[regime]],
        )

        speed_plots_dict[regime] = plot_flow_switch_summaries(
            df_flow_switch,
            df_background,
            output_path=output_path,
            regime_label=regime,
            figure_size=(3.25, 1.25),
            column_names=[Column.OpticalFlow.SPEED_MEAN],
            point_color="black",
            ylabel_rotation=90,
            ylabel_horizontal_alignment="center",
            ylabel_vertical_alignment="bottom",
            yaxis_for_fixed_points=False,
            **placeholders[speed_panel_letters[regime]],
        )

    # Left and right columns, with rho panels above speed panels
    panels = [
        FigurePanel(
            letter=letters[regime],
            path=plots[regime],
            x_position=3.25 * (i % 2),
            y_position=y_start + 1.75 * (i // 2),
            x_offset=0.0,
            y_offset=0.05,
        )
        for plots, letters, y_start in [
            (rho_plots_dict, rho_panel_letters, 0),
            (speed_plots_dict, speed_panel_letters, 3.5),
        ]
        for i, regime in enumerate(regime_order)
    ]

    build_figure_from_panels(
        panels,
        output_path / "supp_figure_shear_switch.svg",
        width=MAX_FIGURE_WIDTH,
        height=MAX_FIGURE_HEIGHT,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
