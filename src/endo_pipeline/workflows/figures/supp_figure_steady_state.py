def main() -> None:
    """
    **Supplemental Figure**. Sensitivity of results to steady state annotation window

    #supp-figure #preprocessing

    | Panel | Description                                                                                      | Notes                |
    | ----- | ------------------------------------------------------------------------------------------------ | -------------------- |


    ## Example usage

    To run the figure workflow:

    ```bash
    uv run endopipe supp-figure-steady-state
    ```

    ## Figure panels

    All panels in this workflow can be run without GPU.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd

    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.window_sweep import SweptBoundary
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.window_sweep_plots import plot_sweep
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.bootstrap_fixed_points import BOOTSTRAP_THRESHOLD
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.manifest_names import (
        BOOTSTRAPPING_MANIFEST_NAMES,
        WINDOW_SWEEP_MANIFEST_NAMES,
    )

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    datasets = get_datasets_in_collection("supp_figure_examples")

    bootstrap_manifest = load_dataframe_manifest(BOOTSTRAPPING_MANIFEST_NAMES["grid_based"])
    start_sweep_manifest = load_dataframe_manifest(
        WINDOW_SWEEP_MANIFEST_NAMES[SweptBoundary.START_STEADY_STATE]
    )
    end_sweep_manifest = load_dataframe_manifest(
        WINDOW_SWEEP_MANIFEST_NAMES[SweptBoundary.START_CELL_PILING]
    )

    figure_paths: list[Path] = []

    for dataset_name in datasets:
        # Load "baseline" bootstrapping manifest, filter to high-confidence fixed points,
        # and rename columns for compatibility with downstream analysis
        bootstrap_df = load_dataframe(bootstrap_manifest.locations[dataset_name])
        bootstrap_df = bootstrap_df[
            bootstrap_df[Column.FIXED_POINT_DETECTION_RATE] >= BOOTSTRAP_THRESHOLD
        ]
        bootstrap_df = bootstrap_df.rename(
            columns={
                ColumnTemplate.BASELINE_FIXED_POINT
                % column_name: (ColumnTemplate.FIXED_POINT % column_name)
                for column_name in DYNAMICS_COLUMN_NAMES
            }
        )
        # Load and filter sweep data for each boundary condition
        sweep_dataframes: dict[SweptBoundary, pd.DataFrame] = {}
        for boundary, manifest in [
            (SweptBoundary.START_STEADY_STATE, start_sweep_manifest),
            (SweptBoundary.START_CELL_PILING, end_sweep_manifest),
        ]:
            # Load results of steady state window sweep
            sweep_df = load_dataframe(manifest.locations[dataset_name])
            # Keep only higher-confidence fixed points (same threshold the other
            # fixed-point workflows use).
            sweep_df = sweep_df[sweep_df[Column.FIXED_POINT_DETECTION_RATE] >= BOOTSTRAP_THRESHOLD]
            sweep_df[Column.SteadyStateWindow.WINDOW_EDGE_HOURS] = (
                sweep_df[Column.SteadyStateWindow.WINDOW_EDGE_MINUTES] / 60
            )
            sweep_dataframes[boundary] = sweep_df
            output_path = get_output_path(__file__, dataset_name)
            fig = plot_sweep(
                sweep_df,
                dataset_name,
                boundary,
                48,
                baseline_df=bootstrap_df,
            )
            panel_path = save_plot_to_path(
                fig,
                output_path,
                f"window_sweep_{boundary}_{dataset_name}",
                tight_layout=False,
                bbox_inches="tight",
            )
            figure_paths.append(panel_path)

    panels = [
        FigurePanel(
            letter="A",
            path=figure_paths[0],
            x_position=0,
            y_position=0,
            x_offset=0,
            y_offset=0,
        )
    ]

    build_figure_from_panels(
        panels,
        output_path / "supp_figure_steady_state.svg",
        width=MAX_FIGURE_WIDTH,
        height=MAX_FIGURE_HEIGHT,
    )
