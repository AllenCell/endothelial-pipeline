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

    import logging
    from pathlib import Path

    import matplotlib.pyplot as plt

    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.window_sweep import SweptBoundary
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.window_sweep_plots import plot_sweep_subset
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.bootstrap_fixed_points import BOOTSTRAP_THRESHOLD
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
    from endo_pipeline.settings.manifest_names import (
        BOOTSTRAPPING_MANIFEST_NAMES,
        WINDOW_SWEEP_MANIFEST_NAMES,
    )

    logger = logging.getLogger(__name__)

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
        dataset_config = load_dataset_config(dataset_name)

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
        # subset to just stable fixed points
        bootstrap_df = bootstrap_df[
            bootstrap_df[Column.FIXED_POINT_STABILITY] == StabilityLabel.STABLE
        ]
        # Load and filter sweep data for each boundary condition
        for boundary, manifest in [
            (SweptBoundary.START_STEADY_STATE, start_sweep_manifest),
            (SweptBoundary.START_CELL_PILING, end_sweep_manifest),
        ]:
            if dataset_name not in manifest.locations:
                logger.warning(
                    "Dataset %s not found in manifest for boundary %s", dataset_name, boundary
                )
                continue
            # Load results of steady state window sweep
            sweep_df = load_dataframe(manifest.locations[dataset_name])
            # Keep only higher-confidence stable fixed points (same threshold
            # the other fixed-point workflows use).
            sweep_df = sweep_df[
                (sweep_df[Column.FIXED_POINT_DETECTION_RATE] >= BOOTSTRAP_THRESHOLD)
                & (sweep_df[Column.FIXED_POINT_STABILITY] == StabilityLabel.STABLE)
            ]
            sweep_df[Column.SteadyStateWindow.WINDOW_EDGE_HOURS] = (
                sweep_df[Column.SteadyStateWindow.WINDOW_EDGE_MINUTES] / 60
            )
            panel_path = plot_sweep_subset(
                sweep_df,
                baseline_df=bootstrap_df,
                dataset_config=dataset_config,
                boundary=boundary,
                output_path=output_path,
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
        ),
        FigurePanel(
            letter="",
            path=figure_paths[1],
            x_position=1.55,
            y_position=0,
            x_offset=0,
            y_offset=0,
        ),
        FigurePanel(
            letter="B",
            path=figure_paths[2],
            x_position=3.0,
            y_position=0,
            x_offset=0,
            y_offset=0,
        ),
        FigurePanel(
            letter="",
            path=figure_paths[3],
            x_position=4.8,
            y_position=0,
            x_offset=0,
            y_offset=0,
        ),
        FigurePanel(
            letter="C",
            path=figure_paths[4],
            x_position=0,
            y_position=2.75,
            x_offset=0,
            y_offset=0,
        ),
        FigurePanel(
            letter="",
            path=figure_paths[5],
            x_position=1.55,
            y_position=2.75,
            x_offset=0,
            y_offset=0,
        ),
        # FigurePanel(
        #     letter="D",
        #     path=figure_paths[6],
        #     x_position=3.0,
        #     y_position=2.75,
        #     x_offset=0,
        #     y_offset=0,
        # ),
        # FigurePanel(
        #     letter="",
        #     path=figure_paths[7],
        #     x_position=4.8,
        #     y_position=2.75,
        #     x_offset=0,
        #     y_offset=0,
        # )
    ]

    build_figure_from_panels(
        panels,
        output_path / "supp_figure_steady_state.svg",
        width=MAX_FIGURE_WIDTH,
        height=MAX_FIGURE_HEIGHT,
    )
