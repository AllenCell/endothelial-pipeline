from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    Compile panels for Figure 3.

    - **Panel A*: Schematic of possible cases for the transition of fixed point
      locations and stability across shear stress conditions.
    - **Panel B**: Example images of several replicates from intermediate shear
      stress conditions.
    - **Panel C**: Summary plot of fixed point locations across all replicates,
      colored by migration coherence (EMA-smoothed optical flow unit vector
      mean).
    - **Panel D**: 3D vector field plot of drift coefficients for example
      intermediate shear stress datasets, with stable fixed points overlaid as a
      scatter marker.

    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    from pandas import DataFrame

    from endo_pipeline.io import get_output_path, load_model
    from endo_pipeline.library.visualize.figure_4 import (
        make_3d_vector_field_plot_panel,
        reconstruct_fixed_points,
    )
    from endo_pipeline.library.visualize.figures import (
        FigurePanel,
        build_figure_from_panels,
        parse_placeholder_panels,
    )
    from endo_pipeline.library.visualize.summary_plot import (
        build_dataframe_for_fixed_point_dataset_summary,
        plot_cross_dataset_summaries,
    )
    from endo_pipeline.manifests import load_dataframe_manifest, load_model_manifest
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.examples import EXAMPLE_DATASET
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.manifest_names import BOOTSTRAPPING_MANIFEST_NAMES
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_PATCH_TYPE
    from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        FEATURES_FILTERED_MANIFEST_NAMES,
    )

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    placeholders = parse_placeholder_panels(include_panels, ["A", "B", "C", "D"])

    # load and instantiate model for generating synthetic images
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    model_location = model_manifest.locations[DEFAULT_MODEL_RUN_NAME]
    model = load_model(model_location, instantiate=True)

    # Load diffae features
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[
        MIGRATION_COHERENCE_PATCH_TYPE
    ]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    fixed_points_bootstrap_dataframe_manifest_name = BOOTSTRAPPING_MANIFEST_NAMES[
        MIGRATION_COHERENCE_PATCH_TYPE
    ]
    fixed_points_bootstrap_dataframe_manifest = load_dataframe_manifest(
        fixed_points_bootstrap_dataframe_manifest_name
    )

    dataset_summary_list = SUMMARY_PLOT_DATASETS["intermediate"]

    BOOTSTRAP_THRESHOLD = 0.4

    # Cross-dataset summary plots
    columns_for_summary_plots = [
        ColumnName.DiffAEData.POLAR_ANGLE,
        ColumnName.DiffAEData.POLAR_RADIUS,
    ]
    dataset_summary_df = build_dataframe_for_fixed_point_dataset_summary(
        dataset_names=dataset_summary_list,
        feature_dataframe_manifest=feature_dataframe_manifest,
        bootstrap_dataframe_manifest=fixed_points_bootstrap_dataframe_manifest,
        column_names=columns_for_summary_plots,
        convert_angle_to_nematic=False,
        unwrap_angle=True,
        stable_only=True,
        bootstrap_threshold=BOOTSTRAP_THRESHOLD,
    )
    summary_plot_path = plot_cross_dataset_summaries(
        dataset_summary_df,
        output_dir=output_path,
        column_names=columns_for_summary_plots,
        axis_mode="replicate",
        figure_size=(MAX_FIGURE_WIDTH * 0.6, 1.4),
        jitter_width=0.2,
        subplot_layout="vertical",
        color_by_column=ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
    )

    vector_field_plot_paths: dict[str, Path] = {}
    stable_fixed_points_dfs: dict[str, DataFrame] = {}
    stable_fixed_point_reconstruction_paths: dict[str, Path] = {}
    example_dataset_12dyn = EXAMPLE_DATASET["FIGURE_3_12_DYN_BISTABLE"]
    example_dataset_15dyn = EXAMPLE_DATASET["FIGURE_3_15_DYN_BISTABLE"]
    for dataset_name in [example_dataset_12dyn, example_dataset_15dyn]:
        # only include colorbar and legend for first of the two plots to save space
        include_colorbar = dataset_name == example_dataset_12dyn
        include_legend = dataset_name == example_dataset_12dyn
        vector_field_plot_paths[dataset_name], stable_fixed_points_dfs[dataset_name] = (
            make_3d_vector_field_plot_panel(
                dataset_name,
                output_path,
                include_colorbar=include_colorbar,
                include_legend=include_legend,
                **placeholders["D"],
            )
        )
        stable_fixed_point_reconstruction_paths[dataset_name] = reconstruct_fixed_points(
            fixed_point_df=stable_fixed_points_dfs[dataset_name],
            model=model,
            fig_savedir=output_path,
            add_fixed_point_coordinate_annotation=False,
        )

    panels = [
        FigurePanel(
            letter="C",
            path=summary_plot_path,
            x_position=0,
            y_position=5.0,
            x_offset=0.1,
            y_offset=0.2,
        ),
        FigurePanel(
            letter="D",
            path=vector_field_plot_paths[example_dataset_12dyn],
            x_position=MAX_FIGURE_WIDTH * 0.66,
            y_position=2.5,
            x_offset=0.15,
            y_offset=0,
        ),
        FigurePanel(
            letter="",
            path=stable_fixed_point_reconstruction_paths[example_dataset_12dyn],
            x_position=MAX_FIGURE_WIDTH * 0.66,
            y_position=4.6,
            x_offset=0.3,
            y_offset=0.0,
        ),
        FigurePanel(
            letter="",
            path=vector_field_plot_paths[example_dataset_15dyn],
            x_position=MAX_FIGURE_WIDTH * 0.64,
            y_position=5.35,
            x_offset=0.3,
            y_offset=0.0,
        ),
        FigurePanel(
            letter="",
            path=stable_fixed_point_reconstruction_paths[example_dataset_15dyn],
            x_position=MAX_FIGURE_WIDTH * 0.66,
            y_position=7.15,
            x_offset=0.3,
            y_offset=0.0,
        ),
    ]

    build_figure_from_panels(
        panels, output_path / "figure_3.svg", width=MAX_FIGURE_WIDTH, height=MAX_FIGURE_HEIGHT
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
