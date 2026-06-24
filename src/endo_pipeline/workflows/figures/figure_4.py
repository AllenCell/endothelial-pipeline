from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    Compile panels for Figure 4.

    - **Panel A**: Summary plot of fixed point locations across all replicates,
      colored by migration coherence (EMA-smoothed optical flow unit vector
      mean).
    - **Panel B**: 3D vector field plot of drift coefficients for example
      bistable intermediate shear stress dataset, with stable fixed points
      overlaid as a scatter marker.
    - **Panel C**: Accompanying 2D-projected streamplot for dynamics projected
      onto plane defined by two stable fixed points and the saddle point that
      connects them via its unstable manifold.

    """
    import matplotlib.pyplot as plt

    from endo_pipeline.io import get_output_path, load_model
    from endo_pipeline.library.visualize.diffae_features.projected_dynamics import (
        visualize_projected_dynamics,
    )
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
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
    from endo_pipeline.settings.manifest_names import BOOTSTRAPPING_MANIFEST_NAMES
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
    from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        FEATURES_FILTERED_MANIFEST_NAMES,
    )

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    placeholders = parse_placeholder_panels(include_panels, ["A", "B", "C"])

    # load and instantiate model for generating synthetic images
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    model_location = model_manifest.locations[DEFAULT_MODEL_RUN_NAME]
    model = load_model(model_location, instantiate=True)

    # Load diffae features
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[
        MIGRATION_COHERENCE_CROP_PATTERN
    ]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    fixed_points_bootstrap_dataframe_manifest_name = BOOTSTRAPPING_MANIFEST_NAMES[
        MIGRATION_COHERENCE_CROP_PATTERN
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
        output_path=output_path,
        column_names=columns_for_summary_plots,
        axis_mode="replicate",
        figure_size=(MAX_FIGURE_WIDTH * 0.6, 1.4),
        jitter_width=0.2,
        subplot_layout="vertical",
        color_by_column=ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
        **placeholders["A"],
    )

    dataset_name = EXAMPLE_DATASET["FIGURE_4_STREAMPLOT"]
    vector_field_plot_path, stable_fixed_points_df = make_3d_vector_field_plot_panel(
        dataset_name,
        output_path,
        include_colorbar=True,
        include_legend=True,
        **placeholders["B"],
    )
    fixed_point_reconstruction_path = reconstruct_fixed_points(
        fixed_point_df=stable_fixed_points_df,
        model=model,
        fig_savedir=output_path,
        add_fixed_point_coordinate_annotation=False,
    )

    projected_streamlines_path = visualize_projected_dynamics(
        dataset_name=dataset_name,
        output_path=output_path,
        figure_size=(2.0, 2.0),
        **placeholders["C"],
    )

    panels = [
        FigurePanel(
            letter="A",
            path=summary_plot_path,
            x_position=0.0,
            y_position=0.0,
            x_offset=0.1,
            y_offset=0.2,
        ),
        FigurePanel(
            letter="B",
            path=vector_field_plot_path,
            x_position=MAX_FIGURE_WIDTH * 0.66,
            y_position=0.0,
            x_offset=0.15,
            y_offset=0,
        ),
        FigurePanel(
            letter="",
            path=fixed_point_reconstruction_path,
            x_position=MAX_FIGURE_WIDTH * 0.66,
            y_position=2.3,
            x_offset=0.3,
            y_offset=0.0,
        ),
        FigurePanel(
            letter="C",
            path=projected_streamlines_path,
            x_position=0.0,
            y_position=3.25,
            x_offset=0.3,
            y_offset=0.0,
        ),
    ]

    build_figure_from_panels(
        panels, output_path / "figure_4.svg", width=MAX_FIGURE_WIDTH, height=5.5
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
