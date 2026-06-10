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

    import matplotlib.pyplot as plt

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.data_example_figures import (
        create_panel_intermediate_examples,
    )
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.summary_plot import (
        build_dataframe_for_fixed_point_dataset_summary,
        plot_cross_dataset_summaries,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.examples import FIGURE_3_EXAMPLE_IMAGES
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.flow_field_dataframes import BOOTSTRAPPING_MANIFEST_NAMES
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
    from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
    from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    # Example images of intermediate shear stress condition
    create_panel_intermediate_examples(
        examples=FIGURE_3_EXAMPLE_IMAGES,
        save_dir=output_path,
        figure_size=(MAX_FIGURE_WIDTH * 0.65, 2.2),
    )

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
        output_dir=output_path,
        column_names=columns_for_summary_plots,
        axis_mode="shear_stress",
        figure_size=(MAX_FIGURE_WIDTH * 0.6, 1.4),
        jitter_width=0.2,
        subplot_layout="vertical",
        color_by_column=ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
    )

    panels = [
        FigurePanel(
            letter="A",
            path=output_path / "intermediate_examples_scale_bar_100um.svg",
            x_position=0,
            y_position=0,
            x_offset=0.2,
            y_offset=0,
        ),
        FigurePanel(
            letter="B",
            path=summary_plot_path,
            x_position=0,
            y_position=2.3,
            x_offset=0,
            y_offset=0.1,
        ),
        FigurePanel(
            letter="C",
            path=output_path / "reconstructed_fp_crop_examples.svg",
            x_position=MAX_FIGURE_WIDTH * 0.6,
            y_position=2.3,
            x_offset=0.1,
            y_offset=0.1,
        ),
    ]

    build_figure_from_panels(
        panels, output_path / "figure_3.svg", width=MAX_FIGURE_WIDTH, height=MAX_FIGURE_HEIGHT
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
