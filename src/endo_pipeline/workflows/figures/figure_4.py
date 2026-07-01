from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    # Figure 4. Properties of the data-driven vector field are consistent with a
    saddle node bifurcation at intermediate shear stress magnitudes

    #main-figure #fixed-points #grid-based

    | Panel | Description                                                                               | Notes      |
    | ----- | ----------------------------------------------------------------------------------------- | ---------- |
    | A     | Summary of fixed point locations across replicated colored by migration coherence         |            |
    | B     | 3D data-driven vector field for representative bistable 12 dyn/cm² shear stress replicate |            |
    | C     | Example reconstruction of VE-cadherin patch using stable fixed point                      | _uses GPU_ |
    | D     | Streamplot for representative bistable 12 dyn/cm² shear stress replicate                  |            |

    ## Example usage

    To run the figure workflow:

    ```bash
    uv run endopipe figure-4
    ```

    To run the figure workflow for a specific panel:

    ```bash
    uv run endopipe figure-4 PANEL
    ```

    ## Figure panels

    Some panels in this workflow should be run with an NVIDIA GPU (as indicated
    by _uses GPU_ in the table above). Run this workflow with the GPU flag (`-g`
    or `--num-gpus`) to make sure GPUs are visible to the workflow. The workflow
    will run without a GPU, but will be noticeably slower. You may want to skip
    generating these panels by excluding them from the list of panels.

    Parameters
    ----------
    include_panels
        List of panels to include in figure. Leave empty to include all panels.
    """

    import matplotlib.pyplot as plt

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.io import get_output_path
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
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.bootstrap_fixed_points import BOOTSTRAP_THRESHOLD
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.examples import EXAMPLE_DATASET
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
    from endo_pipeline.settings.manifest_names import BOOTSTRAPPING_MANIFEST_NAMES
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_PATCH_TYPE
    from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
    from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    placeholders = parse_placeholder_panels(include_panels, ["A", "B", "C", "D"])

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
        figure_size=(3.15, 2.0),
        jitter_width=0.2,
        subplot_layout="vertical",
        color_by_column=ColumnName.OpticalFlow.UNIT_VECTOR_MEAN,
        colorbar_location="bottom",
        **placeholders["A"],
    )

    dataset_name = EXAMPLE_DATASET["FIGURE_4_STREAMPLOT"]
    vector_field_plot_path, stable_fixed_points_df = make_3d_vector_field_plot_panel(
        dataset_name,
        output_path,
        **placeholders["B"],
    )

    fixed_point_reconstruction_path = reconstruct_fixed_points(
        fixed_point_df=stable_fixed_points_df,
        output_path=output_path,
        figure_size=(0.8, 2.25),
        num_gpus=NUM_GPUS,
        **placeholders["C"],
    )

    projected_streamlines_path = visualize_projected_dynamics(
        dataset_name=dataset_name,
        output_path=output_path,
        figure_size=(2.35, 2.35),
        **placeholders["D"],
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
            x_position=3.4,
            y_position=0.0,
            x_offset=0.15,
            y_offset=0,
        ),
        FigurePanel(
            letter="C",
            path=fixed_point_reconstruction_path,
            x_position=5.35,
            y_position=0.0,
            x_offset=0.2,
            y_offset=0.225,
        ),
        FigurePanel(
            letter="D",
            path=projected_streamlines_path,
            x_position=3.4,
            y_position=2.5,
            x_offset=-0.025,
            y_offset=0.05,
        ),
    ]

    build_figure_from_panels(
        panels, output_path / "figure_4.svg", width=MAX_FIGURE_WIDTH, height=4.8
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
