from endo_pipeline.cli import UniqueStrList


def main(include_panels: UniqueStrList | None = None) -> None:
    """
    **Supplemental Figure 2**. Generation of tracked 2D cell segmentations and
    measurements using VE-cadherin

    #supp-figure #cdh5-segmentation

    | Panel | Description                                                                                | Notes                |
    | ----- | ------------------------------------------------------------------------------------------ | -------------------- |
    | A     | Diagram for cell segmentation steps                                                        | _compiled manually_  |
    | B     | Measured features for representative replicate under 6 dyn/cm² shear stress                |                      |
    | C     | Measured features for representative replicate under 21 dyn/cm² shear stress               |                      |
    | D     | Schematic for cell orientation, aspect ratio, and cell-nucleus angle relative to migration | _generated manually_ |

    ## Example usage

    To run the figure workflow:

    ```bash
    uv run endopipe supp-figure-2-segmentations
    ```

    ## Figure panels

    All panels in this workflow can be run without GPU.
    """

    from matplotlib import pyplot as plt

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.figures import (
        FigurePanel,
        build_figure_from_panels,
        get_figure_asset_dir,
        parse_placeholder_panels,
    )
    from endo_pipeline.library.visualize.lib_cdh5_seg_feats_fig_panels import (
        make_feature_contact_sheet,
        make_imaging_panels,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.examples import (
        CDH5_SEG_FIG_CLASSIC_FEAT_EXAMPLES,
        CDH5_SEG_FIG_EXAMPLE,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    placeholders = parse_placeholder_panels(include_panels, ["A", "B", "C", "D"])

    datasets = CDH5_SEG_FIG_CLASSIC_FEAT_EXAMPLES

    # Set the panel sizes
    panel_A_height = 3.05
    panel_B_height = 2.3
    panel_C_height = panel_B_height
    panel_B_and_C_width = 4.8

    # the panels produced by make_classic_feature_panels are arranged into
    # the schematic in panel A using Adobe Illustrator, not here in the code
    schematic_fp = make_imaging_panels(
        output_path=output_path,
        dataset_name=CDH5_SEG_FIG_EXAMPLE.dataset_name,
        position=CDH5_SEG_FIG_EXAMPLE.position,
        timeframe=CDH5_SEG_FIG_EXAMPLE.timepoint,
        **placeholders["A"],
    )

    # make_contact_sheet of select features
    features = [
        Column.SegData.ORIENTATION_DEG,
        Column.SegData.ASPECT_RATIO,
        Column.SegData.CENTROID_VELOCITY_ANGLE_DEG,
        Column.SegData.AREA_UM_SQ,
        Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG,
        Column.SegData.EDGE_FLUOR_MEAN,
    ]

    classic_feat_fig_example_paths = {}
    for dataset in datasets:
        classic_feat_fig_example_path = make_feature_contact_sheet(
            dataset_name=dataset,
            positions=[0],
            features=features,
            ncols=3,
            out_dir=output_path / "feature_contact_sheet",
            figure_width=panel_B_and_C_width,
            figure_height_scaling=0.7,
        )
        classic_feat_fig_example_paths[dataset] = classic_feat_fig_example_path

    assets_dir = get_figure_asset_dir()
    schematic_fp = assets_dir / "cdh5_classic_seg_schematic.svg"
    feature_diagram_fp = assets_dir / "cdh5_seg_feat_diagrams.svg"

    figure_panels = [
        FigurePanel(
            letter="A",
            path=schematic_fp,
            x_position=0,
            y_position=0,
            x_offset=0,
            y_offset=0.15,
        ),
        FigurePanel(
            letter="B",
            path=classic_feat_fig_example_paths[datasets[0]],
            x_position=0,
            y_position=panel_A_height + 0.15,
            x_offset=-0.1,
            y_offset=0.1,
        ),
        FigurePanel(
            letter="C",
            path=classic_feat_fig_example_paths[datasets[1]],
            x_position=0,
            y_position=panel_A_height + panel_B_height + 0.2,
            x_offset=-0.1,
            y_offset=0.1,
        ),
        FigurePanel(
            letter="D",
            path=feature_diagram_fp,
            x_position=panel_B_and_C_width + 0.2,
            y_position=panel_A_height + 0.15,
            x_offset=-0.03,
            y_offset=0.1,
        ),
    ]

    build_figure_from_panels(
        figure_panels,
        output_path / "supp_figure_2_segmentations.svg",
        width=MAX_FIGURE_WIDTH,
        height=panel_A_height + panel_B_height + panel_C_height + 0.3,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
