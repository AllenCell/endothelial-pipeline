def main() -> None:
    """Produces figure panels for the CDH5 segmentation and classic feature workflow figure.
    This includes imaging panels showing the segmentation steps and 2D histograms of classic
    features for each of the PCA reference datasets.

    #test-ready #cpu-only
    """
    from pathlib import Path

    from matplotlib import pyplot as plt

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
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
    from endo_pipeline.workflows.figures import assets as figure_assets

    # set global plotting parameters to be consistent with the other plots in the manuscript
    plt.style.use("endo_pipeline.figure")

    datasets = CDH5_SEG_FIG_CLASSIC_FEAT_EXAMPLES

    out_dir = get_output_path(__file__)

    # Set the panel sizes
    panel_A_height = 3.05
    panel_B_height = 2.3
    panel_C_height = panel_B_height
    panel_B_and_C_width = 4.8

    # the panels produced by make_classic_feature_panels are arranged into
    # the schematic in panel A using Adobe Illustrator, not here in the code
    make_imaging_panels(
        CDH5_SEG_FIG_EXAMPLE.dataset_name,
        CDH5_SEG_FIG_EXAMPLE.position,
        CDH5_SEG_FIG_EXAMPLE.timepoint,
        __file__,
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
            features=features,
            ncols=3,
            out_dir=out_dir / "feature_contact_sheet",
            figure_width=panel_B_and_C_width,
            figure_height_scaling=0.7,
        )
        classic_feat_fig_example_paths[dataset] = classic_feat_fig_example_path

    assets_dir = Path(figure_assets.__path__[0])
    schematic_fp = assets_dir / "cdh5_classic_seg_schematic.svg"
    feature_diagram_fp = assets_dir / "cdh5_seg_feat_diagrams.svg"

    figure_panels = [
        FigurePanel(
            letter="A",
            path=schematic_fp,
            x_position=0,
            y_position=0,
            x_offset=0,
            y_offset=0.0,
        ),
        FigurePanel(
            letter="B",
            path=classic_feat_fig_example_paths[datasets[0]],
            x_position=0,
            y_position=panel_A_height,
            x_offset=-0.2,
            y_offset=0.05,
        ),
        FigurePanel(
            letter="C",
            path=classic_feat_fig_example_paths[datasets[1]],
            x_position=0,
            y_position=panel_A_height + panel_B_height + 0.05,
            x_offset=-0.2,
            y_offset=0.05,
        ),
        FigurePanel(
            letter="D",
            path=feature_diagram_fp,
            x_position=panel_B_and_C_width + 0.2,
            y_position=panel_A_height,
            x_offset=-0.05,
            y_offset=0.05,
        ),
    ]

    build_figure_from_panels(
        figure_panels,
        out_dir / "cdh5_seg_feats_panels.svg",
        width=MAX_FIGURE_WIDTH,
        height=panel_A_height + panel_B_height + panel_C_height + 0.1,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
