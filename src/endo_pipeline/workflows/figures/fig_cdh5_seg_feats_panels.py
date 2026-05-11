from typing import Literal

from endo_pipeline.cli import Datasets, tags

TAGS = [tags.TEST_READY, tags.CPU_ONLY]


def main(datasets: Datasets | None | Literal["figure"] = "figure") -> None:
    """Produces figure panels for the CDH5 segmentation and classic feature workflow figure.
    This includes imaging panels showing the segmentation steps and 2D histograms of classic
    features for each of the PCA reference datasets.
    """
    from pathlib import Path

    from endo_pipeline import figure_assets
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.lib_cdh5_seg_feats_fig_panels import (
        make_classic_feature_panels,
        make_feature_contact_sheet,
        make_imaging_panels,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.examples import (
        CDH5_SEG_FIG_CLASSIC_FEAT_EXAMPLES,
        CDH5_SEG_FIG_EXAMPLE,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH
    from endo_pipeline.settings.workflow_defaults import DEFAULT_SEG_FEATURE_WORKFLOW_DATASETS

    if datasets is None:
        datasets = get_datasets_in_collection(DEFAULT_SEG_FEATURE_WORKFLOW_DATASETS)
    elif datasets == "figure":
        datasets = CDH5_SEG_FIG_CLASSIC_FEAT_EXAMPLES

    out_dir = get_output_path(__file__)

    make_imaging_panels(
        CDH5_SEG_FIG_EXAMPLE.dataset_name,
        CDH5_SEG_FIG_EXAMPLE.position,
        CDH5_SEG_FIG_EXAMPLE.timepoint,
        __file__,
    )

    panels = {}
    for dataset in datasets:
        panels[dataset] = make_classic_feature_panels(dataset, out_dir / "classic_feature_panels")

    # make_contact_sheet of select features
    features = [
        Column.SegData.ORIENTATION_DEG,
        Column.SegData.ASPECT_RATIO,
        Column.SegData.CENTROID_VELOCITY_ANGLE_DEG,
        Column.SegData.AREA_UM_SQ,
        Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG,
        Column.SegData.EDGE_FLUOR_MEAN,
    ]

    classic_feat_fig_example_path = make_feature_contact_sheet(
        datasets=list(datasets),
        features=features,
        out_dir=out_dir / "feature_contact_sheet",
        figure_width=MAX_FIGURE_WIDTH,
    )

    schematic_dir = [Path(fp) for fp in figure_assets.__path__]
    schematic_name = "cdh5_classic_seg_schematic.svg"
    schematic_fps = [fp for fdir in schematic_dir for fp in fdir.rglob(schematic_name)]
    if len(schematic_fps) == 1:
        schematic_fp = schematic_fps[0]
    else:
        raise FileNotFoundError(
            f"Expected to find exactly one file matching {schematic_name} in {schematic_dir}, but found {len(schematic_fps)}: {schematic_fps}"
        )
    panel_A_height = 3

    figure_panels = [
        FigurePanel(
            letter="A",
            path=schematic_fp,
            x_position=0,
            y_position=0,
            x_offset=0,
            y_offset=0.2,
        ),
        FigurePanel(
            letter="B",
            path=classic_feat_fig_example_path,
            x_position=0,
            y_position=panel_A_height,
            x_offset=0,
            y_offset=0,
        ),
    ]

    build_figure_from_panels(
        figure_panels,
        out_dir / "cdh5_seg_feats_panels.svg",
        width=MAX_FIGURE_WIDTH,
        height=panel_A_height + MAX_FIGURE_WIDTH * len(datasets) / len(features),
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
