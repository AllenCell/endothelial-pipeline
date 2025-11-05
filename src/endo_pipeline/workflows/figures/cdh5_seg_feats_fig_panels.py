from endo_pipeline.cli import tags

TAGS = [tags.CPU_ONLY]


def main() -> None:
    """Produces figure panels for the CDH5 segmentation and classic feature workflow figure.
    This includes imaging panels showing the segmentation steps and 2D histograms of classic
    features for each of the PCA reference datasets.
    """
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.lib_cdh5_seg_feats_fig_panels import (
        make_classic_feature_panels,
        make_imaging_panels,
    )
    from endo_pipeline.settings.examples import CDH5_SEG_FIG_EXAMPLE

    out_dir = get_output_path(__file__)

    make_imaging_panels(
        CDH5_SEG_FIG_EXAMPLE.dataset_name,
        CDH5_SEG_FIG_EXAMPLE.position,
        CDH5_SEG_FIG_EXAMPLE.timepoint,
        out_dir,
    )

    make_classic_feature_panels(out_dir / "classic_feature_panels")


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
