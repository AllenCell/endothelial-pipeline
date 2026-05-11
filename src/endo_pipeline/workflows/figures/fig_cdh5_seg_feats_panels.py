from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None) -> None:
    """Produces figure panels for the CDH5 segmentation and classic feature workflow figure.
    This includes imaging panels showing the segmentation steps and 2D histograms of classic
    features for each of the PCA reference datasets.

    #test-ready #cpu-only
    """
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.lib_cdh5_seg_feats_fig_panels import (
        make_classic_feature_panels,
        make_imaging_panels,
    )
    from endo_pipeline.settings import CDH5_SEG_FIG_EXAMPLE, DEFAULT_SEG_FEATURE_WORKFLOW_DATASETS

    if datasets is None:
        datasets = get_datasets_in_collection(DEFAULT_SEG_FEATURE_WORKFLOW_DATASETS)

    out_dir = get_output_path(__file__)

    make_imaging_panels(
        CDH5_SEG_FIG_EXAMPLE.dataset_name,
        CDH5_SEG_FIG_EXAMPLE.position,
        CDH5_SEG_FIG_EXAMPLE.timepoint,
        __file__,
    )

    make_classic_feature_panels(datasets, out_dir / "classic_feature_panels")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
