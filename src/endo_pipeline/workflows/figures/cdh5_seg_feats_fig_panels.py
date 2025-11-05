"""
Note
Becky says 20250326 (15 dyn) is probably the overall most ideal dataset for
creating panels depicting the segmentation workflow. The no flow dataset from
20250728 is also quite good but has some quirks around the feedings.
I have used the 20250728 dataset for the current panels since it is no flow.
It is worth noting that the plots use the PCA reference datasets which does NOT
include 20250728.
"""

from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.lib_fig_cdh5_classic_feat_workflow import (
    make_classic_feature_panels,
    make_imaging_panels,
)


def main() -> None:

    dataset_name = "20250818_20X"  # showing the no-flow dataset from PCA reference collection
    position = 4

    out_dir = get_output_path(__file__)

    make_imaging_panels(dataset_name, position, out_dir)

    make_classic_feature_panels(out_dir / "classic_feature_panels")


if __name__ == "__main__":
    main()
