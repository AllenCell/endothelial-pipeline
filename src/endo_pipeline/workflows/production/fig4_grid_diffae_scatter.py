import seaborn as sns
from matplotlib import pyplot as plt

from endo_pipeline.io import get_output_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.manifests import (
    ModelManifest,
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    NUM_PCS_TO_ANALYZE,
    ColumnName,
)


def main(
    dataset_name: str,
    model_manifest_name: ModelManifest = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    seg_feature_manifest_name: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    collection_name_for_pca: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    num_pcs: int = NUM_PCS_TO_ANALYZE,
    drop_rows_without_diffae_feats: bool = True,
    filtered: bool = False,
) -> None:

    model_manifest = load_model_manifest(model_manifest_name)
    grid_diffae_feat_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )

    outdir = get_output_path(__file__)
    FIG_DPI = 600

    # fit the PCA
    pca = fit_pca(
        dataset_collection_name=collection_name_for_pca,
        dataframe_manifest_name=grid_diffae_feat_manifest_name,
        num_pcs=num_pcs,
    )

    # read in the grid crop-based diffae features
    grid_diffae_manifest = load_dataframe_manifest(grid_diffae_feat_manifest_name)
    diffae_grid_crops = get_dataframe_for_dynamics_workflows(
        dataset_name,
        grid_diffae_manifest,
        pca,
        include_cell_piling=False,
        include_not_steady_state=False,
    )

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    sns.scatterplot(
        data=diffae_grid_crops,
        x="pc_1",
        y="pc_2",
        hue=ColumnName.TIMEPOINT,
        palette="flare",
        legend=False,
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    fig.savefig(
        outdir / f"{dataset_name}_grid_diffae_pc1_pc2_scatter.png", dpi=FIG_DPI, facecolor="white"
    )

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    sns.scatterplot(
        data=diffae_grid_crops,
        x="pc_1",
        y="pc_3",
        hue=ColumnName.TIMEPOINT,
        palette="flare",
        legend=False,
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 3")
    fig.savefig(
        outdir / f"{dataset_name}_grid_diffae_pc1_pc3_scatter.png", dpi=FIG_DPI, facecolor="white"
    )
