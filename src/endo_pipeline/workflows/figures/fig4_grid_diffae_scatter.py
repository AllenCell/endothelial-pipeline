from endo_pipeline.settings.diffae_feature_dataframes import NUM_PCS_TO_ANALYZE
from endo_pipeline.settings.workflow_defaults import DEFAULT_PCA_DATASET_COLLECTION_NAME


def main(
    dataset_name: str = "20250818_20X",
    collection_name_for_pca: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    num_pcs: int = NUM_PCS_TO_ANALYZE,
) -> None:
    """Creates scatter plots in DiffAE PC-space for grid crops colored by timepoint."""

    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import (
        get_no_flow_pc_space_example_points_fig4,
        make_pc_scatter_fig4a,
    )
    from endo_pipeline.library.visualize.seg_features.general_standard_plots import save_colorbar
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
    from endo_pipeline.settings.figures import FIGURE_SAVE_DPI
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    grid_diffae_feat_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
    )

    outdir = get_output_path(__file__)

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

    example_points, target_points = get_no_flow_pc_space_example_points_fig4(
        diffae_grid_crops, radius=2.2, origin_xy=(0, 0)
    )

    hue = ColumnName.TIMEPOINT
    color_palette = "inferno_r"

    fig1 = make_pc_scatter_fig4a(
        df=diffae_grid_crops,
        pc_col_for_xaxis="pc_1",
        pc_col_for_yaxis="pc_2",
        hue=hue,
        color_palette=color_palette,
    )
    fig2 = make_pc_scatter_fig4a(
        df=diffae_grid_crops,
        pc_col_for_xaxis="pc_1",
        pc_col_for_yaxis="pc_3",
        hue=hue,
        color_palette=color_palette,
    )
    for filetype in [".png", ".pdf"]:
        save_plot_to_path(
            figure=fig1,
            output_path=outdir,
            figure_name=f"{dataset_name}_grid_diffae_pc1_pc2_scatter",
            file_format=filetype,
            dpi=FIGURE_SAVE_DPI,
        )
        save_plot_to_path(
            figure=fig2,
            output_path=outdir,
            figure_name=f"{dataset_name}_grid_diffae_pc1_pc3_scatter",
            file_format=filetype,
            dpi=FIGURE_SAVE_DPI,
        )
        save_colorbar(
            outdir=outdir,
            colormap_name=color_palette,
            filename=f"{hue}_colorbar",
            filetype=filetype,
        )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
