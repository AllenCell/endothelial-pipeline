from endo_pipeline.manifests import ModelManifest
from endo_pipeline.settings import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    NUM_PCS_TO_ANALYZE,
    ColumnName,
)


def main(
    dataset_name: str = "20250818_20X",
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    collection_name_for_pca: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    num_pcs: int = NUM_PCS_TO_ANALYZE,
) -> None:

    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MultipleLocator

    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
    )
    from endo_pipeline.library.visualize.seg_features.general_standard_plots import save_colorbar
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
    from endo_pipeline.settings.figures import FIGURE_SAVE_DPI

    def make_pc_scatter(
        pc_col_for_xaxis: str,
        pc_col_for_yaxis: str,
        hue: str | ColumnName = ColumnName.TIMEPOINT,
        figsize=(2.5, 2.5),
        color_palette="viridis",
        marker=".",
        marker_size=5,
        linewidth=0,
        alpha=0.5,
    ) -> plt.Figure:

        if pc_col_for_xaxis not in DIFFAE_PC_COLUMN_NAMES:
            raise ValueError(f"pc_col_for_xaxis must be one of: {DIFFAE_PC_COLUMN_NAMES}")
        if pc_col_for_yaxis not in DIFFAE_PC_COLUMN_NAMES:
            raise ValueError(f"pc_col_for_yaxis must be one of: {DIFFAE_PC_COLUMN_NAMES}")
        if hue not in [x.value for x in ColumnName]:
            raise ValueError(f"hue must be one of: {[x.value for x in ColumnName]}")

        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(
            data=diffae_grid_crops,
            x=pc_col_for_xaxis,
            y=pc_col_for_yaxis,
            hue=hue,
            palette=color_palette,
            marker=marker,
            s=marker_size,
            alpha=alpha,
            linewidth=linewidth,
            legend=False,
            ax=ax,
        )
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.set_xlabel(pc_col_for_xaxis.upper().replace("_", " "))
        ax.set_ylabel(pc_col_for_yaxis.upper().replace("_", " "))
        ax.set_aspect("equal")

        return fig

    model_manifest = load_model_manifest(model_manifest_name)
    grid_diffae_feat_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
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

    hue = ColumnName.TIMEPOINT
    color_palette = "inferno_r"

    fig1 = make_pc_scatter(
        pc_col_for_xaxis="pc_1", pc_col_for_yaxis="pc_2", hue=hue, color_palette=color_palette
    )
    fig2 = make_pc_scatter(
        pc_col_for_xaxis="pc_1", pc_col_for_yaxis="pc_3", hue=hue, color_palette=color_palette
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
