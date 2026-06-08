def main() -> None:
    """Main function to general Supp. Fig. showing PC-based feature derivation and interpretation."""

    import matplotlib.pyplot as plt
    import pandas as pd

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.pca import fit_pca
    from endo_pipeline.library.visualize.columns import get_label_for_column
    from endo_pipeline.library.visualize.diffae_features import feature_viz
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.latent_walk import perform_and_plot_latent_walk_for_figures
    from endo_pipeline.library.visualize.multi_feature_correlation import (
        make_feature_correlation_panel,
    )
    from endo_pipeline.library.visualize.supp_fig_features import plot_2d_latent_walk
    from endo_pipeline.settings.diffae_feature_dataframes import (
        DIFFAE_PC_COLUMN_NAME_GROUPS,
        DIFFAE_PC_COLUMN_NAMES,
        NUM_LATENT_FEATURES,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.workflow_defaults import SEGMENTATION_FEATURE_COLUMNS

    plt.style.use("endo_pipeline.figure")

    save_dir = get_output_path(__file__)

    # plot cumulative explained variance ratio of PCA components
    pca = fit_pca(num_pcs=NUM_LATENT_FEATURES)
    fig, _ = feature_viz.plot_explained_variance(pca.explained_variance_ratio_, figsize=(2.3, 2.5))
    save_plot_to_path(
        fig,
        save_dir,
        "explained_variance_ratio",
        file_format=".svg",
        pad_inches=0,
        transparent=True,
    )

    # Correlation heatmap of ML-based features vs. measured features
    ml_columns = DIFFAE_PC_COLUMN_NAME_GROUPS["supp_figure"]
    measured_feature_columns = SEGMENTATION_FEATURE_COLUMNS["supp_figure"]
    ml_columns_100_pcs = (
        DIFFAE_PC_COLUMN_NAME_GROUPS["main_figure"] + DIFFAE_PC_COLUMN_NAME_GROUPS["first_100_pcs"]
    )

    # call feature correlation workflow to get the max correlation value for the
    # PCs that are not in the top 10 PCs up to PC 100, which we will report in
    # the figure legend
    _ = make_feature_correlation_panel(
        pc_columns=ml_columns_100_pcs,
        seg_columns=measured_feature_columns,
        output_path=save_dir,
    )
    corr_matrix_100pcs = (
        save_dir / "correlation_ml_based_features_100_vs_measured_features_correlation_matrix.csv"
    )
    correlation_matrix_100_pcs = pd.read_csv(corr_matrix_100pcs)
    non_fig_pcs = set(ml_columns_100_pcs) - set(ml_columns)
    non_fig_labels = [get_label_for_column(col) for col in non_fig_pcs]
    biggest_corr_mag = correlation_matrix_100_pcs[non_fig_labels].abs().max().max()
    print(f"Biggest correlation magnitude for non-figure PCs up to PC 100: {biggest_corr_mag}")

    # Now call on just the supplementary figure PCs to get the correlation
    # heatmap for the figure
    feature_correlations_path = make_feature_correlation_panel(
        pc_columns=ml_columns,
        seg_columns=measured_feature_columns,
        output_path=save_dir,
        figure_size=(6.0, 2.75),
    )

    # perform latent walk along top 10 PCs and save the resulting contact sheet
    latent_walk_filename = "latent_walk_top_10_pcs"
    latent_walk_path, walk_img_grid = perform_and_plot_latent_walk_for_figures(
        save_path=save_dir,
        filename=latent_walk_filename,
        walk_column_names=DIFFAE_PC_COLUMN_NAMES[:10],  # walk along top 10 PCs
        figsize=(3.8, 5.2),
        sigma=3,
        n_steps=7,
        scale_bar_um=20,
        num_gpus=NUM_GPUS,
    )

    # Take the images from the latent walk along PCs 1 and 2 and plot them as a
    # "2D" walk to motivate the polar coordinate transform. Just (-3 sigma, 0,
    # +3 sigma) along each PC, so the grid is 3x3 with the center image repeated
    # in the middle (showcase the extreme points along with the origin).
    latent_walk_2d_filename = "latent_walk_pc1_pc2_2d"
    n_steps = walk_img_grid[0].shape[0]
    center = n_steps // 2
    images_pc1 = walk_img_grid[0][[0, center, -1]]
    images_pc2 = walk_img_grid[1][[0, center, -1]]

    latent_walk_2d_path = plot_2d_latent_walk(
        images_pc1,
        images_pc2,
        save_dir,
        latent_walk_2d_filename,
    )

    # build figure with panels
    panels = [
        FigurePanel(
            letter="A",
            path=save_dir / "explained_variance_ratio.svg",
            x_position=0,
            y_position=0,
            x_offset=-0.1,
            y_offset=-0.05,
        ),
        FigurePanel(
            letter="B",
            path=latent_walk_path,
            x_position=2.3,
            y_position=0.0,
            x_offset=0.05,
            y_offset=0.1,
        ),
        FigurePanel(
            letter="C",
            path=latent_walk_2d_path,
            x_position=0.0,
            y_position=2.5,
            x_offset=0.05,
            y_offset=0.1,
        ),
        FigurePanel(
            letter="D",
            path=feature_correlations_path,
            x_position=0,
            y_position=5.2,
            x_offset=0.1,
            y_offset=0,
        ),
    ]

    build_figure_from_panels(
        panels,
        save_dir / "Figure_S5_features.svg",
        width=MAX_FIGURE_WIDTH,
        height=MAX_FIGURE_HEIGHT,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
