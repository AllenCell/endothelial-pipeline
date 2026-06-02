def main() -> None:
    """Main function to general Supp. Fig. showing PC-based feature derivation and interpretation."""
    import logging

    import matplotlib.pyplot as plt
    import pandas as pd

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.cli.demo_mode_defaults import use_default_collection
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.pca import fit_pca
    from endo_pipeline.library.visualize.columns import get_label_for_column
    from endo_pipeline.library.visualize.diffae_features import feature_viz
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.multi_feature_correlation_viz import (
        get_df_for_feature_correlation_viz,
        visualize_correlation_heatmaps,
    )
    from endo_pipeline.library.visualize.supp_fig_features import (
        perform_latent_walk_along_top_pcs,
        plot_2d_latent_walk,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import (
        DIFFAE_PC_COLUMN_NAME_GROUPS,
        NUM_LATENT_FEATURES,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
    from endo_pipeline.settings.workflow_defaults import (
        DATASET_INFO_COLUMNS,
        DEFAULT_PCA_DATASET_COLLECTION_NAME,
        SEGMENTATION_FEATURE_COLUMNS,
    )

    logger = logging.getLogger(__name__)

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
    dataset_name_list = use_default_collection(None, DEFAULT_PCA_DATASET_COLLECTION_NAME)
    ml_columns = DIFFAE_PC_COLUMN_NAME_GROUPS["supp_figure"]
    measured_feature_columns = SEGMENTATION_FEATURE_COLUMNS["supp_figure"]
    ml_columns_100_pcs = (
        DIFFAE_PC_COLUMN_NAME_GROUPS["main_figure"] + DIFFAE_PC_COLUMN_NAME_GROUPS["first_100_pcs"]
    )

    df = get_df_for_feature_correlation_viz(
        dataset_name_list=dataset_name_list,
        dataset_info_columns=DATASET_INFO_COLUMNS,
        segmentation_feature_columns=measured_feature_columns,
        pc_columns=ml_columns_100_pcs,
    )

    label_column_tuples = [
        (
            "ML-based Features 100",
            [get_label_for_column(col, single_line=True) for col in ml_columns_100_pcs],
        ),
        (
            "Measured Features",
            [get_label_for_column(col, single_line=True) for col in measured_feature_columns],
        ),
    ]

    visualize_correlation_heatmaps(
        dataset_name="aggregate",
        df_dataset=df,
        label_column_tuples=label_column_tuples,
        out_dir=save_dir,
        cross_correlation_only=True,
        figsize_cluster_heatmap=(4.35, 2.75),
        y_axis_label_coords=None,
    )
    # report the max correlation value in the heatmap for all PCs vs measured features
    # that are not in the supplementary figure
    corr_matrix_100pcs = (
        save_dir / "correlation_ml_based_features_100_vs_measured_features_correlation_matrix.csv"
    )
    df_100_pcs = pd.read_csv(corr_matrix_100pcs)
    non_fig_pcs = set(ml_columns_100_pcs) - set(ml_columns)
    biggest_corr_mag = (
        df_100_pcs[[get_label_for_column(col, single_line=True) for col in non_fig_pcs]]
        .abs()
        .max()
        .max()
    )
    logger.info(biggest_corr_mag)

    df = get_df_for_feature_correlation_viz(
        dataset_name_list=dataset_name_list,
        dataset_info_columns=DATASET_INFO_COLUMNS,
        segmentation_feature_columns=measured_feature_columns,
        pc_columns=ml_columns,
    )

    label_column_tuples = [
        ("ML-based Features", [get_label_for_column(col, single_line=True) for col in ml_columns]),
        (
            "Measured Features",
            [get_label_for_column(col, single_line=True) for col in measured_feature_columns],
        ),
    ]

    visualize_correlation_heatmaps(
        dataset_name="aggregate",
        df_dataset=df,
        label_column_tuples=label_column_tuples,
        out_dir=save_dir,
        cross_correlation_only=True,
        figsize_cluster_heatmap=(4.35, 2.75),
        y_axis_label_coords=None,
    )

    # perform latent walk along top 10 PCs and save the resulting contact sheet
    latent_walk_filename = "latent_walk_top_10_pcs"

    walk_img_grid = perform_latent_walk_along_top_pcs(
        save_dir, latent_walk_filename, num_pcs=10, figsize=(3.8, 5.2), num_gpus=NUM_GPUS
    )
    latent_walk_path = save_dir / f"{latent_walk_filename}_scale_bar_10um.svg"

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
            path=save_dir / "correlation_ml_based_features_vs_measured_features_heatmap.svg",
            x_position=0.0,
            y_position=5.2,
            x_offset=-0.05,
            y_offset=-0.1,
        ),
    ]

    build_figure_from_panels(
        panels,
        save_dir / "Supplemental_Figure_3.svg",
        width=MAX_FIGURE_WIDTH,
        height=MAX_FIGURE_HEIGHT,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
