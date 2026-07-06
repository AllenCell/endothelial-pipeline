def main() -> None:
    """
    **Supplemental Figure 5**. Derivation and validation of a low-dimensional,
    interpretable morphological feature space from DiffAE latent vector
    representations

    #supp-figure #pca #diffae #correlation-analysis

    | Panel | Description                                                                              | Notes      |
    | ----- | ---------------------------------------------------------------------------------------- | ---------- |
    | A     | Cumulative explained variance ratio for the DiffAE PCs                                   |            |
    | B     | Latent walk along the top 10 PCs                                                         | _uses GPU_ |
    | C     | 2D latent walk along PC1 and PC2                                                         | _uses GPU_ |
    | D     | Pearson correlation coefficient for ML-based features (DiffAE PCs) and measured features |            |

    ## Example usage

    To run the figure workflow:

    ```bash
    uv run endopipe supp-figure-5-features
    ```

    ## Figure panels

    Some panels in this workflow should be run with an NVIDIA GPU (as indicated
    by _uses GPU_ in the table above). Run this workflow with the GPU flag (`-g`
    or `--num-gpus`) to make sure GPUs are visible to the workflow. The workflow
    will run without a GPU, but will be noticeably slower.
    """

    import matplotlib.pyplot as plt
    import pandas as pd

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.pca import fit_pca
    from endo_pipeline.library.visualize.diffae_features import feature_viz
    from endo_pipeline.library.visualize.figures import (
        FigurePanel,
        build_figure_from_panels,
        parse_placeholder_panels,
    )
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
    from endo_pipeline.settings.plot_defaults import RECONSTRUCTION_RANDOM_SEED
    from endo_pipeline.settings.workflow_defaults import SEGMENTATION_FEATURE_COLUMNS

    plt.style.use("endo_pipeline.figure")

    output_path = get_output_path(__file__)

    placeholders = parse_placeholder_panels(None, ["A", "B", "C", "D"])

    # plot cumulative explained variance ratio of PCA components
    pca = fit_pca(num_pcs=NUM_LATENT_FEATURES)
    fig, _ = feature_viz.plot_explained_variance(pca.explained_variance_ratio_, figsize=(2.3, 2.5))
    save_plot_to_path(
        fig,
        output_path,
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

    # Call feature correlation workflow to get the max correlation value for the
    # PCs that are not in the top 10 PCs up to PC 100, which we will report in
    # the figure legend. Only do this if panel D is not a placeholder, since if
    # it is a placeholder then we won't have the correlation matrix CSV saved as
    # part of the workflow outputs and it would error when trying to read it.
    if not placeholders["D"]["placeholder"]:
        output_path_base = make_feature_correlation_panel(
            pc_columns=ml_columns_100_pcs,
            seg_columns=measured_feature_columns,
            output_path=output_path,
            force_labels_single_line=True,
            **placeholders["D"],
        )
        # replace "heatmap.svg" with "correlation_matrix.csv" to get the path to the
        # correlation matrix CSV that was saved as part of the workflow
        correlation_matrix_path = output_path_base.parent / (
            output_path_base.stem.replace("heatmap", "correlation_matrix") + ".csv"
        )
        correlation_matrix_100_pcs = pd.read_csv(correlation_matrix_path)
        non_fig_pcs = list(set(ml_columns_100_pcs) - set(ml_columns))
        biggest_corr_mag = correlation_matrix_100_pcs[non_fig_pcs].abs().max().max()
        print(f"Biggest correlation magnitude for non-figure PCs up to PC 100: {biggest_corr_mag}")

    # Now call on just the supplementary figure PCs to get the correlation
    # heatmap for the figure
    feature_correlations_path = make_feature_correlation_panel(
        pc_columns=ml_columns,
        seg_columns=measured_feature_columns,
        output_path=output_path,
        figure_size=(6.0, 2.75),
        force_labels_single_line=True,
        **placeholders["D"],
    )

    # perform latent walk along top 10 PCs and save the resulting contact sheet
    latent_walk_filename = "latent_walk_top_10_pcs"
    latent_walk_path, walk_img_grid = perform_and_plot_latent_walk_for_figures(
        output_path=output_path,
        filename=latent_walk_filename,
        walk_column_names=DIFFAE_PC_COLUMN_NAMES[:10],  # walk along top 10 PCs
        figure_size=(3.8, 5.2),
        sigma=3,
        n_steps=7,
        scale_bar_um=20,
        random_seed=RECONSTRUCTION_RANDOM_SEED,
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
        output_path,
        latent_walk_2d_filename,
    )

    # build figure with panels
    panels = [
        FigurePanel(
            letter="A",
            path=output_path / "explained_variance_ratio.svg",
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
        output_path / "supp_figure_5_features.svg",
        width=MAX_FIGURE_WIDTH,
        height=MAX_FIGURE_HEIGHT,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
