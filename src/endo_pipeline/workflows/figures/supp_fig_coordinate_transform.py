def main() -> None:
    """Generate the panels for the supplementary figure on coordinate transforms."""
    import matplotlib.pyplot as plt

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.supp_fig_coordinate_transform import (
        perform_latent_walk_along_top_pcs,
        plot_2d_latent_walk,
    )
    from endo_pipeline.settings.figures import MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH

    plt.style.use("endo_pipeline.figure")
    output_path = get_output_path("supp_fig_coords")

    # perform latent walk along top 3 PCs and save the resulting contact sheet
    latent_walk_filename = "latent_walk_top_3_pcs"

    walk_img_grid = perform_latent_walk_along_top_pcs(output_path, latent_walk_filename)
    latent_walk_path = output_path / f"{latent_walk_filename}_scale_bar_10um.svg"

    # take the images from the latent walk along PCs 1 and 2 and plot them
    # as a "2D" walk to motivate the polar coordinate transform
    latent_walk_2d_filename = "latent_walk_pc1_pc2_2d"

    images_pc1 = walk_img_grid[0]
    images_pc2 = walk_img_grid[1]
    # plot walk along PC 1 horizontally and along PC 2 vertically to create a "2D" walk
    # on the PC1 and PC2 axes with the center image for both at the origin
    latent_walk_2d_path = plot_2d_latent_walk(
        images_pc1,
        images_pc2,
        output_path,
        latent_walk_2d_filename,
        fig_kwargs={"figsize": (4.5, 4.5), "layout": "constrained"},
        gridspec_kwargs={"wspace": 0, "hspace": 0},
    )

    panels = [
        # --- Low flow dataset (row 1) ---
        FigurePanel(
            letter="A",
            path=latent_walk_path,
            x_position=0,
            y_position=0.0,
            x_offset=0.0,
            y_offset=0.0,
        ),
        FigurePanel(
            letter="B",
            path=latent_walk_2d_path,
            x_position=0.0,
            y_position=3.0,
            x_offset=0.00,
            y_offset=0.00,
        ),
    ]

    build_figure_from_panels(
        panels,
        output_path / "supp_fig_coordinate_transform.svg",
        width=MAX_FIGURE_WIDTH,
        height=MAX_FIGURE_HEIGHT,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
