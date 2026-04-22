def main():
    import matplotlib.pyplot as plt

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH

    plt.style.use("endo_pipeline.figure")

    save_dir = get_output_path("figure_4")

    panels = [
        FigurePanel(
            letter="A",
            path=save_dir / "20250618_20X_FPT_fp_0_stable_mean_correlation.svg",
            x_position=0,
            y_position=0,
            x_offset=0,  # 0.2,
            y_offset=0,  # 0.08,
        ),
        FigurePanel(
            letter="B",
            path=save_dir / "20250611_20X_FPT_fp_3_stable_mean_correlation.svg",
            x_position=2,
            y_position=0,
            x_offset=0,  # 0,
            y_offset=0,  # 0.2,
        ),
        FigurePanel(
            letter="C",
            path=save_dir / "20250813_20X_FPT_fp_0_stable_mean_correlation.svg",
            x_position=4,
            y_position=0,
            x_offset=0,  # 0,
            y_offset=0,  # 0.2,
        ),
        FigurePanel(
            letter="D",
            path=save_dir / "FPT_correlation_summary.svg",
            x_position=0,
            y_position=3,
            x_offset=0,  # 0,
            y_offset=0,  # 0.2,
        ),
    ]

    build_figure_from_panels(
        figure_panels=panels,
        output_path=save_dir / "figure_4.svg",
        width=MAX_FIGURE_WIDTH,
        height=6,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
