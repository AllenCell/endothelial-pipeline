def main():
    import matplotlib.pyplot as plt

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.figure_fpt import generate_first_passage_time_example
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.settings.figures import MAX_FIGURE_WIDTH

    plt.style.use("endo_pipeline.figure")

    save_dir = get_output_path("figure_4")

    example_dataset_name = "20250618_20X"
    generate_first_passage_time_example(dataset_name=example_dataset_name, out_dir=save_dir)

    panels = [
        FigurePanel(
            letter="A",
            path=save_dir
            / example_dataset_name
            / f"{example_dataset_name}_FPT_fp_0_mean_3d_scatter.svg",
            x_position=0,
            y_position=0,
            x_offset=0,  # 0.2,
            y_offset=0,  # 0.08,
        ),
        FigurePanel(
            letter="B",
            path=save_dir / "20250618_20X_FPT_fp_0_stable_mean_correlation_for_figure.svg",
            x_position=2,
            y_position=0,
            x_offset=0,  # 0,
            y_offset=0.2,  # 0.2,
        ),
        FigurePanel(
            letter="C",
            path=save_dir / "20250611_20X_FPT_fp_3_stable_mean_correlation_for_figure.svg",
            x_position=4,
            y_position=0,
            x_offset=0,  # 0,
            y_offset=0.2,  # 0.2,
        ),
        FigurePanel(
            letter="D",
            path=save_dir / "FPT_correlation_summary_for_figure.svg",
            x_position=0,
            y_position=2.2,
            x_offset=0,  # 0,
            y_offset=0,  # 0.2,
        ),
    ]

    build_figure_from_panels(
        figure_panels=panels,
        output_path=save_dir / "figure_4.svg",
        width=MAX_FIGURE_WIDTH,
        height=6.3,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
