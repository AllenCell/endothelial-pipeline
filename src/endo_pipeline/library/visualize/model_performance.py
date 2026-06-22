"""Plotting methods for model performance."""

from pathlib import Path
from textwrap import wrap
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from matplotlib.figure import Figure
from matplotlib.layout_engine import LayoutEngine
from matplotlib.lines import Line2D

from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.model.model_comparison import ModelComparisonMetrics
from endo_pipeline.library.visualize.figure_utils import make_contact_sheet
from endo_pipeline.settings.examples import ExampleImage
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_SMALL, FONTSIZE_XSMALL

if TYPE_CHECKING:
    from numpy.typing import NDArray

MODEL_PERFORMANCE_CONTACT_SHEET_METRIC_KWARGS: dict[str, Any] = {
    "color": "yellow",
    "va": "bottom",
    "weight": "bold",
    "size": FONTSIZE_XSMALL * 0.5,
}
"""Kwargs for model performance contact sheet metric overlay text."""


def add_model_and_example_metadata_to_plot(
    fig: Figure, model_manifest_name: str, run_name: str, example: ExampleImage
):
    """Add metadata about the model and example to the plot."""

    model_text = f"Model = {model_manifest_name} ({run_name})"
    example_text = f"Example = {str(example).replace('_', ' ')}"

    fig.text(x=0.01, y=0.97, s=model_text, fontsize=FONTSIZE_XSMALL)
    fig.text(x=0.01, y=0.945, s=example_text, fontsize=FONTSIZE_XSMALL)


def plot_model_performance_negative_control_contact_sheet(
    output_path: Path,
    model_manifest_name: str,
    run_name: str,
    example: ExampleImage,
    conditioning_example: np.ndarray,
    diffusion_example: np.ndarray,
    noised_examples: list[np.ndarray],
    denoised_examples: list[np.ndarray],
    denoised_scrambled_latent: list[np.ndarray],
    denoised_scrambled_input: list[np.ndarray],
    noise_levels: list[float],
    conditioning_label: str,
    figure_size: tuple[float, float] = (6, 4.4),
) -> None:
    """
    Create and save the negative-control contact sheet for one example.

    Parameters
    ----------
    output_path
        Output path to save contact sheet.
    model_manifest_name
        Name of model training manifest.
    run_name
        Name of model training run.
    example
        Example image defining specific crop example.
    conditioning_example
        Cropped conditioning input example image.
    diffusion_example
        Cropped ground-truth diffusion input example image.
    noised_examples
        Noised example images at each noise level.
    denoised_examples
        Denoised example images from normal conditioning.
    denoised_scrambled_latent
        Denoised outputs from scrambled latent embedding.
    denoised_scrambled_input
        Denoised outputs from scrambled input image.
    noise_levels
        List of fractional noise levels.
    conditioning_label
        Label for the conditioning channel.
    figure_size
        Size of contact sheet.
    """

    num_noise_levels = len(noise_levels)

    panels = [
        *[conditioning_example.squeeze()] * num_noise_levels,
        *[diffusion_example.squeeze()] * num_noise_levels,
        *[img.squeeze() for img in noised_examples],
        *[img.squeeze() for img in denoised_examples],
        *[img.squeeze() for img in denoised_scrambled_latent],
        *[img.squeeze() for img in denoised_scrambled_input],
    ]

    col_titles = [
        f"{conditioning_label} input",
        "Original CDH5",
        "Noised CDH5",
        f"{conditioning_label}\nembedding",
        "Scrambled\nlatent embedding",
        "Scrambled\ninput image",
    ]

    row_titles = [f"{level * 100:.0f}% Noise" for level in noise_levels]

    fig = make_contact_sheet(
        panels=panels,
        max_rows=num_noise_levels,
        max_cols=len(col_titles),
        col_titles=col_titles,
        row_titles=row_titles,
        direction="top-down first",
        font_size=FONTSIZE_SMALL,
        subplot_kwargs={"frame_on": False},
        fig_kwargs={"figsize": figure_size},
        use_constrained_layout=True,
    )

    # Adjust the layout to make space for title
    layout_engine = cast(LayoutEngine, fig.get_layout_engine())
    layout_engine.set(**{"rect": (0, 0, 1, 0.95), "h_pad": 0.02, "w_pad": 0.02})
    fig.canvas.draw()

    # Add metadata about the model manifest name and run name
    add_model_and_example_metadata_to_plot(fig, model_manifest_name, run_name, example)

    # Add a title above the last three columns of predicted images
    column_width = fig.get_axes()[4].get_position().width
    center_x = fig.get_axes()[4].get_position().x0 + (column_width / 2)
    fig.text(
        x=center_x,
        y=0.97,
        s="Predicted CDH5 images",
        ha="center",
        fontsize=FONTSIZE_MEDIUM,
    )

    # Draw a line to group the last three columns of predicted images
    left_x = fig.get_axes()[3].get_position().x0
    right_x = fig.get_axes()[5].get_position().x1
    line = Line2D([left_x, right_x], [0.96, 0.96], lw=0.5, color="k")
    fig.add_artist(line)

    save_plot_to_path(
        fig,
        output_path,
        f"{example}_negative_control",
        tight_layout=False,
        show_and_close=True,
    )


def plot_model_performance_intermediate_level_contact_sheet(
    output_path: Path,
    model_manifest_name: str,
    run_name: str,
    example: ExampleImage,
    conditioning_example: np.ndarray,
    diffusion_example: np.ndarray,
    noised_examples: list[np.ndarray],
    denoised_examples: list[np.ndarray],
    noise_levels: list[float],
    conditioning_label: str,
    comparison_metrics: list[ModelComparisonMetrics] | None = None,
    figure_size: tuple[float, float] = (4.1, 4.4),
) -> None:
    """
    Create and save the intermediate noise level contact sheet for one example.

    Parameters
    ----------
    output_path
        Output path to save contact sheet.
    model_manifest_name
        Name of model training manifest.
    run_name
        Name of model training run.
    example
        Example image defining specific crop example.
    conditioning_example
        Cropped conditioning input example image.
    diffusion_example
        Cropped ground-truth diffusion input example image.
    noised_examples
        Noised example images at each noise level.
    denoised_examples
        Denoised example images from normal conditioning.
    noise_levels
        List of fractional noise levels.
    conditioning_label
        Label for the conditioning channel.
    comparison_metrics
        Calculate comparison metrics for each noise level, if available.
    figure_size
        Size of contact sheet.
    """

    num_noise_levels = len(noise_levels)

    panels = [
        *[conditioning_example.squeeze()] * num_noise_levels,
        *[diffusion_example.squeeze()] * num_noise_levels,
        *[img.squeeze() for img in noised_examples],
        *[img.squeeze() for img in denoised_examples],
    ]

    col_titles = [
        f"{conditioning_label} input",
        "Original CDH5",
        "Noised CDH5",
        "Predicted CDH5",
    ]

    row_titles = [f"{level * 100:.0f}% Noise" for level in noise_levels]

    fig = make_contact_sheet(
        panels=panels,
        max_rows=num_noise_levels,
        max_cols=len(col_titles),
        col_titles=col_titles,
        row_titles=row_titles,
        direction="top-down first",
        font_size=FONTSIZE_SMALL,
        subplot_kwargs={"frame_on": False},
        fig_kwargs={"figsize": figure_size},
        use_constrained_layout=True,
    )

    # Adjust the layout to make space for title
    layout_engine = cast(LayoutEngine, fig.get_layout_engine())
    layout_engine.set(**{"rect": (0, 0, 1, 0.92), "h_pad": 0.02, "w_pad": 0.02})
    fig.canvas.draw()

    # Add metadata about the model manifest name and run name
    add_model_and_example_metadata_to_plot(fig, model_manifest_name, run_name, example)

    if comparison_metrics is not None:
        for row_index, metrics in enumerate(comparison_metrics):
            x_pos = fig.get_axes()[3].get_position().x0
            y_pos = fig.get_axes()[row_index * len(col_titles)].get_position().y0
            text = "\n".join(
                [
                    f"Corr: {metrics.correlation:.3f}",
                    f"SSIM: {metrics.ssim:.3f}",
                    f"LPIPS: {metrics.lpips:.3f}",
                ]
            )
            fig.text(x=x_pos, y=y_pos, s=text, **MODEL_PERFORMANCE_CONTACT_SHEET_METRIC_KWARGS)

    save_plot_to_path(
        fig,
        output_path,
        f"{example}_intermediate_levels",
        tight_layout=False,
        show_and_close=True,
    )


def plot_model_performance_summary_contact_sheet(
    output_path: Path,
    model_manifest_name: str,
    run_name: str,
    example_groups: dict[str, list[ExampleImage]],
    conditioning_examples: list["NDArray"],
    diffusion_examples: list["NDArray"],
    denoised_examples: list["NDArray"],
    conditioning_label: str,
    comparison_metrics: list[ModelComparisonMetrics] | None = None,
    figure_size: tuple[float, float] | None = None,
) -> None:
    """
    Create and save model performance summary contact sheet for one example.

    Parameters
    ----------
    output_path
        Output path to save contact sheet.
    model_manifest_name
        Name of model training manifest.
    run_name
        Name of model training run.
    example
        Example image defining specific crop example.
    conditioning_crop
        Cropped conditioning input image.
    diffusion_crop
        Cropped ground-truth diffusion input image.
    noised_crops
        Noised images at each noise level.
    denoised_images
        Denoised outputs from normal conditioning.
    noise_levels
        List of fractional noise levels.
    conditioning_label
        Label for the conditioning channel.
    comparison_metrics
        Calculated comparison metrics for each example, if available.
    figure_size
        Size of contact sheet.
    """

    # Calculate number of groups and maximum number of examples in a group
    num_groups = len(example_groups)
    max_num_examples = max([len(examples) for examples in example_groups.values()])

    # Create empty crop to pad different number of examples in each group and
    # to space out the example label columns
    empty_axes = np.empty((128, 128))
    empty_axes[:] = np.nan
    empty_column = [empty_axes] * max_num_examples

    # Build out panels structure
    panels = []
    index_start = 0

    for examples in example_groups.values():
        # Add empty column to space out groups
        panels.extend(empty_column)

        # Calculate indices to use based on number of examples in the group
        num_examples_in_group = len(examples)
        index_end = index_start + num_examples_in_group

        # Add panels for conditioning examples and pad with empty axes
        panels.extend([ex.squeeze() for ex in conditioning_examples[index_start:index_end]])
        panels.extend([empty_axes] * (max_num_examples - num_examples_in_group))

        # Add panels for diffusion examples and pad with empty axes
        panels.extend([ex.squeeze() for ex in diffusion_examples[index_start:index_end]])
        panels.extend([empty_axes] * (max_num_examples - num_examples_in_group))

        # Add panels for denoising examples and pad with empty axes
        panels.extend([ex.squeeze() for ex in denoised_examples[index_start:index_end]])
        panels.extend([empty_axes] * (max_num_examples - num_examples_in_group))

        # Increment index for new example group
        index_start = index_start + num_examples_in_group

    # Build colume titles for each example group
    col_titles = [
        "",
        f"{conditioning_label}\ninput",
        "Original\nCDH5",
        "Predicted\nCDH5",
    ] * num_groups

    # Estimate figure size based on max number of examples in group
    figure_size = figure_size or (1.9 * num_groups, 0.59 + (max_num_examples * 0.52))

    # Build contact sheet
    fig = make_contact_sheet(
        panels=panels,
        max_rows=max_num_examples,
        max_cols=3 * num_groups,
        col_titles=col_titles,
        direction="top-down first",
        font_size=FONTSIZE_XSMALL,
        subplot_kwargs={"frame_on": False},
        gridspec_kwargs={"width_ratios": [0.5, 1, 1, 1] * num_groups},
        fig_kwargs={"figsize": figure_size},
        use_constrained_layout=True,
    )

    # Adjust the layout to make space for title
    layout_engine = cast(LayoutEngine, fig.get_layout_engine())
    width, height = figure_size
    top_padding = 0.25  # inches from top
    relative_top_padding = (height - top_padding) / height
    layout_engine.set(**{"rect": (0, 0, 1, relative_top_padding), "h_pad": 0.02, "w_pad": 0.02})
    fig.canvas.draw()

    # Calculate relative positions for title and grouping line
    title_position = 0.15  # inches from top
    relative_title_position = (height - title_position) / height
    line_position = 0.2
    relative_line_position = (height - line_position) / height

    # Get axes sizes for positioning labels
    label_width = fig.get_axes()[0].get_position().width
    column_width = fig.get_axes()[1].get_position().width
    row_height = fig.get_axes()[1].get_position().height

    flat_index = 0
    for group_index, (group_name, examples) in enumerate(example_groups.items()):
        # Add group name for set of three columns
        center_x = fig.get_axes()[group_index * 4 + 2].get_position().x0 + (column_width / 2)
        fig.text(
            x=center_x,
            y=relative_title_position,
            s=group_name,
            ha="center",
            fontsize=FONTSIZE_SMALL,
        )

        # Draw a line across the set of three columns
        left_x = fig.get_axes()[group_index * 4 + 1].get_position().x0
        right_x = fig.get_axes()[group_index * 4 + 3].get_position().x1
        line = Line2D([left_x, right_x], [relative_line_position] * 2, lw=0.5, color="k")
        fig.add_artist(line)

        # Add a label for each example
        for example_index, example in enumerate(examples):
            x = fig.get_axes()[group_index * 4].get_position().x0 + label_width / 2
            center_y = (
                fig.get_axes()[example_index * num_groups * 4 + 1].get_position().y0
                + row_height / 2
            )

            fig.text(
                x=x,
                y=center_y,
                s="\n".join(wrap(example.description, 14)),
                ha="center",
                va="center",
                rotation=90,
                fontsize=FONTSIZE_XSMALL * 0.75,
            )

            if comparison_metrics is not None:
                metrics = comparison_metrics[flat_index]
                x_pos = fig.get_axes()[group_index * 4 + 3].get_position().x0
                y_pos = fig.get_axes()[example_index * 4 * num_groups + 1].get_position().y0
                text = "\n".join(
                    [
                        f"Corr: {metrics.correlation:.3f}",
                        f"SSIM: {metrics.ssim:.3f}",
                        f"LPIPS: {metrics.lpips:.3f}",
                    ]
                )
                fig.text(x=x_pos, y=y_pos, s=text, **MODEL_PERFORMANCE_CONTACT_SHEET_METRIC_KWARGS)
                flat_index = flat_index + 1

    file_name_suffix = "_with_metrics" if comparison_metrics is not None else ""
    save_plot_to_path(
        fig,
        output_path,
        f"{model_manifest_name}_{run_name}_model_performance_summary{file_name_suffix}",
        tight_layout=False,
        show_and_close=True,
    )
