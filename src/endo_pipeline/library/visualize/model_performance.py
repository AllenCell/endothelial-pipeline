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
from endo_pipeline.library.visualize.figures import figure_panel
from endo_pipeline.settings.examples import ExampleImage
from endo_pipeline.settings.figures import (
    FONTSIZE_LARGE,
    FONTSIZE_MEDIUM,
    FONTSIZE_SMALL,
    FONTSIZE_XSMALL,
)

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
    Create and save model performance summary contact sheet for examples.

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


@figure_panel("Thumbnails for DiffAE architecture diagram")
def make_model_training_architecture_panel(
    output_path: Path,
    num_gpus: int | None = None,
    figure_size: tuple[float, float] = (6.5, 3.2),
) -> Path:

    from typing import cast

    import matplotlib.pyplot as plt
    from matplotlib import patches
    from numpy.random import default_rng
    from omegaconf import DictConfig

    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import load_image, load_model
    from endo_pipeline.io.load_models import instantiate_model_target_class
    from endo_pipeline.io.output import save_plot_to_path
    from endo_pipeline.library.model.diffae.eval_diffae import get_latent_vector_from_crop
    from endo_pipeline.library.model.diffae.generate_image import (
        generate_from_coords_and_noised_image,
    )
    from endo_pipeline.library.model.model_comparison import (
        load_transformed_conditioning_example_image,
        load_transformed_diffusion_example_image,
    )
    from endo_pipeline.library.process.image_processing import contrast_stretching
    from endo_pipeline.library.visualize.figure_utils import plot_image_thumbnail
    from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
        apply_img_transforms,
        create_data_dict_loaded_image,
        get_image_transforms,
        get_target_image_from_sample,
    )
    from endo_pipeline.manifests import get_zarr_location_for_position, load_model_manifest
    from endo_pipeline.settings.examples import EXAMPLES_DIFFAE_TRAINING_ARCHITECTURE_EXAMPLE
    from endo_pipeline.settings.image_data import (
        DIFFAE_ZARR_RESOLUTION_LEVEL,
        Z_SLICE_OFFSETS,
        PIXEL_SIZE_3i_20x_RESOLUTION_1,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT,
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        RANDOM_SEED,
    )

    # Set RNG seed
    rng = default_rng(seed=RANDOM_SEED)

    # Load model manifest and get run name (if not provided)
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    run_name = DEFAULT_MODEL_RUN_NAME

    # Load model for run and get model config. First load the model without
    # instantiation to grab the model config, then instantiate for use later.
    model_location = model_manifest.locations[run_name]
    model_ = load_model(model_location, instantiate=False)
    model_config: DictConfig = cast(DictConfig, model_.cfg)
    model = instantiate_model_target_class(model_)

    # Load dataset config for example
    example = EXAMPLES_DIFFAE_TRAINING_ARCHITECTURE_EXAMPLE
    dataset_config = load_dataset_config(example.dataset_name)
    assert dataset_config.center_z_plane is not None
    center_slice = dataset_config.center_z_plane[example.position]

    # Load raw image
    zarr_loc = get_zarr_location_for_position(dataset_config, example.position)
    raw_image = load_image(
        zarr_loc,
        level=DIFFAE_ZARR_RESOLUTION_LEVEL,
        timepoints=example.timepoint,
        squeeze=True,
        compute=True,
    )

    # Get slices from raw image
    cdh5_lower_slice = raw_image[0, center_slice - Z_SLICE_OFFSETS[0], :, :].squeeze()
    cdh5_slice = raw_image[0, center_slice, :, :].squeeze()
    cdh5_upper_slice = raw_image[0, center_slice + Z_SLICE_OFFSETS[1], :, :].squeeze()
    bf_lower_slice = raw_image[1, center_slice - Z_SLICE_OFFSETS[0], :, :].squeeze()
    bf_slice = raw_image[1, center_slice, :, :].squeeze()
    bf_upper_slice = raw_image[1, center_slice + Z_SLICE_OFFSETS[1], :, :].squeeze()

    # Save image thumbnail for each raw image slice
    for image, image_name, outline_color in [
        (cdh5_lower_slice, "cdh5_lower_slice", "white"),
        (cdh5_slice, "cdh5_slice", "white"),
        (cdh5_upper_slice, "cdh5_upper_slice", "white"),
        (bf_lower_slice, "bf_lower_slice", "black"),
        (bf_slice, "bf_slice", "black"),
        (bf_upper_slice, "bf_upper_slice", "black"),
    ]:
        image = contrast_stretching(image)
        plot_image_thumbnail(
            image,
            f"{image_name}_{dataset_config.name}_T{example.timepoint}",
            output_path,
            figsize=(0.7, 0.7),
            scalebar_size_um=100,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            file_format=".svg",
            outline_color=outline_color,
            bar_padding=30,
            bar_thickness=20,
            scalebar_location="lower right",
        )

    # Extract transformation steps and apply to image
    data = create_data_dict_loaded_image(raw_image)
    transforms = get_image_transforms(model_config)
    sample = apply_img_transforms(transforms, data)

    # Extract the target images
    crop_size = model_config.model.image_shape[-1]
    diffusion_fov = get_target_image_from_sample(sample, DEFAULT_CHANNEL_KEY_FOR_DIFFUSION_INPUT)
    conditioning_fov = get_target_image_from_sample(sample, model_config.model.condition_key)

    # Save image thumbnail for each model input crop with outline
    for image, image_name in [
        (diffusion_fov, "diffusion_input_fov"),
        (conditioning_fov, "conditioning_input_fov"),
    ]:
        fig, ax = plot_image_thumbnail(
            image.squeeze(),
            f"{image_name}_{dataset_config.name}_T{example.timepoint}",
            None,
            figsize=(0.7, 0.7),
            scalebar_size_um=100,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            file_format=".svg",
            bar_thickness=20,
            bar_padding=30,
            scalebar_location="lower right",
        )
        rect = patches.Rectangle(
            (example.crop_x_start, example.crop_y_start),
            crop_size,
            crop_size,
            linewidth=0.5,
            edgecolor="yellow",
            facecolor="none",
        )
        ax.add_patch(rect)
        save_plot_to_path(fig, output_path, image_name, file_format=".svg", pad_inches=0)

    # Load transformed conditioning and diffusion examples
    conditioning_ex = load_transformed_conditioning_example_image(example, model_config)
    diffusion_ex = load_transformed_diffusion_example_image(example, model_config)

    # Apply noise to conditioning image and then denoise
    noise = rng.standard_normal(size=conditioning_ex.shape)
    latent = get_latent_vector_from_crop(model, conditioning_ex, num_gpus=num_gpus)
    denoised_ex = generate_from_coords_and_noised_image(model, latent, noise, num_gpus=num_gpus)

    for image, image_name in [
        (conditioning_ex, "conditioning_input_crop"),
        (diffusion_ex, "diffusion_input_crop"),
        (denoised_ex, "denoised_image_by_bf_cond"),
        (noise, "noise_image"),
    ]:
        plot_image_thumbnail(
            image.squeeze(),
            image_name,
            output_path,
            figsize=(0.7, 0.7),
            scalebar_size_um=20,
            bar_padding=5,
            bar_thickness=5,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            file_format=".svg",
            scalebar_location="lower right",
        )

    # Create figure with just the panel title
    fig, ax = plt.subplots(figsize=figure_size, layout="constrained")
    ax.set_axis_off()
    fig.text(
        x=0.04,
        y=0.94,
        s="DiffAE training architecture and data preparation",
        fontweight="bold",
        fontsize=FONTSIZE_LARGE * 1.2,
    )

    return save_plot_to_path(
        fig,
        output_path,
        "model_training_architecture",
        file_format=".svg",
        tight_layout=False,
        show_and_close=True,
    )


@figure_panel("Contact sheet showing DiffAE model performance examples")
def make_model_performance_examples_panel(
    output_path: Path,
    num_gpus: int | None = None,
    figure_size: tuple[float, float] = (6.5, 4.75),
) -> Path:
    """
    Create contact sheet showing DiffAE model performance examples.

    Parameters
    ----------
    output_path
        Output path to save contact sheet.
    num_gpus
        Number of GPUs to use. If None, run on CPU.
    figure_size
        Size of contact sheet.

    Returns
    -------
    :
        Path to output contact sheet.
    """

    from typing import cast

    from numpy.random import default_rng
    from omegaconf import DictConfig

    from endo_pipeline.io import load_model
    from endo_pipeline.io.load_models import instantiate_model_target_class
    from endo_pipeline.library.model.diffae.eval_diffae import get_latent_vector_from_crop
    from endo_pipeline.library.model.diffae.generate_image import (
        generate_from_coords_and_noised_image,
    )
    from endo_pipeline.library.model.model_comparison import (
        load_transformed_conditioning_example_image,
        load_transformed_diffusion_example_image,
    )
    from endo_pipeline.library.model.model_performance import (
        denoise_with_scrambled_conditioning_input,
        denoise_with_scrambled_latent_vector,
    )
    from endo_pipeline.library.visualize.figure_utils import add_scalebar
    from endo_pipeline.manifests import load_model_manifest
    from endo_pipeline.settings.examples import DIFFAE_MODEL_PERFORMANCE_PANEL_EXAMPLES
    from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x_RESOLUTION_1
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        RANDOM_SEED,
    )

    # Set RNG seed
    rng = default_rng(seed=RANDOM_SEED)

    # Load model manifest and get run name (if not provided)
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    run_name = DEFAULT_MODEL_RUN_NAME

    # Load model for run and get model config. First load the model without
    # instantiation to grab the model config, then instantiate for use later.
    model_location = model_manifest.locations[run_name]
    model_ = load_model(model_location, instantiate=False)
    model_config: DictConfig = cast(DictConfig, model_.cfg)
    model = instantiate_model_target_class(model_)

    # Collect examples for panel
    all_conditioning_examples = []
    all_diffusion_examples = []
    all_denoised_examples = []
    all_denoised_scrambled_latent_examples = []
    all_denoised_scrambled_input_examples = []

    for example in DIFFAE_MODEL_PERFORMANCE_PANEL_EXAMPLES:
        # Load transformed conditioning and diffusion examples
        conditioning_ex = load_transformed_conditioning_example_image(example, model_config)
        diffusion_ex = load_transformed_diffusion_example_image(example, model_config)

        # Apply noise to conditioning image and then denoise
        noise = rng.standard_normal(size=conditioning_ex.shape)
        latent = get_latent_vector_from_crop(model, conditioning_ex, num_gpus=num_gpus)
        denoised_ex = generate_from_coords_and_noised_image(model, latent, noise, num_gpus=num_gpus)

        # Scrambled latent vector and scrambled input image
        denoised_scrambled_latent = denoise_with_scrambled_latent_vector(
            rng, model, [noise], latent, num_gpus
        )
        denoised_scrambled_input = denoise_with_scrambled_conditioning_input(
            rng, model, [noise], conditioning_ex, num_gpus
        )

        # Add examples to list for use in summary figure
        all_conditioning_examples.append(conditioning_ex)
        all_diffusion_examples.append(diffusion_ex)
        all_denoised_examples.append(denoised_ex)
        all_denoised_scrambled_latent_examples.append(denoised_scrambled_latent[0])
        all_denoised_scrambled_input_examples.append(denoised_scrambled_input[0])

    # Build panels and set column titles
    panels = [
        *[img.squeeze() for img in all_conditioning_examples],
        *[img.squeeze() for img in all_diffusion_examples],
        *[img.squeeze() for img in all_denoised_examples],
        *[img.squeeze() for img in all_denoised_scrambled_latent_examples],
        *[img.squeeze() for img in all_denoised_scrambled_input_examples],
    ]

    fig = make_contact_sheet(
        panels=panels,
        max_rows=len(DIFFAE_MODEL_PERFORMANCE_PANEL_EXAMPLES),
        max_cols=5,
        direction="top-down first",
        subplot_kwargs={"frame_on": False},
        fig_kwargs={"figsize": figure_size},
        use_constrained_layout=True,
    )

    # Add column titles with adjusted sizing
    titles = [
        ("Brightfield\nencoder input", FONTSIZE_MEDIUM),
        ("Target\nVE-cadherin", FONTSIZE_MEDIUM),
        ("latent vector\nfrom brightfield", FONTSIZE_SMALL),
        ("scrambled latent vector\nfrom brightfield", FONTSIZE_SMALL),
        ("latent vector from\nscrambled brightfield", FONTSIZE_SMALL),
    ]

    for index, (title, fontsize) in enumerate(titles):
        ax = fig.get_axes()[index]
        ax.set_title(title, fontsize=fontsize, weight="normal")

    # Adjust the layout to make space for title
    layout_engine = cast(LayoutEngine, fig.get_layout_engine())
    layout_engine.set(**{"rect": (0, 0, 1, 0.89), "h_pad": 0.02, "w_pad": 0.02})
    fig.canvas.draw()

    # Add a title above the last three columns of predicted images
    column_width = fig.get_axes()[3].get_position().width
    center_x = fig.get_axes()[3].get_position().x0 + (column_width / 2)
    fig.text(
        x=center_x,
        y=0.91,
        s="Predicted VE-Cadherin conditioned on:",
        ha="center",
        fontweight="bold",
        fontsize=FONTSIZE_MEDIUM,
    )

    # Draw a line to group the last three columns of predicted images
    left_x = fig.get_axes()[2].get_position().x0
    right_x = fig.get_axes()[4].get_position().x1
    line = Line2D([left_x, right_x], [0.90, 0.90], lw=0.5, color="k")
    fig.add_artist(line)

    # Add scale bar on every image with text label only on the top-left panel
    for i, ax in enumerate(fig.get_axes()):
        add_scalebar(
            ax,
            pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
            scale_bar_um=20,
            bar_thickness=3,
            padding=5,
            location="lower right",
            include_label=(i == 0),
        )

    # Add panel title
    fig.text(
        x=0.04,
        y=0.96,
        s="DiffAE performance evaluation and conditioning validation",
        fontweight="bold",
        fontsize=FONTSIZE_LARGE * 1.2,
    )

    return save_plot_to_path(
        fig,
        output_path,
        "model_performance_examples",
        file_format=".svg",
        tight_layout=False,
        show_and_close=True,
    )
