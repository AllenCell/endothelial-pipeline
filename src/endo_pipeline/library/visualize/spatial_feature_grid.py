"""Helper function for visualizing spatial feature values on a crop grid."""

from pathlib import Path

import colorcet as cc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm as mpl_cm
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.visualize.figure_utils import add_scalebar
from endo_pipeline.settings.column_metadata import COLUMN_METADATA, ColumnMetadata
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_XSMALL
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x


def _get_colormap_for_feature(
    feature: str, default_colormap: mpl_cm.ScalarMappable
) -> mpl_cm.ScalarMappable:
    """Return the appropriate colormap for a given feature column."""
    if feature == Column.DiffAEData.POLAR_ANGLE:
        return cc.cm.CET_C8
    if feature == Column.OpticalFlow.UNIT_VECTOR_MEAN:
        return mpl_cm.get_cmap("cool")
    return default_colormap


def _add_image_rows(
    fig: plt.Figure,
    axes: np.ndarray,
    gs: GridSpec,
    image_rows: dict[str, list[np.ndarray]],
    n_examples: int,
    crop_size: int,
    example_labels: list[str] | None,
) -> None:
    """Render image rows at the top of the figure."""
    for img_row_idx, (row_label, img_list) in enumerate(image_rows.items()):
        for col_idx, img in enumerate(img_list):
            ax = axes[img_row_idx, col_idx]
            ax.imshow(img, cmap="gray", aspect="equal")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if img_row_idx == 0 and example_labels is not None:
                ax.set_title(example_labels[col_idx], fontsize=FONTSIZE_MEDIUM)

            # Draw grid lines every crop_size pixels
            im_height, im_width = img.shape[:2]
            for x in range(crop_size, im_width, crop_size):
                ax.axvline(x, color="yellow", linewidth=0.5)
            for y in range(crop_size, im_height, crop_size):
                ax.axhline(y, color="yellow", linewidth=0.5)

            # Add scale bar to every image, label only on first image of first row
            add_scalebar(
                ax,
                scale_bar_um=20,
                pixel_size=PIXEL_SIZE_3i_20x,
                location="lower right",
                bar_thickness=25,
                padding=25,
                include_label=(img_row_idx == 0 and col_idx == 0),
                label_xy=(0.98, 0.06),
                label_fontsize=FONTSIZE_XSMALL,
            )

        # Row label
        axes[img_row_idx, 0].set_ylabel(row_label, fontsize=FONTSIZE_MEDIUM)

        # Hide the colorbar column cell for image rows
        empty_ax = fig.add_subplot(gs[img_row_idx, n_examples])
        empty_ax.set_visible(False)


def _filter_grid_data(
    df_example: pd.DataFrame,
    feature: str,
    start_x_col: str,
    start_y_col: str,
    grid_positions: list[tuple[int, int]] | None,
) -> pd.DataFrame:
    """Filter a dataframe to the relevant grid positions for a given feature."""
    if grid_positions is not None:
        mask = pd.Series(False, index=df_example.index)
        for gx, gy in grid_positions:
            mask |= (df_example[start_x_col] == gx) & (df_example[start_y_col] == gy)
        return df_example.loc[mask, [start_x_col, start_y_col, feature]].dropna(subset=[feature])
    return df_example[[start_x_col, start_y_col, feature]].dropna(subset=[feature])


def _plot_feature_patches(
    ax: Axes,
    grid_data: pd.DataFrame,
    feature: str,
    start_x_col: str,
    start_y_col: str,
    crop_size: int,
    colormap: mpl_cm.ScalarMappable,
    norm: Normalize,
) -> None:
    """Draw colored patches on *ax* for each crop position in *grid_data*."""
    for _, data_row in grid_data.iterrows():
        sx, sy = data_row[start_x_col], data_row[start_y_col]
        color = colormap(norm(data_row[feature]))
        rect = mpatches.FancyBboxPatch(
            (sx, sy),
            crop_size,
            crop_size,
            boxstyle="square,pad=0",
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.add_patch(rect)

    # Set axis limits based on grid extent
    x_min = grid_data[start_x_col].min()
    x_max = grid_data[start_x_col].max() + crop_size
    y_min = grid_data[start_y_col].min()
    y_max = grid_data[start_y_col].max() + crop_size
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # invert y for image coords


def _get_feature_limits(
    metadata: ColumnMetadata | None,
) -> tuple[float | None, float | None]:
    """Extract vmin/vmax from column metadata, returning None when unavailable."""
    if metadata is not None and metadata.min is not None and metadata.max is not None:
        return float(metadata.min), float(metadata.max)
    return None, None


def _configure_feature_axis(
    ax: Axes,
    col_idx: int,
    row_idx: int,
    metadata: ColumnMetadata | None,
    feature: str,
    n_image_rows: int,
    example_labels: list[str] | None,
) -> None:
    """Style a feature axis: remove spines/ticks and add labels."""
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if col_idx == 0:
        label = metadata.label_with_unit if metadata is not None else feature
        ax.set_ylabel(label, fontsize=FONTSIZE_MEDIUM)

    if n_image_rows == 0 and row_idx == 0 and example_labels is not None:
        ax.set_title(example_labels[col_idx], fontsize=FONTSIZE_MEDIUM)


def _add_feature_colorbar(
    fig: plt.Figure,
    gs: GridSpec,
    ax_row: int,
    n_examples: int,
    colormap: mpl_cm.ScalarMappable,
    vmin: float | None,
    vmax: float | None,
    metadata: ColumnMetadata | None,
) -> None:
    """Add a colorbar in the dedicated GridSpec column for a feature row."""
    col_vmin = vmin if vmin is not None else 0
    col_vmax = vmax if vmax is not None else 1
    norm = Normalize(vmin=col_vmin, vmax=col_vmax)
    sm = mpl_cm.ScalarMappable(cmap=colormap, norm=norm)

    # Create a host axis in the colorbar column, then place a shorter
    # inset axis inside it so the colorbar doesn't force the row taller.
    host_ax = fig.add_subplot(gs[ax_row, n_examples])
    host_ax.set_frame_on(False)
    host_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cbar_ax = host_ax.inset_axes([0.0, 0.1, 1.0, 0.8])  # [x0, y0, width, height] in axes frac
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=FONTSIZE_XSMALL)
    if metadata is not None and metadata.ticks is not None:
        cbar.set_ticks(metadata.ticks)
        if metadata.tick_labels is not None:
            cbar.set_ticklabels(metadata.tick_labels, fontsize=FONTSIZE_XSMALL)


def _load_example_data(
    example_images: list,
    include_bf_images: bool = False,
    image_crop_size: int = 768,
) -> tuple[list[pd.DataFrame], dict[str, list[np.ndarray]], list[str]]:
    """Load images, feature dataframes, and labels for a list of ExampleImage objects.

    Uses lazy imports to avoid coupling the visualization module to the full
    data-loading stack at import time.

    Parameters
    ----------
    example_images
        List of ``ExampleImage`` named-tuples (from settings.examples).
    include_bf_images
        If True, also load BF standard-deviation projection images and
        include them as a second image row.
    image_crop_size
        Pixel size of the square crop loaded for each example image.

    Returns
    -------
    (example_dfs, image_rows, example_labels)
    """
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
        add_optical_flow_features,
    )
    from endo_pipeline.library.process.image_processing import (
        load_processed_bf_std_dev_image_crop,
        load_processed_egfp_image_crop,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.dynamics_workflows import (
        DYNAMICS_COLUMN_NAMES,
        METADATA_COLUMNS_TO_KEEP,
    )
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
    from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

    manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[MIGRATION_COHERENCE_CROP_PATTERN]
    manifest = load_dataframe_manifest(manifest_name)
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP["grid"], *DYNAMICS_COLUMN_NAMES]

    example_dfs: list[pd.DataFrame] = []
    example_labels: list[str] = []
    gfp_images: list[np.ndarray] = []
    bf_images: list[np.ndarray] = []

    for i, example in enumerate(example_images):
        dataset_name = example.dataset_name
        dataset_config = load_dataset_config(dataset_name)

        # Load VE-cadherin MIP image
        gfp_mip = load_processed_egfp_image_crop(
            dataset_config,
            example.position,
            example.timepoint,
            example.crop_x_start,
            example.crop_y_start,
            crop_size=image_crop_size,
        )
        gfp_images.append(gfp_mip)

        # Optionally load BF std dev projection
        if include_bf_images:
            bf_std = load_processed_bf_std_dev_image_crop(
                dataset_config,
                example.position,
                example.timepoint,
                example.crop_x_start,
                example.crop_y_start,
                crop_size=image_crop_size,
            )
            bf_images.append(bf_std)

        # Load selected columns from feature dataframe
        df_delay = load_dataframe(manifest.locations[dataset_name], delay=True)
        df_features = df_delay[columns_to_compute].compute()
        df_features = add_optical_flow_features(df_features, datasets=[dataset_name])
        df_example = df_features[df_features[f"{Column.POSITION}"] == example.position]
        df_example = df_example[df_example[f"{Column.TIMEPOINT}"] == example.timepoint]

        example_dfs.append(df_example)
        shear_stress = dataset_config.flow_conditions[0].shear_stress_bin
        example_labels.append(f"{shear_stress} dyn/cm\u00b2\nExample {i + 1}")

    image_rows: dict[str, list[np.ndarray]] = {"VE-cadherin\nMIP": gfp_images}
    if include_bf_images:
        image_rows["BF\nstd. dev. proj."] = bf_images

    return example_dfs, image_rows, example_labels


def create_panel_spatial_feature_grid(
    feature_columns: list[str],
    example_images: list | None = None,
    example_dataframes: list[pd.DataFrame] | None = None,
    example_labels: list[str] | None = None,
    image_rows: dict[str, list[np.ndarray]] | None = None,
    include_bf_images: bool = False,
    image_crop_size: int = 768,
    crop_size: int = 256,
    grid_start_xy: tuple[int, int] | None = None,
    grid_dimensions: tuple[int, int] = (3, 3),
    grid_spacing: int | None = None,
    start_x_col: str = "start_x",
    start_y_col: str = "start_y",
    cmap: str = "viridis",
    figure_size: tuple[float, float] | None = None,
    save_dir: Path | None = None,
    filename: str = "spatial_feature_grid",
) -> plt.Figure:
    """Create a figure showing spatial feature values on a grid for multiple examples.

    Rows correspond to features, columns correspond to examples.
    Each panel shows a grid of colored patches where color encodes the
    feature value at that spatial crop position.

    Data can be supplied in two ways:

    1. **From ``example_images``** (preferred) — pass a list of
       ``ExampleImage`` named-tuples and the function will load VE-cadherin
       MIP images, feature dataframes, and example labels automatically.
       Set ``include_bf_images=True`` to also load BF std-dev projections.
    2. **Pre-loaded** — pass ``example_dataframes`` (and optionally
       ``image_rows`` / ``example_labels``) directly.

    Parameters
    ----------
    feature_columns
        List of column names to visualize (one per row).
    example_images
        List of ``ExampleImage`` named-tuples (from ``settings.examples``).
        When provided, images, dataframes, and labels are loaded
        automatically and ``example_dataframes`` / ``image_rows`` /
        ``example_labels`` are ignored.
    example_dataframes
        List of DataFrames, one per example. Each should be pre-filtered
        to a single position/timepoint. Only used when ``example_images``
        is not provided.
    example_labels
        Optional labels for each example (used as column titles).
    image_rows
        Optional dict mapping row label (e.g. "VE-cadherin MIP") to a list
        of 2D image arrays, one per example. Each entry becomes a row of
        images at the top of the figure. Only used when ``example_images``
        is not provided.
    include_bf_images
        If True, load BF standard-deviation projection images and include
        them as an additional image row. Only used with ``example_images``.
    image_crop_size
        Pixel size of the square crop loaded for each example image.
        Only used with ``example_images``.
    crop_size
        Size of each crop in pixels (used for patch width/height).
    grid_start_xy
        Optional (start_x, start_y) of the upper-left crop in the grid.
        Combined with ``grid_dimensions`` and ``grid_spacing`` to compute
        which grid positions to include. If None, all positions present
        in the data are plotted.
    grid_dimensions
        (n_cols, n_rows) number of crops in the x and y directions.
        Only used when ``grid_start_xy`` is provided.
    grid_spacing
        Distance in pixels between adjacent crop start positions.
        Defaults to ``crop_size`` if not provided (i.e. non-overlapping grid).
        Set to a smaller value (e.g. 128) for overlapping crops on a
        finer grid.
    start_x_col
        Column name for the x-coordinate of each crop.
    start_y_col
        Column name for the y-coordinate of each crop.
    cmap
        Matplotlib colormap name.
    figure_size
        Total figure size (width, height) in inches. If None, auto-computed
        as 2.5 inches per example (columns) and 2.5 inches per feature (rows).
    save_dir
        Directory to save the output figure. If None, figure is not saved.
    filename
        Filename (without extension) for saving.

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    # Load data from example images if provided
    if example_images is not None:
        example_dataframes, image_rows, example_labels = _load_example_data(
            example_images,
            include_bf_images=include_bf_images,
            image_crop_size=image_crop_size,
        )
    if example_dataframes is None:
        raise ValueError("Either example_images or example_dataframes must be provided.")
    # Compute grid positions from start_xy and dimensions
    spacing = grid_spacing if grid_spacing is not None else crop_size
    grid_positions: list[tuple[int, int]] | None = None
    if grid_start_xy is not None:
        n_cols_grid, n_rows_grid = grid_dimensions
        sx0, sy0 = grid_start_xy
        grid_positions = [
            (sx0 + col * spacing, sy0 + row * spacing)
            for row in range(n_rows_grid)
            for col in range(n_cols_grid)
        ]

    n_features = len(feature_columns)
    n_examples = len(example_dataframes)
    image_row_labels = list(image_rows.keys()) if image_rows else []
    n_image_rows = len(image_row_labels)
    n_rows = n_features + n_image_rows

    if figure_size is None:
        figure_size = (2.5 * n_examples, 2.5 * n_rows)

    fig = plt.figure(figsize=figure_size, layout="constrained")
    gs = GridSpec(
        n_rows,
        n_examples + 1,
        figure=fig,
        width_ratios=[1] * n_examples + [0.06],
        height_ratios=[1] * n_rows,
        hspace=0.05,
        wspace=0.02,
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(w_pad=0.01, h_pad=0.01)

    # Create axes from GridSpec
    axes = np.empty((n_rows, n_examples), dtype=object)
    for row in range(n_rows):
        for col in range(n_examples):
            axes[row, col] = fig.add_subplot(gs[row, col])

    # Display image rows at the top
    image_row_offset = n_image_rows
    if image_rows:
        _add_image_rows(fig, axes, gs, image_rows, n_examples, crop_size, example_labels)

    colormap_default = mpl_cm.get_cmap(cmap)

    for row_idx, feature in enumerate(feature_columns):
        ax_row = row_idx + image_row_offset
        metadata = COLUMN_METADATA.get(feature)
        vmin, vmax = _get_feature_limits(metadata)
        colormap = _get_colormap_for_feature(feature, colormap_default)

        for col_idx, df_example in enumerate(example_dataframes):
            ax = axes[ax_row, col_idx]

            if feature not in df_example.columns:
                ax.set_visible(False)
                continue

            grid_data = _filter_grid_data(
                df_example, feature, start_x_col, start_y_col, grid_positions
            )

            if grid_data.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                col_vmin = vmin if vmin is not None else grid_data[feature].min()
                col_vmax = vmax if vmax is not None else grid_data[feature].max()
                norm = Normalize(vmin=col_vmin, vmax=col_vmax)
                _plot_feature_patches(
                    ax, grid_data, feature, start_x_col, start_y_col, crop_size, colormap, norm
                )

            _configure_feature_axis(
                ax, col_idx, row_idx, metadata, feature, n_image_rows, example_labels
            )

        _add_feature_colorbar(fig, gs, ax_row, n_examples, colormap, vmin, vmax, metadata)

    if save_dir is not None:
        save_plot_to_path(
            fig,
            save_dir,
            filename,
            file_format=".svg",
            tight_layout=False,
            pad_inches=0,
        )

    return fig
