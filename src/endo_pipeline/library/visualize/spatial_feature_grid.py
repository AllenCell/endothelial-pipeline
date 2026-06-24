"""Helper function for visualizing spatial feature values on a crop grid."""

import colorcet as cc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm as mpl_cm
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.gridspec import GridSpec

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
    add_optical_flow_features,
)
from endo_pipeline.library.process.image_processing import (
    load_processed_bf_std_dev_image_crop,
    load_processed_egfp_image_crop,
)
from endo_pipeline.library.visualize.figure_utils import add_scalebar
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_metadata import COLUMN_METADATA, ColumnMetadata
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    METADATA_COLUMNS_TO_KEEP,
)
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_XSMALL
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_PATCH_TYPE
from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES


def _get_colormap_for_feature(feature: str, default_colormap: Colormap) -> Colormap:
    """Return the appropriate colormap for a given feature column.

    Parameters
    ----------
    feature
        Column name to look up.
    default_colormap
        Colormap returned when no feature-specific override exists.

    Returns
    -------
    Colormap
        ``CET_C8`` for polar angle, ``"cool"`` for unit-vector mean,
        otherwise *default_colormap*.
    """
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
    example_labels: list[str],
    example_subtitles: list[str] | None = None,
) -> None:
    """Render image rows at the top of the figure.

    Each entry in *image_rows* becomes one row of grayscale images with
    yellow grid lines, scale bars, and a y-axis row label.

    Parameters
    ----------
    fig
        Parent figure.
    axes
        2-D array of axes (rows x columns).
    gs
        GridSpec used to create the colorbar-column placeholder.
    image_rows
        Mapping of row label to a list of images (one per example).
    n_examples
        Number of example columns.
    crop_size
        Grid-line spacing in pixels.
    example_labels
        Column titles placed above the first image row.
    """
    for img_row_idx, (row_label, img_list) in enumerate(image_rows.items()):
        for col_idx, img in enumerate(img_list):
            ax = axes[img_row_idx, col_idx]
            ax.imshow(img, cmap="gray", aspect="equal")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if img_row_idx == 0:
                ax.set_title(example_labels[col_idx], fontsize=FONTSIZE_MEDIUM, pad=10)
                if example_subtitles and col_idx < len(example_subtitles):
                    ax.text(
                        0.5,
                        1.01,
                        example_subtitles[col_idx],
                        transform=ax.transAxes,
                        fontsize=FONTSIZE_XSMALL,
                        fontweight="normal",
                        ha="center",
                        va="bottom",
                    )

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


def _draw_colored_feature_patch(
    ax: Axes,
    df_grid: pd.DataFrame,
    feature: str,
    crop_size: int,
    colormap: Colormap,
    norm: Normalize,
    col_idx: int,
    row_idx: int,
    metadata: ColumnMetadata | None,
    n_image_rows: int,
    example_labels: list[str],
    example_subtitles: list[str] | None = None,
) -> None:
    """Draw coloured feature patches and style the axis.

    Parameters
    ----------
    ax
        Axes to draw on and configure.
    df_grid
        DataFrame with start_x, start_y, and *feature* columns.
    feature
        Column name whose values determine patch colour.
    crop_size
        Width and height of each patch in pixels.
    colormap
        Colormap used to map normalised feature values to colours.
    norm
        Normalisation applied to feature values before colour mapping.
    col_idx
        Column index of this axis in the grid.
    row_idx
        Row index (within feature rows, 0-based).
    metadata
        Optional column metadata providing a display label.
    n_image_rows
        Number of image rows above the feature rows.  Column titles
        are added only when there are no image rows.
    example_labels
        Column titles for each example.
    """
    # Draw coloured patches
    for _, data_row in df_grid.iterrows():
        sx, sy = data_row[Column.DiffAEData.START_X], data_row[Column.DiffAEData.START_Y]
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
    x_min = df_grid[Column.DiffAEData.START_X].min()
    x_max = df_grid[Column.DiffAEData.START_X].max() + crop_size
    y_min = df_grid[Column.DiffAEData.START_Y].min()
    y_max = df_grid[Column.DiffAEData.START_Y].max() + crop_size
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # invert y for image coords

    # Style axis
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if col_idx == 0:
        label = metadata.label_with_unit if metadata is not None else feature
        ax.set_ylabel(label, fontsize=FONTSIZE_MEDIUM)

    if n_image_rows == 0 and row_idx == 0:
        ax.set_title(example_labels[col_idx], fontsize=FONTSIZE_MEDIUM)
        if example_subtitles and col_idx < len(example_subtitles):
            ax.text(
                0.5,
                1.02,
                example_subtitles[col_idx],
                transform=ax.transAxes,
                fontsize=FONTSIZE_XSMALL,
                fontweight="normal",
                ha="center",
                va="bottom",
            )


def _add_feature_colorbar(
    fig: plt.Figure,
    gs: GridSpec,
    ax_row: int,
    n_examples: int,
    colormap: Colormap,
    vmin: float | None,
    vmax: float | None,
    metadata: ColumnMetadata | None,
) -> None:
    """Add a colorbar in the dedicated GridSpec column for a feature row.

    Parameters
    ----------
    fig
        Parent figure.
    gs
        GridSpec layout.
    ax_row
        Row index in *gs* for the colorbar.
    n_examples
        Number of example columns (the colorbar lives in column
        ``n_examples``).
    colormap
        Colormap matching the feature patches.
    vmin
        Minimum value for the colour scale.  Defaults to 0 if None.
    vmax
        Maximum value for the colour scale.  Defaults to 1 if None.
    metadata
        Optional column metadata for custom tick positions and labels.
    """
    col_vmin = vmin if vmin is not None else 0
    col_vmax = vmax if vmax is not None else 1
    norm = Normalize(vmin=col_vmin, vmax=col_vmax)
    sm = mpl_cm.ScalarMappable(cmap=colormap, norm=norm)

    host_ax = fig.add_subplot(gs[ax_row, n_examples])
    host_ax.set_frame_on(False)
    host_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cbar_ax = host_ax.inset_axes((0.0, 0.1, 1.0, 0.8))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.outline.set_visible(False)  # type: ignore[operator]
    cbar.ax.tick_params(labelsize=FONTSIZE_XSMALL)
    if metadata is not None and metadata.ticks is not None:
        cbar.set_ticks(metadata.ticks)
        if metadata.tick_labels is not None:
            cbar.set_ticklabels(metadata.tick_labels, fontsize=FONTSIZE_XSMALL)


def _load_example_data(
    example_images: list,
    include_bf_images: bool = False,
    image_crop_size: int = 768,
) -> tuple[list[pd.DataFrame], dict[str, list[np.ndarray]], list[str], list[str]]:
    """Load images, feature dataframes, and labels for a list of ExampleImage objects.

    Parameters
    ----------
    example_images
        List of ``ExampleImage`` named-tuples (from settings.examples).
    include_bf_images
        If True, also load BF standard-deviation projection images.
    image_crop_size
        Pixel size of the square crop loaded for each example image.

    Returns
    -------
    example_dfs : list[pd.DataFrame]
        Feature dataframes, one per example image.
    image_rows : dict[str, list[np.ndarray]]
        Mapping of row label to loaded images.
    example_labels : list[str]
        Column titles derived from shear-stress conditions.
    """
    manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[MIGRATION_COHERENCE_PATCH_TYPE]
    manifest = load_dataframe_manifest(manifest_name)
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP["grid_based"], *DYNAMICS_COLUMN_NAMES]

    example_dfs: list[pd.DataFrame] = []
    example_labels: list[str] = []
    example_subtitles: list[str] = []
    gfp_images: list[np.ndarray] = []
    bf_images: list[np.ndarray] = []

    for i, example in enumerate(example_images):
        dataset_name = example.dataset_name
        dataset_config = load_dataset_config(dataset_name)

        gfp_images.append(
            load_processed_egfp_image_crop(
                dataset_config,
                example.position,
                example.timepoint,
                example.crop_x_start,
                example.crop_y_start,
                crop_size=image_crop_size,
            )
        )

        if include_bf_images:
            bf_images.append(
                load_processed_bf_std_dev_image_crop(
                    dataset_config,
                    example.position,
                    example.timepoint,
                    example.crop_x_start,
                    example.crop_y_start,
                    crop_size=image_crop_size,
                )
            )

        # Load and filter feature dataframe
        df_delay = load_dataframe(manifest.locations[dataset_name], delay=True)
        df_features = df_delay[columns_to_compute].compute()
        df_features = add_optical_flow_features(df_features, datasets=[dataset_name])
        df_example = df_features[
            (df_features[Column.POSITION] == example.position)
            & (df_features[Column.TIMEPOINT] == example.timepoint)
        ]
        example_dfs.append(df_example)

        shear_stress = dataset_config.flow_conditions[0].shear_stress_bin
        example_labels.append(f"{shear_stress} dyn/cm{Unicode.SQUARED}")
        example_subtitles.append(f"Replicate {dataset_config.replicate_number}")

    image_rows: dict[str, list[np.ndarray]] = {"VE-cadherin\nmax int. proj.": gfp_images}
    if include_bf_images:
        image_rows["Brightfield\nstd. dev. proj."] = bf_images

    return example_dfs, image_rows, example_labels, example_subtitles


def create_panel_spatial_feature_grid(
    feature_columns: list[str],
    example_images: list,
    include_bf_images: bool = False,
    image_crop_size: int = 768,
    grid_start_xy: tuple[int, int] = (128, 128),
    grid_dimensions: tuple[int, int] = (3, 3),
    cmap: str = "viridis",
    figure_size: tuple[float, float] | None = None,
) -> plt.Figure:
    """Create a figure showing spatial feature values on a grid for multiple examples.

    Parameters
    ----------
    feature_columns
        Column names to visualize (one row per feature).
    example_images
        ``ExampleImage`` named-tuples. Images, feature dataframes, and
        labels are loaded automatically from these.
    include_bf_images
        Also load BF std-dev projection images.
    image_size
        Pixel size for loaded images.
    grid_start_xy
        (start_x, start_y) of the upper-left crop in the grid.
    grid_dimensions
        (n_cols, n_rows) of the feature grid.
    cmap
        Default matplotlib colormap name.
    figure_size
        Figure size in inches. Auto-computed if None.

    Returns
    -------
    plt.Figure
        The assembled figure.
    """
    example_dataframes, image_rows, example_labels, example_subtitles = _load_example_data(
        example_images,
        include_bf_images=include_bf_images,
        image_crop_size=image_crop_size,
    )

    # Compute grid positions from start_xy and dimensions
    res_level_0_patch_size = 256
    n_cols_grid, n_rows_grid = grid_dimensions
    sx0, sy0 = grid_start_xy
    grid_positions = [
        (sx0 + col * res_level_0_patch_size, sy0 + row * res_level_0_patch_size)
        for row in range(n_rows_grid)
        for col in range(n_cols_grid)
    ]
    positions_df = pd.DataFrame(
        grid_positions, columns=[Column.DiffAEData.START_X, Column.DiffAEData.START_Y]
    )

    n_features = len(feature_columns)
    n_examples = len(example_dataframes)
    n_image_rows = len(image_rows)
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
        layout_engine.set(w_pad=0.01, h_pad=0.01)  # type: ignore[call-arg]

    # Create axes grid
    axes = np.empty((n_rows, n_examples), dtype=object)
    for row in range(n_rows):
        for col in range(n_examples):
            axes[row, col] = fig.add_subplot(gs[row, col])

    # Image rows at the top
    _add_image_rows(
        fig,
        axes,
        gs,
        image_rows,
        n_examples,
        res_level_0_patch_size,
        example_labels,
        example_subtitles,
    )

    # Feature rows
    colormap_default = mpl_cm.get_cmap(cmap)
    for row_idx, feature in enumerate(feature_columns):
        ax_row = row_idx + n_image_rows
        metadata = COLUMN_METADATA.get(feature)
        vmin = getattr(metadata, "min", None)
        vmax = getattr(metadata, "max", None)
        colormap = _get_colormap_for_feature(feature, colormap_default)

        for col_idx, df_example in enumerate(example_dataframes):
            ax = axes[ax_row, col_idx]

            grid_data = df_example[
                [Column.DiffAEData.START_X, Column.DiffAEData.START_Y, feature]
            ].dropna(subset=[feature])
            grid_data = grid_data.merge(
                positions_df, on=[Column.DiffAEData.START_X, Column.DiffAEData.START_Y]
            )

            col_vmin = vmin if vmin is not None else grid_data[feature].min()
            col_vmax = vmax if vmax is not None else grid_data[feature].max()
            norm = Normalize(vmin=col_vmin, vmax=col_vmax)
            _draw_colored_feature_patch(
                ax,
                grid_data,
                feature,
                res_level_0_patch_size,
                colormap,
                norm,
                col_idx,
                row_idx,
                metadata,
                n_image_rows,
                example_labels,
                example_subtitles,
            )

        _add_feature_colorbar(fig, gs, ax_row, n_examples, colormap, vmin, vmax, metadata)

    return fig
