"""Methods for optical-flow comparison figures.

Each function in this module produces / plots a single component and is
ignorant of the overall figure layout.  Workflows are responsible for
orchestrating these building blocks (loading picks, allocating axes,
saving and composing the final figure).
"""

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_dataframe, load_image
from endo_pipeline.library.analyze.optical_flow import build_crop_grid
from endo_pipeline.library.visualize.figure_utils import add_scalebar
from endo_pipeline.manifests import get_zarr_location_for_position, load_dataframe_manifest
from endo_pipeline.settings import DIMENSION_ORDER
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_SMALL
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x_RESOLUTION_1
from endo_pipeline.settings.optical_flow import (
    OPTICAL_FLOW_COLUMNS_TO_COMPUTE,
    QUIVER_GRID_DIVISIONS,
)
from endo_pipeline.settings.workflow_defaults import GRID_BASED_FEATURES_UNFILTERED_MANIFEST_NAME


def load_optical_flow_feature_df(dataset_name: str) -> pd.DataFrame:
    """Load the per-crop feature DataFrame used to build the crop grid."""
    manifest = load_dataframe_manifest(GRID_BASED_FEATURES_UNFILTERED_MANIFEST_NAME)
    cols = list(OPTICAL_FLOW_COLUMNS_TO_COMPUTE["grid"])
    return load_dataframe(manifest.locations[dataset_name], delay=True)[cols].compute()


def build_bf_frame_cache(
    dataset_name: str,
    position: int,
    level: int,
    timepoints: list[int],
    df_dataset: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """Load and z-normalize the requested *timepoints* of a position's BF
    std-dev z-projection.

    Returns
    -------
    cache
        Mapping ``{timepoint: 2D float32 frame}`` (z-scored, percentile-clipped).
    crop_grid
        The position's crop grid (one row per crop, with START_X / START_Y /
        end_x / end_y / CROP_INDEX columns).
    """
    dataset_config = load_dataset_config(dataset_name)
    df_position = df_dataset[
        (df_dataset[Column.POSITION] == position) & (df_dataset[Column.TIMEPOINT].isin(timepoints))
    ].copy()
    crop_grid = build_crop_grid(df_position)

    zarr_path = get_zarr_location_for_position(dataset_config, position)
    image_dask = load_image(zarr_path, channels=["BF"], level=level, compute=False)
    z_axis = DIMENSION_ORDER.index("Z")
    z_projection = da.log(image_dask.std(axis=z_axis) + 1e-12)

    needed_indices = sorted({int(t) for t in timepoints})
    needed_frames = z_projection[needed_indices, 0].compute(scheduler="synchronous")
    cache: dict[int, np.ndarray] = {}
    for j, t in enumerate(needed_indices):
        frame = needed_frames[j].astype(np.float32, copy=False)
        lo, hi = np.percentile(frame, [0.1, 99.9])
        frame = np.clip(frame, lo, hi)
        std = frame.std()
        frame = (frame - frame.mean()) / (std if std > 0 else 1.0)
        cache[t] = frame
    return cache, crop_grid


def resolve_grid_crop(
    crop_grid: pd.DataFrame,
    *,
    grid_row: int,
    grid_col: int,
) -> tuple[int, int, int, int]:
    """Return the pixel bbox ``(sx, sy, ex, ey)`` of the crop at the
    1-indexed (``grid_row``, ``grid_col``) position in the regular crop grid.

    Row 1 / col 1 is the top-left crop; rows increase downward (sorted by
    ``START_Y``), cols increase rightward (sorted by ``START_X``).
    """
    grid = crop_grid.copy()
    sy_unique = sorted(grid[Column.DiffAEData.START_Y].unique().tolist())
    sx_unique = sorted(grid[Column.DiffAEData.START_X].unique().tolist())
    sy_rank = {v: i for i, v in enumerate(sy_unique)}
    sx_rank = {v: i for i, v in enumerate(sx_unique)}
    grid["_row"] = grid[Column.DiffAEData.START_Y].map(sy_rank)
    grid["_col"] = grid[Column.DiffAEData.START_X].map(sx_rank)

    target = grid[(grid["_row"] == grid_row - 1) & (grid["_col"] == grid_col - 1)]
    if target.empty:
        raise RuntimeError(
            f"No crop at row={grid_row} col={grid_col} "
            f"(grid is {len(sy_unique)}x{len(sx_unique)})"
        )
    sx = int(target.iloc[0][Column.DiffAEData.START_X])
    sy = int(target.iloc[0][Column.DiffAEData.START_Y])
    ex = int(target.iloc[0][Column.DiffAEData.END_X])
    ey = int(target.iloc[0][Column.DiffAEData.END_Y])
    return sx, sy, ex, ey


def rbar(uf: np.ndarray, vf: np.ndarray) -> float:
    """Mean resultant length (a.k.a. migration coherence, R-bar) of a flow field.

    Each pixel's flow vector is reduced to its unit direction (magnitude
    discarded), then averaged across all moving pixels.  The returned
    value is the magnitude of that average direction:

        R-bar = || mean( (uf, vf) / ||(uf, vf)|| ) ||,  range [0, 1].

    R-bar = 1 -> every pixel moves in the same direction (perfectly
    coherent migration); R-bar ~ 0 -> directions cancel out (incoherent /
    isotropic flow).  Returns NaN if no pixel has non-zero flow.
    """
    sp = np.sqrt(uf**2 + vf**2)
    nz = sp > 0
    if not nz.any():
        return float("nan")
    u = uf[nz] / sp[nz]
    v = vf[nz] / sp[nz]
    return float(np.sqrt(u.mean() ** 2 + v.mean() ** 2))


def plot_optical_flow_composite(
    ax: plt.Axes,
    frame_t0: np.ndarray,
    frame_t1: np.ndarray,
    *,
    title: str,
    scale_bar_um: int = 10,
    pixel_size_um: float = PIXEL_SIZE_3i_20x_RESOLUTION_1,
) -> None:
    """Render a magenta/green composite of two consecutive frames on ``ax``.

    ``frame_t0`` is shown in magenta, ``frame_t1`` in green; pixels
    bright in both appear white.  A scale bar is drawn in the lower-right
    corner.  Both frames must be 2D and the same shape.
    """

    def _norm(im: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(im, [2, 99.5])
        return np.clip((im - lo) / (hi - lo + 1e-9), 0, 1)

    cy, cx = frame_t0.shape
    rgb = np.zeros((cy, cx, 3), dtype=np.float32)
    t0_norm = _norm(frame_t0)
    rgb[..., 0] = t0_norm
    rgb[..., 1] = _norm(frame_t1)
    rgb[..., 2] = t0_norm

    ax.set_facecolor("white")
    ax.imshow(rgb, origin="upper")
    ax.set_title(title, fontweight="bold", fontsize=FONTSIZE_MEDIUM)
    ax.legend(
        handles=[
            Patch(facecolor="magenta", label="t"),
            Patch(facecolor="green", label="t+1"),
            Patch(facecolor="white", label="overlap"),
        ],
        loc="upper right",
        framealpha=0.35,
        facecolor="black",
        edgecolor="none",
        labelcolor="white",
        fontsize=FONTSIZE_SMALL,
        handlelength=1.0,
        handleheight=1.0,
        borderpad=0.4,
        borderaxespad=0.4,
        labelspacing=0.4,
    )
    add_scalebar(
        ax,
        scale_bar_um=scale_bar_um,
        pixel_size=pixel_size_um,
        location="lower right",
        bar_thickness=2.5,
        padding=5,
        include_label=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def plot_optical_flow_quiver(
    ax: plt.Axes,
    uf: np.ndarray,
    vf: np.ndarray,
    *,
    title: str,
    grid_divisions: int = QUIVER_GRID_DIVISIONS,
) -> None:
    """Render a downsampled quiver of the flow field on ``ax``.

    Arrows are anchored at cell centres (so edge arrows can extend a half
    cell outward without crossing the panel border) and per-arrow length
    is capped at one grid cell so a few extreme vectors don't blow off
    the axes.  Length encodes pixel speed; colour is fixed black.
    """
    cy, cx = uf.shape
    step = max(1, cy // grid_divisions)
    half = step // 2
    Y, X = np.mgrid[half:cy:step, half:cx:step]
    uf_sub = uf[half::step, half::step].copy()
    vf_sub = vf[half::step, half::step].copy()
    sp_sub = np.sqrt(uf_sub**2 + vf_sub**2)

    # Per-panel scaling: a median-magnitude arrow spans ~0.6 of a grid cell.
    med_sp = float(np.median(sp_sub[sp_sub > 0])) if (sp_sub > 0).any() else 1.0
    q_scale = med_sp / (step * 0.6) if med_sp > 0 else 1.0

    # Clip outliers so a few extreme vectors don't extend past the crop.
    cap = q_scale * step
    over = sp_sub > cap
    if over.any():
        scale_down = np.where(over, cap / np.maximum(sp_sub, 1e-12), 1.0)
        uf_sub *= scale_down
        vf_sub *= scale_down

    ax.set_facecolor("white")
    ax.quiver(
        X,
        Y,
        uf_sub,
        vf_sub,
        color="black",
        angles="xy",
        scale_units="xy",
        scale=q_scale,
        width=0.008,
        headwidth=4,
        headlength=5,
        minshaft=1.5,
        alpha=0.95,
    )
    # Pad the axes by a fraction of a cell so arrowheads at the panel
    # edge (especially in incoherent fields, where arrows can point
    # outward) draw fully inside the frame.
    head_pad = step * 0.4
    ax.set_xlim(-head_pad, cx + head_pad)
    ax.set_ylim(cy + head_pad, -head_pad)
    ax.set_aspect("equal")
    ax.set_title(title, fontweight="bold", fontsize=FONTSIZE_MEDIUM)
    ax.set_xticks([])
    ax.set_yticks([])
