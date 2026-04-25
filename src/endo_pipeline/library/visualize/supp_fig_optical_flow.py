"""Builders for the supplementary optical-flow comparison figure.

Renders a 2x2 figure with one *coherent* and one *incoherent*
(crop, timepoint-pair) example, arranged as **two columns**:

    column 0 -- "Coherent Example":   composite (top), quiver (bottom)
    column 1 -- "Incoherent Example": composite (top), quiver (bottom)

The composite uses a magenta/green merge of consecutive BF frames
(overlap = white) and carries a scale bar.  The quiver shows the TVL1
flow field with arrow length encoding pixel speed and a single CVD-safe
colour; the migration coherence (R-bar) is reported in the panel legend.

Coherent / incoherent picks are passed in as
:class:`endo_pipeline.settings.examples.OpticalFlowExample` instances so the
figure inputs live with the rest of the project's example configuration.
"""

from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import load_dataframe, load_image
from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.analyze.optical_flow import build_crop_grid, resolve_attachment
from endo_pipeline.library.analyze.optical_flow.compute import compute_tvl1
from endo_pipeline.library.visualize.figure_utils import add_scalebar
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.manifests import get_zarr_location_for_position, load_dataframe_manifest
from endo_pipeline.settings import DIFFAE_ZARR_RESOLUTION_LEVEL, DIMENSION_ORDER
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.examples import OpticalFlowExample
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_SMALL
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x_RESOLUTION_1
from endo_pipeline.settings.optical_flow import (
    OPTICAL_FLOW_COLUMNS_TO_COMPUTE,
    QUIVER_GRID_DIVISIONS,
)
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

_SCALE_BAR_UM: int = 10


def _load_feature_df(dataset_name: str) -> pd.DataFrame:
    base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_grid"
    manifest = load_dataframe_manifest(f"{base_name}_pca")
    cols = list(OPTICAL_FLOW_COLUMNS_TO_COMPUTE["grid"])
    return load_dataframe(manifest.locations[dataset_name], delay=True)[cols].compute()


def _build_frame_cache(
    dataset_name: str,
    position: int,
    level: int,
    timepoints: list[int],
    df_dataset: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """Load and normalize the requested *timepoints* (BF std-dev z-projection),
    returning a ``{tp: 2D frame}`` cache plus the position's crop grid.
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


def _rbar(uf: np.ndarray, vf: np.ndarray) -> float:
    """Mean resultant length (a.k.a. migration coherence, R-bar) of a flow field.

    Each pixel's flow vector ``(uf, vf)`` is reduced to its unit direction
    (magnitude discarded), then averaged across all moving pixels.  The
    returned value is the magnitude of that average direction:

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


def _resolve_pick(
    example: OpticalFlowExample,
    cache: dict[int, np.ndarray],
    crop_grid: pd.DataFrame,
    attachment: float,
) -> pd.Series:
    """Resolve an :class:`OpticalFlowExample` into a single scan row containing
    the crop pixel bbox and the R-bar value for that crop / timepoint pair.
    """
    grid_sorted = crop_grid.copy()
    sy_unique = sorted(grid_sorted[Column.DiffAEData.START_Y].unique().tolist())
    sx_unique = sorted(grid_sorted[Column.DiffAEData.START_X].unique().tolist())
    sy_rank = {v: i for i, v in enumerate(sy_unique)}
    sx_rank = {v: i for i, v in enumerate(sx_unique)}
    grid_sorted["_row"] = grid_sorted[Column.DiffAEData.START_Y].map(sy_rank)
    grid_sorted["_col"] = grid_sorted[Column.DiffAEData.START_X].map(sx_rank)

    target = grid_sorted[
        (grid_sorted["_row"] == example.crop_row - 1)
        & (grid_sorted["_col"] == example.crop_col - 1)
    ]
    if target.empty:
        raise RuntimeError(
            f"No crop at row={example.crop_row} col={example.crop_col} "
            f"for pos={example.position} (grid is {len(sy_unique)}x{len(sx_unique)})"
        )
    cid = int(target.iloc[0][Column.CROP_INDEX])
    sx = int(target.iloc[0][Column.DiffAEData.START_X])
    sy = int(target.iloc[0][Column.DiffAEData.START_Y])
    ex = int(target.iloc[0]["end_x"])
    ey = int(target.iloc[0]["end_y"])

    f0, f1 = cache[example.t0], cache[example.t1]
    uf_full, vf_full = compute_tvl1(f0, f1, attachment=attachment)
    uf = uf_full[sy:ey, sx:ex]
    vf = vf_full[sy:ey, sx:ex]
    rbar = _rbar(uf, vf)
    if np.isnan(rbar):
        raise RuntimeError(
            f"R-bar undefined for pos={example.position} "
            f"t={example.t0}->{example.t1} crop={cid}"
        )
    sp = np.sqrt(uf**2 + vf**2)
    nz = sp[sp > 0]
    sp_p95 = float(np.percentile(nz, 95)) if nz.size else 0.0
    return pd.Series(
        {
            "position": example.position,
            "crop": cid,
            "t0": example.t0,
            "t1": example.t1,
            "sx": sx,
            "sy": sy,
            "ex": ex,
            "ey": ey,
            "rbar": rbar,
            "sp_p95": sp_p95,
        }
    )


def _plot_column(
    axes,
    row: pd.Series,
    title: str,
    cache: dict[int, np.ndarray],
    attachment: float,
) -> None:
    """Plot one column (composite on top, quiver on bottom) of the figure.

    TVL1 flow is computed on the full image and sliced to the crop
    (avoids edge artefacts at the crop border).
    """
    t0, t1 = int(row["t0"]), int(row["t1"])
    _sx, _sy = int(row["sx"]), int(row["sy"])
    _ex, _ey = int(row["ex"]), int(row["ey"])
    cy, cx = _ey - _sy, _ex - _sx

    f0, f1 = cache[t0], cache[t1]
    c0, c1 = f0[_sy:_ey, _sx:_ex], f1[_sy:_ey, _sx:_ex]
    uf_full, vf_full = compute_tvl1(f0, f1, attachment=attachment)
    uf, vf = uf_full[_sy:_ey, _sx:_ex], vf_full[_sy:_ey, _sx:_ex]
    sp = np.sqrt(uf**2 + vf**2)
    nz = sp > 0
    if nz.any():
        u = uf[nz] / sp[nz]
        v = vf[nz] / sp[nz]
        rbar_val = float(np.sqrt(u.mean() ** 2 + v.mean() ** 2))
    else:
        rbar_val = 0.0

    def _norm(im: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(im, [2, 99.5])
        return np.clip((im - lo) / (hi - lo + 1e-9), 0, 1)

    # --- Composite (top): magenta/green merge, overlap = white. ----------
    ax = axes[0]
    ax.set_facecolor("white")
    rgb = np.zeros((cy, cx, 3), dtype=np.float32)
    t0_norm = _norm(c0)
    rgb[..., 0] = t0_norm
    rgb[..., 1] = _norm(c1)
    rgb[..., 2] = t0_norm
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
        scale_bar_um=_SCALE_BAR_UM,
        pixel_size=PIXEL_SIZE_3i_20x_RESOLUTION_1,
        location="lower right",
        bar_thickness=2.5,
        padding=5,
    )
    ax.text(
        0.96,
        0.08,
        f"{_SCALE_BAR_UM} {Unicode.MU}m",
        color="white",
        transform=ax.transAxes,
        fontsize=FONTSIZE_SMALL,
        va="bottom",
        ha="right",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Quiver (bottom): single CVD-safe colour, R-bar in the legend. ---
    ax = axes[1]
    ax.set_facecolor("white")
    step = max(1, cy // QUIVER_GRID_DIVISIONS)
    half = step // 2
    Y, X = np.mgrid[half:cy:step, half:cx:step]
    uf_sub = uf[half::step, half::step].copy()
    vf_sub = vf[half::step, half::step].copy()
    sp_sub = np.sqrt(uf_sub**2 + vf_sub**2)
    # Per-panel scaling: a median-magnitude arrow spans ~0.6 of a grid cell.
    med_sp = float(np.median(sp_sub[sp_sub > 0])) if (sp_sub > 0).any() else 1.0
    q_scale = med_sp / (step * 0.6) if med_sp > 0 else 1.0
    # Clip outliers so a few extreme vectors don't extend past the crop;
    # cap arrow length at ~1 grid cell in data units (the centred grid
    # gives each arrow half a cell of room on either side).
    cap = q_scale * step
    over = sp_sub > cap
    if over.any():
        scale_down = np.where(over, cap / np.maximum(sp_sub, 1e-12), 1.0)
        uf_sub *= scale_down
        vf_sub *= scale_down
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
    ax.set_xlim(0, cx)
    ax.set_ylim(cy, 0)
    ax.set_aspect("equal")
    ax.set_title(
        f"Migration Coherence = {rbar_val:.2f}",
        fontweight="bold",
        fontsize=FONTSIZE_MEDIUM,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def build_supp_fig_optical_flow(
    coherent: OpticalFlowExample,
    incoherent: OpticalFlowExample,
    output_dir: Path,
    figure_size: tuple[float, float] = (5.0, 5.0),
) -> None:
    """Build the supplementary optical-flow figure and write it to ``output_dir``.

    Parameters
    ----------
    coherent
        Example with high migration coherence (top row of the 2x2 figure).
    incoherent
        Example with low migration coherence (bottom row of the 2x2 figure).
    output_dir
        Directory to write panel + composed SVGs / PNG to.
    figure_size
        ``(width, height)`` in inches for both the matplotlib figure and the
        composed SVG canvas.
    """
    attachment = resolve_attachment("BF")

    # Per-(dataset, position) frame cache, loaded with only the timepoints
    # actually needed by the picks.
    examples: list[tuple[OpticalFlowExample, str]] = [
        (coherent, "Coherent Example"),
        (incoherent, "Incoherent Example"),
    ]
    df_cache: dict[str, pd.DataFrame] = {}
    per_key_cache: dict[
        tuple[str, int],
        tuple[dict[int, np.ndarray], pd.DataFrame],
    ] = {}
    needed_tps: dict[tuple[str, int], set[int]] = {}
    for ex, _label in examples:
        needed_tps.setdefault((ex.dataset_name, ex.position), set()).update([ex.t0, ex.t1])

    for (ds, pos), tps in needed_tps.items():
        if ds not in df_cache:
            df_cache[ds] = _load_feature_df(ds)
        cache, crop_grid = _build_frame_cache(
            dataset_name=ds,
            position=pos,
            level=DIFFAE_ZARR_RESOLUTION_LEVEL,
            timepoints=sorted(tps),
            df_dataset=df_cache[ds],
        )
        per_key_cache[(ds, pos)] = (cache, crop_grid)

    picks: list[tuple[pd.Series, str, OpticalFlowExample]] = []
    for ex, label in examples:
        cache, grid = per_key_cache[(ex.dataset_name, ex.position)]
        row = _resolve_pick(ex, cache, grid, attachment)
        row["dataset"] = ex.dataset_name
        picks.append((row, label, ex))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figure_size,
        facecolor="white",
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )
    for col_idx, (row, label, _ex) in enumerate(picks):
        cache, _grid = per_key_cache[(row["dataset"], int(row["position"]))]
        _plot_column(
            axes[:, col_idx],
            row,
            title=label,
            cache=cache,
            attachment=attachment,
        )

    base_name = "optical_flow_panels"
    # PNG keeps the tight crop (used as a quick-look raster).
    save_plot_to_path(
        fig,
        output_dir,
        base_name,
        file_format=".png",
        dpi=300,
        show_and_close=False,
        tight_layout=False,
        bbox_inches="tight",
    )
    # SVG must NOT be tight-cropped: build_figure_from_panels assumes the
    # embedded panel is exactly ``figure_size`` inches.  A tight crop would
    # shrink the panel and leave a margin on the composed canvas.
    save_plot_to_path(
        fig,
        output_dir,
        base_name,
        file_format=".svg",
        dpi=300,
        show_and_close=False,
        tight_layout=False,
        bbox_inches=None,
    )
    plt.close(fig)

    # Compose into the standard figure canvas (single panel, no letter).
    # Future panels can be added alongside without changing the per-panel
    # rendering above.
    panels = [
        FigurePanel(
            letter="",
            path=output_dir / f"{base_name}.svg",
            x_position=0,
            y_position=0,
            x_offset=0,
            y_offset=0,
        ),
    ]
    build_figure_from_panels(
        panels,
        output_dir / "supp_fig_optical_flow.svg",
        width=figure_size[0],
        height=figure_size[1],
    )


__all__ = ["build_supp_fig_optical_flow"]
