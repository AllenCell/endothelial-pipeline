"""Supplementary figure: optical-flow coherent vs incoherent example panels.

Renders a 2x2 figure showing one COHERENT (high migration coherence) and
one INCOHERENT (low migration coherence) crop/timepoint pair.  Each row
contains the red/green composite of consecutive frames (std dev projection of
the brightfield channel) and the TVL1 quiver plot.

The (dataset, position, timepoint, crop row/col) for both panels are
hard-coded in :data:`MANUAL_PICKS` -- this script intentionally has no
search / scan / demo path.  The picks were chosen using TFE!

Saves the figure under ``results/<date>/supp_fig_optical_flow/`` as both
PNG and SVG, following the standard endo_pipeline figures convention.
"""

import logging

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe, load_image
from endo_pipeline.io.output import save_plot_to_path
from endo_pipeline.library.analyze.optical_flow import build_crop_grid, resolve_attachment
from endo_pipeline.library.analyze.optical_flow.compute import compute_tvl1
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.manifests import get_zarr_location_for_position, load_dataframe_manifest
from endo_pipeline.settings import DIFFAE_ZARR_RESOLUTION_LEVEL, DIMENSION_ORDER
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.optical_flow import (
    OPTICAL_FLOW_COLUMNS_TO_COMPUTE,
    QUIVER_GRID_DIVISIONS,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

plt.style.use("endo_pipeline.figure")

logger = logging.getLogger(__name__)

# Crop pairs that we are using for the figures.
#
# Each entry identifies a single crop/timepoint pair:
#   - dataset, position, t0, t1: source frames (BF) at this position
#     and the consecutive timepoints used to compute optical flow.
#   - row, col: 1-indexed (row, col) of the crop within the
#     position's regular crop grid (NOT pixel coordinates).
#     Row 1 / col 1 is the top-left crop; rows increase downward
#     (sorted by START_Y), cols increase rightward (sorted by START_X).
#     Resolved to pixel bbox in :func:`_resolve_pick`.

MANUAL_PICKS: dict[str, dict] = {
    "COHERENT": {
        "dataset": "20250409_20X",
        "position": 2,
        "t0": 150,
        "t1": 151,
        "row": 5,
        "col": 4,
    },
    "INCOHERENT": {
        "dataset": "20251001_20X",
        "position": 1,
        "t0": 198,
        "t1": 199,
        "row": 2,
        "col": 5,
    },
}


# Helper functions
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
    sp = np.sqrt(uf**2 + vf**2)
    nz = sp > 0
    if not nz.any():
        return float("nan")
    u = uf[nz] / sp[nz]
    v = vf[nz] / sp[nz]
    return float(np.sqrt(u.mean() ** 2 + v.mean() ** 2))


def _resolve_pick(
    spec: dict,
    cache: dict[int, np.ndarray],
    crop_grid: pd.DataFrame,
    attachment: float,
) -> pd.Series:
    """Resolve a hard-coded pick (``position``, ``t0``, ``t1``, ``row``, ``col``)
    into a single scan row containing the crop bbox and the R-bar value.
    """
    position = int(spec["position"])
    t0 = int(spec["t0"])
    t1 = int(spec.get("t1", t0 + 1))
    grid_row = int(spec["row"])
    grid_col = int(spec["col"])

    grid_sorted = crop_grid.copy()
    sy_unique = sorted(grid_sorted[Column.DiffAEData.START_Y].unique().tolist())
    sx_unique = sorted(grid_sorted[Column.DiffAEData.START_X].unique().tolist())
    sy_rank = {v: i for i, v in enumerate(sy_unique)}
    sx_rank = {v: i for i, v in enumerate(sx_unique)}
    grid_sorted["_row"] = grid_sorted[Column.DiffAEData.START_Y].map(sy_rank)
    grid_sorted["_col"] = grid_sorted[Column.DiffAEData.START_X].map(sx_rank)

    target = grid_sorted[
        (grid_sorted["_row"] == grid_row - 1) & (grid_sorted["_col"] == grid_col - 1)
    ]
    if target.empty:
        raise RuntimeError(
            f"No crop at row={grid_row} col={grid_col} for pos={position} "
            f"(grid is {len(sy_unique)}x{len(sx_unique)})"
        )
    cid = int(target.iloc[0][Column.CROP_INDEX])
    sx = int(target.iloc[0][Column.DiffAEData.START_X])
    sy = int(target.iloc[0][Column.DiffAEData.START_Y])
    ex = int(target.iloc[0]["end_x"])
    ey = int(target.iloc[0]["end_y"])

    f0, f1 = cache[t0], cache[t1]
    uf_full, vf_full = compute_tvl1(f0, f1, attachment=attachment)
    uf = uf_full[sy:ey, sx:ex]
    vf = vf_full[sy:ey, sx:ex]
    rbar = _rbar(uf, vf)
    if np.isnan(rbar):
        raise RuntimeError(f"R-bar undefined for pos={position} t={t0}->{t1} crop={cid}")
    return pd.Series(
        {
            "position": position,
            "crop": cid,
            "t0": t0,
            "t1": t1,
            "sx": sx,
            "sy": sy,
            "ex": ex,
            "ey": ey,
            "rbar": rbar,
        }
    )


def _plot_pair(
    axes,
    row: pd.Series,
    label: str,
    cache: dict[int, np.ndarray],
    attachment: float,
) -> None:
    """Plot one (composite, quiver) pair for a single scan row.

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

    # Composite
    ax = axes[0]
    ax.set_facecolor("white")
    rgb = np.zeros((cy, cx, 3), dtype=np.float32)
    rgb[..., 0] = _norm(c0)
    rgb[..., 1] = _norm(c1)
    ax.imshow(rgb, origin="upper")
    ax.set_title("Composite", fontweight="bold")
    ax.set_ylabel(label, fontweight="bold")
    ax.legend(
        handles=[
            Patch(facecolor="red", label="t"),
            Patch(facecolor="green", label="t+1"),
            Patch(facecolor="yellow", label="overlap"),
        ],
        loc="upper right",
        framealpha=0.0,
        labelcolor="white",
        handlelength=0.9,
        handleheight=0.9,
        borderpad=0.3,
        borderaxespad=0.3,
        labelspacing=0.3,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # Quiver
    ax = axes[1]
    ax.set_facecolor("white")
    step = max(1, cy // QUIVER_GRID_DIVISIONS)
    Y, X = np.mgrid[0:cy:step, 0:cx:step]
    sp_sub = sp[::step, ::step]
    med_sp = float(np.median(sp_sub[sp_sub > 0])) if (sp_sub > 0).any() else 1.0
    q_scale = med_sp / (step * 0.6) if med_sp > 0 else 1.0
    ax.quiver(
        X,
        Y,
        uf[::step, ::step],
        vf[::step, ::step],
        sp_sub,
        cmap="autumn",
        clim=[0, np.percentile(sp, 97)],
        angles="xy",
        scale_units="xy",
        scale=q_scale,
        width=0.008,
        headwidth=4,
        headlength=5,
        minshaft=1.5,
        alpha=0.9,
    )
    ax.set_xlim(0, cx)
    ax.set_ylim(cy, 0)
    ax.set_aspect("equal")
    ax.set_title(f"Migration Coherence = {rbar_val:.2f}")
    ax.set_xticks([])
    ax.set_yticks([])


def main(
    figure_size: tuple[float, float] = (5.0, 5.0),
) -> None:
    """Render the 2x2 supplementary figure (see module docstring).

    Parameters
    ----------
    figure_size
        ``(width, height)`` in inches passed to ``plt.subplots``.
    """
    attachment = resolve_attachment("BF")

    logger.info("Building supplementary optical-flow figure")

    # Build per-(dataset, position) frame cache, loading only the
    # timepoints actually needed by the picks.
    df_cache: dict[str, pd.DataFrame] = {}
    per_key_cache: dict[
        tuple[str, int],
        tuple[dict[int, np.ndarray], pd.DataFrame],
    ] = {}
    needed_tps: dict[tuple[str, int], set[int]] = {}
    for spec in MANUAL_PICKS.values():
        ds = str(spec["dataset"])
        pos = int(spec["position"])
        t0 = int(spec["t0"])
        t1 = int(spec.get("t1", t0 + 1))
        needed_tps.setdefault((ds, pos), set()).update([t0, t1])

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

    picks: list[tuple[pd.Series, str, str]] = []
    for tag, label in (("COHERENT", "Coherent Example"), ("INCOHERENT", "Incoherent Example")):
        spec = MANUAL_PICKS[tag]
        ds = str(spec["dataset"])
        pos = int(spec["position"])
        cache, grid = per_key_cache[(ds, pos)]
        row = _resolve_pick(spec, cache, grid, attachment)
        row["dataset"] = ds
        picks.append((row, label, tag))
        logger.info(
            "%s: R_bar=%.4f (ds=%s pos %d, crop %d, t=%d->%d)",
            tag,
            row["rbar"],
            ds,
            int(row["position"]),
            int(row["crop"]),
            int(row["t0"]),
            int(row["t1"]),
        )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figure_size,
        facecolor="white",
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.05, "hspace": 0.1},
    )
    for row_idx, (row, label, _tag) in enumerate(picks):
        cache, _grid = per_key_cache[(row["dataset"], int(row["position"]))]
        _plot_pair(
            axes[row_idx],
            row,
            label,
            cache=cache,
            attachment=attachment,
        )

    save_dir = get_output_path("supp_fig_optical_flow")
    base_name = "optical_flow_panels"
    # PNG keeps the tight crop (used as a quick-look raster).
    save_plot_to_path(
        fig,
        save_dir,
        base_name,
        file_format=".png",
        dpi=300,
        show_and_close=False,
        tight_layout=False,
        bbox_inches="tight",
    )
    # SVG must NOT be tight-cropped: build_figure_from_panels assumes the
    # embedded panel is exactly ``figure_size`` inches.  A tight crop
    # would shrink the panel and leave a margin on the composed canvas.
    save_plot_to_path(
        fig,
        save_dir,
        base_name,
        file_format=".svg",
        dpi=300,
        show_and_close=False,
        tight_layout=False,
        bbox_inches=None,
    )
    plt.close(fig)

    # Compose into the standard figure canvas (single panel, no letter).
    # Future panels can be added alongside without changing
    # the per-panel rendering above.
    panels = [
        FigurePanel(
            letter="",
            path=save_dir / f"{base_name}.svg",
            x_position=0,
            y_position=0,
            x_offset=0,
            y_offset=0,
        ),
    ]
    build_figure_from_panels(
        panels,
        save_dir / "supp_fig_optical_flow.svg",
        width=figure_size[0],
        height=figure_size[1],
    )
    logger.info("Saved supplementary optical-flow figure to %s", save_dir)


if __name__ == "__main__":

    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
