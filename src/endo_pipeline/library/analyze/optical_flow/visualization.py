import logging

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.optical_flow import (
    DEMO_SCAN_N_CROPS,
    DEMO_SCAN_N_PAIRS,
    QUIVER_GRID_DIVISIONS,
)
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

from .compute import compute_tvl1

logger = logging.getLogger(__name__)


def _scan_crop_pairs(
    cache: dict[int, np.ndarray],
    crop_grid: pd.DataFrame,
    thresh: float,
    attachment: float,
) -> pd.DataFrame:
    """Subsample (crop, timepoint) pairs and compute R-bar for each.

    Returns a DataFrame with columns ci_idx, crop, t0, t1, sx, sy, ex, ey,
    circ_std, rbar — sorted by rbar descending.
    """
    sorted_tp = sorted(cache.keys())
    cids = crop_grid[ColumnName.CROP_INDEX].values
    sx_arr = crop_grid[ColumnName.DiffAEData.START_X].values.astype(int)
    sy_arr = crop_grid[ColumnName.DiffAEData.START_Y].values.astype(int)
    ex_arr = crop_grid[ColumnName.DiffAEData.END_X].values.astype(int)
    ey_arr = crop_grid[ColumnName.DiffAEData.END_Y].values.astype(int)

    crop_step = max(1, len(cids) // DEMO_SCAN_N_CROPS)
    scan_cids = range(0, len(cids), crop_step)

    all_pairs = [(sorted_tp[i], sorted_tp[i + 1]) for i in range(len(sorted_tp) - 1)]
    pair_step = max(1, len(all_pairs) // DEMO_SCAN_N_PAIRS)
    scan_pairs = all_pairs[::pair_step]

    records: list[dict] = []
    _image_flow_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for ci_idx in scan_cids:
        _sx, _sy = int(sx_arr[ci_idx]), int(sy_arr[ci_idx])
        _ex, _ey = int(ex_arr[ci_idx]), int(ey_arr[ci_idx])
        _cidx = int(cids[ci_idx])
        for t0, t1 in scan_pairs:
            f0, f1 = cache[t0], cache[t1]
            c0, c1 = f0[_sy:_ey, _sx:_ex], f1[_sy:_ey, _sx:_ex]

            if (t0, t1) not in _image_flow_cache:
                _image_flow_cache[(t0, t1)] = compute_tvl1(f0, f1, attachment=attachment)

            uf_full, vf_full = _image_flow_cache[(t0, t1)]
            uf, vf = uf_full[_sy:_ey, _sx:_ex], vf_full[_sy:_ey, _sx:_ex]

            ang = np.arctan2(vf, uf)
            sp_scan = np.sqrt(uf**2 + vf**2)
            mask = (c0 > thresh) | (c1 > thresh)
            cstd = float(sp_stats.circstd(ang[mask])) if mask.any() else float("nan")

            nz_mask = mask & (sp_scan > 0)
            if nz_mask.any():
                unit_u = uf[nz_mask] / sp_scan[nz_mask]
                unit_v = vf[nz_mask] / sp_scan[nz_mask]
                rbar = float(np.sqrt(unit_u.mean() ** 2 + unit_v.mean() ** 2))
            else:
                rbar = float("nan")

            records.append(
                {
                    "ci_idx": ci_idx,
                    "crop": _cidx,
                    "t0": t0,
                    "t1": t1,
                    "sx": _sx,
                    "sy": _sy,
                    "ex": _ex,
                    "ey": _ey,
                    "circ_std": cstd,
                    "rbar": rbar,
                }
            )

    scan_df = pd.DataFrame(records).dropna(subset=["rbar"])
    return scan_df.sort_values("rbar", ascending=False).reset_index(drop=True)


def plot_demo_summary(
    cache: dict[int, np.ndarray],
    crop_grid: pd.DataFrame,
    ds_name: str,
    position: int,
    thresh: float,
    out_dir,
    channel: list[str],
    attachment: float = 7.5,
) -> None:
    """Produce a multi-crop diagnostic figure (up to 3 rows x 5 cols).

    Scans a subsample of (crop, timepoint) pairs, sorts by R-bar
    (mean resultant length of unit flow vectors), and picks up to three
    representative pairs — most coherent (high R-bar), median, and most
    incoherent (low R-bar) — then plots for each:
      (a) Red/Green composite of crop at t0 (red) and t1 (green).
      (b) Quiver plot of TVL1 flow.
      (c) Angle histogram (masked pixels only) with R-bar annotation.
      (d) Speed distribution with mean/median lines.
      (e) Per-crop R-bar distribution across all scanned time pairs.

    Parameters
    ----------
    cache
        Mapping from timepoint index to its 2-D intensity frame.
    crop_grid
        One row per spatial crop (see :func:`build_crop_grid`).
    ds_name
        Dataset name for figure title and filename.
    position
        Integer position index.
    thresh
        Intensity threshold for foreground masking.
    out_dir
        Directory where the PNG figure is saved.
    channel
        Imaging channel name(s) (e.g. ``["BF"]``).
    attachment
        TVL1 attachment (lambda) value.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    out_dir = Path(out_dir)

    sorted_tp = sorted(cache.keys())
    if len(sorted_tp) < 2:
        logger.warning("Only %d cached frame(s) — skipping demo plot", len(sorted_tp))
        return

    scan_df = _scan_crop_pairs(cache, crop_grid, thresh, attachment)
    if len(scan_df) < 2:
        logger.warning("Scan produced <2 valid records — skipping demo plot")
        return
    n_scanned = len(scan_df)

    if n_scanned >= 3:
        picks = [
            (scan_df.iloc[0], r"COHERENT (high $\bar{R}$)", "COHERENT"),
            (scan_df.iloc[n_scanned // 2], r"MEDIAN $\bar{R}$", "MEDIAN"),
            (scan_df.iloc[-1], r"INCOHERENT (low $\bar{R}$)", "INCOHERENT"),
        ]
    else:
        picks = [
            (scan_df.iloc[0], r"COHERENT (high $\bar{R}$)", "COHERENT"),
            (scan_df.iloc[-1], r"INCOHERENT (low $\bar{R}$)", "INCOHERENT"),
        ]

    logger.info(
        "Demo scan: %d valid pairs, picked %d crops for plot",
        n_scanned,
        len(picks),
    )
    for row, _label, tag in picks:
        logger.info(
            "%s: R_bar=%.4f (crop %d, t=%d->%d)",
            tag,
            row["rbar"],
            int(row["crop"]),
            int(row["t0"]),
            int(row["t1"]),
        )

    # Helper: plot one row (5 panels)
    def _plot_row(axes, row, label):  # noqa: C901
        t0, t1 = int(row["t0"]), int(row["t1"])
        _sx, _sy = int(row["sx"]), int(row["sy"])
        _ex, _ey = int(row["ex"]), int(row["ey"])
        _cidx = int(row["crop"])
        cy, cx = _ey - _sy, _ex - _sx

        f0, f1 = cache[t0], cache[t1]
        c0, c1 = f0[_sy:_ey, _sx:_ex], f1[_sy:_ey, _sx:_ex]
        uf_full, vf_full = compute_tvl1(f0, f1, attachment=attachment)
        uf, vf = uf_full[_sy:_ey, _sx:_ex], vf_full[_sy:_ey, _sx:_ex]

        sp = np.sqrt(uf**2 + vf**2)
        ang = np.arctan2(vf, uf)
        mask = (c0 > thresh) | (c1 > thresh)

        # Compute R̄ for this crop
        nz_mask = mask & (sp > 0)
        if nz_mask.any():
            _unit_u = uf[nz_mask] / sp[nz_mask]
            _unit_v = vf[nz_mask] / sp[nz_mask]
            rbar_val = float(np.sqrt(_unit_u.mean() ** 2 + _unit_v.mean() ** 2))
        else:
            rbar_val = 0.0

        def _norm(im):
            lo, hi = np.percentile(im, [2, 99.5])
            return np.clip((im - lo) / (hi - lo + 1e-9), 0, 1)

        # (a) Red/Green composite
        ax = axes[0]
        ax.set_facecolor("white")
        rgb = np.zeros((cy, cx, 3), dtype=np.float32)
        rgb[..., 0] = _norm(c0)
        rgb[..., 1] = _norm(c1)
        ax.imshow(rgb, origin="upper")
        ax.set_title(f"(a) Composite  crop {_cidx}\n{label}", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"$\\bar{{R}}$ = {rbar_val:.4f}", fontsize=10, fontstyle="italic")
        ax.legend(
            handles=[
                Patch(facecolor="red", label=f"t={t0}"),
                Patch(facecolor="green", label=f"t={t1}"),
                Patch(facecolor="yellow", label="overlap"),
            ],
            fontsize=6,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            framealpha=0.7,
            borderaxespad=0,
        )
        ax.tick_params(labelsize=7)

        # (b) Quiver
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
            alpha=0.85,
        )
        ax.set_xlim(0, cx)
        ax.set_ylim(cy, 0)
        ax.set_aspect("equal")
        ax.set_title(f"(b) Quiver  t={t0}{Unicode.RIGHT_ARROW}{t1}", fontsize=9)
        ax.tick_params(labelsize=7)

        # (c) Angle histogram
        ax = axes[2]
        ax.set_facecolor("white")
        if mask.any():
            ang_m = ang[mask]
            ax.hist(ang_m, bins=72, color="salmon", edgecolor="white", linewidth=0.3, density=True)
            cmean = float(np.arctan2(np.sin(ang_m).mean(), np.cos(ang_m).mean()))
            ax.axvline(
                cmean, color="red", ls="--", lw=1.5, label=f"circ {Unicode.MU} = {cmean:.2f}"
            )
            ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel(f"{Unicode.THETA} (rad)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"(c) {Unicode.THETA} distribution  $\\bar{{R}}$ = {rbar_val:.4f}", fontsize=9)
        ax.tick_params(labelsize=7)

        # (d) Speed distribution
        ax = axes[3]
        ax.set_facecolor("white")
        if mask.any():
            sp_m = sp[mask] if sp.shape == mask.shape else sp
            ax.hist(
                sp_m, bins=60, color="steelblue", edgecolor="white", linewidth=0.3, density=True
            )
            ax.axvline(
                float(sp_m.mean()),
                color="navy",
                ls="--",
                lw=1.5,
                label=f"{Unicode.MU} = {float(sp_m.mean()):.3f}",
            )
            ax.axvline(
                float(np.median(sp_m)),
                color="dodgerblue",
                ls=":",
                lw=1.5,
                label=f"med = {float(np.median(sp_m)):.3f}",
            )
            ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel("Speed (px/frame)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title("(d) Speed distribution", fontsize=9)
        ax.tick_params(labelsize=7)

        # (e) R-bar distribution for this crop across all scanned time pairs
        ax = axes[4]
        ax.set_facecolor("white")
        crop_rbar = scan_df.loc[scan_df["crop"] == _cidx, "rbar"].dropna().values
        if len(crop_rbar) > 0:
            ax.hist(
                crop_rbar,
                bins=max(8, len(crop_rbar) // 2),
                color="mediumseagreen",
                edgecolor="white",
                linewidth=0.3,
                density=True,
                rwidth=0.75,
            )
            ax.axvline(
                float(np.mean(crop_rbar)),
                color="black",
                ls="--",
                lw=1.5,
                label=f"mean $\\bar{{R}}$ = {float(np.mean(crop_rbar)):.3f}",
            )
            ax.axvline(
                float(np.median(crop_rbar)),
                color="forestgreen",
                ls=":",
                lw=1.5,
                label=f"median = {float(np.median(crop_rbar)):.3f}",
            )
            ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel(r"$\bar{R}$", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"(e) $\\bar{{R}}$ distribution  crop {_cidx}", fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.tick_params(labelsize=7)

    # Build the figure
    n_rows = len(picks)
    fig, axes = plt.subplots(
        n_rows, 5, figsize=(30, 4.5 * n_rows), facecolor="white", squeeze=False
    )

    for row_idx, (row, label, _tag) in enumerate(picks):
        _plot_row(axes[row_idx], row, label)

    fig.suptitle(
        f"Coherent vs Incoherent : {ds_name} / pos {position}  [{', '.join(channel)}]",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_plot_to_path(
        fig,
        out_dir,
        f"demo_coherent_vs_incoherent_{ds_name}_{position}_{'_'.join(channel)}",
        dpi=300,
        show_and_close=False,
    )
    logger.info("Saved coherent-vs-incoherent figure to %s", out_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tracked-crop coherence time series (demo mode)
# ---------------------------------------------------------------------------
def plot_tracked_crop_coherence_timeseries(
    df: pd.DataFrame,
    ds_name: str,
    position: int,
    out_dir,
    ema_alphas: list[float] | tuple[float, ...] = (0.1,),
    compute_radial_coherence: bool = False,
    max_crops: int = 2,
    max_dt: int = 1,
) -> None:
    """Plot coherence metrics over time for individual tracked crops.

    Produces a multi-panel figure showing how optical-flow coherence
    evolves across timepoints for a subset of tracked crops.  The
    crops with the longest tracks are selected (up to *max_crops*).

    Each panel plots the raw coherence metric together with its
    EMA-smoothed variants.

    Parameters
    ----------
    df
        Merged position DataFrame with flow features and EMA columns
        already computed.  Must contain ``crop_index``, ``frame_number``,
        and the relevant flow feature columns.
    ds_name
        Dataset name for figure title and filename.
    position
        Integer position index.
    out_dir
        Directory where the PNG figure is saved.
    ema_alphas
        EMA alpha values used for smoothing (for column name generation).
    compute_radial_coherence
        Whether radial coherence columns are present.
    max_crops
        Maximum number of tracked crops to plot.
    max_dt
        Maximum temporal gap to plot.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)

    crop_col = ColumnName.CROP_INDEX
    time_col = ColumnName.TIMEPOINT

    if crop_col not in df.columns or time_col not in df.columns:
        logger.warning("Missing %s or %s columns — skipping time series plot", crop_col, time_col)
        return

    # Select crops with the longest tracks
    crop_lengths = df.groupby(crop_col)[time_col].nunique().sort_values(ascending=False)
    selected_crops = crop_lengths.head(max_crops).index.tolist()
    if not selected_crops:
        logger.warning("No tracked crops found — skipping time series plot")
        return

    logger.info(
        "Plotting coherence time series for %d tracked crop(s): %s",
        len(selected_crops),
        selected_crops,
    )

    # Build the list of (raw_col, label, ema_cols_and_labels) metric groups
    # to plot. Each group gets its own subplot row.
    metric_groups: list[tuple[str, str, list[tuple[str, str]]]] = []
    for d in range(1, max_dt + 1):
        dt_tag = f"_dt{d}"

        # 1) Raw coherence (mean unit vector)
        raw_col = f"optical_flow_mean_unit_vector{dt_tag}"
        ema_variants = []
        for alpha in ema_alphas:
            atag = str(alpha).replace(".", "")
            ema_col = f"ema{atag}_optical_flow_mean_unit_vector{dt_tag}"
            ema_variants.append((ema_col, f"EMA \u03b1={alpha}"))
        metric_groups.append((raw_col, rf"Coherence ($\bar{{R}}$, dt={d})", ema_variants))

        # 2) Fast coherence
        raw_fast = f"optical_flow_mean_unit_vector_fast{dt_tag}"
        ema_fast = []
        for alpha in ema_alphas:
            atag = str(alpha).replace(".", "")
            ema_col = f"ema{atag}_optical_flow_mean_unit_vector_fast{dt_tag}"
            ema_fast.append((ema_col, f"EMA \u03b1={alpha}"))
        metric_groups.append((raw_fast, f"Fast Coherence (speed > thr, dt={d})", ema_fast))

        # 3) Radial coherence (if enabled)
        if compute_radial_coherence:
            raw_rad = f"optical_flow_radial_coherence{dt_tag}"
            ema_rad = []
            for alpha in ema_alphas:
                atag = str(alpha).replace(".", "")
                ema_col = f"ema{atag}_optical_flow_radial_coherence{dt_tag}"
                ema_rad.append((ema_col, f"EMA \u03b1={alpha}"))
            metric_groups.append((raw_rad, f"Radial Coherence (dt={d})", ema_rad))

    # Filter to metric groups whose raw column actually exists
    metric_groups = [(r, lbl, e) for r, lbl, e in metric_groups if r in df.columns]
    if not metric_groups:
        logger.warning("No coherence columns found in dataframe — skipping time series plot")
        return

    n_metrics = len(metric_groups)
    n_crops = len(selected_crops)
    fig, axes = plt.subplots(
        n_metrics,
        1,
        figsize=(14, 4 * n_metrics),
        facecolor="white",
        squeeze=False,
        sharex=True,
    )

    # Use the colour cycle from the active style so that this respects
    # any axes.prop_cycle set in the endo_pipeline.figure mplstyle.
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for row_idx, (raw_col, metric_label, ema_variants) in enumerate(metric_groups):
        ax = axes[row_idx, 0]
        ax.set_facecolor("white")

        for ci, crop_id in enumerate(selected_crops):
            color = prop_cycle[ci % len(prop_cycle)]
            df_crop = df[df[crop_col] == crop_id].sort_values(time_col)
            t = df_crop[time_col].values

            # Raw (thin, more transparent)
            if raw_col in df_crop.columns:
                ax.plot(
                    t,
                    df_crop[raw_col].values,
                    color=color,
                    alpha=0.35,
                    linewidth=0.8,
                    label=f"crop {crop_id} (raw)" if row_idx == 0 else None,
                )

            # EMA (thicker, opaque)
            for ema_col, ema_label in ema_variants:
                if ema_col in df_crop.columns:
                    ax.plot(
                        t,
                        df_crop[ema_col].values,
                        color=color,
                        alpha=0.9,
                        linewidth=1.5,
                        label=f"crop {crop_id} ({ema_label})" if row_idx == 0 else None,
                    )

        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Timepoint", fontsize=10)

    # Legend on first subplot only (can be large; place outside)
    if n_crops <= 8:
        axes[0, 0].legend(
            fontsize=6,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            framealpha=0.7,
        )

    fig.suptitle(
        f"Tracked Crop Coherence Over Time : {ds_name} / pos {position}\n"
        f"({n_crops} crops, EMA \u03b1 = {list(ema_alphas)})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_plot_to_path(
        fig,
        out_dir,
        f"demo_tracked_coherence_timeseries_{ds_name}_{position}",
        dpi=300,
        show_and_close=False,
    )
    logger.info("Saved tracked-crop coherence time series to %s", out_dir)
    plt.close(fig)
