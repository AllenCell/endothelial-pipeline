import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from endo_pipeline.io import save_plot_to_path
from endo_pipeline.library.analyze.optical_flow import OpticalFlowImagePair, compute_tvl1
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.optical_flow import DEFAULT_EMA_ALPHA, QUIVER_GRID_DIVISIONS
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

logger = logging.getLogger(__name__)


def plot_optical_flow_summary(
    crop_picks: list,
    image_pairs: list[OpticalFlowImagePair],
    pick_labels: list[str],
    image_cache: dict[int, np.ndarray],
    feature_data: pd.DataFrame,
    output_name: str,
    output_dir: Path,
    attachment: float,
    intensity_threshold: float,
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
    crop_picks
        List of crops to plot.
    image_pairs
        List of images pair timepoints and temporal strides.
    pick_labels
        Plot labels for each picked crop.
    image_cache
        Mapping from timepoint index to its 2-D intensity frame.
    feature_data
        Optical flow feature data.
    output_name
        Plot output name.
    output_dir
        Plot output directory.
    attachment
        TVL1 data-fidelity weight (λ).
    intensity_threshold
        Intensity threshold for foreground masking.
    """

    rbar_col = ColumnName.OpticalFlow.UNIT_VECTOR_MEAN

    # Helper: plot one row (5 panels)
    def _plot_row(axes, crop, pair, label):
        t0 = pair.t0
        t1 = pair.t1
        _sx = int(crop[ColumnName.DiffAEData.START_X])
        _sy = int(crop[ColumnName.DiffAEData.START_Y])
        _ex = int(crop[ColumnName.DiffAEData.END_X])
        _ey = int(crop[ColumnName.DiffAEData.END_Y])
        _cidx = int(crop[ColumnName.CROP_INDEX])
        cy, cx = _ey - _sy, _ex - _sx

        f0, f1 = image_cache[t0], image_cache[t1]
        c0, c1 = f0[_sy:_ey, _sx:_ex], f1[_sy:_ey, _sx:_ex]

        uf_full, vf_full = compute_tvl1(f0, f1, attachment=attachment)
        uf, vf = uf_full[_sy:_ey, _sx:_ex], vf_full[_sy:_ey, _sx:_ex]

        sp = np.sqrt(uf**2 + vf**2)
        ang = np.arctan2(vf, uf)
        mask = (c0 > intensity_threshold) | (c1 > intensity_threshold)

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
        crop_rbar = (
            feature_data[feature_data[ColumnName.CROP_INDEX] == _cidx][rbar_col].dropna().values
        )
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
                float(np.mean(crop_rbar)),  # type: ignore[arg-type]
                color="black",
                ls="--",
                lw=1.5,
                label=f"mean $\\bar{{R}}$ = {float(np.mean(crop_rbar)):.3f}",  # type: ignore[arg-type]
            )
            ax.axvline(
                float(np.median(crop_rbar)),  # type: ignore[arg-type]
                color="forestgreen",
                ls=":",
                lw=1.5,
                label=f"median = {float(np.median(crop_rbar)):.3f}",  # type: ignore[arg-type]
            )
            ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel(r"$\bar{R}$", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"(e) $\\bar{{R}}$ distribution crop {_cidx}", fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.tick_params(labelsize=7)

    # Build the figure
    n_rows = len(crop_picks)
    fig, axes = plt.subplots(
        n_rows,
        5,
        figsize=(30, 4.5 * n_rows),
        facecolor="white",
        squeeze=False,
        layout="constrained",
    )

    for row_idx, (crop, pair, label) in enumerate(
        zip(crop_picks, image_pairs, pick_labels, strict=False)
    ):
        _plot_row(axes[row_idx], crop, pair, label)

    fig.suptitle(
        f"Coherent vs. Incoherent: {output_name}",
        fontsize=12,
        fontweight="bold",
    )

    save_plot_to_path(
        fig,
        output_dir,
        f"{output_name}_coherent_vs_incoherent",
        dpi=300,
        show_and_close=True,
        tight_layout=False,
    )


def plot_optical_flow_coherence_over_time(
    feature_data: pd.DataFrame,
    output_name: str,
    output_dir: Path,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
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
    feature_data
        Optical flow feature data.
    output_name
        Plot output name.
    output_dir
        Plot output directory.
    ema_alpha
        EMA alpha value used for smoothing (for column name generation).
    max_crops
        Maximum number of crops to plot.
    max_dt
        Maximum temporal gap to plot.
    """

    crop_col = ColumnName.CROP_INDEX
    time_col = ColumnName.TIMEPOINT

    if crop_col not in feature_data.columns or time_col not in feature_data.columns:
        logger.warning("Missing %s or %s columns — skipping time series plot", crop_col, time_col)
        return

    # Select crops with the longest tracks
    crop_lengths = feature_data.groupby(crop_col)[time_col].nunique().sort_values(ascending=False)
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
        atag = str(ema_alpha).replace(".", "")

        # 1) Raw coherence (mean unit vector)
        raw_col = f"optical_flow_mean_unit_vector{dt_tag}"
        ema_variants = []
        ema_col = f"ema{atag}_optical_flow_mean_unit_vector{dt_tag}"
        ema_variants.append((ema_col, f"EMA \u03b1={ema_alpha}"))
        metric_groups.append((raw_col, rf"Coherence ($\bar{{R}}$, dt={d})", ema_variants))

        # 2) Fast coherence
        raw_fast = f"optical_flow_mean_unit_vector_fast{dt_tag}"
        ema_fast = []
        ema_col = f"ema{atag}_optical_flow_mean_unit_vector_fast{dt_tag}"
        ema_fast.append((ema_col, f"EMA \u03b1={ema_alpha}"))
        metric_groups.append((raw_fast, f"Fast Coherence (speed > thr, dt={d})", ema_fast))

    # Filter to metric groups whose raw column actually exists
    metric_groups = [(r, lbl, e) for r, lbl, e in metric_groups if r in feature_data.columns]
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
        layout="constrained",
    )

    # Use the colour cycle from the active style so that this respects
    # any axes.prop_cycle set in the endo_pipeline.figure mplstyle.
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for row_idx, (raw_col, metric_label, ema_variants) in enumerate(metric_groups):
        ax = axes[row_idx, 0]
        ax.set_facecolor("white")

        for ci, crop_id in enumerate(selected_crops):
            color = prop_cycle[ci % len(prop_cycle)]
            df_crop = feature_data[feature_data[crop_col] == crop_id].sort_values(time_col)
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
        f"Crop Coherence Over Time: {output_name}\n" f"({n_crops} crops, EMA \u03b1 = {ema_alpha})",
        fontsize=12,
        fontweight="bold",
    )
    save_plot_to_path(
        fig,
        output_dir,
        f"{output_name}_coherence_over_time",
        dpi=300,
        show_and_close=True,
        tight_layout=False,
    )
