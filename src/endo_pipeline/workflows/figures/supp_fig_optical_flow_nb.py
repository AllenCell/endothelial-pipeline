"""Supplementary figure: optical-flow coherent vs. incoherent example panels.

Renders a 2x2 figure with two columns:

    column 0 -- "Coherent Example":   composite (top), quiver (bottom)
    column 1 -- "Incoherent Example": composite (top), quiver (bottom)

The composite is a magenta/green merge of consecutive BF frames
(overlap = white) with a scale bar; the quiver shows the TVL1 flow
field annotated with the migration coherence (R-bar).

The picks live in :mod:`endo_pipeline.settings.examples` as
``SUPP_FIG_OPTICAL_FLOW_{COHERENT,INCOHERENT}_EXAMPLE``; the per-panel
plotting helpers live in
:mod:`endo_pipeline.library.visualize.supp_fig_optical_flow`.

"""

# %%
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from endo_pipeline.io import get_output_path
from endo_pipeline.io.output import save_plot_to_path, slugify
from endo_pipeline.library.analyze.optical_flow import compute_tvl1
from endo_pipeline.library.visualize.data_example_figures import (
    create_panel_retraction_fiber_blob_example,
)
from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
from endo_pipeline.library.visualize.migration_coherence import make_example_migration_coherence
from endo_pipeline.library.visualize.supp_fig_optical_flow import (
    build_bf_frame_cache,
    load_optical_flow_feature_df,
    plot_optical_flow_composite,
    plot_optical_flow_quiver,
    rbar,
    resolve_grid_crop,
)
from endo_pipeline.settings import DIFFAE_ZARR_RESOLUTION_LEVEL
from endo_pipeline.settings.column_metadata import COLUMN_METADATA
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.examples import (
    SUPP_FIG_OPTICAL_FLOW_COHERENT_EXAMPLE,
    SUPP_FIG_OPTICAL_FLOW_INCOHERENT_EXAMPLE,
    SUPP_FIG_RETRACTION_FIBER_BLOB,
)
from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, MAX_FIGURE_HEIGHT, MAX_FIGURE_WIDTH
from endo_pipeline.settings.optical_flow import OPTICAL_FLOW_CHANNEL_ATTACHMENT

plt.style.use("endo_pipeline.figure")
output_dir = get_output_path("migration_coherence_blob")
optical_flow_panel_size = (3.5, 3.5)
migration_coherence_panel_size = (3, 3)
# %%
# The picks are ExampleImage instances whose
# crop_x_start / crop_y_start fields are interpreted as the
# 1-indexed (col, row) of the chosen crop within the position's regular
# crop grid, and the timepoint pair is taken as (timepoint, timepoint + 1).

picks = {
    "Coherent Example": SUPP_FIG_OPTICAL_FLOW_COHERENT_EXAMPLE,
    "Incoherent Example": SUPP_FIG_OPTICAL_FLOW_INCOHERENT_EXAMPLE,
}
attachment = OPTICAL_FLOW_CHANNEL_ATTACHMENT["BF"]

# Load per-pick BF frames + crop bbox + flow.
panels: list[dict] = []
for label, example in picks.items():
    feature_df = load_optical_flow_feature_df(example.dataset_name)
    t0, t1 = example.timepoint, example.timepoint + 1
    cache, crop_grid = build_bf_frame_cache(
        dataset_name=example.dataset_name,
        position=example.position,
        level=DIFFAE_ZARR_RESOLUTION_LEVEL,
        timepoints=[t0, t1],
        df_dataset=feature_df,
    )
    sx, sy, ex, ey = resolve_grid_crop(
        crop_grid,
        grid_row=example.crop_y_start,
        grid_col=example.crop_x_start,
    )
    # Compute TVL1 on the full image and slice to the crop, which
    # avoids edge artefacts at the crop border.
    uf_full, vf_full = compute_tvl1(cache[t0], cache[t1], attachment=attachment)
    uf, vf = uf_full[sy:ey, sx:ex], vf_full[sy:ey, sx:ex]
    panels.append(
        {
            "label": label,
            "frame_t0": cache[t0][sy:ey, sx:ex],
            "frame_t1": cache[t1][sy:ey, sx:ex],
            "uf": uf,
            "vf": vf,
        }
    )

# 2x2 figure: row 0 = composites, row 1 = quivers.
fig, axes = plt.subplots(
    2,
    2,
    figsize=optical_flow_panel_size,
    facecolor="white",
    squeeze=False,
    constrained_layout=True,
    gridspec_kw={"wspace": 0.05, "hspace": 0.05},
)
for col_idx, panel in enumerate(panels):
    plot_optical_flow_composite(
        axes[0, col_idx],
        panel["frame_t0"],
        panel["frame_t1"],
        title=panel["label"],
    )
    plot_optical_flow_quiver(
        axes[1, col_idx],
        panel["uf"],
        panel["vf"],
        title=f"Migration Coherence = {rbar(panel['uf'], panel['vf']):.2f}",
    )

# Save raster (tight) + SVG (untight) for downstream composition.
base_name = "optical_flow_panels"
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
# embedded panel is exactly ``figure_size`` inches.
save_plot_to_path(
    fig,
    output_dir,
    base_name,
    file_format=".svg",
    show_and_close=False,
    tight_layout=False,
    bbox_inches=None,
)
plt.close(fig)

# %%
# create and plot an example of the coherence data in theta, r, rho space
# with the bin drawn around the fixed point
for label, example in picks.items():
    dataset_name = example.dataset_name
    coherence_example_fig_name = slugify(f"{dataset_name}_3D_scatter_{label}")
    make_example_migration_coherence(
        dataset_name=dataset_name,
        figure_size=migration_coherence_panel_size,
        output_dir=output_dir,
        fig_name=coherence_example_fig_name,
    )

# %% Save standalone colorbar to go with TFE examples
of_metadata = COLUMN_METADATA[Column.OpticalFlow.UNIT_VECTOR_MEAN]
cmap = LinearSegmentedColormap.from_list("cyan_magenta", ["cyan", "magenta"])
fig_cb, ax_cb = plt.subplots(figsize=(1.6, 0.1), layout="constrained")
norm = plt.Normalize(vmin=of_metadata.min, vmax=of_metadata.max)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig_cb.colorbar(sm, cax=ax_cb, orientation="horizontal")
cbar.set_ticks(list(of_metadata.ticks))
cbar.set_label(of_metadata.label, fontsize=FONTSIZE_MEDIUM)
save_plot_to_path(
    fig_cb,
    output_dir,
    "colorbar_cyan_magenta_0_1",
    file_format=".svg",
    tight_layout=False,
    pad_inches=0,
    transparent=True,
)

# %% create and plot an example of the retraction fibers and blobs
t = SUPP_FIG_RETRACTION_FIBER_BLOB.timepoint
create_panel_retraction_fiber_blob_example(
    example=SUPP_FIG_RETRACTION_FIBER_BLOB,
    timepoints=list(range(t, t + 15, 3)),
    save_dir=output_dir,
    figure_size=(4.3, 2.5),
)

# %% Put the panels together into a single figure
build_figure_from_panels(
    [
        FigurePanel(
            letter="A",
            path=output_dir / f"{base_name}.svg",
            x_position=0,
            y_position=0,
            x_offset=0,
            y_offset=0.1,
        ),
        FigurePanel(
            letter="B",
            path=output_dir / "20250409_20x_3d_scatter_coherent_example.svg",
            x_position=3.5,
            y_position=0,
            x_offset=0,
            y_offset=-0.3,
        ),
        FigurePanel(
            letter="",
            path=output_dir / "20251001_20x_3d_scatter_incoherent_example.svg",
            x_position=3.5,
            y_position=2.1,
            x_offset=0,
            y_offset=0,
        ),
        FigurePanel(
            letter="C",
            path=output_dir / "retraction_fiber_blob_example.svg",
            x_position=0,
            y_position=5,
            x_offset=0.1,
            y_offset=0.1,
        ),
        FigurePanel(
            letter="",
            path=output_dir / "colorbar_cyan_magenta_0_1.svg",
            x_position=0,
            y_position=3.8,
            x_offset=0,
            y_offset=0.1,
        ),
    ],
    output_dir / "supp_fig_migration_coherence.svg",
    width=MAX_FIGURE_WIDTH,
    height=MAX_FIGURE_HEIGHT,
)
# %%
