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


def main() -> None:
    """Build the supplementary optical-flow figure."""
    import matplotlib.pyplot as plt

    from endo_pipeline.io import get_output_path
    from endo_pipeline.io.output import save_plot_to_path
    from endo_pipeline.library.analyze.optical_flow import resolve_attachment
    from endo_pipeline.library.analyze.optical_flow.compute import compute_tvl1
    from endo_pipeline.library.visualize.figures import FigurePanel, build_figure_from_panels
    from endo_pipeline.library.visualize.supp_fig_optical_flow import (
        build_bf_frame_cache,
        load_optical_flow_feature_df,
        plot_optical_flow_composite,
        plot_optical_flow_quiver,
        rbar,
        resolve_grid_crop,
    )
    from endo_pipeline.settings import DIFFAE_ZARR_RESOLUTION_LEVEL
    from endo_pipeline.settings.examples import (
        SUPP_FIG_OPTICAL_FLOW_COHERENT_EXAMPLE,
        SUPP_FIG_OPTICAL_FLOW_INCOHERENT_EXAMPLE,
    )

    plt.style.use("endo_pipeline.figure")
    output_dir = get_output_path("supp_fig_optical_flow")
    figure_size = (5.0, 5.0)

    # The picks are ExampleImage instances whose
    # crop_x_start / crop_y_start fields are interpreted as the
    # 1-indexed (col, row) of the chosen crop within the position's regular
    # crop grid, and the timepoint pair is taken as (timepoint, timepoint + 1).

    picks = [
        (SUPP_FIG_OPTICAL_FLOW_COHERENT_EXAMPLE, "Coherent Example"),
        (SUPP_FIG_OPTICAL_FLOW_INCOHERENT_EXAMPLE, "Incoherent Example"),
    ]
    attachment = resolve_attachment("BF")

    # Load per-pick BF frames + crop bbox + flow.
    panels: list[dict] = []
    for example, label in picks:
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
        figsize=figure_size,
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
        dpi=300,
        show_and_close=False,
        tight_layout=False,
        bbox_inches=None,
    )
    plt.close(fig)

    # Compose into the standard figure canvas (single panel, no letter).
    build_figure_from_panels(
        [
            FigurePanel(
                letter="",
                path=output_dir / f"{base_name}.svg",
                x_position=0,
                y_position=0,
                x_offset=0,
                y_offset=0,
            ),
        ],
        output_dir / "supp_fig_optical_flow.svg",
        width=figure_size[0],
        height=figure_size[1],
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
