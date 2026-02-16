"""Visualization of first-pass CDH5 puncta quantifications."""


def main():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
        calculate_derived_data_dynamics_dependent,
    )

    outdir = get_output_path(__file__)

    low_flow_fp = Path(
        "//allen/aics/users/serge.parent/cellsmap/results/2026-02-16/cdh5_puncta_quant/20250402_20X_edge_intensities.parquet"
    )
    high_flow_fp = Path(
        "//allen/aics/users/serge.parent/cellsmap/results/2026-02-16/cdh5_puncta_quant/20250611_20X_edge_intensities.parquet"
    )

    low_flow_df = pd.read_parquet(low_flow_fp)
    high_flow_df = pd.read_parquet(high_flow_fp)

    for df in [low_flow_df, high_flow_df]:
        df = calculate_derived_data_dynamics_dependent(df)
        df["intens_angle_rel_migration"] = df.apply(
            lambda x: x["angle"] - x["centroid_velocity_angle"], axis=1
        )

    df = pd.concat([low_flow_df["intensity"], high_flow_df["intensity"]], ignore_index=True)
    intens_all = df.explode("intensity")
    intens_lim = np.percentile(intens_all.values, 99.9)

    for df_sub in [low_flow_df, high_flow_df]:
        dataset_name = df_sub.dataset.unique().item()
        if df_sub is low_flow_df:
            flow_cond = "low_flow"
        elif df_sub is high_flow_df:
            flow_cond = "high_flow"
        else:
            flow_cond = "unknown_flow"

        cmap = "inferno"
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        intens = np.concatenate(df_sub.intensity.values)
        ang = np.concatenate(df_sub.angle.values)
        ax.hist2d(
            x=ang,
            y=intens,
            bins=[72, 100],
            range=[[-np.pi, np.pi], [0, intens_lim]],
            cmap=cmap,
        )
        for tick_label in ax.get_yticklabels():
            tick_label.set_color("lightgrey")
        fig.suptitle(f"{dataset_name} {flow_cond.replace('_', ' ').title()}")
        plt.tight_layout()
        fname = f"cdh5_edge_intens_quant_angle_rel_flow_{flow_cond}.png"
        save_plot_to_path(fig, outdir, fname)
        plt.close(fig)

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        intens = np.concatenate(df_sub.intensity.values)
        ang = np.concatenate(df_sub.intens_angle_rel_migration.values)
        ax.hist2d(
            x=ang,
            y=intens,
            bins=[72, 100],
            range=[[-np.pi, np.pi], [0, intens_lim]],
            cmap=cmap,
        )
        for tick_label in ax.get_yticklabels():
            tick_label.set_color("lightgrey")
        fig.suptitle(f"{dataset_name} {flow_cond.replace('_', ' ').title()}")
        plt.tight_layout()
        fname = f"cdh5_edge_intens_quant_angle_rel_migration_{flow_cond}.png"
        save_plot_to_path(fig, outdir, fname)
        plt.close(fig)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
