"""Visualization of first-pass CDH5 puncta quantifications."""


def main():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.intensity_features import (
        EXAMPLES,
        get_peaks_of_edge_intensities,
        show_intensity_measure_example,
    )
    from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
        calculate_derived_data_dynamics_dependent,
        get_smallest_angle_difference,
    )

    outdir = get_output_path(__file__)

    low_flow_fp = Path(
        "//allen/aics/users/serge.parent/cellsmap/results/2026-02-17/cdh5_puncta_quant/20250402_20X_edge_intensities.parquet"
    )
    high_flow_fp = Path(
        "//allen/aics/users/serge.parent/cellsmap/results/2026-02-17/cdh5_puncta_quant/20250611_20X_edge_intensities.parquet"
    )
    interm_flow_fp = Path(
        "//allen/aics/users/serge.parent/cellsmap/results/2026-02-17/cdh5_puncta_quant/20250818_20X_edge_intensities.parquet"
    )

    low_flow_df = pd.read_parquet(low_flow_fp)
    high_flow_df = pd.read_parquet(high_flow_fp)
    interm_flow_df = pd.read_parquet(interm_flow_fp)

    for df in tqdm([low_flow_df, high_flow_df, interm_flow_df]):
        df = calculate_derived_data_dynamics_dependent(df)
        df[["peak_angles", "peak_intensities", "peak_details", "peak_width_details"]] = df.apply(
            lambda x: get_peaks_of_edge_intensities(x["angle"], x["intensity"])[1:],
            axis=1,
            result_type="expand",
        )
        df["num_peaks"] = df["peak_intensities"].transform(lambda x: len(x))
        df["peak_prominences"] = df["peak_details"].transform(lambda x: x["prominences"])
        df["peak_widths"] = df["peak_width_details"].transform(lambda x: x["widths"])
        df["peak_prominence_max"] = df.peak_prominences.transform(
            lambda x: max(x) if len(x) > 0 else 0
        )
        df["peak_intensity_max"] = df.peak_intensities.transform(
            lambda x: max(x) if len(x) > 0 else 0
        )
        df["edge_fluorescence_au_mean"] = df.intensity.transform(lambda x: np.mean(x))
        df["edge_fluorescence_au_median"] = df.intensity.transform(lambda x: np.median(x))
        df["peak_angle_rel_migration"] = df.apply(
            lambda x: get_smallest_angle_difference(
                angles=np.array(x["peak_angles"]),
                reference_angles=np.array([x["centroid_velocity_angle"]] * len(x["peak_angles"])),
                units="rad",
            ),
            axis=1,
        )

        df_sub = df[["peak_angle_rel_migration", "peak_intensities", "peak_prominences"]]
        intens_all = df_sub.explode(
            ["peak_angle_rel_migration", "peak_intensities", "peak_prominences"]
        ).dropna()
        intens_all = intens_all.sort_values("peak_angle_rel_migration")

        peak_angle_bin_locs = np.linspace(-np.pi, np.pi, 361, endpoint=True)
        peak_angle_bin_widths = np.deg2rad(5)
        peak_angle_rolling_means = []
        peak_angle_rolling_quantile = []
        # peak_angle_rolling_stds = []
        for bin_center in peak_angle_bin_locs:
            bin_mask = (
                intens_all["peak_angle_rel_migration"] >= bin_center - peak_angle_bin_widths / 2
            ) & (intens_all["peak_angle_rel_migration"] < bin_center + peak_angle_bin_widths / 2)
            peak_angle_rolling_means.append(intens_all.loc[bin_mask].peak_intensities.mean())
            peak_angle_rolling_quantile.append(
                intens_all.loc[bin_mask].peak_intensities.quantile(0.8)
            )
            # peak_angle_rolling_stds.append(intens_all.loc[bin_mask].peak_intensities.std())

        intens_lim = np.percentile(intens_all["peak_intensities"].values, 99.9)

        dataset_name = df.dataset.unique().item()
        position = df.position.unique().item()

        examples_to_plot = []
        if df is low_flow_df:
            flow_cond = "low_flow"
            for ex in EXAMPLES["low_flow"]:
                if ex["dataset_name"] == dataset_name and ex["position"] == position:
                    examples_to_plot.append(ex)
                    break  # only plot the first example for this proof-of-concept
        elif df is high_flow_df:
            flow_cond = "high_flow"
            for ex in EXAMPLES["high_flow"]:
                if ex["dataset_name"] == dataset_name and ex["position"] == position:
                    examples_to_plot.append(ex)
                    break  # only plot the first example for this proof-of-concept
        elif df is interm_flow_df:
            flow_cond = "interm_flow"
        else:
            flow_cond = "unknown_flow"

        cmap = "inferno"
        scatter_alpha = 0.2
        polar_hist_bins = [72, 100]
        polar_range = [[-np.pi, np.pi], [0, intens_lim]]

        fig, ax = plt.subplots()
        intens = np.concatenate(df.peak_intensities.values)
        ang = np.concatenate(df.peak_angles.values)
        ax.hist2d(x=ang, y=intens, bins=polar_hist_bins, range=polar_range, cmap=cmap, norm="log")
        fig.suptitle(f"{dataset_name} P{position} {flow_cond.replace('_', ' ').title()}")
        plt.tight_layout()
        fname = f"cdh5_edge_intens_quant_angle_rel_flow_{flow_cond}.png"
        save_plot_to_path(fig, outdir, fname)
        plt.close(fig)

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        intens = np.concatenate(df.peak_intensities.values)
        ang = np.concatenate(df.peak_angles.values)
        ax.hist2d(x=ang, y=intens, bins=polar_hist_bins, range=polar_range, cmap=cmap, norm="log")
        for tick_label in ax.get_yticklabels():
            tick_label.set_color("lightgrey")
        fig.suptitle(f"{dataset_name} P{position} {flow_cond.replace('_', ' ').title()}")
        plt.tight_layout()
        fname = f"cdh5_edge_intens_quant_angle_rel_flow_{flow_cond}_polar.png"
        save_plot_to_path(fig, outdir, fname)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.set_xlim(-np.pi, np.pi)
        ax.scatter(x=ang, y=intens, c="black", alpha=scatter_alpha, marker=".")
        fig.suptitle(f"{dataset_name} P{position} {flow_cond.replace('_', ' ').title()}")
        plt.tight_layout()
        fname = f"cdh5_edge_intens_quant_angle_rel_flow_{flow_cond}_scatter.png"
        save_plot_to_path(fig, outdir, fname)
        plt.close(fig)

        fig, ax = plt.subplots()
        intens = np.concatenate(df.peak_intensities.values)
        ang = np.concatenate(df.peak_angle_rel_migration.values)
        ax.hist2d(x=ang, y=intens, bins=polar_hist_bins, range=polar_range, cmap=cmap, norm="log")
        fig.suptitle(f"{dataset_name} P{position} {flow_cond.replace('_', ' ').title()}")
        plt.tight_layout()
        fname = f"cdh5_edge_intens_quant_angle_rel_migration_{flow_cond}.png"
        save_plot_to_path(fig, outdir, fname)
        plt.close(fig)

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        intens = np.concatenate(df.peak_intensities.values)
        ang = np.concatenate(df.peak_angle_rel_migration.values)
        ax.hist2d(x=ang, y=intens, bins=polar_hist_bins, range=polar_range, cmap=cmap, norm="log")
        for tick_label in ax.get_yticklabels():
            tick_label.set_color("lightgrey")
        fig.suptitle(f"{dataset_name} P{position} {flow_cond.replace('_', ' ').title()}")
        plt.tight_layout()
        fname = f"cdh5_edge_intens_quant_angle_rel_migration_{flow_cond}_polar.png"
        save_plot_to_path(fig, outdir, fname)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.set_xlim(-np.pi, np.pi)
        ax.scatter(x=ang, y=intens, c="black", alpha=scatter_alpha, marker=".")
        fig.suptitle(f"{dataset_name} P{position} {flow_cond.replace('_', ' ').title()}")
        plt.tight_layout()
        fname = f"cdh5_edge_intens_quant_angle_rel_migration_{flow_cond}_scatter.png"
        save_plot_to_path(fig, outdir, fname)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.set_xlim(-np.pi, np.pi)
        sns.lineplot(
            x=peak_angle_bin_locs,
            y=peak_angle_rolling_means,
            c="black",
            marker=".",
            ax=ax,
        )
        fig.suptitle(f"{dataset_name} P{position} {flow_cond.replace('_', ' ').title()}")
        plt.tight_layout()
        fname = f"cdh5_edge_intens_quant_angle_rel_migration_{flow_cond}_rolling_mean.png"
        save_plot_to_path(fig, outdir, fname)
        plt.close(fig)

        for ex in examples_to_plot:
            timepoint = ex["timepoint"]
            seg_label = ex["label"]
            fig, axs = show_intensity_measure_example(
                df=df,
                dataset_name=dataset_name,
                position=position,
                timepoint=timepoint,
                seg_label=seg_label,
            )
            fname = (
                f"cdh5_edge_intens_quant_example_{dataset_name}_T{timepoint}_label{seg_label}.png"
            )
            save_plot_to_path(fig, outdir, fname)
            plt.close(fig)

    fig, ax = plt.subplots()
    for df_sub in [low_flow_df, high_flow_df, interm_flow_df]:
        sns.histplot(
            df_sub.num_peaks,
            binwidth=1,
            label=df_sub.dataset.unique().item(),
            discrete=True,
            stat="percent",
            alpha=0.3,
            ax=ax,
        )
    ax.set_xlim(-0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("Number of peaks in edge intensity distribution")
    ax.legend()
    plt.tight_layout()
    fname = "cdh5_edge_intens_quant_num_peaks.png"
    save_plot_to_path(fig, outdir, fname)
    plt.close(fig)

    fig, ax = plt.subplots()
    for df_sub in [low_flow_df, high_flow_df, interm_flow_df]:
        peak_prominences = [
            item
            for sublist in df_sub.peak_prominences.values.tolist()
            for item in sublist
            if len(sublist) > 0
        ]
        sns.histplot(
            peak_prominences,
            binwidth=1,
            label=df_sub.dataset.unique().item(),
            stat="percent",
            alpha=0.3,
            kde=True,
            ax=ax,
        )
    ax.set_xlim(0, np.percentile(peak_prominences, 99.9))
    ax.set_xlabel("Peak prominence value")
    ax.legend()
    plt.tight_layout()
    fname = "cdh5_edge_intens_quant_peak_prom.png"
    save_plot_to_path(fig, outdir, fname)
    plt.close(fig)

    fig, ax = plt.subplots()
    for df_sub in [low_flow_df, high_flow_df, interm_flow_df]:
        peak_widths_flat = [
            item
            for sublist in df_sub.peak_widths.values.tolist()
            for item in sublist
            if len(sublist) > 0
        ]
        sns.histplot(
            peak_widths_flat,
            binwidth=1,
            label=df_sub.dataset.unique().item(),
            stat="percent",
            alpha=0.3,
            kde=True,
            ax=ax,
        )
    ax.set_xlim(0)
    ax.set_xlabel("Peak width value")
    ax.legend()
    plt.tight_layout()
    fname = "cdh5_edge_intens_quant_peak_width.png"
    save_plot_to_path(fig, outdir, fname)
    plt.close(fig)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
