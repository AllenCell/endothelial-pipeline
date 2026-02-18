"""Visualization of first-pass CDH5 puncta quantifications."""


def main():
    from functools import reduce
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
        df["intens_angle_rel_migration"] = df.apply(
            lambda x: x["angle"] - x["centroid_velocity_angle"], axis=1
        )
        df[["peak_angles", "peak_intensities", "peak_details"]] = df.apply(
            lambda x: get_peaks_of_edge_intensities(x["angle"], x["intensity"])[1:],
            axis=1,
            result_type="expand",
        )
        df["num_peaks"] = df["peak_intensities"].transform(lambda x: len(x))
        df["peak_prominences"] = df["peak_details"].transform(lambda x: x["prominences"])
        df["peak_prominence_max"] = df.peak_prominences.transform(
            lambda x: max(x) if len(x) > 0 else 0
        )
        df["peak_intensity_max"] = df.peak_intensities.transform(
            lambda x: max(x) if len(x) > 0 else 0
        )
        df["edge_fluorescence_au_mean"] = df.intensity.transform(lambda x: np.mean(x))
        df["edge_fluorescence_au_median"] = df.intensity.transform(lambda x: np.median(x))

    df = pd.concat([low_flow_df["intensity"], high_flow_df["intensity"]], ignore_index=True)
    intens_all = df.explode("intensity")
    intens_lim = np.percentile(intens_all.values, 99.9)

    for df_sub in [low_flow_df, high_flow_df, interm_flow_df]:
        dataset_name = df_sub.dataset.unique().item()
        position = df_sub.position.unique().item()

        examples_to_plot = []
        if df_sub is low_flow_df:
            flow_cond = "low_flow"
            for ex in EXAMPLES["low_flow"]:
                if ex["dataset_name"] == dataset_name and ex["position"] == position:
                    examples_to_plot.append(ex)
                    break  # only plot the first example for this proof-of-concept
        elif df_sub is high_flow_df:
            flow_cond = "high_flow"
            for ex in EXAMPLES["high_flow"]:
                if ex["dataset_name"] == dataset_name and ex["position"] == position:
                    examples_to_plot.append(ex)
                    break  # only plot the first example for this proof-of-concept
        elif df_sub is interm_flow_df:
            flow_cond = "interm_flow"
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
            # norm="log",
        )
        for tick_label in ax.get_yticklabels():
            tick_label.set_color("lightgrey")
        fig.suptitle(f"{dataset_name} P{position} {flow_cond.replace('_', ' ').title()}")
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
            # norm="log",
        )
        # ax.semilogy()
        # ax.set_xlim(-np.pi, np.pi)
        # ax.set_ylim(0, intens_lim)
        # ax.scatter(
        #     x=ang,
        #     y=intens,
        #     c="black",
        #     alpha=0.3,
        # )
        for tick_label in ax.get_yticklabels():
            tick_label.set_color("lightgrey")
        fig.suptitle(f"{dataset_name} P{position} {flow_cond.replace('_', ' ').title()}")
        plt.tight_layout()
        fname = f"cdh5_edge_intens_quant_angle_rel_migration_{flow_cond}.png"
        save_plot_to_path(fig, outdir, fname)
        plt.close(fig)

        for ex in examples_to_plot:
            timepoint = ex["timepoint"]
            seg_label = ex["label"]
            fig, axs = show_intensity_measure_example(
                df=df_sub,
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
    sns.histplot(
        low_flow_df.num_peaks,
        binwidth=1,
        color="tab:blue",
        label=low_flow_df.dataset.unique().item(),
        discrete=True,
        ax=ax,
    )
    sns.histplot(
        high_flow_df.num_peaks,
        binwidth=1,
        color="tab:orange",
        label=high_flow_df.dataset.unique().item(),
        discrete=True,
        ax=ax,
    )
    ax.set_xlabel("Number of peaks in edge intensity distribution")
    ax.legend()
    plt.tight_layout()
    fname = "cdh5_edge_intens_quant_num_peaks.png"
    save_plot_to_path(fig, outdir, fname)
    plt.close(fig)

    # fig, ax = plt.subplots()
    # sns.histplot(
    #     low_flow_df.num_peaks,
    #     binwidth=1,
    #     color="tab:blue",
    #     label=low_flow_df.dataset.unique().item(),
    #     element="step",
    #     fill=False,
    #     discrete=True,
    #     cumulative=True,
    #     ax=ax,
    # )
    # sns.histplot(
    #     high_flow_df.num_peaks,
    #     binwidth=1,
    #     color="tab:orange",
    #     label=high_flow_df.dataset.unique().item(),
    #     element="step",
    #     fill=False,
    #     discrete=True,
    #     cumulative=True,
    #     ax=ax,
    # )
    # ax.set_ylabel("Cumulative count")
    # ax.set_xlabel("Number of peaks in edge intensity distribution")
    # ax.legend()
    # plt.tight_layout()
    # fname = "cdh5_edge_intens_quant_num_peaks_cumulative.png"
    # save_plot_to_path(fig, outdir, fname)
    # plt.close(fig)

    fig, ax = plt.subplots()
    sns.histplot(
        reduce(lambda a, b: a + b, low_flow_df.peak_prominences.values.tolist(), []),
        binwidth=1,
        color="tab:blue",
        label=low_flow_df.dataset.unique().item(),
        discrete=True,
        ax=ax,
    )
    sns.histplot(
        reduce(lambda a, b: a + b, high_flow_df.peak_prominences.values.tolist(), []),
        binwidth=1,
        color="tab:orange",
        label=high_flow_df.dataset.unique().item(),
        discrete=True,
        ax=ax,
    )
    ax.set_xlim(0, intens_lim)
    ax.set_ylabel("Cumulative count")
    ax.set_xlabel("Peak prominence value")
    ax.legend()
    plt.tight_layout()
    fname = "cdh5_edge_intens_quant_peak_prom.png"
    save_plot_to_path(fig, outdir, fname)
    plt.close(fig)

    # fig, ax = plt.subplots()
    # sns.histplot(
    #     reduce(lambda a, b: a + b, low_flow_df.peak_prominences.values.tolist(), []),
    #     binwidth=1,
    #     color="tab:blue",
    #     label=low_flow_df.dataset.unique().item(),
    #     element="step",
    #     fill=False,
    #     discrete=True,
    #     cumulative=True,
    #     ax=ax,
    # )
    # sns.histplot(
    #     reduce(lambda a, b: a + b, high_flow_df.peak_prominences.values.tolist(), []),
    #     binwidth=1,
    #     color="tab:orange",
    #     label=high_flow_df.dataset.unique().item(),
    #     element="step",
    #     fill=False,
    #     discrete=True,
    #     cumulative=True,
    #     ax=ax,
    # )
    # ax.set_ylabel("Cumulative count")
    # ax.set_xlabel("Peak prominence value")
    # ax.legend()
    # plt.tight_layout()
    # fname = "cdh5_edge_intens_quant_peak_prom_cumulative.png"
    # save_plot_to_path(fig, outdir, fname)
    # plt.close(fig)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
