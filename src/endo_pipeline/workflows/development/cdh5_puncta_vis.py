from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main():
    low_flow_fp = Path(
        "//allen/aics/users/serge.parent/cellsmap/results/2026-02-14/cdh5_puncta_quant/20250402_20X_edge_intensities.parquet"
    )
    high_flow_fp = Path(
        "//allen/aics/users/serge.parent/cellsmap/results/2026-02-14/cdh5_puncta_quant/20250611_20X_edge_intensities.parquet"
    )

    low_flow_df = pd.read_parquet(low_flow_fp)
    high_flow_df = pd.read_parquet(high_flow_fp)

    df = pd.concat([low_flow_df, high_flow_df], ignore_index=True)
    df.drop(
        columns=[
            "start_x_cdh5_seg",
            "start_y_cdh5_seg",
            "end_x_cdh5_seg",
            "end_y_cdh5_seg",
            "centroid_X",
            "centroid_Y",
            "frame_number",
        ],
        inplace=True,
    )
    df = df.explode(["intensity", "angle"]).reset_index(drop=True)

    intens_sorted = np.sort(df.intensity.values)
    intens_lim = np.percentile(intens_sorted, 99.9)

    # fig, ax = plt.subplots()
    # ax.semilogy()
    # sns.histplot(data=df, x="intensity", hue="dataset", binwidth=1)
    # ax.set_xlim(0, intens_lim)

    # fig, ax = plt.subplots()
    # # ax.semilogy()
    # for df_sub in [low_flow_df, high_flow_df]:
    #     intens_sorted = np.sort(np.concatenate(df_sub.intensity.values))
    #     sns.kdeplot(intens_sorted)
    # ax.set_xlim(0, intens_lim)

    # ax.semilogy()
    for df_sub in [low_flow_df, high_flow_df]:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        intens = np.concatenate(df_sub.intensity.values)
        ang = np.concatenate(df_sub.angle.values)
        # ax.scatter(x=ang, y=intens, s=1, alpha=0.1)
        ax.hist2d(
            x=ang,
            y=intens,
            bins=[72, 100],
            range=[[-np.pi, np.pi], [0, intens_lim]],
            # norm=plt.matplotlib.colors.LogNorm(),
        )
        plt.show()
        # break
