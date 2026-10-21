"""
Visualization script for bandwidth and binwidth sweep results.
"""


def main():
    """Plot the kernel bandwidth and binwidth sweep results."""
    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.bandwidth_binwidth_sweep_viz import (
        load_sweep_fixed_points,
        plot_fixed_point_count_heatmaps,
        plot_fixed_point_count_scatterplots,
        plot_fixed_point_location_summary,
    )
    from endo_pipeline.settings.bootstrap_fixed_points import BOOTSTRAP_THRESHOLD
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.examples import SUPP_FIG_PARAM_SWEEP

    out_dir = get_output_path(__file__)

    file_format = ".png" if DEMO_MODE else ".pdf"

    # load the bootstrapped fixed points from every sweep run for the datasets in the figure
    df_sweep = load_sweep_fixed_points()
    df_sweep = df_sweep.loc[df_sweep[ColumnName.DATASET].isin(SUPP_FIG_PARAM_SWEEP)]

    for stability in sorted(df_sweep[ColumnName.FIXED_POINT_STABILITY].unique()):
        df_stability = df_sweep.loc[df_sweep[ColumnName.FIXED_POINT_STABILITY] == stability]

        # skip this stability if no parameter combination has a detection rate below the threshold
        if not (df_stability[ColumnName.FIXED_POINT_DETECTION_RATE] < BOOTSTRAP_THRESHOLD).any():
            continue

        plot_fixed_point_count_heatmaps(
            df_sweep, stability, out_dir, datasets=SUPP_FIG_PARAM_SWEEP, file_format=file_format
        )
        plot_fixed_point_count_scatterplots(
            df_sweep, stability, out_dir, datasets=SUPP_FIG_PARAM_SWEEP, file_format=file_format
        )
        plot_fixed_point_location_summary(
            df_sweep, stability, out_dir, datasets=SUPP_FIG_PARAM_SWEEP, file_format=file_format
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
