def main():
    """Development workflow to test calculation of classic feature dynamics."""
    import seaborn as sns
    from matplotlib import pyplot as plt

    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
        calculate_derived_data_dynamics_dependent,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.workflow_defaults import (
        CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME,
        SEGMENTATION_FEATURE_COLUMNS,
    )

    out_dir = get_output_path(__file__)

    dataset_name = "20251001_20X"

    cell_centric_feats_manifest = load_dataframe_manifest(
        CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME
    )
    cell_centric_feats_location = get_dataframe_location_for_dataset(
        cell_centric_feats_manifest, dataset_name
    )
    df_delayed = load_dataframe(cell_centric_feats_location, delay=True)
    df_delayed = df_delayed.reset_index(drop=True)

    cols_to_compute = SEGMENTATION_FEATURE_COLUMNS["dynamics_calculation_prereq"] + ["is_included"]
    df = df_delayed[cols_to_compute].compute()
    df = df[df.is_included]
    df = calculate_derived_data_dynamics_dependent(df)

    # for nm, df_grp in df.groupby(["dataset_name", "position"]):
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="time_minutes",
        y="dnum_nuclei_in_crop_dt",
        hue="position",
        ax=ax,
    )
    ax.axhline(0, color="grey", linestyle="--")
    save_plot_to_path(fig, out_dir, "dnum_nuclei_in_crop_dt")

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="time_minutes",
        y="dmean_EGFP_intensity_dt",
        hue="position",
        ax=ax,
    )
    ax.axhline(0, color="grey", linestyle="--")
    save_plot_to_path(fig, out_dir, "dmean_EGFP_intensity_dt")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
