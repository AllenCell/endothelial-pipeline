# %%
from endo_pipeline.io.output import get_output_path, save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.library.analyze.numerics import get_3d_bounds_from_data
from endo_pipeline.library.analyze.z_slice_feats.compare_feats import (
    feature_density,
    plot_scatter_by_position_and_frame,
)
from endo_pipeline.manifests import DataframeManifest, load_dataframe_manifest

# %%
TIMEPOINTS = [0, 250, 500]

# %%
model_name = "diffae_04_10"
dataset_list = ["20250331_20X", "20250402_20X", "20250409_20X"]

pca = fit_pca()

manifest1 = load_dataframe_manifest(model_name)
manifest2 = load_dataframe_manifest(f"{model_name}_z_stack_5_10")
manifest3 = load_dataframe_manifest(f"{model_name}_z_stack_0_16")
manifest4 = load_dataframe_manifest(f"{model_name}_z_stack_9_24")

# Generate a dynamic dataframe manifest that combines the above manifests for
# different z stack offsets. All locations in each manifest are combined into
# a single location dictionary, with an index appended to the dataset name to
# differentiate the different conditions.
#
#   SEPARATE: manifest1.location["dataset_a"] = (manifest 1 location a)
#   COMBINED: combined_manifest.locations["dataset_a-0"] = (manifest 1 location a)
#
# This could probably be streamlined, but since this is just used for getting
# bounds, this is the version that make the minimal changes.
combined_manifest = DataframeManifest(
    name=f"{model_name}_z_stacks",
    workflow="-",
    locations={
        f"{key}-{index}": location
        for index, manifest in enumerate([manifest1, manifest2, manifest3, manifest4])
        for key, location in manifest.locations.items()
    },
)

for dataset_name in dataset_list:
    save_dir = get_output_path("visualize_z_stack_feats", model_name, dataset_name)

    df1 = get_dataframe_for_dynamics_workflows(dataset_name, manifest1, pca, filter_to_valid=False)
    df1 = df1[df1["frame_number"].isin(TIMEPOINTS)].reset_index(drop=True)
    df2 = get_dataframe_for_dynamics_workflows(dataset_name, manifest2, pca, filter_to_valid=False)
    df3 = get_dataframe_for_dynamics_workflows(dataset_name, manifest3, pca, filter_to_valid=False)
    df4 = get_dataframe_for_dynamics_workflows(dataset_name, manifest4, pca, filter_to_valid=False)

    assert df1.shape == df2.shape == df3.shape == df4.shape, "DataFrames have different shapes!"

    manifest_list = [manifest1, manifest2, manifest3, manifest4]
    df_list = [df1, df2, df3, df4]
    df_info = [
        "all slices",
        "centered slices (-5, +10)",
        "bottom slices (0, 16)",
        "top slices (9, 24)",
    ]

    datasets_for_bounds = [f"{dataset_name}-{index}" for index in range(4)]
    bounds = get_3d_bounds_from_data(
        datasets_for_bounds, combined_manifest, pca, filter_to_valid=False
    )

    for target_frame in TIMEPOINTS:
        for df, info in zip(df_list, df_info, strict=True):
            fig, ax = plot_scatter_by_position_and_frame(
                df, target_frame, bounds, info, dataset_name
            )
            save_plot_to_path(
                fig,
                save_dir,
                f"{dataset_name}_{info}_pc1_T{target_frame}_scatter_plot",
                transparent=True,
            )

    for target_frame in TIMEPOINTS:
        for df, info in zip(df_list, df_info, strict=True):
            df = df[df["frame_number"] == target_frame]
            for pc, bound in zip(["pc1", "pc2", "pc3"], bounds, strict=True):
                fig, ax = feature_density(
                    df, pc, bound, title=f"{dataset_name} {info}, T={target_frame} (frames)"
                )
                save_plot_to_path(
                    fig,
                    save_dir,
                    f"{dataset_name}_{info}_{pc}_T{target_frame}_density_plot",
                    transparent=True,
                )

# %%
