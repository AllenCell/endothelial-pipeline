# %%
from src.endo_pipeline.configs import get_model_manifest, load_model_config
from src.endo_pipeline.io.output import get_output_path, save_plot_to_path
from src.endo_pipeline.library.analyze.diffae_manifest import (
    fit_pca,
    get_manifest_for_dynamics_workflows,
)
from src.endo_pipeline.library.analyze.numerics import get_3d_bounds_from_data
from src.endo_pipeline.library.analyze.z_slice_feats.compare_feats import (
    feature_density,
    plot_distribution_by_frame,
    plot_scatter_by_position_and_frame,
)

# %%
model_config = load_model_config("diffae_04_10")
# dataset_name = "20241016_20X"
dataset_name = "20250331_20X"

save_dir = get_output_path("visualize_z_stack_feats", model_config.name, dataset_name)

model_manifest1 = get_model_manifest(dataset_name, model_config)
model_manifest2 = get_model_manifest(dataset_name, model_config, [5, 10])
model_manifest3 = get_model_manifest(dataset_name, model_config, [0, 16])
model_manifest4 = get_model_manifest(dataset_name, model_config, [9, 24])

for model_manifest in [model_manifest1, model_manifest2, model_manifest3, model_manifest4]:
    print(model_manifest.z_stack_offsets)
    print(model_manifest.fmsid)

pca = fit_pca()

df1 = get_manifest_for_dynamics_workflows(model_manifest1, pca, filter_to_valid=False)
df1 = df1[df1["frame_number"].isin([0, 250, 500])].reset_index(drop=True)
df2 = get_manifest_for_dynamics_workflows(model_manifest2, pca, filter_to_valid=False)
df3 = get_manifest_for_dynamics_workflows(model_manifest3, pca, filter_to_valid=False)
df4 = get_manifest_for_dynamics_workflows(model_manifest4, pca, filter_to_valid=False)

assert df1.shape == df2.shape == df3.shape == df4.shape, "DataFrames have different shapes!"
# %%
manifest_list = [model_manifest1, model_manifest2, model_manifest3, model_manifest4]
df_list = [df1, df2, df3, df4]
df_info = ["all slices", "centered slices (-5, +10)", "bottom slices (0, 16)", "top slices (9, 24)"]
bounds = get_3d_bounds_from_data(manifest_list, pca)

# %%
for target_frame in [0, 250, 500]:
    for df, info in zip(df_list, df_info):
        fig, ax = plot_scatter_by_position_and_frame(df, target_frame, bounds, info, dataset_name)

# %%
# plot_distribution_by_frame(df_list, df_info, target_frame=0)
# %%
for target_frame in [0, 250, 500]:
    for df, info in zip(df_list, df_info):
        df = df[df["frame_number"] == target_frame]
        fig, ax = feature_density(
            df, "pc1", bounds[0], title=f"{dataset_name} {info}, T={target_frame} (frames)"
        )
        save_plot_to_path(
            fig,
            save_dir,
            f"{dataset_name}_{info}_pc1_T{target_frame}_density_plot",
        )
# %%
for target_frame in [0, 250, 500]:
    for df, info in zip(df_list, df_info):
        df = df[df["frame_number"] == target_frame]
        fig, ax = feature_density(
            df, "pc2", bounds[1], title=f"{dataset_name} {info}, T={target_frame} (frames)"
        )
        save_plot_to_path(
            fig,
            save_dir,
            f"{dataset_name}_{info}_pc2_T{target_frame}_density_plot",
        )
# %%
for target_frame in [0, 250, 500]:
    for df, info in zip(df_list, df_info):
        df = df[df["frame_number"] == target_frame]
        fig, ax = feature_density(
            df, "pc3", bounds[2], title=f"{dataset_name} {info}, T={target_frame} (frames)"
        )
        save_plot_to_path(
            fig,
            save_dir,
            f"{dataset_name}_{info}_pc3_T{target_frame}_density_plot",
        )
# %%
