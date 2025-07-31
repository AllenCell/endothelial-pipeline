# %%
from src.endo_pipeline.configs import get_model_manifest, load_model_config
from src.endo_pipeline.library.analyze.diffae_manifest import (
    fit_pca,
    get_manifest_for_dynamics_workflows,
)
from src.endo_pipeline.library.analyze.numerics import get_3d_bounds_from_data
from src.endo_pipeline.library.analyze.z_slice_feats.compare_feats import (
    feature_density,
    plot_distribution_by_frame,
    plot_distribution_by_position_and_frame,
    plot_scatter_by_position_and_frame,
)

# %%
model_config = load_model_config("diffae_04_10")
dataset_name = "20241016_20X"
# dataset_name = "20250331_20X"
# %%
model_manifest1 = get_model_manifest(dataset_name, model_config)
model_manifest2 = get_model_manifest(dataset_name, model_config, [5, 10])
model_manifest3 = get_model_manifest(dataset_name, model_config, [0, 16])
model_manifest4 = get_model_manifest(dataset_name, model_config, [9, 25])
# %%
for model_manifest in [model_manifest1, model_manifest2, model_manifest3, model_manifest4]:
    print(model_manifest.z_stack_offsets)
    print(model_manifest.fmsid)
# %%
pca = fit_pca()
# %%
df1 = get_manifest_for_dynamics_workflows(model_manifest1, pca)
df2 = get_manifest_for_dynamics_workflows(model_manifest2, pca)
df3 = get_manifest_for_dynamics_workflows(model_manifest3, pca)
df4 = get_manifest_for_dynamics_workflows(model_manifest4, pca)
# %%
manifest_list = [model_manifest1, model_manifest2, model_manifest3, model_manifest4]
df_list = [df1, df2, df3, df4]
df_info = ["all slices", "centered slices (-5, +10)", "bottom slices (0, 16)", "top slices (9, 25)"]
bounds = get_3d_bounds_from_data(manifest_list, pca)

# %%
target_frame = 0
fig, ax = plot_scatter_by_position_and_frame(df1, target_frame, bounds)
fig, ax = plot_scatter_by_position_and_frame(df2, target_frame, bounds)
# %%
target_frame = 0
fig, ax = plot_distribution_by_position_and_frame(df1, target_frame)
fig, ax = plot_distribution_by_position_and_frame(df1, target_frame)
# %%
plot_distribution_by_frame(df_list, df_info, target_frame=0)
# %%
for target_frame in [0, 250, 500]:
    print(target_frame)
    df = df1[df1["frame_number"] == target_frame]
    fig, ax = feature_density(df, "pc1", bounds[0])

    df = df2[df2["frame_number"] == target_frame]
    fig, ax = feature_density(df, "pc1", bounds[0])
# %%
for target_frame in [0, 250, 500]:
    print(target_frame)
    df = df1[df1["frame_number"] == target_frame]
    fig, ax = feature_density(df, "pc2", bounds[1])

    df = df2[df2["frame_number"] == target_frame]
    fig, ax = feature_density(df, "pc2", bounds[1])
# %%
for target_frame in [0, 250, 500]:
    print(target_frame)
    df = df1[df1["frame_number"] == target_frame]
    fig, ax = feature_density(df, "pc3", bounds[2])

    df = df2[df2["frame_number"] == target_frame]
    fig, ax = feature_density(df, "pc3", bounds[2])
# %%
for target_frame in [0, 250, 500]:
    print(f"Comparing frame: {target_frame}")

    # Filter rows for the current frame
    df3_frame = df3[df3["frame_number"] == target_frame]
    df2_frame = df2[df2["frame_number"] == target_frame]

    # Find rows in df1 but not in df2
    diff_df3 = df3_frame[~df3_frame.isin(df2_frame.to_dict(orient="list")).all(axis=1)]
    print(f"Rows in df1 but not in df2 for frame {target_frame}:")
    print(diff_df1)

    # Find rows in df2 but not in df1
    diff_df2 = df2_frame[~df2_frame.isin(df3_frame.to_dict(orient="list")).all(axis=1)]
    print(f"Rows in df2 but not in df1 for frame {target_frame}:")
    print(diff_df2)
# %%
