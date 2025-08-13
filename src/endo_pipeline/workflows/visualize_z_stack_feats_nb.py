# %%
from typing import cast

from src.endo_pipeline.configs import CytoDLModelConfig, get_model_manifest, load_model_config
from src.endo_pipeline.io.output import get_output_path, save_plot_to_path
from src.endo_pipeline.library.analyze.diffae_manifest import (
    fit_pca,
    get_manifest_for_dynamics_workflows,
)
from src.endo_pipeline.library.analyze.numerics import get_3d_bounds_from_data
from src.endo_pipeline.library.analyze.z_slice_feats.compare_feats import (
    feature_density,
    plot_scatter_by_position_and_frame,
)

# %%
TIMEPOINTS = [0, 250, 500]

# %%
model_config = cast(CytoDLModelConfig, load_model_config("diffae_04_10"))
dataset_list = ["20241016_20X", "20250331_20X", "20250402_20X", "20250409_20X"]

pca = fit_pca()

for dataset_name in dataset_list:
    save_dir = get_output_path("visualize_z_stack_feats", model_config.name, dataset_name)

    model_manifest1 = get_model_manifest(dataset_name, model_config)
    model_manifest2 = get_model_manifest(dataset_name, model_config, [5, 10])
    model_manifest3 = get_model_manifest(dataset_name, model_config, [0, 16])
    model_manifest4 = get_model_manifest(dataset_name, model_config, [9, 24])

    df1 = get_manifest_for_dynamics_workflows(model_manifest1, pca, filter_to_valid=False)
    df1 = df1[df1["frame_number"].isin(TIMEPOINTS)].reset_index(drop=True)
    df2 = get_manifest_for_dynamics_workflows(model_manifest2, pca, filter_to_valid=False)
    df3 = get_manifest_for_dynamics_workflows(model_manifest3, pca, filter_to_valid=False)
    df4 = get_manifest_for_dynamics_workflows(model_manifest4, pca, filter_to_valid=False)

    assert df1.shape == df2.shape == df3.shape == df4.shape, "DataFrames have different shapes!"

    manifest_list = [model_manifest1, model_manifest2, model_manifest3, model_manifest4]
    df_list = [df1, df2, df3, df4]
    df_info = [
        "all slices",
        "centered slices (-5, +10)",
        "bottom slices (0, 16)",
        "top slices (9, 24)",
    ]
    bounds = get_3d_bounds_from_data(manifest_list, pca)

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
