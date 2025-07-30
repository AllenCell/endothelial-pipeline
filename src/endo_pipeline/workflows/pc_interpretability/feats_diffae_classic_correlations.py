from pathlib import Path

# for data exploration; remove later
import seaborn as sns
from matplotlib import pyplot as plt

from src.endo_pipeline.configs import load_dataset_collection_config
from src.endo_pipeline.io import configure_logging, get_output_path
from src.endo_pipeline.library.analyze.integration.track_integration import (  # get_approx_point_from_grid,; get_approx_vec_from_grid,; get_gridcrop_and_cellcentric_trajectories_and_flow_fields,; get_vector_angles_as_grid,; get_vector_dot_products_as_grid,; get_vector_vector_angle_fast,; make_angular_deviation_test,
    get_preprocessed_manifests_and_km_bounds,
)

if __name__ == "__main__":

    dataset_name_list = load_dataset_collection_config("pca_reference").datasets
    dataset_name = dataset_name_list[0]  # for testing purposes
    out_subdir = get_output_path(Path(__file__).stem, dataset_name, include_timestamp=False)

    # load and preprocess the different diffae manifests and PCA pipeline
    # NOTE: this takes a little over a minute to load; we can consider
    # using dask dataframes and only computing the desired columns
    merged_feats_df, diffae_grid_crops, bounds = get_preprocessed_manifests_and_km_bounds(
        dataset_name, datasets_for_bounds=dataset_name_list
    )

    # # keep only the columns that are needed for the analysis to reduce memory usage
    # cols_to_keep = [
    #     "dataset_name",
    #     "position",
    #     "position_as_str",
    #     "track_id",
    #     "label",
    #     "crop_index",
    #     "mlflow_id",
    #     "model_name",
    #     "image_index",
    #     "frame_number",
    #     "time_hours",
    #     "time_minutes",
    #     "track_duration",
    # ] + [col for col in merged_feats_df.columns if "feat" in col or "pc" in col]
    # merged_feats_df = merged_feats_df[cols_to_keep]

    for i in range(1, 6):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"PC{i} vs. Alignment to Flow")
        sns.lineplot(data=merged_feats_df, x="time_hours", y=f"pc{i}")
        ax2 = ax.twinx()
        sns.lineplot(
            data=merged_feats_df,
            x="time_hours",
            y="alignment_deg_rel_to_flow",
            ax=ax2,
            c="tab:orange",
        )
        plt.show()
