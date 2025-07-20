import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.endo_pipeline.configs import get_model_manifest, load_dataset_config, load_model_config
from src.endo_pipeline.configs.dataset_io import (
    get_live_segmentation_features_manifest,
    get_reference_datasets,
    ipython_cli_flexecute,
)
from src.endo_pipeline.configs.dynamics_io import load_dynamics_config
from src.endo_pipeline.io import get_output_path, load_dataframe_from_fms
from src.endo_pipeline.library.analyze.diffae_features import regression_helper as rh
from src.endo_pipeline.library.analyze.diffae_manifest import preprocessing as diffae_preproc
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import (
    get_manifest_for_dynamics_workflows,
    project_manifest_to_pcs,
)
from src.endo_pipeline.library.analyze.numerics import data_driven_flow_field as ddff
from src.endo_pipeline.library.visualize.diffae_features.flow_field_viz import (
    get_slice_indexes,
    plot_quiver_slices,
    set_slice_plot_bounds_and_labels,
)
from src.endo_pipeline.workflows.feats_diffae_classic_comparison import (  # get_traj_and_flowfield_from_manifest,
    get_traj_and_flowfield,
)

logger = logging.getLogger(__name__)


def merge_diffae_feats_liveseg_feats_tables(
    diffae_tracking_df: pd.DataFrame,
    live_seg_feats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges the DiffAE tracking data with the live segmentation features data.

    Parameters:
        diffae_tracking_df (pd.DataFrame): DataFrame containing DiffAE tracking data.
        live_seg_feats_df (pd.DataFrame): DataFrame containing live segmentation features data.

    Returns:
        pd.DataFrame: Merged DataFrame with DiffAE and live segmentation features.
    """
    logging.debug("processing the diffae tracking data...")
    # process the diffae tracking data
    diffae_tracking_df["is_unique"] = diffae_tracking_df.groupby(
        ["dataset", "position", "frame_number", "track_id"]
    )["frame_number"].transform(lambda t: t.nunique() == t.size)
    diffae_tracking_df = diffae_tracking_df[diffae_tracking_df["is_unique"]]

    # give the crop_index column the same value as the track_ids
    diffae_tracking_df["crop_index"] = diffae_tracking_df["track_id"]
    diffae_tracking_df = diffae_preproc.add_description_column(
        diffae_tracking_df, dataset_name, simple=True
    )  # add description column (e.g., 48hr_High)
    diffae_tracking_df["track_id"] = diffae_tracking_df["track_id"].astype(int)
    diffae_tracking_df.rename(columns={"position": "position_as_str"}, inplace=True)

    logging.debug("processing the live segmentation features data...")
    live_seg_feats_df["position_as_str"] = live_seg_feats_df["position"].transform(
        lambda x: "P" + str(x)
    )
    live_seg_feats_df["track_id"] = live_seg_feats_df["track_id"].astype(int)

    logging.debug("merging segmentation properties and track-based DiffAE data...")
    merged_feats_df = pd.merge(
        left=live_seg_feats_df,
        right=diffae_tracking_df,
        how="left",
        left_on=["dataset_name", "position_as_str", "image_index", "track_id"],
        right_on=["dataset", "position_as_str", "frame_number", "track_id"],
        validate="one_to_one",
    )

    return merged_feats_df


def get_diffae_feats_liveseg_feats_merged_table(dataset_name: str) -> pd.DataFrame:

    logging.debug(f"Loading dataset config file for dataset: {dataset_name}...")
    dataset_config = load_dataset_config(dataset_name)

    # read in the segmentation-based diffae features if available
    logging.debug("loading diffae features from tracking data...")
    diffae_fms_id = dataset_config.diffae_tracking_integration_fmsid
    if diffae_fms_id is None:
        logging.warning(
            f"No DiffAE track integration FMS ID for {dataset_name}. Returning empty dataframe."
        )
        return pd.DataFrame()
    diffae_tracking_df = load_dataframe_from_fms(diffae_fms_id)

    # load the tracking data of the measured features and merge them
    logging.debug("loading segmentation property data...")
    live_seg_fmsid = dataset_config.live_merged_seg_features_manifest_fmsid
    if live_seg_fmsid is None:
        logging.warning(
            f"No live segmentation features FMS ID for {dataset_name}. Returning empty dataframe."
        )
        return pd.DataFrame()
    live_seg_feats_df = load_dataframe_from_fms(live_seg_fmsid)  # this takes a minute

    # merge the two tables
    merged_feats_df = merge_diffae_feats_liveseg_feats_tables(diffae_tracking_df, live_seg_feats_df)

    return merged_feats_df


dataset_name = "20241120_20X"

## NOTE CODE FOR DEV ONLY
live_seg_feats_df = pd.read_csv(
    r"C:\Users\serge.parent\Documents\projects\cellsmap\results\2025-07-16\make_seg_feats_manifest\segmentation_features_manifests\20241120_20X_live_segmentation_features.tsv",
    sep="\t",
)
diffae_tracking_df = pd.read_parquet(
    r"C:\Users\serge.parent\Documents\projects\cellsmap\results\models\diffae_04_10\20241120_20X\predict_20241120_20X_diffae_04_10_tracked_crop_features.parquet"
)

merged_feats_df = merge_diffae_feats_liveseg_feats_tables(diffae_tracking_df, live_seg_feats_df)
## NOTE END OF DEV CODE


# load the tables
merged_feats_df = get_diffae_feats_liveseg_feats_merged_table(dataset_name)

# filter the merged table
merged_feats_df = merged_feats_df[~merged_feats_df["filter_global"]]

# remove any rows that were not evaluated by the model and thus have no mlflow_id
merged_feats_df.dropna(axis="index", how="any", subset="mlflow_id", inplace=True)

# fit the PCA (uses the reference datasets)
pca = fit_pca()

# read in the grid crop-based diffae features
model_name = diffae_tracking_df["model_name"].unique()[0]
model_config = load_model_config(model_name)
model_manifest = get_model_manifest(dataset_name, model_config)
diffae_grid_crops = get_manifest_for_dynamics_workflows(model_manifest, pca)

# add the PC columns to the track-based DiffAE table
# (the grid-based DiffAE table already has them, but
# but I believe that the columns are named "feat_0",
# "feat_1", etc. when they should be named "pc1",
# "pc2", etc.)
merged_feats_df = project_manifest_to_pcs(merged_feats_df, pca)
# df_all_positions.dropna(axis="index", how="any", subset="is_unique", inplace=True)

# use the full set of datasets to be analyzed for the bounds
bounds = ddff.set_3d_bounds_from_data([model_manifest], pca)

logger.debug("getting trajectory and flow field for grid-based crops...")
traj_grids, flow_field_dict_grids = get_traj_and_flowfield(diffae_grid_crops, bounds)

logger.debug("getting trajectory and flow field for tracks-based crops...")
traj_tracks, _ = get_traj_and_flowfield(merged_feats_df, bounds)

logger.debug("saving the trajectory data from the track-based crops...")
out_subdir_traj = get_output_path(Path(__file__).stem, dataset_name)
np.save(out_subdir_traj / f"{dataset_name}_traj_tracks.npy", traj_tracks)
