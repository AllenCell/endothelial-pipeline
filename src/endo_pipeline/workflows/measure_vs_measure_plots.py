import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    calculate_derived_data_dynamics_dependent,
)
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings import DEFAULT_SEG_FEATURE_MANIFEST_NAME

# dataset_name = "20250618_20X"
dataset_name = "20250818_20X"
seg_feature_manifest_name = DEFAULT_SEG_FEATURE_MANIFEST_NAME
live_seg_manifest = load_dataframe_manifest(seg_feature_manifest_name)
live_seg_location = get_dataframe_location_for_dataset(live_seg_manifest, dataset_name)
live_seg_feats_df = load_dataframe(live_seg_location)
live_seg_feats_df = live_seg_feats_df.query("is_included")
live_seg_feats_df = calculate_derived_data_dynamics_dependent(live_seg_feats_df)

live_seg_feats_df["orientation_deg"] = np.rad2deg(live_seg_feats_df["orientation"] + np.pi / 2)

# sns.histplot(data=live_seg_feats_df, x="orientation_deg", y="centroid_velocity_angle_deg")


live_seg_feats_df["centroid_velocity_angle_deg_shifted"] = (
    live_seg_feats_df[live_seg_feats_df["centroid_velocity_angle_deg"] >= 0][
        "centroid_velocity_angle_deg"
    ]
    + live_seg_feats_df[live_seg_feats_df["centroid_velocity_angle_deg"] < 0][
        "centroid_velocity_angle_deg"
    ]
    * -1
)

live_seg_feats_df["centroid_velocity_angle_deg_shifted"] = live_seg_feats_df[
    "centroid_velocity_angle_deg"
].transform(lambda x: x if x >= 0 else x + 360)


fig, ax = plt.subplots()
sns.histplot(
    data=live_seg_feats_df, x="nuc_pos_rel_cell_angle_deg", y="centroid_velocity_angle_deg"
)

fig, ax = plt.subplots()
sns.histplot(
    data=live_seg_feats_df, x="nuc_pos_rel_cell_angle_deg", y="centroid_velocity_angle_deg_shifted"
)

live_seg_feats_df[["nuc_pos_rel_cell_angle_deg", "centroid_velocity_angle_deg_shifted"]].corr(
    "pearson"
)

sns.histplot(
    data=live_seg_feats_df,
    x="alignment_deg_rel_to_flow",
    y="dalignment_dt_deg_rel_to_flow",
)

fig, ax = plt.subplots()
sns.histplot(
    x=live_seg_feats_df["orientation_deg"],
    y=live_seg_feats_df.groupby(["position", "track_id"])["orientation_deg"].transform(
        lambda x: np.diff(x, prepend=np.nan)
    ),
    ax=ax,
)
ax.set_ylim(-25, 25)

# sns.histplot(
#     x=live_seg_feats_df["alignment_deg_rel_to_flow"],
#     y=live_seg_feats_df.groupby(["position", "track_id"])["alignment_deg_rel_to_flow"].transform(
#         lambda x: np.diff(x, prepend=np.nan)
#     ),
# )

sns.histplot(x=live_seg_feats_df["time_hours"], y=live_seg_feats_df["centroid_velocity_angle_deg"])

fig, ax = plt.subplots()
sns.histplot(
    x=np.cos(np.deg2rad(live_seg_feats_df["centroid_velocity_angle_deg"])),
    y=np.sin(np.deg2rad(live_seg_feats_df["centroid_velocity_angle_deg"])),
    ax=ax,
)
ax.set_aspect("equal")

fig, ax = plt.subplots()
sns.histplot(
    x=live_seg_feats_df["time_hours"],
    y=np.cos(np.deg2rad(live_seg_feats_df["centroid_velocity_angle_deg"])),
    ax=ax,
)

fig, ax = plt.subplots()
sns.histplot(
    x=live_seg_feats_df["time_hours"],
    y=np.sin(np.deg2rad(live_seg_feats_df["centroid_velocity_angle_deg"])),
    ax=ax,
)


fig, ax = plt.subplots()
sns.histplot(
    x=np.cos(live_seg_feats_df["nuc_pos_rel_cell_angle"]),
    y=np.cos(live_seg_feats_df["centroid_velocity_angle"]),
    binwidth=(np.deg2rad(1), np.deg2rad(1)),
    ax=ax,
)
ax.set_aspect("equal")
ax.set_xlabel("Nucleus Position Relative to Cell Angle Cosine")
ax.set_ylabel("Centroid Velocity Angle Cosine")

fig, ax = plt.subplots()
sns.histplot(
    x=np.sin(live_seg_feats_df["nuc_pos_rel_cell_angle"]),
    y=np.sin(live_seg_feats_df["centroid_velocity_angle"]),
    binwidth=(np.deg2rad(1), np.deg2rad(1)),
    ax=ax,
)
ax.set_aspect("equal")
ax.set_xlabel("Nucleus Position Relative to Cell Angle Sine")
ax.set_ylabel("Centroid Velocity Angle Sine")

y = np.sin(live_seg_feats_df[["nuc_pos_rel_cell_angle", "centroid_velocity_angle"]].dropna())
x = np.cos(live_seg_feats_df[["nuc_pos_rel_cell_angle", "centroid_velocity_angle"]].dropna())

y.corr()
x.corr()

sns.histplot(x=x.sum(axis=1), y=y.sum(axis=1))

sns.histplot(x=np.linalg.norm(x, axis=1), y=np.linalg.norm(y, axis=1))


sns.histplot(
    data=live_seg_feats_df,
    x="alignment_deg_rel_to_flow",
    y="aspect_ratio",
)

# I wonder what this plot below looks like for the intermediate SS condition...
fig, ax = plt.subplots()
sns.histplot(
    x=live_seg_feats_df["alignment_deg_rel_to_flow"],
    y=np.log10(1 - live_seg_feats_df["eccentricity"]),
    ax=ax,
)
ax.set_ylabel("log10(1 - eccentricity)")
