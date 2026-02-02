import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib import pyplot as plt
from tqdm import tqdm

from endo_pipeline.configs import (
    get_datasets_in_collection,
    get_subset_of_timepoint_annotations,
    load_dataset_config,
)
from endo_pipeline.io import get_output_path, save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import filter_dataframe_by_annotations
from endo_pipeline.library.analyze.integration.track_integration import (
    load_pc_diffae_liveseg_feats_merged_table,
)
from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.polar_coords import THETA_RESCALED_PERIOD


def vector_mean_angle_and_mag(angles: np.ndarray) -> tuple[float, float]:
    """From a distribution of angles get the vector mean and return its angle and magnitude.
    Input angles must be in radians.
    The returned angle is in the range of [-pi, pi] and the magnitude is in the range of [0, 1].
    """
    # test line below
    # angles = np.deg2rad(np.random.randint(low=0, high=180, size=100000))
    xs = np.cos(angles)
    ys = np.sin(angles)

    # the x and y components of the vector mean:
    x_mean = xs.mean()
    y_mean = ys.mean()

    vector_mean_angle = float(np.arctan2(y_mean, x_mean))
    vector_mean_mag = float(np.linalg.norm([x_mean, y_mean]))

    return (vector_mean_angle, vector_mean_mag)


def generate_test_angles() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = int(1e5)
    # random distribution in the range of [0,180] degrees.
    angles_half_rand = np.deg2rad(np.random.randint(low=0, high=180, size=n_samples))

    # random distribution in the range of [0,360] degrees.
    angles_full_rand = np.deg2rad(np.random.randint(low=0, high=360, size=n_samples))

    # all angles are 45 degrees.
    angles_45 = np.deg2rad(np.linspace(45, 45, n_samples))

    # half of angles are 135 degrees, other half are 315 degrees.
    angles_135 = np.deg2rad(np.linspace(135, 135, n_samples // 2))
    angles_315 = np.deg2rad(np.linspace(315, 315, n_samples // 2))
    angles_bimodal = np.concatenate([angles_135, angles_315])

    return (angles_full_rand, angles_half_rand, angles_45, angles_bimodal)


outdir = get_output_path(__file__)

datasets = get_datasets_in_collection("live_cdh5_seg_based_feat_datasets")


dataset_info_cols = [
    ColumnName.DATASET.value,
    ColumnName.POSITION.value,
    ColumnName.TIMEPOINT.value,
]
density_cols = [
    "num_unique_tracks_before_filtering_at_T",
    "num_unique_tracks_after_filtering_at_T",
    "num_nuclei_in_crop",
    "total_nuclei_count_at_T",
]
filter_cols = ["is_included"]
feature_cols = ["alignment_rel_to_flow", "orientation", "polar_theta"]
other_cols = ["track_id"]

cols_to_compute = dataset_info_cols + density_cols + filter_cols + feature_cols + other_cols


summary_df = pd.DataFrame(columns=cols_to_compute)
df_subset = pd.DataFrame(columns=cols_to_compute)
for dataset_name in tqdm(datasets):
    config = load_dataset_config(dataset_name)
    if len(config.flow_conditions) != 1:
        print(f"Dataset {dataset_name} is not monoflow, skipping.")
        continue
    else:
        shear_stress = config.flow_conditions[0].shear_stress

    df_delayed = load_pc_diffae_liveseg_feats_merged_table(dataset_name)

    df = df_delayed[cols_to_compute].compute()
    df.dataset = dataset_name
    annotations_to_ignore: list = []
    timepoint_annotations = get_subset_of_timepoint_annotations(annotations_to_ignore)
    df_filtered = filter_dataframe_by_annotations(
        df,
        config,
        timepoint_annotations=timepoint_annotations,
    )
    df_filtered = df_filtered[df_filtered.is_included]

    # the original orientation feature is in radians
    # and the y-axis is defined as 0 degrees
    # this keeps the orientation angle range between 0-180 degrees
    df_filtered["orientation"] += np.pi / 2
    # df_filtered["orientation_deg"] = np.rad2deg(df_filtered["orientation"] + np.pi / 2)
    df_filtered["shear_stress"] = shear_stress

    df_filtered["orientation"] = df_filtered.groupby([ColumnName.POSITION, "track_id"])[
        "orientation"
    ].transform(lambda x: np.unwrap(x, period=THETA_RESCALED_PERIOD))
    # ].transform(
    #     lambda x: unwrap_nonsequential_array(x, period=THETA_RESCALED_PERIOD, reference_angle=0)
    # )

    df_filtered["polar_theta"] = df_filtered.groupby([ColumnName.POSITION, "track_id"])[
        "polar_theta"
    ].transform(lambda x: np.unwrap(x, period=THETA_RESCALED_PERIOD))
    # ].transform(
    #     lambda x: unwrap_nonsequential_array(x, period=THETA_RESCALED_PERIOD, reference_angle=0)
    # )

    df_subset = pd.concat([df_subset, df_filtered[df_filtered.is_included]])

    df = df.dropna(subset="total_nuclei_count_at_T")
    first_t = df[ColumnName.TIMEPOINT].min()
    df_first_t = df[df[ColumnName.TIMEPOINT] == first_t]

    groups = df_first_t.groupby(dataset_info_cols)

    for (ds, pos, tp), df_grp in groups:
        pos_num_nuclei = sequence_to_scalar(df_grp["total_nuclei_count_at_T"])
        pos_nuc_density = df_grp.num_nuclei_in_crop.mean()
        pos_num_seg_unfilt = sequence_to_scalar(df_grp["num_unique_tracks_before_filtering_at_T"])
        pos_num_seg_filt = sequence_to_scalar(df_grp["num_unique_tracks_after_filtering_at_T"])

        df_features = df_filtered[df_filtered[ColumnName.POSITION] == pos]
        vector_means: dict = {}
        vector_means_multipos: dict = {}
        for feature in ["orientation", "polar_theta"]:
            vec_mean_ang, vec_mean_mag = vector_mean_angle_and_mag(
                df_features[feature].dropna() * 2
            )
            vector_means[f"{feature}_vec_mean_angle"] = vec_mean_ang / 2
            vector_means[f"{feature}_vec_mean_magnitude"] = vec_mean_mag

            vec_mean_ang_multipos, vec_mean_mag_multipos = vector_mean_angle_and_mag(
                df_filtered[feature].dropna()
            )
            vector_means_multipos[f"{feature}_vec_mean_multipos_angle"] = vec_mean_ang_multipos
            vector_means_multipos[f"{feature}_vec_mean_multipos_magnitude"] = vec_mean_mag_multipos

        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame(
                    {
                        ColumnName.DATASET.value: [ds],
                        ColumnName.POSITION.value: [pos],
                        ColumnName.TIMEPOINT.value: [tp],
                        "shear_stress": [shear_stress],
                        "total_nuclei_count_at_T": [pos_num_nuclei],
                        "num_unique_tracks_before_filtering_at_T": [pos_num_seg_unfilt],
                        "num_unique_tracks_after_filtering_at_T": [pos_num_seg_filt],
                        "num_nuclei_in_crop": [pos_nuc_density],
                        **vector_means,
                        **vector_means_multipos,
                    }
                ),
            ],
            ignore_index=True,
        )

for dens_col in density_cols:
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=summary_df,
        x="shear_stress",
        y=dens_col,
        hue=ColumnName.DATASET,
        marker="o",
        # s=20,
        ax=ax,
        legend=False,
    )
    ax.set_xlabel("Shear Stress (dyn/cm²)")
    # ax.set_ylabel("Cell Density (nuclei/FOV)")
    plt.show()


fig, ax = plt.subplots(figsize=(3, 3))
sns.scatterplot(
    data=summary_df,
    x="shear_stress",
    y="total_nuclei_count_at_T",
    hue="polar_theta_vec_mean_magnitude",
    marker="o",
    # s=20,
    ax=ax,
    legend=False,
)
ax.set_xlabel("Shear Stress (dyn/cm²)")
ax.set_ylabel("Initial Cell Density (nuclei/FOV)")


fig, ax = plt.subplots(figsize=(3, 3))
sns.scatterplot(
    data=summary_df,
    x="shear_stress",
    y="total_nuclei_count_at_T",
    hue="alignment_rel_to_flow_vec_mean_magnitude",
    marker="o",
    # palette='Spectral',
    # s=20,
    ax=ax,
    legend=False,
)
ax.set_xlabel("Shear Stress (dyn/cm²)")
ax.set_ylabel("Initial Cell Density (nuclei/FOV)")
# ax.colorba
# r(label="Alignment Magnitude")


fig, ax = plt.subplots(figsize=(3, 3))
sns.scatterplot(
    data=summary_df,
    x="shear_stress",
    y="total_nuclei_count_at_T",
    hue="orientation_vec_mean_magnitude",
    marker="o",
    # palette='Spectral',
    # s=20,
    ax=ax,
    legend=False,
)
ax.set_xlabel("Shear Stress (dyn/cm²)")
ax.set_ylabel("Initial Cell Density (nuclei/FOV)")
# ax.colorbar(label="Alignment Magnitude")

# random distribution in the range of [0,180] degrees.
np.random.seed(0)
angles_half_rand = np.deg2rad(np.random.randint(low=0, high=180, size=int(1e6)))
vec_mean_angle_expect, vec_mean_mag_expect = vector_mean_angle_and_mag(angles_half_rand)

# angles_quarter_rand = np.deg2rad(np.random.randint(low=0, high=90, size=int(1e6)))
# vec_mean_angle_expect, vec_mean_mag_expect = vector_mean_angle_and_mag(angles_quarter_rand)


d = summary_df.groupby(ColumnName.DATASET)["dataset"].apply(lambda x: x.unique()[0])
x = summary_df.groupby(ColumnName.DATASET)["total_nuclei_count_at_T"].sum()
y = summary_df.groupby(ColumnName.DATASET)["shear_stress"].apply(lambda x: float(x.unique()))
c1 = summary_df.groupby(ColumnName.DATASET)["polar_theta_vec_mean_multipos_magnitude"].apply(
    lambda x: float(x.unique())
)

c3 = summary_df.groupby(ColumnName.DATASET)[
    "alignment_rel_to_flow_vec_mean_multipos_magnitude"
].apply(lambda x: float(x.unique()))
c4 = summary_df.groupby(ColumnName.DATASET)["orientation_vec_mean_multipos_magnitude"].apply(
    lambda x: float(x.unique())
)
c5 = df_subset.groupby(ColumnName.DATASET)["alignment_rel_to_flow"].std()
summary_df_multipos = pd.DataFrame(
    {
        "dataset": d.values,
        "total_nuclei_count_at_T": x.values,
        "shear_stress": y.values,
        "polar_theta_vec_mean_multipos_magnitude": c1.values,
        "alignment_rel_to_flow_vec_mean_multipos_magnitude": c3.values,
        "orientation_vec_mean_multipos_magnitude": c4.values,
        "alignment_rel_to_flow_std": c5.values,
        "polar_theta_vec_mean_multipos_magnitude_dist_from_expect": abs(
            c1.values - vec_mean_mag_expect
        ),
        "orientation_vec_mean_multipos_magnitude_dist_from_expect": abs(
            c4.values - vec_mean_mag_expect
        ),
    }
)

hue_norm = colors.TwoSlopeNorm(vmin=0, vcenter=vec_mean_mag_expect, vmax=1)
# cmap = "Spectral"
cmap = "seismic"
sm = plt.cm.ScalarMappable(cmap=cmap, norm=hue_norm)
# cmap = "viridis"

fig, ax = plt.subplots(figsize=(3, 3))
sns.scatterplot(
    data=summary_df_multipos,
    # data=summary_df,
    x="shear_stress",
    y="total_nuclei_count_at_T",
    # hue="orientation_vec_mean_multipos_magnitude_dist_from_expect",
    hue="orientation_vec_mean_multipos_magnitude",
    # hue="orientation_vec_mean_magnitude",
    marker="o",
    edgecolor="black",
    hue_norm=hue_norm,
    palette=cmap,
    s=20,
    ax=ax,
    legend=False,
)
ax.set_title("Orientation Vector Mean Magnitude")
ax.set_xlabel("Shear Stress (dyn/cm²)")
ax.set_ylabel("Initial Cell Density (nuclei/FOV)")
cbar = ax.figure.colorbar(sm, ax=ax)
cbar.set_ticks(cbar.get_ticks().tolist() + [vec_mean_mag_expect])
plt.show()

fig, ax = plt.subplots(figsize=(3, 3))
sns.scatterplot(
    data=summary_df_multipos,
    # data=summary_df,
    x="shear_stress",
    y="total_nuclei_count_at_T",
    # hue="polar_theta_vec_mean_multipos_magnitude_dist_from_expect",
    hue="polar_theta_vec_mean_multipos_magnitude",
    # hue="polar_theta_vec_mean_magnitude",
    marker="o",
    edgecolor="black",
    hue_norm=hue_norm,
    palette=cmap,
    s=20,
    ax=ax,
    legend=False,
)
ax.set_title(f"{get_label_for_column('polar_theta').capitalize()} Vector Mean Magnitude")
ax.set_xlabel("Shear Stress (dyn/cm²)")
ax.set_ylabel("Initial Cell Density (nuclei/FOV)")
cbar = ax.figure.colorbar(sm, ax=ax)
cbar.set_ticks(cbar.get_ticks().tolist() + [vec_mean_mag_expect])
plt.show()

test_angles = generate_test_angles()

for angles in test_angles:
    vec_mean_ang, vec_mean_mag = vector_mean_angle_and_mag(angles)
    print(np.rad2deg(vec_mean_ang), vec_mean_mag)


for nm, df_grp in df_subset.groupby(ColumnName.DATASET):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    sns.histplot(
        data=df_grp,
        x="polar_theta",
        stat="probability",
        binwidth=np.deg2rad(5),
        color="tab:blue",
        alpha=0.5,
        lw=0.2,
        ax=ax,
    )
    sns.histplot(
        data=df_grp,
        x="orientation",
        stat="probability",
        binwidth=np.deg2rad(5),
        color="tab:orange",
        alpha=0.5,
        lw=0.2,
        ax=ax,
    )
    ax.set_title(
        f"{nm}, {sequence_to_scalar(df_grp.shear_stress.unique())} dyn/cm²",
    )
    ax.set_xlim(0, np.pi)
    ax.yaxis.set_tick_params(rotation=45)
    fig.legend(
        labels=[get_label_for_column("polar_theta"), get_label_for_column("orientation")],
        loc="lower center",
        ncols=2,
    )
    plt.tight_layout()
    save_plot_to_path(fig, outdir, f"{nm}_polar_histograms")

    # break


# for nm, df in df_subset.groupby([ColumnName.DATASET, ColumnName.POSITION, "track_id"]):
#     df["orientation"]
#     break

# dataset_mixed = "20250813_20X"
# dataset_low = "20250618_20X"
# dataset_no = "20250818_20X"
# dataset_high = "20251001_20X"
# dataset_high2 = "20250611_20X"


# df_mixed = df_subset[df_subset.dataset == dataset_mixed]
# df_low = df_subset[df_subset.dataset == dataset_low]
# df_no = df_subset[df_subset.dataset == dataset_no]
# df_high = df_subset[df_subset.dataset == dataset_high]
# df_high2 = df_subset[df_subset.dataset == dataset_high]

# vector_mean_angle_and_mag(df_low["alignment_rel_to_flow"])
# vector_mean_angle_and_mag(df_no["alignment_rel_to_flow"])
# vector_mean_angle_and_mag(df_high["alignment_rel_to_flow"])
# vector_mean_angle_and_mag(df_high2["alignment_rel_to_flow"])
# vector_mean_angle_and_mag(df_mixed["alignment_rel_to_flow"])

# vector_mean_angle_and_mag(df_low["orientation"])
# vector_mean_angle_and_mag(df_no["orientation"])
# vector_mean_angle_and_mag(df_high["orientation"])
# vector_mean_angle_and_mag(df_high2["orientation"])
# vector_mean_angle_and_mag(df_mixed["orientation"])

# for nm, df_grp in [
#     (dataset_low, df_low),
#     (dataset_high, df_high),
#     (dataset_high2, df_high2),
#     (dataset_no, df_no),
#     (dataset_mixed, df_mixed),
# ]:
#     fig, ax = plt.subplots()
#     sns.histplot(data=df_grp, x="alignment_rel_to_flow", bins=30, stat="density")
#     ax.set_title(
#         f"{nm}, {sequence_to_scalar(df_grp.shear_stress.unique())} dyn/cm²",
#     )
#     plt.show()


# angles = df_no["alignment_rel_to_flow"]


# # # fig, ax = plt.subplots(figsize=(3, 3))
# # # sns.scatterplot(
# # #     data=summary_df_multipos,
# # #     x="shear_stress",
# # #     y="total_nuclei_count_at_T",
# # #     hue="alignment_rel_to_flow_std",
# # #     marker="o",
# # #     # hue_norm=plt.Normalize(0, 1),
# # #     palette="viridis",
# # #     s=20,
# # #     ax=ax,
# # #     legend=False,
# # # )
# # # ax.set_xlabel("Shear Stress (dyn/cm²)")
# # # ax.set_ylabel("Initial Cell Density (nuclei/FOV)")

# # for nm, df in df_subset.groupby(ColumnName.DATASET):
# #     print(nm, len(df))

# # test_df = (
# #     df_subset.groupby([ColumnName.DATASET, "shear_stress"])[other_cols]
# #     .apply(lambda x: x.std() / len(x))
# #     .reset_index()
# # )
# # test_df = test_df.merge(summary_df_multipos, on=[ColumnName.DATASET, "shear_stress"])

# # fig, ax = plt.subplots(figsize=(3, 3))
# # sns.scatterplot(
# #     data=test_df,
# #     x="shear_stress",
# #     y="total_nuclei_count_at_T",
# #     hue="alignment_rel_to_flow",
# #     marker="o",
# #     # hue_norm=plt.Normalize(0, 1),
# #     palette="viridis",
# #     s=20,
# #     ax=ax,
# #     legend=False,
# # )
# # ax.set_xlabel("Shear Stress (dyn/cm²)")
# # ax.set_ylabel("Initial Cell Density (nuclei/FOV)")
