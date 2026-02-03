import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib import pyplot as plt
from tqdm import tqdm

from endo_pipeline.cli import Datasets
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


def vector_mean_angle_and_mag(angles: np.ndarray) -> tuple[float, float]:
    """From a distribution of angles get the vector mean and return its angle and magnitude.
    Input angles must be in radians.
    The returned angle is in the range of [-pi, pi] and the magnitude is in the range of [0, 1].
    """
    # test line below
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


def create_summary_df(datasets: Datasets, cols_to_compute) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        # df_filtered = df_filtered[df_filtered.is_included]

        # the original orientation feature is in radians
        # and the y-axis is defined as 0 degrees
        # this keeps the orientation angle range between 0-180 degrees
        df_filtered["orientation"] += np.pi / 2
        # df_filtered["orientation_deg"] = np.rad2deg(df_filtered["orientation"] + np.pi / 2)
        df_filtered["shear_stress"] = shear_stress

        df_subset = pd.concat([df_subset, df_filtered])

        df = df.dropna(subset="total_nuclei_count_at_T")
        first_t = df[ColumnName.TIMEPOINT].min()
        df_first_t = df[df[ColumnName.TIMEPOINT] == first_t]

        groups = df_first_t.groupby(dataset_info_cols)

        for (ds, pos, tp), df_grp in groups:
            pos_num_nuclei = sequence_to_scalar(df_grp["total_nuclei_count_at_T"])
            pos_nuc_density = df_grp.num_nuclei_in_crop.mean()
            pos_num_seg_unfilt = sequence_to_scalar(
                df_grp["num_unique_tracks_before_filtering_at_T"]
            )
            pos_num_seg_filt = sequence_to_scalar(df_grp["num_unique_tracks_after_filtering_at_T"])

            df_features = df_filtered[df_filtered[ColumnName.POSITION] == pos]
            vector_means: dict = {}
            vector_means_multipos: dict = {}
            for feature in ["orientation", "polar_theta"]:
                vec_mean_ang, vec_mean_mag = vector_mean_angle_and_mag(
                    df_features[feature].dropna()
                )
                vector_means[f"{feature}_vec_mean_angle"] = (vec_mean_ang + 2 * np.pi) / 2
                vector_means[f"{feature}_vec_mean_magnitude"] = vec_mean_mag

                vec_mean_ang_multipos, vec_mean_mag_multipos = vector_mean_angle_and_mag(
                    df_filtered[feature].dropna() * 2
                )
                vector_means_multipos[f"{feature}_vec_mean_multipos_angle"] = (
                    (vec_mean_ang_multipos + 2 * np.pi) % (2 * np.pi)
                ) / 2
                vector_means_multipos[f"{feature}_vec_mean_multipos_magnitude"] = (
                    vec_mean_mag_multipos
                )

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
    return summary_df, df_subset


def create_multipos_summary_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    df_grpd = summary_df.groupby(ColumnName.DATASET)

    polar_theta_vec_mean_mag = df_grpd["polar_theta_vec_mean_multipos_magnitude"].apply(
        lambda x: float(x.unique())
    )
    orientation_vec_mean_mag = df_grpd["orientation_vec_mean_multipos_magnitude"].apply(
        lambda x: float(x.unique())
    )
    polar_theta_vec_mean_ang = df_grpd["polar_theta_vec_mean_multipos_angle"].apply(
        lambda x: float(x.unique())
    )
    orientation_vec_mean_ang = df_grpd["orientation_vec_mean_multipos_angle"].apply(
        lambda x: float(x.unique())
    )
    summary_df_multipos = pd.DataFrame(
        {
            "dataset": df_grpd["dataset"].apply(lambda x: x.unique()[0]),
            "shear_stress_regime": df_grpd["shear_stress_regime"].apply(lambda x: x.unique()[0]),
            "total_nuclei_count_at_T": df_grpd["total_nuclei_count_at_T"].sum(),
            "shear_stress": df_grpd["shear_stress"].apply(lambda x: float(x.unique())),
            "polar_theta_vec_mean_multipos_magnitude": polar_theta_vec_mean_mag,
            "orientation_vec_mean_multipos_magnitude": orientation_vec_mean_mag,
            "polar_theta_vec_mean_multipos_angle": polar_theta_vec_mean_ang,
            "orientation_vec_mean_multipos_angle": orientation_vec_mean_ang,
        }
    )

    return summary_df_multipos


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
other_cols = ["track_id", "shear_stress_regime"]

cols_to_compute = dataset_info_cols + density_cols + filter_cols + feature_cols + other_cols

summary_df, df_subset = create_summary_df(datasets, cols_to_compute)
summary_df_multipos = create_multipos_summary_df(summary_df)


# def make_summary_plots(df: pd.DataFrame, x:str, y:str, hue:str|None, outdir: Path):

for dens_col in density_cols:
    out_subdir = outdir / dens_col
    out_subdir.mkdir(parents=True, exist_ok=True)

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
    hue="orientation_vec_mean_magnitude",
    marker="o",
    # palette='Spectral',
    # s=20,
    ax=ax,
    legend=False,
)
ax.set_xlabel("Shear Stress (dyn/cm²)")
ax.set_ylabel("Initial Cell Density (nuclei/FOV)")


cmap = "inferno"
hue_norm = colors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=hue_norm)

cmap_ang = "hsv"
hue_norm_angle = colors.Normalize(vmin=0, vmax=np.pi)
sm_angle = plt.cm.ScalarMappable(cmap=cmap_ang, norm=hue_norm_angle)


fig, ax = plt.subplots(figsize=(3, 3))
sns.scatterplot(
    data=summary_df_multipos,
    # data=summary_df,
    x="shear_stress",
    y="total_nuclei_count_at_T",
    hue="dataset",
    marker="o",
    edgecolor="black",
    hue_norm=hue_norm,
    palette=cmap,
    s=20,
    ax=ax,
    legend=False,
)
ax.set_title("")
ax.set_xlabel("Shear Stress (dyn/cm²)")
ax.set_ylabel("Initial Cell Density (nuclei/FOV)")
save_plot_to_path(fig, outdir, "orientation_vector_mean_magnitude_vs_init_density_vs_flow")


fig, ax = plt.subplots(figsize=(3, 3))
sns.scatterplot(
    data=summary_df_multipos,
    # data=summary_df,
    x="shear_stress",
    y="total_nuclei_count_at_T",
    hue="orientation_vec_mean_multipos_magnitude",
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
save_plot_to_path(fig, outdir, "orientation_vector_mean_magnitude_vs_init_density_vs_flow")


fig, ax = plt.subplots(figsize=(3, 3))
sns.scatterplot(
    data=summary_df_multipos,
    # data=summary_df,
    x="shear_stress",
    y="total_nuclei_count_at_T",
    hue="polar_theta_vec_mean_multipos_magnitude",
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
cbar = ax.figure.colorbar(sm, ax=ax, label="Vector Mean Magnitude")
save_plot_to_path(fig, outdir, "polar_theta_vector_mean_magnitude_vs_init_density_vs_flow")


fig, ax = plt.subplots(figsize=(3, 3))
sns.scatterplot(
    data=summary_df_multipos,
    # data=summary_df,
    x="shear_stress",
    y="total_nuclei_count_at_T",
    hue="orientation_vec_mean_multipos_angle",
    marker="o",
    edgecolor="black",
    hue_norm=hue_norm_angle,
    palette=cmap_ang,
    s=20,
    ax=ax,
    legend=False,
)
ax.set_title("Orientation Vector Mean Angle")
ax.set_xlabel("Shear Stress (dyn/cm²)")
ax.set_ylabel("Initial Cell Density (nuclei/FOV)")
cbar = ax.figure.colorbar(sm_angle, ax=ax)
cbar.set_ticks(np.linspace(0, np.pi, 7, endpoint=True), labels=range(0, 181, 30))
save_plot_to_path(fig, outdir, "orientation_vector_mean_angle_vs_init_density_vs_flow")


fig, ax = plt.subplots(figsize=(3, 3))
sns.scatterplot(
    data=summary_df_multipos,
    # data=summary_df,
    x="shear_stress",
    y="total_nuclei_count_at_T",
    hue="polar_theta_vec_mean_multipos_angle",
    marker="o",
    edgecolor="black",
    hue_norm=hue_norm_angle,
    palette=cmap_ang,
    s=20,
    ax=ax,
    legend=False,
)
ax.set_title(f"{get_label_for_column('polar_theta').capitalize()} Vector Mean Angle")
ax.set_xlabel("Shear Stress (dyn/cm²)")
ax.set_ylabel("Initial Cell Density (nuclei/FOV)")
cbar = ax.figure.colorbar(sm_angle, ax=ax)
cbar.set_ticks(np.linspace(0, np.pi, 7, endpoint=True), labels=range(0, 181, 30))
save_plot_to_path(fig, outdir, "polar_theta_vector_mean_angle_vs_init_density_vs_flow")


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

# random distribution in the range of [0,180] degrees.
# np.random.seed(0)
# angles_half_rand = np.random.random_sample(size=int(1e6)) * np.pi
# vec_mean_angle_expect, vec_mean_mag_expect = vector_mean_angle_and_mag(angles_half_rand * 2)
# vec_mean_angle_expect = (vec_mean_angle_expect + 2 * np.pi) % (2 * np.pi) / 2

test_angles = generate_test_angles()

for angles in test_angles:
    vec_mean_ang, vec_mean_mag = vector_mean_angle_and_mag(angles)
    print(np.rad2deg(vec_mean_ang), vec_mean_mag)
