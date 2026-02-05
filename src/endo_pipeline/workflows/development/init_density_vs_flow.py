from pathlib import Path

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
from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES, ColumnName


def test_vector_mean_angle_and_mag():
    test_angles = generate_test_angles()

    for angles in test_angles:
        vec_mean_ang, vec_mean_mag = vector_mean_angle_and_mag(angles)
        print(np.rad2deg(vec_mean_ang), vec_mean_mag)


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
            shear_stress_regime = config.shear_stress_regime[0].value

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
        df_filtered["shear_stress_regime"] = shear_stress_regime

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
            pc3 = df_grp[DIFFAE_PC_COLUMN_NAMES[2]].mean()

            df_features = df_filtered[df_filtered[ColumnName.POSITION] == pos]
            vector_means: dict = {}
            vector_means_multipos: dict = {}
            for feature in ["orientation", "polar_theta"]:
                vec_mean_ang, vec_mean_mag = vector_mean_angle_and_mag(
                    df_features[feature].dropna() * 2
                )
                vector_means[f"{feature}_vec_mean_angle"] = (
                    (vec_mean_ang + 2 * np.pi) % (2 * np.pi) / 2
                )
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
                            "shear_stress_regime": [shear_stress_regime],
                            "total_nuclei_count_at_T": [pos_num_nuclei],
                            "num_unique_tracks_before_filtering_at_T": [pos_num_seg_unfilt],
                            DIFFAE_PC_COLUMN_NAMES[2]: [pc3],
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
            "num_unique_tracks_before_filtering_at_T": df_grpd[
                "num_unique_tracks_before_filtering_at_T"
            ].sum(),
            DIFFAE_PC_COLUMN_NAMES[2]: df_grpd[DIFFAE_PC_COLUMN_NAMES[2]].mean(),
            "num_nuclei_in_crop": df_grpd["num_nuclei_in_crop"].sum(),
            "shear_stress": df_grpd["shear_stress"].apply(lambda x: float(x.unique())),
            "polar_theta_vec_mean_multipos_magnitude": polar_theta_vec_mean_mag,
            "orientation_vec_mean_multipos_magnitude": orientation_vec_mean_mag,
            "polar_theta_vec_mean_multipos_angle": polar_theta_vec_mean_ang,
            "orientation_vec_mean_multipos_angle": orientation_vec_mean_ang,
        }
    )

    return summary_df_multipos


def make_summary_plots(
    out_dir: Path,
    filename: str,
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None,
    hue_norm: colors.Normalize,
    cmap: str,
    cbar_scalarmap: plt.cm.ScalarMappable | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    legend: bool = False,
):
    x_label = x_label or get_label_for_column(x)
    y_label = y_label or get_label_for_column(y)

    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        marker="o",
        edgecolor="black",
        hue_norm=hue_norm,
        palette=cmap,
        s=20,
        ax=ax,
        legend=legend,
    )
    ax.set_title(hue.replace("_", " ").capitalize())
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if cbar_scalarmap is not None:
        cbar = ax.figure.colorbar(cbar_scalarmap, ax=ax)
        if "angle" in hue:
            cmin, cmax = sm_angle.get_clim()
            cbar.set_ticks(
                np.linspace(cmin, cmax, 5, endpoint=True),
                labels=np.linspace(np.rad2deg(cmin), np.rad2deg(cmax), 5, endpoint=True, dtype=int),
            )
    save_plot_to_path(fig, out_dir, filename)


outdir = get_output_path(__file__)

datasets = get_datasets_in_collection("live_cdh5_seg_based_feat_datasets")


dataset_info_cols = [
    ColumnName.DATASET.value,
    ColumnName.POSITION.value,
    ColumnName.TIMEPOINT.value,
]
density_cols = [
    "num_unique_tracks_before_filtering_at_T",
    DIFFAE_PC_COLUMN_NAMES[2],
    "num_nuclei_in_crop",
    "total_nuclei_count_at_T",
]
filter_cols = ["is_included"]
feature_cols = ["alignment_rel_to_flow", "orientation", "polar_theta"]
other_cols = ["track_id", "shear_stress_regime"]

cols_to_compute = dataset_info_cols + density_cols + filter_cols + feature_cols + other_cols

summary_df, df_subset = create_summary_df(datasets, cols_to_compute)
summary_df_multipos = create_multipos_summary_df(summary_df)


cmap_mag = "inferno"
hue_norm_mag = colors.Normalize(vmin=0, vmax=1)
sm_mag = plt.cm.ScalarMappable(cmap=cmap_mag, norm=hue_norm_mag)

cmap_ang = "hsv"
hue_norm_angle = colors.Normalize(vmin=0, vmax=np.pi)
sm_angle = plt.cm.ScalarMappable(cmap=cmap_ang, norm=hue_norm_angle)

hue_groups_multiposition = [
    (ColumnName.DATASET.value, "tab20", None, None, True),
    ("shear_stress_regime", "tab20", None, None, True),
    ("orientation_vec_mean_multipos_magnitude", cmap_mag, hue_norm_mag, sm_mag, False),
    ("polar_theta_vec_mean_multipos_magnitude", cmap_mag, hue_norm_mag, sm_mag, False),
    ("orientation_vec_mean_multipos_angle", cmap_ang, hue_norm_angle, sm_angle, False),
    ("polar_theta_vec_mean_multipos_angle", cmap_ang, hue_norm_angle, sm_angle, False),
]

hue_groups_single_position = [
    ("orientation_vec_mean_magnitude", cmap_mag, hue_norm_mag, sm_mag, False),
    ("polar_theta_vec_mean_magnitude", cmap_mag, hue_norm_mag, sm_mag, False),
    ("orientation_vec_mean_angle", cmap_ang, hue_norm_angle, sm_angle, False),
    ("polar_theta_vec_mean_angle", cmap_ang, hue_norm_angle, sm_angle, False),
]


for dens_col in density_cols:
    for hue_col, cmap, norm, cbar, legend in hue_groups_single_position:
        out_subdir = outdir / "single_position"
        out_subdir.mkdir(parents=True, exist_ok=True)

        make_summary_plots(
            out_dir=out_subdir,
            filename=f"{hue_col}_vs_{dens_col}_vs_flow",
            df=summary_df.dropna(subset=hue_col),
            x="shear_stress",
            x_label="Shear Stress (dyn/cm²)",
            y=dens_col,
            hue=hue_col,
            hue_norm=norm,
            cmap=cmap,
            cbar_scalarmap=cbar,
            legend=legend,
        )

    for hue_col, cmap, norm, cbar, legend in hue_groups_multiposition:
        out_subdir = outdir / "multiposition"
        out_subdir.mkdir(parents=True, exist_ok=True)

        make_summary_plots(
            out_dir=out_subdir,
            filename=f"{hue_col}_vs_{dens_col}_vs_flow",
            df=summary_df_multipos,
            x="shear_stress",
            x_label="Shear Stress (dyn/cm²)",
            y=dens_col,
            hue=hue_col,
            hue_norm=norm,
            cmap=cmap,
            cbar_scalarmap=cbar,
            legend=legend,
        )

print("Done.")
