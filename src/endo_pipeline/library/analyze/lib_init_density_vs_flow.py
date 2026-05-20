import logging
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from endo_pipeline.cli import Datasets
from endo_pipeline.configs import get_subset_of_timepoint_annotations, load_dataset_config
from endo_pipeline.io import load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_annotations
from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.workflow_defaults import CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME

logger = logging.getLogger(__name__)


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


def create_summary_dfs(
    datasets: Datasets,
    cols_to_compute: list[str],
    dataset_manifest_name: str = CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    dataset_info_cols = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
    ]

    for col in dataset_info_cols[::-1]:
        if col not in cols_to_compute:
            cols_to_compute.insert(0, col)

    summary_df = pd.DataFrame(columns=cols_to_compute)
    df_subset = pd.DataFrame(columns=cols_to_compute)
    for dataset_name in tqdm(datasets):
        config = load_dataset_config(dataset_name)
        if len(config.flow_conditions) != 1:
            logger.debug(f"Dataset {dataset_name} is not monoflow, skipping.")
            continue
        else:
            shear_stress = config.flow_conditions[0].shear_stress
            shear_stress_regime = config.shear_stress_regime[0].value

        dataset_manifest = load_dataframe_manifest(dataset_manifest_name)
        dataset_location = get_dataframe_location_for_dataset(dataset_manifest, dataset_name)
        df_delayed = load_dataframe(dataset_location, delay=True)

        df = df_delayed[cols_to_compute].compute().reset_index()
        df[Column.DATASET] = dataset_name
        annotations_to_ignore: list = []
        timepoint_annotations = get_subset_of_timepoint_annotations(annotations_to_ignore)
        df_filtered = filter_dataframe_by_annotations(
            df,
            config,
            timepoint_annotations=timepoint_annotations,
        )

        df_filtered[Column.SHEAR_STRESS] = shear_stress
        df_filtered[Column.SHEAR_STRESS_REGIME] = shear_stress_regime

        if df_subset.empty:
            df_subset = df_filtered
        else:
            df_subset = pd.concat([df_subset, df_filtered])

        df = df.dropna(subset=Column.SegData.NUM_NUCLEI_AT_TIMEPOINT)
        first_t = df[Column.TIMEPOINT].min()
        df_first_t = df[df[Column.TIMEPOINT] == first_t]

        groups = df_first_t.groupby(dataset_info_cols)

        for (ds, pos, tp), df_grp in groups:
            pos_num_nuclei = sequence_to_scalar(df_grp[Column.SegData.NUM_NUCLEI_AT_TIMEPOINT])
            pos_nuc_density = df_grp[Column.SegData.NUM_NUCLEI_IN_CROP].mean()
            pos_num_seg_unfilt = sequence_to_scalar(
                df_grp[Column.SegData.NUM_TRACKS_BEFORE_FILTERING]
            )
            rho = df_grp[Column.DiffAEData.PC3_FLIPPED].mean()

            df_features = df_filtered[df_filtered[Column.POSITION] == pos]
            vector_means: dict = {}
            vector_means_multipos: dict = {}
            for feature in [Column.SegData.ORIENTATION, Column.SegData.POLAR_ANGLE]:
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

            new_records = pd.DataFrame(
                {
                    Column.DATASET: [ds],
                    Column.POSITION: [pos],
                    Column.TIMEPOINT: [tp],
                    Column.SHEAR_STRESS: [shear_stress],
                    Column.SHEAR_STRESS_REGIME: [shear_stress_regime],
                    Column.SegData.NUM_NUCLEI_AT_TIMEPOINT: [pos_num_nuclei],
                    Column.SegData.NUM_TRACKS_BEFORE_FILTERING: [pos_num_seg_unfilt],
                    Column.DiffAEData.PC3_FLIPPED: [rho],
                    Column.SegData.NUM_NUCLEI_IN_CROP: [pos_nuc_density],
                    **vector_means,
                    **vector_means_multipos,
                }
            )
            if summary_df.empty:
                summary_df = new_records
            else:
                summary_df = pd.concat([summary_df, new_records], ignore_index=True)

    df_grpd = summary_df.groupby(Column.DATASET)
    df_grpd_for_means = df_subset.groupby(Column.DATASET)

    polar_theta_vec_mean_mag = df_grpd[
        f"{Column.DiffAEData.POLAR_ANGLE}_vec_mean_multipos_magnitude"
    ].apply(lambda x: float(x.unique().item()))
    orientation_vec_mean_mag = df_grpd[
        f"{Column.SegData.ORIENTATION}_vec_mean_multipos_magnitude"
    ].apply(lambda x: float(x.unique().item()))
    polar_theta_vec_mean_ang = df_grpd[
        f"{Column.DiffAEData.POLAR_ANGLE}_vec_mean_multipos_angle"
    ].apply(lambda x: float(x.unique().item()))
    orientation_vec_mean_ang = df_grpd[
        f"{Column.SegData.ORIENTATION}_vec_mean_multipos_angle"
    ].apply(lambda x: float(x.unique().item()))

    summary_df_agg = pd.DataFrame(
        {
            Column.DATASET: df_grpd[Column.DATASET].apply(lambda x: x.unique().item()),
            Column.SHEAR_STRESS_REGIME: df_grpd[Column.SHEAR_STRESS_REGIME].apply(
                lambda x: x.unique().item()
            ),
            Column.SegData.NUM_NUCLEI_AT_TIMEPOINT: df_grpd[
                Column.SegData.NUM_NUCLEI_AT_TIMEPOINT
            ].sum(),
            Column.SegData.NUM_TRACKS_BEFORE_FILTERING: df_grpd[
                Column.SegData.NUM_TRACKS_BEFORE_FILTERING
            ].sum(),
            Column.DiffAEData.PC3_FLIPPED: df_grpd_for_means[Column.DiffAEData.PC3_FLIPPED].mean(),
            Column.SegData.NUM_NUCLEI_IN_CROP: df_grpd_for_means[
                Column.SegData.NUM_NUCLEI_IN_CROP
            ].mean(),
            Column.SHEAR_STRESS: df_grpd[Column.SHEAR_STRESS].apply(
                lambda x: float(x.unique().item())
            ),
            f"{Column.DiffAEData.POLAR_ANGLE}_vec_mean_multipos_magnitude": polar_theta_vec_mean_mag,
            f"{Column.SegData.ORIENTATION}_vec_mean_multipos_magnitude": orientation_vec_mean_mag,
            f"{Column.DiffAEData.POLAR_ANGLE}_vec_mean_multipos_angle": polar_theta_vec_mean_ang,
            f"{Column.SegData.ORIENTATION}_vec_mean_multipos_angle": orientation_vec_mean_ang,
        }
    )

    return summary_df_agg, summary_df


def make_summary_plots(
    out_dir: Path,
    filename: str,
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None,
    hue_norm: mpl.colors.Normalize,
    cmap: str,
    cbar_scalarmap: plt.cm.ScalarMappable | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    legend: bool = False,
):
    mpl.rcParams["figure.max_open_warning"] = 0
    hue_as_str: str = "" if hue is None else hue

    x_label = x_label or get_label_for_column(x)
    y_label = y_label or get_label_for_column(y)

    fig, ax = plt.subplots(figsize=(3.5, 3))
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue_as_str,
        marker="o",
        edgecolor="black",
        hue_norm=hue_norm,
        palette=cmap,
        s=20,
        ax=ax,
        legend=legend,
    )
    if legend:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_title(hue_as_str.replace("_", " ").capitalize())
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_adjustable("box")
    if cbar_scalarmap is not None:
        cbar = ax.figure.colorbar(cbar_scalarmap, ax=ax)
        if "angle" in hue_as_str:
            cmin, cmax = cbar_scalarmap.get_clim()
            num_steps = 7
            cbar.set_ticks(
                np.linspace(cmin, cmax, num_steps, endpoint=True),
                labels=np.linspace(
                    np.rad2deg(cmin), np.rad2deg(cmax), num_steps, endpoint=True, dtype=int
                ),
            )
    save_plot_to_path(fig, out_dir, filename)
