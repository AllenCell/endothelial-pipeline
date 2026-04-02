from matplotlib import pyplot as plt

from endo_pipeline.cli import Datasets
from endo_pipeline.io.output import get_output_path, save_plot_to_path


def main(
    datasets: Datasets | None = None,
    n_proc: int = 1,
) -> None:

    from concurrent.futures import ProcessPoolExecutor

    import numpy as np
    import pandas as pd
    import seaborn as sns
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.library.analyze.data_driven_flow_field import (
        compute_extrapolated_vector_field,
    )
    from endo_pipeline.library.analyze.diffae_dataframe_utils import rewrap_polar_angle
    from endo_pipeline.library.analyze.integration.track_integration import (
        get_drift_values_and_grid,
        get_flow_field_and_fixed_points,
        solve_ddff_from_trajectory_initial_condition_helper,
    )
    from endo_pipeline.library.visualize.integration.track_integration_viz import (
        plot_quiver_slices_from_flow_field_dict,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.flow_field_3d import DATASET_COLLECTION_FOR_3D_DYNAMICS
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED,
    )

    dataset_names = datasets or get_datasets_in_collection(DATASET_COLLECTION_FOR_3D_DYNAMICS)

    # DEMO_MODE = True

    if DEMO_MODE:
        # dataset_names = dataset_names[:1]
        dataset_names = ["20250618_20X"]

    for dataset_name in dataset_names:
        outdir = get_output_path(dataset_name)

        # load the dynamics features from the grid-based dataframe
        dynamics_manifest_grid = load_dataframe_manifest(
            DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED
        )
        dynamics_loc_grid = get_dataframe_location_for_dataset(dynamics_manifest_grid, dataset_name)
        df_grid = load_dataframe(dynamics_loc_grid)
        # the loaded grid-based dynamics dataframe is disorderedby default so
        # sort the grid-based dynamics dataframe by crop index and timepoint
        df_grid = df_grid.sort_values(by=[Column.CROP_INDEX, Column.TIMEPOINT])

        # load the flow field and fixed points
        drift_values, grid_points_1d = get_drift_values_and_grid(
            dataset_name, column_names=DYNAMICS_COLUMN_NAMES
        )
        flow_field_dict_grid, fixed_points_df = get_flow_field_and_fixed_points(
            dataset_name, column_names=DYNAMICS_COLUMN_NAMES
        )

        ## ODE solver: dx/dt = f(x) (drift, first Kramers-Moyal coefficient) ##
        # with initial conditions given by init solve IVP, get back trajectory
        extrapolated_flow_field_dict_reg = compute_extrapolated_vector_field(
            drift_values, grid_points_1d, method="linear", for_vtk_files=False
        )

        # add the track durations
        df_grid[Column.TRACK_LENGTH] = df_grid.groupby(Column.CROP_INDEX)[
            Column.TIMEPOINT
        ].transform(lambda t: t.max() - t.min())

        # get the initial conditions for the simulation from the dynamics features dataframe
        df_grid_t_init = df_grid[df_grid[Column.TIMEPOINT] == df_grid[Column.TIMEPOINT].min()]
        crop_index, initial_conditions = list(
            zip(*df_grid_t_init.groupby(Column.CROP_INDEX)[[*DYNAMICS_COLUMN_NAMES]], strict=True)
        )

        crop_indices_and_initial_conditions = list(
            df_grid_t_init.groupby([Column.CROP_INDEX, Column.TRACK_LENGTH, Column.TIMEPOINT])[
                list(DYNAMICS_COLUMN_NAMES)
            ]
        )

        if DEMO_MODE:
            crop_index = crop_index[:1]
            initial_conditions = initial_conditions[:1]
            crop_indices_and_initial_conditions = crop_indices_and_initial_conditions[:1]

        ivp_args_mp: list[dict] = []
        for (crop_i, track_duration, timepoint), init_df in crop_indices_and_initial_conditions:
            ivp_args_mp.append(
                {
                    "crop_index": crop_i,
                    "flow_field_dict": extrapolated_flow_field_dict_reg,
                    "initial_condition": init_df.values.flatten(),
                    "timepoint_initial": timepoint,
                    "trajectory_duration": track_duration,
                    "simulation_results_column_names": list(DYNAMICS_COLUMN_NAMES),
                }
            )

        with ProcessPoolExecutor(max_workers=n_proc) as executor:
            results = list(
                tqdm(
                    executor.map(solve_ddff_from_trajectory_initial_condition_helper, ivp_args_mp),
                    desc=f"Solving ODEs for dataset {dataset_name}",
                    total=len(ivp_args_mp),
                )
            )

        traj_sim_df = pd.concat(map(pd.DataFrame, results)).reset_index(drop=True)
        df_grid_sub = df_grid[
            df_grid[Column.CROP_INDEX].isin(traj_sim_df[Column.CROP_INDEX].unique())
        ]
        merging_columns = [Column.CROP_INDEX, Column.TRACK_LENGTH, Column.TIMEPOINT]
        df_grid_sub = df_grid_sub.merge(traj_sim_df, on=merging_columns, how="outer")

        # the simulations polar angle needs to be rewrapped because of its periodic nature
        # (the simulation normally lets the angle go beyond the original range of 0 to pi)
        df_grid_sub[f"{Column.DiffAEData.POLAR_ANGLE}_simulated_unwrapped"] = df_grid_sub[
            f"{Column.DiffAEData.POLAR_ANGLE}_simulated"
        ].copy()
        df_grid_sub[f"{Column.DiffAEData.POLAR_ANGLE}_simulated"] = rewrap_polar_angle(
            df_grid_sub[f"{Column.DiffAEData.POLAR_ANGLE}_simulated"].values,
            original_range=(0, np.pi),
        )

        for i, fp_row in fixed_points_df.iterrows():
            flow_field_slices = (
                fp_row[DYNAMICS_COLUMN_NAMES[2]],
                fp_row[DYNAMICS_COLUMN_NAMES[1]],
            )  # feature 3, feature 2
            fixed_points_at_slices = (
                fp_row[list(map(str, DYNAMICS_COLUMN_NAMES))].drop(
                    index=[DYNAMICS_COLUMN_NAMES[2]]
                ),
                fp_row[list(map(str, DYNAMICS_COLUMN_NAMES))].drop(
                    index=[DYNAMICS_COLUMN_NAMES[1]]
                ),
            )

            fig, axs = plot_quiver_slices_from_flow_field_dict(
                dataset_name=dataset_name,
                flow_field_dict_grids=flow_field_dict_grid,
                feature_vals=flow_field_slices,
                column_names=DYNAMICS_COLUMN_NAMES,
            )
            for j, ax in enumerate(axs):
                # add the fixed points
                ax.scatter(*fixed_points_at_slices[j], c="k", s=50)

                cols_measured = fixed_points_at_slices[j].index.tolist()
                cols_simulated = [f"{col}_simulated" for col in cols_measured]
                for crop_i, traj_df in df_grid_sub.groupby(Column.CROP_INDEX):
                    sns.scatterplot(
                        data=traj_df,
                        x=cols_measured[0],
                        y=cols_measured[1],
                        hue=Column.TIMEPOINT,
                        palette="flare",
                        marker="D",
                        edgecolor="black",
                        alpha=0.7,
                        s=10,
                        ax=ax,
                    )
                    unwrapped_angle_diff = (
                        traj_df[f"{Column.DiffAEData.POLAR_ANGLE}_simulated_unwrapped"]
                        .diff()
                        .replace(np.nan, True)
                    )
                    wrapped_angle_diff = (
                        traj_df[f"{Column.DiffAEData.POLAR_ANGLE}_simulated"]
                        .diff()
                        .replace(np.nan, True)
                    )
                    wrap_discontinuity = np.logical_not(unwrapped_angle_diff == wrapped_angle_diff)
                    angle_segments_to_plot_indices = np.split(
                        wrap_discontinuity, wrap_discontinuity.index[wrap_discontinuity]
                    )
                    for segment_indices in angle_segments_to_plot_indices:
                        data_segment = traj_df.iloc[segment_indices.index]
                        ax.plot(
                            data_segment[cols_simulated[0]].values,
                            data_segment[cols_simulated[1]].values,
                            ls="--",
                            lw=1,
                            alpha=0.7,
                            c="black",
                            zorder=10,
                        )
                    sns.scatterplot(
                        data=traj_df,
                        x=cols_simulated[0],
                        y=cols_simulated[1],
                        hue=Column.TIMEPOINT,
                        palette="flare",
                        edgecolor=None,
                        marker="o",
                        alpha=0.7,
                        s=10,
                        zorder=9,
                        ax=ax,
                    )
            save_plot_to_path(
                figure=fig,
                output_path=outdir,
                figure_name=f"{dataset_name}_cropindex{crop_i}_traj_meas_vs_sim.png",
            )
            plt.close(fig)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
