from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    n_proc: int = 1,
    make_trajectory_validation_plots: bool = False,
) -> None:

    from concurrent.futures import ProcessPoolExecutor

    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from tslearn.metrics import dtw, frechet

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.io.output import get_output_path
    from endo_pipeline.library.analyze.data_driven_flow_field import (
        compute_extrapolated_vector_field,
        get_drift_df,
        get_drift_flow_field_as_dict,
        get_drift_values_and_grid,
        get_fixed_points_df,
    )
    from endo_pipeline.library.analyze.integration.track_integration import (
        solve_ddff_from_trajectory_initial_condition_helper,
    )
    from endo_pipeline.library.analyze.polar_coords import (
        rewrap_polar_angle,
        unwrap_nonsequential_array,
    )
    from endo_pipeline.library.visualize.integration.track_integration_viz import (
        plot_trajectory_measured_vs_simulation_over_flow_field_helper,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        DYNAMICS_COLUMN_NAMES,
        PERIOD_THETA_RESCALED,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_3d import DATASET_COLLECTION_FOR_3D_DYNAMICS
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED,
    )

    dataset_names = datasets or get_datasets_in_collection(DATASET_COLLECTION_FOR_3D_DYNAMICS)

    if DEMO_MODE:
        dataset_names = dataset_names[:1]

    for dataset_name in dataset_names:
        outdir = get_output_path(__file__, dataset_name)

        # load the dynamics features from the grid-based dataframe
        dynamics_manifest_grid = load_dataframe_manifest(
            DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED
        )
        dynamics_loc_grid = get_dataframe_location_for_dataset(dynamics_manifest_grid, dataset_name)
        df_grid = load_dataframe(dynamics_loc_grid)
        # the loaded grid-based dynamics dataframe is disorderedby default so
        # sort the grid-based dynamics dataframe by crop index and timepoint
        df_grid = df_grid.sort_values(by=[Column.CROP_INDEX, Column.TIMEPOINT])

        # load the flow field dictionaries and fixed points
        drift_df = get_drift_df(dataset_name)
        drift_values, grid_points_1d = get_drift_values_and_grid(
            flow_field_dataframe=drift_df, column_names=DYNAMICS_COLUMN_NAMES
        )
        flow_field_dict_grid = get_drift_flow_field_as_dict(
            flow_field_dataframe=drift_df, column_names=DYNAMICS_COLUMN_NAMES
        )
        fixed_points_df = get_fixed_points_df(dataset_name)

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
            crop_index = crop_index[:10]
            initial_conditions = initial_conditions[:10]
            crop_indices_and_initial_conditions = crop_indices_and_initial_conditions[:10]

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
                    "time_limit": 10,  # seconds
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

        # combine the simulation results with the measured trajectories in a single dataframe
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

        # compute the Frechet distance between the measured and simulated trajectories
        theta_period = PERIOD_THETA_RESCALED if RESCALE_THETA else 2 * np.pi

        df_grid_sub[f"{Column.DiffAEData.POLAR_ANGLE}_unwrapped"] = df_grid_sub.groupby(
            Column.CROP_INDEX
        )[f"{Column.DiffAEData.POLAR_ANGLE}"].transform(
            lambda x, theta_period=theta_period: unwrap_nonsequential_array(
                x.values, period=theta_period
            )
        )
        frechet_distances = []
        dtw_distances = []
        for crop_i, traj_df in df_grid_sub.groupby(Column.CROP_INDEX):
            traj_df = traj_df.dropna(
                subset=[
                    f"{Column.DiffAEData.POLAR_ANGLE}_unwrapped",
                    Column.DiffAEData.POLAR_RADIUS,
                ]
            )
            traj_measured = traj_df[
                [f"{Column.DiffAEData.POLAR_ANGLE}_unwrapped", Column.DiffAEData.POLAR_RADIUS]
            ]
            traj_simulated = traj_df[
                [
                    f"{Column.DiffAEData.POLAR_ANGLE}_simulated_unwrapped",
                    f"{Column.DiffAEData.POLAR_RADIUS}_simulated",
                ]
            ]

            # try the Frechet distance:
            frechet_dist = frechet(traj_measured, traj_simulated)
            frechet_distances.append(frechet_dist)

            # try dynamic time warping:
            dtw_dist = dtw(traj_measured, traj_simulated)
            dtw_distances.append(dtw_dist)

            # break

        #     measured_traj = traj_df[[Column.DiffAEData.X, Column.DiffAEData.Y]].values
        #     simulated_traj = traj_df[
        #         [f"{Column.DiffAEData.X}_simulated", f"{Column.DiffAEData.Y}_simulated"]
        #     ].values
        #     if len(measured_traj) > 0 and len(simulated_traj) > 0:
        #         frechet_distances.append(frechet_distance(measured_traj, simulated_traj))
        # df_grid_sub["frechet_distance"] = np.nan
        # if frechet_distances:
        #     df_grid_sub.loc[
        #         df_grid_sub[Column.CROP_INDEX].isin(df_grid_sub[Column.CROP_INDEX].unique()),
        #         "frechet_distance",
        #     ] = frechet_distances

        if make_trajectory_validation_plots:
            # plot overlays of the tracks with the fixed points on the flow field slices
            for i, fp_row in fixed_points_df.iterrows():
                out_subdir = outdir / f"fixed_point_{i}"
                out_subdir.mkdir(parents=True, exist_ok=True)

                # prepare arguments for multiprocessing plotting
                plotting_args_mp: list[dict] = []
                for crop_i, traj_df in df_grid_sub.groupby(Column.CROP_INDEX):
                    plotting_args_mp.append(
                        {
                            "crop_index": crop_i,
                            "traj_df": traj_df,
                            "fixed_point_id": i,
                            "fixed_point_row": fp_row,
                            "flow_field_dict_grid": flow_field_dict_grid,
                            "out_dir": out_subdir,
                        }
                    )

                with ProcessPoolExecutor(max_workers=n_proc) as executor:
                    list(
                        tqdm(
                            executor.map(
                                plot_trajectory_measured_vs_simulation_over_flow_field_helper,
                                plotting_args_mp,
                            ),
                            desc=f"Plotting trajectories for fixed point {i}",
                            total=len(plotting_args_mp),
                        )
                    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
