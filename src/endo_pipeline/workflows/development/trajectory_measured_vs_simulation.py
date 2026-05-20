"""This workflow compares the measured trajectories with ones simulated from the flow field"""

from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    n_proc: int = 1,
) -> None:

    from concurrent.futures import ProcessPoolExecutor

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.io.output import get_output_path
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_steady_state
    from endo_pipeline.library.analyze.numerics.fixed_points import (
        load_fixed_points_dataframe_for_dataset,
    )
    from endo_pipeline.library.analyze.polar_coords import rewrap_polar_angle
    from endo_pipeline.library.analyze.track_integration import (
        solve_ddff_from_trajectory_initial_condition_helper,
    )
    from endo_pipeline.library.analyze.vector_field_estimation import (
        compute_extrapolated_vector_field,
        get_reshaped_vector_field_and_grid,
        get_vector_field_as_dict_from_dataframe,
        load_drift_dataframe_for_dataset,
    )
    from endo_pipeline.library.visualize.integration.track_integration_viz import (
        plot_trajectory_measured_vs_simulation_over_flow_field_helper,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES, TIME_STEP_IN_HOURS
    from endo_pipeline.settings.workflow_defaults import GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME

    dataset_names = datasets or get_datasets_in_collection("3d_flow_field_analysis")

    if DEMO_MODE:
        dataset_names = dataset_names[:1]

    for dataset_name in dataset_names:
        outdir = get_output_path(__file__, dataset_name)

        # load the dataset config to get the time interval in minutes for converting time units for the ODE solver
        dataset_config = load_dataset_config(dataset_name)
        timepoint_units = TIME_STEP_IN_HOURS

        # load the dynamics features from the grid-based dataframe
        dynamics_manifest_grid = load_dataframe_manifest(GRID_BASED_FEATURES_FILTERED_MANIFEST_NAME)
        dynamics_loc_grid = get_dataframe_location_for_dataset(dynamics_manifest_grid, dataset_name)
        df_grid = load_dataframe(dynamics_loc_grid)
        # the loaded grid-based dynamics dataframe is disorderedby default so
        # sort the grid-based dynamics dataframe by crop index and timepoint
        df_grid = df_grid.sort_values(by=[Column.CROP_INDEX, Column.TIMEPOINT])

        # filter the grid-based dynamics dataframe to only include timepoints from steady state
        dataset_config = load_dataset_config(dataset_name)
        df_grid = filter_dataframe_to_steady_state(dataframe=df_grid, dataset_config=dataset_config)

        # load the flow field dictionaries and fixed points
        drift_df = load_drift_dataframe_for_dataset(dataset_name)
        drift_values, grid_points_1d = get_reshaped_vector_field_and_grid(
            flow_field_dataframe=drift_df, column_names=DYNAMICS_COLUMN_NAMES
        )
        flow_field_dict_grid = get_vector_field_as_dict_from_dataframe(
            flow_field_dataframe=drift_df, column_names=DYNAMICS_COLUMN_NAMES
        )
        fixed_points_df = load_fixed_points_dataframe_for_dataset(dataset_name)

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
        df_grid_t_init = (
            df_grid.groupby(Column.POSITION, as_index=False)
            .apply(
                lambda grp: (df := pd.DataFrame(grp))[
                    df[Column.TIMEPOINT] == df[Column.TIMEPOINT].min()
                ]
            )
            .reset_index(drop=True)
        )

        crop_indices_and_initial_conditions = list(
            df_grid_t_init.groupby([Column.CROP_INDEX, Column.TRACK_LENGTH, Column.TIMEPOINT])[
                list(DYNAMICS_COLUMN_NAMES)
            ]
        )

        if DEMO_MODE:
            num_traj = 10
            crop_indices_and_initial_conditions = crop_indices_and_initial_conditions[:num_traj]

        ivp_args_mp: list[dict] = []
        maximum_multiple_of_trajectory_duration_for_simulation = 6
        for (crop_i, track_duration, timepoint), init_df in crop_indices_and_initial_conditions:
            ivp_args_mp.append(
                {
                    "crop_index": crop_i,
                    "flow_field_dict": extrapolated_flow_field_dict_reg,
                    "initial_condition": init_df.values.flatten(),
                    "timepoint_initial": timepoint,
                    "trajectory_duration": track_duration
                    * maximum_multiple_of_trajectory_duration_for_simulation,
                    # convert time units to hours for the ODE solver
                    "time_units_for_solver": timepoint_units,
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
