"""This workflow computes the time of first passage for each track in the dataset."""

from typing import Literal

from endo_pipeline.cli import Datasets


def main(datasets: Datasets, n_proc: int = 1, crop_pattern: Literal["grid", "tracked"] = "grid"):

    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd
    from tqdm import tqdm

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs.dataset_config_io import (
        get_datasets_in_collection,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.data_driven_flow_field import (
        compute_extrapolated_vector_field,
        get_drift_df,
        get_drift_values_and_grid_from_drift_df,
        get_fixed_points_df,
    )
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_steady_state
    from endo_pipeline.library.analyze.integration.track_integration import (
        add_distance_to_fixed_points_columns,
        get_time_of_first_passage,
        solve_ddff_from_trajectory_initial_condition_helper,
    )
    from endo_pipeline.library.visualize.integration.track_integration_viz import (
        plot_time_of_first_passage,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.flow_field_3d import DATASET_COLLECTION_FOR_3D_DYNAMICS
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_COLORMAP_BIN_SIZE
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED,
    )

    if crop_pattern == "tracked":
        raise ValueError("Tracked crop pattern is not supported yet.")

    dataset_names = datasets or get_datasets_in_collection(DATASET_COLLECTION_FOR_3D_DYNAMICS)

    if DEMO_MODE:
        dataset_names = dataset_names[:1]

    for dataset_name in dataset_names:
        out_dir = get_output_path(__file__, dataset_name)

        # load the dynamics features from the grid-based dataframe
        dynamics_manifest_grid = load_dataframe_manifest(
            DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED
        )
        dynamics_loc_grid = get_dataframe_location_for_dataset(dynamics_manifest_grid, dataset_name)
        df_grid = load_dataframe(dynamics_loc_grid)
        # the loaded grid-based dynamics dataframe is disordered by default so
        # sort the grid-based dynamics dataframe by crop index and timepoint
        df_grid = df_grid.sort_values(by=[Column.CROP_INDEX, Column.TIMEPOINT])

        # filter the grid-based dynamics dataframe to only include timepoints from steady state
        dataset_config = load_dataset_config(dataset_name)
        df_grid = filter_dataframe_to_steady_state(dataframe=df_grid, dataset_config=dataset_config)

        # load the flow field dictionaries and fixed points
        drift_df = get_drift_df(dataset_name)
        drift_values, grid_points_1d = get_drift_values_and_grid_from_drift_df(
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

        df_grid_sub = add_distance_to_fixed_points_columns(
            trajectory_df=df_grid_sub,
            fixed_point_df=fixed_points_df,
            trajectory_columns=DYNAMICS_COLUMN_NAMES,
            column_suffix="grid",
        )

        simulated_columns = [f"{col}_simulated" for col in DYNAMICS_COLUMN_NAMES]
        df_grid_sub = add_distance_to_fixed_points_columns(
            trajectory_df=df_grid_sub,
            fixed_point_df=fixed_points_df,
            trajectory_columns=simulated_columns,
            fixed_point_columns=DYNAMICS_COLUMN_NAMES,
            column_suffix="_simulated",
        )

        for i in fixed_points_df.index:
            time_of_first_passage = []
            time_of_first_passage.append(
                get_time_of_first_passage(
                    trajectory_df=df_grid_sub,
                    column=f"dist_from_fp_{i}_grid",
                    threshold=MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
                )
            )
            time_of_first_passage.append(
                get_time_of_first_passage(
                    trajectory_df=df_grid_sub,
                    column=f"dist_from_fp_{i}_simulated",
                    threshold=MIGRATION_COHERENCE_COLORMAP_BIN_SIZE,
                )
            )
            time_of_first_passage_df = pd.concat(time_of_first_passage, axis=1).reset_index()

            plot_time_of_first_passage(
                fixed_point_id=i,
                dataset_config=dataset_config,
                time_of_first_passage_df=time_of_first_passage_df,
                out_dir=out_dir,
            )
