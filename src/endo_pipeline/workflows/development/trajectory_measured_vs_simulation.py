from endo_pipeline.cli import Datasets
from endo_pipeline.library.visualize.integration.track_integration_viz import (
    plot_quiver_slices_from_flow_field_dict,
)


def main(
    datasets: Datasets | None = None,
    n_proc: int = 1,
) -> None:

    from concurrent.futures import ProcessPoolExecutor

    from endo_pipeline.settings import ColumnName as Column
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.manifests import get_dataframe_location_for_dataset
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.library.analyze.integration.track_integration import (
        get_drift_values_and_grid,
        get_flow_field_and_fixed_points,
    )

    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED,
    )
    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.settings.flow_field_3d import DATASET_COLLECTION_FOR_3D_DYNAMICS
    from endo_pipeline.library.analyze.data_driven_flow_field import (
        compute_extrapolated_vector_field,
        solve_ddff_ode,
    )
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from tqdm import tqdm
    import numpy as np
    import pandas as pd

    def solve_ddff_from_trajectory_initial_condition(
        crop_index: int,
        flow_field_dict: dict,
        initial_condition: np.ndarray,
        trajectory_duration: float,
    ) -> dict:
        trajectory_simulation = solve_ddff_ode(
            flow_field_dict=flow_field_dict,
            init=initial_condition,
            t_span=(0, trajectory_duration),
            num_t=trajectory_duration,
        )
        simulation_as_df_record = {
            Column.CROP_INDEX: crop_index,
            Column.TRACK_LENGTH: trajectory_duration,
            "drift_prediction": trajectory_simulation,
        }
        return simulation_as_df_record

    dataset_names = datasets or get_datasets_in_collection(DATASET_COLLECTION_FOR_3D_DYNAMICS)

    if DEMO_MODE:
        dataset_names = dataset_names[:1]

    for dataset_name in dataset_names:
        # load the dynamics features from the grid-based dataframe
        dynamics_manifest_grid = load_dataframe_manifest(
            DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED
        )
        dynamics_loc_grid = get_dataframe_location_for_dataset(dynamics_manifest_grid, dataset_name)
        df_grid = load_dataframe(dynamics_loc_grid)

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

        # get the initial conditions for the simulation from the dynamics features dataframe
        df_grid_t_init = df_grid[df_grid[Column.TIMEPOINT] == df_grid[Column.TIMEPOINT].min()]
        crop_index, initial_conditions = list(
            zip(*df_grid_t_init.groupby(Column.CROP_INDEX)[[*DYNAMICS_COLUMN_NAMES]])
        )

        crop_indices_and_initial_conditions = list(
            df_grid_t_init.groupby([Column.CROP_INDEX, Column.TRACK_LENGTH])[
                list(DYNAMICS_COLUMN_NAMES)
            ]
        )

        if DEMO_MODE:
            crop_index = crop_index[:1]
            initial_conditions = initial_conditions[:1]
            crop_indices_and_initial_conditions = crop_indices_and_initial_conditions[:1]

        # result = solve_ddff_ode(
        #     extrapolated_flow_field_dict_reg,
        #     initial_conditions[0].values.flatten(),
        #     TRAJECTORY_TIME_SPAN,
        # )

        # results = []
        ivp_args_mp: list[dict] = []
        for (crop_i, track_duration), init_df in crop_indices_and_initial_conditions:
            # results.append(
            #     solve_ddff_from_trajectory_initial_condition(
            #         crop_index=crop_i,
            #         flow_field_dict=extrapolated_flow_field_dict_reg,
            #         initial_condition=init_df.values.flatten(),
            #         trajectory_duration=track_duration,
            #     )
            # )
            ivp_args_mp.append(
                {
                    "crop_index": crop_i,
                    "flow_field_dict": extrapolated_flow_field_dict_reg,
                    "initial_condition": init_df.values.flatten(),
                    "trajectory_duration": track_duration,
                }
            )

        # TODO FIND A WAY TO ASSOCIATE CROP INDEX WITH SIMULATED TRAJECTORIES
        # will also need the track duration to pass along to the ODE solver so
        # that simulated trajectories have same duration as measured tracks

        with ProcessPoolExecutor(max_workers=n_proc) as executor:
            # results = list(
            #     tqdm(
            #         executor.map(
            #             solve_ddff_from_trajectory_initial_condition,
            #             [extrapolated_flow_field_dict_reg] * len(initial_conditions),
            #             initial_conditions,
            #             [TRAJECTORY_TIME_SPAN] * len(initial_conditions),
            #         ),
            #         desc=f"Solving ODEs for dataset {dataset_name}",
            #         total=len(initial_conditions),
            #     )
            # )

            results = list(
                tqdm(
                    executor.map(
                        solve_ddff_from_trajectory_initial_condition,
                        **ivp_args_mp,
                    ),
                    desc=f"Solving ODEs for dataset {dataset_name}",
                    total=len(ivp_args_mp),
                )
            )
        traj_sim_df = pd.DataFrame(results)

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
                ax.scatter(*fixed_points_at_slices[j], c="k", s=50)
