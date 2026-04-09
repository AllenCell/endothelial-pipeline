"""This workflow compares the measured trajectories with ones simulated from the flow field"""

from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    n_proc: int = 1,
    make_trajectory_validation_plots: bool = False,
) -> None:

    from concurrent.futures import ProcessPoolExecutor

    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import directed_hausdorff
    from tqdm import tqdm
    from tslearn.metrics import dtw, frechet

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import load_dataframe
    from endo_pipeline.io.output import get_output_path
    from endo_pipeline.library.analyze.data_driven_flow_field import (
        compute_extrapolated_vector_field,
        get_drift_df,
        get_drift_flow_field_as_dict,
        get_drift_values_and_grid_from_drift_df,
        get_fixed_points_df,
    )
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_steady_state
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

        # filter the grid-based dynamics dataframe to only include timepoints from steady state
        dataset_config = load_dataset_config(dataset_name)
        df_grid = filter_dataframe_to_steady_state(dataframe=df_grid, dataset_config=dataset_config)

        # load the flow field dictionaries and fixed points
        drift_df = get_drift_df(dataset_name)
        drift_values, grid_points_1d = get_drift_values_and_grid_from_drift_df(
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

        measured_cols = [
            f"{Column.DiffAEData.POLAR_ANGLE}_unwrapped",
            Column.DiffAEData.POLAR_RADIUS,
        ]
        simulated_cols = [
            f"{Column.DiffAEData.POLAR_ANGLE}_simulated_unwrapped",
            f"{Column.DiffAEData.POLAR_RADIUS}_simulated",
        ]

        def compute_frechet_distance(curve1: pd.DataFrame, curve2: pd.DataFrame) -> float:
            return float(frechet(curve1.values, curve2.values))

        def compute_dtw_distance(curve1: pd.DataFrame, curve2: pd.DataFrame) -> float:
            return float(dtw(curve1.values, curve2.values))

        def compute_hausdorff_distance(curve1: pd.DataFrame, curve2: pd.DataFrame) -> float:
            return float(directed_hausdorff(curve1.values, curve2.values)[0])

        distances_df = (
            df_grid_sub.groupby(Column.CROP_INDEX)
            .apply(
                lambda traj_df, measured_cols=measured_cols, simulated_cols=simulated_cols: (
                    lambda tdf, simulated_cols=simulated_cols: pd.Series(
                        {
                            Column.DATASET: str(traj_df[Column.DATASET].iloc[0]),
                            Column.POSITION: int(traj_df[Column.POSITION].iloc[0]),
                            "frechet_distance": compute_frechet_distance(
                                tdf[measured_cols], tdf[simulated_cols]
                            ),
                            "frechet_distance_sim_vs_meas": compute_frechet_distance(
                                tdf[simulated_cols], tdf[measured_cols]
                            ),
                            "dtw_distance": compute_dtw_distance(
                                tdf[measured_cols], tdf[simulated_cols]
                            ),
                            "dtw_distance_sim_vs_meas": compute_dtw_distance(
                                tdf[simulated_cols], tdf[measured_cols]
                            ),
                            "directed_hausdorff_distance_meas_vs_sim": compute_hausdorff_distance(
                                tdf[measured_cols], tdf[simulated_cols]
                            ),
                            "directed_hausdorff_distance_sim_vs_meas": compute_hausdorff_distance(
                                tdf[simulated_cols], tdf[measured_cols]
                            ),
                            "frechet_distance_resampled": compute_frechet_distance(
                                tdf[measured_cols].sample(frac=1),
                                tdf[simulated_cols].sample(frac=1),
                            ),
                            "dtw_distance_resampled": compute_dtw_distance(
                                tdf[measured_cols].sample(frac=1),
                                tdf[simulated_cols].sample(frac=1),
                            ),
                            "dir_hausdorff_dist_meas_vs_sim_resampled": compute_hausdorff_distance(
                                tdf[measured_cols].sample(frac=1),
                                tdf[simulated_cols].sample(frac=1),
                            ),
                            "dir_hausdorff_dist_sim_vs_meas_resampled": compute_hausdorff_distance(
                                tdf[simulated_cols].sample(frac=1),
                                tdf[measured_cols].sample(frac=1),
                            ),
                        }
                    )
                )(traj_df.dropna(subset=measured_cols))
            )
            .reset_index()
        )

        df_grid_sub = df_grid_sub.merge(
            distances_df, on=[Column.DATASET, Column.POSITION, Column.CROP_INDEX], how="left"
        )

        import seaborn as sns
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df_grid_sub, x="directed_hausdorff_distance_meas_vs_sim", y="frechet_distance"
        )
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c="red", alpha=0.5, zorder=0)
        plt.show()

        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df_grid_sub,
            x="directed_hausdorff_distance_sim_vs_meas",
            y="frechet_distance_sim_vs_meas",
        )
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c="red", alpha=0.5, zorder=0)
        plt.show()

        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df_grid_sub, x="directed_hausdorff_distance_meas_vs_sim", y="dtw_distance"
        )
        plt.show()

        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df_grid_sub,
            x="directed_hausdorff_distance_sim_vs_meas",
            y="dtw_distance_sim_vs_meas",
        )
        plt.show()

        fig, ax = plt.subplots()
        sns.histplot(distances_df["directed_hausdorff_distance_meas_vs_sim"], alpha=0.7, ax=ax)
        sns.histplot(distances_df["dir_hausdorff_dist_meas_vs_sim_resampled"], alpha=0.7, ax=ax)
        plt.show()

        fig, ax = plt.subplots()
        sns.histplot(distances_df["directed_hausdorff_distance_sim_vs_meas"], alpha=0.7, ax=ax)
        sns.histplot(distances_df["dir_hausdorff_dist_sim_vs_meas_resampled"], alpha=0.7, ax=ax)
        plt.show()

        fig, ax = plt.subplots()
        sns.histplot(distances_df["frechet_distance"], alpha=0.7, ax=ax)
        sns.histplot(distances_df["frechet_distance_resampled"], alpha=0.7, ax=ax)
        plt.show()

        fig, ax = plt.subplots()
        sns.histplot(distances_df["dtw_distance"], alpha=0.7, ax=ax)
        sns.histplot(distances_df["dtw_distance_resampled"], alpha=0.7, ax=ax)
        plt.show()

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

    def trajectory_similarity_metrics_test(print_outputs: bool = True) -> pd.DataFrame:
        import numpy as np
        import pandas as pd
        from matplotlib import pyplot as plt

        # define some lines
        # punctual makes a round trip and arrives exactly on time
        punctual_x = [*np.linspace(0, 10, 11), *np.linspace(9, 0, 10)]
        punctual_y = [0] * len(punctual_x)
        punctual = {"punctual": list(zip(punctual_x, punctual_y, strict=True))}

        # lazy moves slower than punctual and gives up half way
        quitter_x = [*np.linspace(0, 10, 21)]
        quitter_y = [0] * len(quitter_x)
        quitter = {"quitter": list(zip(quitter_x, quitter_y, strict=True))}

        # impatient moves faster than punctual and ends up waiting at the end
        impatient_x = [*np.linspace(0, 10, 6), *np.linspace(8, 0, 5), *([0] * 10)]
        impatient_y = [0] * len(impatient_x)
        impatient = {"impatient": list(zip(impatient_x, impatient_y, strict=True))}

        # overachiever moves faster than punctual and ends up doing an extra lap
        overachiever_x = [
            *np.linspace(0, 10, 6),
            *np.linspace(8, 0, 5),
            *np.linspace(2, 10, 5),
            *np.linspace(8, 0, 5),
        ]
        overachiever_y = [0] * len(overachiever_x)
        overachiever = {"overachiever": list(zip(overachiever_x, overachiever_y, strict=True))}

        # lost moves randomly without following the path everyone else takes
        lost_x = [*np.random.randint(-10, 10, 21)]
        lost_y = [*np.random.randint(-10, 10, 21)]
        lost = {"lost": list(zip(lost_x, lost_y, strict=True))}

        # rebel moves in the opposite direction of everyone else
        rebel_x = [*np.linspace(0, -10, 11), *np.linspace(-9, 0, 10)]
        rebel_y = [0] * len(rebel_x)
        rebel = {"rebel": list(zip(rebel_x, rebel_y, strict=True))}

        line_comparisons = (
            (punctual, punctual),
            (punctual, quitter),
            (punctual, impatient),
            (punctual, overachiever),
            (punctual, lost),
            (punctual, rebel),
        )
        color_dict = {
            "punctual": "tab:blue",
            "quitter": "tab:orange",
            "impatient": "tab:green",
            "overachiever": "tab:red",
            "lost": "tab:purple",
            "rebel": "tab:brown",
        }

        distances = []
        for line_pair in line_comparisons:
            line1_name = list(line_pair[0].keys())[0]
            line2_name = list(line_pair[1].keys())[0]
            line1_vals = line_pair[0][line1_name]
            line2_vals = line_pair[1][line2_name]

            frechet_dist = frechet(
                line1_vals,
                line2_vals,
            )
            dtw_dist = dtw(
                line1_vals,
                line2_vals,
            )
            hausdorff_dist = directed_hausdorff(
                line1_vals,
                line2_vals,
            )
            hausdorff_dist_rev = directed_hausdorff(
                line2_vals,
                line1_vals,
            )

            distances.append(
                {
                    "line_1": line1_name,
                    "line_2": line2_name,
                    "color": color_dict[line2_name],
                    "Frechet_distance": frechet_dist,
                    "DTW_distance": dtw_dist,
                    "Hausdorff_distance": hausdorff_dist[0],
                    "Hausdorff_distance_rev": hausdorff_dist_rev[0],
                }
            )

        dist_metrics_df = pd.DataFrame(distances)

        if print_outputs:
            print(dist_metrics_df)

            fig, ax = plt.subplots()
            for _, row in dist_metrics_df.iterrows():
                ax.scatter(
                    x=row["Hausdorff_distance"],
                    y=row["Frechet_distance"],
                    color=color_dict[row["line_2"]],
                    marker="o",
                    s=30,
                    label=f"{row['line_2']} Hausdorff vs. Frechet",
                    linewidth=1,
                    alpha=0.7,
                )
                ax.scatter(
                    x=row["Hausdorff_distance"],
                    y=row["DTW_distance"],
                    marker="D",
                    s=30,
                    edgecolors=color_dict[row["line_2"]],
                    facecolor="none",
                    label=f"{row['line_2']} Hausdorff vs. DTW",
                    linewidth=1,
                    alpha=0.7,
                )
            ax.plot(ax.get_ylim(), ax.get_ylim(), ls=":", c="black")
            ax.axhline(0, ls="--", c="lightgrey")
            ax.axvline(0, ls="--", c="lightgrey")
            ax.set_xlabel("Hausdorff distance")
            ax.set_ylabel("Distance metric")
            ax.legend()
            plt.show()

            fig, ax = plt.subplots()
            for _, row in dist_metrics_df.iterrows():
                ax.scatter(
                    x=row["Hausdorff_distance_rev"],
                    y=row["Frechet_distance"],
                    color=color_dict[row["line_2"]],
                    marker="o",
                    s=30,
                    label=f"{row['line_2']} Hausdorff (reversed) vs. Frechet",
                    linewidth=1,
                    alpha=0.7,
                )
                ax.scatter(
                    x=row["Hausdorff_distance_rev"],
                    y=row["DTW_distance"],
                    marker="D",
                    s=30,
                    edgecolors=color_dict[row["line_2"]],
                    facecolor="none",
                    label=f"{row['line_2']} Hausdorff (reversed) vs. DTW",
                    linewidth=1,
                    alpha=0.7,
                )
            ax.plot(ax.get_ylim(), ax.get_ylim(), ls=":", c="black")
            ax.axhline(0, ls="--", c="lightgrey")
            ax.axvline(0, ls="--", c="lightgrey")
            ax.set_xlabel("Hausdorff distance (inverted comparison)")
            ax.set_ylabel("Distance metric")
            ax.legend()
            plt.show()

        return dist_metrics_df


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
