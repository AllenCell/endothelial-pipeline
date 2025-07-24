from pathlib import Path

import fire
import numpy as np
from sklearn.pipeline import Pipeline

from src.endo_pipeline.configs import (
    ModelManifest,
    dynamics_io,
    get_model_manifest,
    get_pca_reference_model_manifests,
    load_dataset_config,
    load_model_config,
)
from src.endo_pipeline.io import get_output_path, save_plot_to_path
from src.endo_pipeline.library.analyze.diffae_features import (
    compute_extrapolated_vector_field,
    get_traj_and_diff,
    solve_ddff_ode,
)
from src.endo_pipeline.library.analyze.diffae_manifest import (
    fit_pca,
    get_dataset_descriptions,
    get_manifest_for_dynamics_workflows,
    get_pc_column_names,
    get_timepoints_for_plotting_pcs,
)
from src.endo_pipeline.library.analyze.kramersmoyal import get_kramers_moyal
from src.endo_pipeline.library.analyze.numerics import get_3d_bounds_from_data, get_bins
from src.endo_pipeline.library.visualize.diffae_features import flow_field_viz, manifest_viz, vtk_io


def _ddff_model_analysis(
    model_manifest: ModelManifest,
    pca: Pipeline,
    kernel_params: dict,
    dt: float,
    bins: list[np.ndarray],
    centers: list[np.ndarray],
    time_span: list,
    init: np.ndarray,
    fig_savedir: Path,
    vtk_savedir: Path,
    output_savedir: Path,
) -> np.ndarray | list[np.ndarray]:
    """
    Get 3D flow field (drift coefficient) from data
    projected onto the 3D principal component feature space
    and output summary figures and vtk files for visualization.

    Inputs:
    - model_manifest: ModelManifest object for the dataset
    - pca: PCA model to use for transforming the data
    - kernel_params: parameters for the kernel-based
        estimation of Kramers-Moyal coefficients
    - dt: time step for the Kramers-Moyal coefficients
    - bins: list of numpy arrays with the bin edges
        for each dimension in the 3D state space
        (computed via get_bins)
    - centers: list of numpy arrays with the
        centers of the bins in each dimension
        (computed via get_bins)
    - time_span: time span for the ODE solver
        (list of two floats)
    - init: initial condition for the trajectory
        (numpy array of shape (3,))
    - fig_savedir: directory to save figures
    - vtk_savedir: directory to save vtk files
    - output_savedir: directory to save output files
        (.npy files with flow field and diffusion field)

    Outputs:
    - traj: trajectory in 3D state space for the
        given initial condition and time span
        according to the dynamics given by the
        approximated flow field for the dataset
        (numpy array of shape (num_t, 3))
        - if name is "20250319_20X" or "20250326_20X",
            returns a list of two trajectories
            (trajectories going towards each of the
            two stable fixed points for these conditions)
    """
    # load dataframe and get top 3 PCs
    df = get_manifest_for_dynamics_workflows(model_manifest, pca)
    pc_column_names = get_pc_column_names(df, pc_axes=[0, 1, 2])

    # get list of per-crop trajectories, the corresponding
    # displacement vectors, and time differences
    traj_list, d_traj_list = get_traj_and_diff(df, pc_column_names)
    # get drift and diffusion estimates
    # (Kramers-Moyal coefficients)
    drift_km, diff_km = get_kramers_moyal(
        traj_list, d_traj_list, bins=bins, dt=dt, kernel_params=kernel_params
    )

    # compute interpolated flow field - drift
    flow_field_dict = compute_extrapolated_vector_field(drift_km, centers, interpolator="nearest")
    # save flow field dictionary as npy
    np.save(
        output_savedir / f"flow_field_dict_{model_manifest.dataset_name}.npy",
        flow_field_dict,  # type: ignore
        allow_pickle=True,
    )
    # save flow field as vtk image data
    vtk_io.save_vector_field_as_vtk(
        flow_field_dict, vtk_savedir / f"flow_field_{model_manifest.dataset_name}.vtk"
    )

    # compute interpolated diffusion field
    # (diagonal diffusion tensor represented as 3D vector field)
    diffusion_field_dict = compute_extrapolated_vector_field(
        diff_km, centers, interpolator="nearest"
    )
    # save diffusion field dictionary as npy
    np.save(
        output_savedir / f"diffusion_field_dict_{model_manifest.dataset_name}.npy",
        diffusion_field_dict,  # type: ignore
        allow_pickle=True,
    )
    # save diffusion field as vtk image data
    vtk_io.save_vector_field_as_vtk(
        diffusion_field_dict, vtk_savedir / f"diffusion_field_{model_manifest.dataset_name}.vtk"
    )

    ## ODE solver: dx/dt = f(x) (drift, first Kramers-Moyal coefficient) ##
    # with initial conditions given by init
    # solve IVP, get back trajectory
    traj = solve_ddff_ode(flow_field_dict, init, time_span)

    # call main flow field viz function (makes and saves plots)
    flow_field_viz.flow_field_viz_main(flow_field_dict, df, traj, fig_savedir)

    # hack-y work around for intermediate shear stress
    # simulate second trajectory to get second stable point
    if model_manifest.dataset_name == "20250319_20X":
        init = np.array([1.1, 0.0, -0.2])
        time_span = [0, 5000]
        traj_2 = solve_ddff_ode(flow_field_dict, init, time_span)
        traj_list = [traj, traj_2]  # return both trajectories
        return traj_list
    else:
        return traj


def _get_and_analyze_ddff(
    model_manifest_list: list[ModelManifest],
    pca: Pipeline,
    kernel_params: dict,
    dt: float,
    time_span: list,
    init: np.ndarray,
    fig_savedir: Path,
    vtk_savedir: Path,
    output_savedir: Path,
) -> None:
    """
    Run main workflow for computing and visualizing
    the "data-driven flow field" (DDFF) for a list of datasets.

    Inputs:
    - model_manifest_list: list of ModelManifest objects
        - each manifest contains the dataset name and
            the fmsid of the model manifest for the dataset
    - pca: PCA model to use for transforming the data
    - kernel_params: parameters for the kernel-based
        estimation of Kramers-Moyal coefficients
    - dt: time step for the Kramers-Moyal coefficients
    - time_span: time span for the ODE solver
    - init: initial condition for the trajectory
    - fig_savedir: directory to save figures
    - vtk_savedir: directory to save vtk files
    - output_savedir: directory to save other output files

    Outputs:
    - None.
    - This function saves out the trajectories for each dataset
        in a dictionary, where keys are dataset descriptions
        and values are trajectories in 3D state space.
        - see docstring for `_ddff_model_analysis` for details
            of what other files are saved out for each dataset
    """
    # get bins for KMCs
    bounds = get_3d_bounds_from_data(model_manifest_list, pca)
    num_bins = [50, 50, 50]
    bins, centers = get_bins(num_bins, bin_limits=bounds)

    # get experimental condition
    # descriptions of each dataset
    condition_dict = get_dataset_descriptions(
        [model_manifest.dataset_name for model_manifest in model_manifest_list], simple=True
    )

    # initialize dict to save trajectories
    # used for crop reconstruction
    traj_dict = {}
    for model_manifest in model_manifest_list:
        print(f"******** Processing dataset: {model_manifest.dataset_name} ******** \n")
        traj = _ddff_model_analysis(
            model_manifest,
            pca,
            kernel_params,
            dt,
            bins,
            centers,
            time_span,
            init,
            fig_savedir,
            vtk_savedir,
            output_savedir,
        )

        # save out using dataset descriptions
        condition = condition_dict[model_manifest.dataset_name]
        traj_dict[condition] = traj

    np.save(output_savedir / "traj_dict", traj_dict, allow_pickle=True)  # type: ignore

    # generate plot of stable fixed points
    # for low, high, and 12dyn datasets
    flow_field_viz.plot_stable_fixed_points_together(fig_savedir, output_savedir)


def main(dataset_names: str | list[str] | None = None, model_name: str = "diffae_04_10") -> None:
    """
    Visualize 3D (drift) flow fields for the dynamics of the
    DiffAE crop-based features for each of the single flow datasets.
    """
    # Create output folder if does not exist yet
    workflow_name = "flow_field_3d"
    output_savedir = get_output_path(workflow_name, model_name, "outputs", include_timestamp=False)
    fig_savedir = get_output_path(workflow_name, model_name, "figs", include_timestamp=False)
    vtk_savedir = get_output_path(
        workflow_name, model_name, "outputs", "vtk", include_timestamp=False
    )

    if isinstance(dataset_names, str):
        # if a single dataset is provided, convert to list
        dataset_names = [dataset_names]
    elif dataset_names is None:
        # if not provided in command line, run
        # on default list of datasets
        dataset_names = [
            "20241120_20X",
            "20250409_20X",
            "20241217_20X",
            "20250428_20X",
            "20250319_20X",
            "20250326_20X",
        ]
    pca = fit_pca(model_name=model_name)

    # plot scatter of PCA components
    # for a) just the datasets used to fit PCA
    # and b) all datasets specified in the command line
    #   (or default list, if not specified)
    # get timepoints to use for scatter plots
    # all timepoints except no flow
    model_config = load_model_config(model_name)
    pca_ref_model_manifest_list = get_pca_reference_model_manifests(model_config)
    pca_ref_configs = [
        load_dataset_config(model_manifest.dataset_name)
        for model_manifest in pca_ref_model_manifest_list
    ]
    restrict_no_flow = True  # restrict plot to subset of no flow timepoints

    # get timepoints to use for scatter plots
    # this can definitely be written into a wrapper function
    # maybe make a dictionary instead of a list?
    timepoints_refs = get_timepoints_for_plotting_pcs(
        pca_ref_configs, restrict_no_flow=restrict_no_flow
    )

    # scatter plot of pca reference datasets
    fig, _ = manifest_viz.plot_pc_scatter(
        pca, pca_ref_model_manifest_list, timepoints_to_use=timepoints_refs
    )
    save_plot_to_path(fig, fig_savedir, "pca_scatter_ref")

    # scatter plot of all datasets specified in command line
    model_manifest_list = [
        get_model_manifest(dataset_name, model_config) for dataset_name in dataset_names
    ]
    fig, _ = manifest_viz.plot_pc_scatter(
        pca,
        model_manifest_list,  # all datasets specified and all timepoints
    )
    save_plot_to_path(fig, fig_savedir, "pca_scatter_all")

    # load default config, get kernel params
    dynamics_config = dynamics_io.load_dynamics_config("default")
    kernel_params = dynamics_config["kramers_moyal"]["kernel_params"]

    # get time between frames
    # in minutes
    dt = dynamics_config["dt"]

    # time span for the ODE solver
    # units for time steps are in minutes
    # 48 hours in minutes =
    # 48 * 60 = 2880 time steps
    time_span = [0, 2880]

    # initial condition for the ODE solver
    # this is fixed across datasets /
    # shear stress conditions
    init = np.array([-0.1, -0.7, -0.1])

    _get_and_analyze_ddff(
        model_manifest_list,
        pca,
        kernel_params,
        dt,
        time_span,
        init,
        fig_savedir,
        vtk_savedir,
        output_savedir,
    )


if __name__ == "__main__":
    fire.Fire(main)
