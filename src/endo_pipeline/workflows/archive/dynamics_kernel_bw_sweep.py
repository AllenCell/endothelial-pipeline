def main(
    dataset_names: str | list[str] | None = None,
    model_name: str = "diffae_04_10",
    bw_range: list[float] | None = None,
) -> None:
    """
    Get and visualize data-driven flow fields for
    all datasets in the manifest using various kernel bandwidths.

    Includes model summary and comparison to data as used in, e.g.,
    the `summarize_sde` workflow.
    """
    import numpy as np

    from src.endo_pipeline.configs import dynamics_io
    from src.endo_pipeline.io import get_output_path
    from src.endo_pipeline.library.analyze.diffae_features import get_and_analyze_ddff
    from src.endo_pipeline.library.analyze.diffae_manifest import fit_pca
    from src.endo_pipeline.manifests import load_dataframe_manifest

    from .data_driven_dynamics_summary import _get_and_analyze_ddd

    # if not provided in command line, run
    # on default list of datasets
    if dataset_names is None:
        dataset_names = [
            "20241120_20X",  # 48hr High
            "20241217_20X",  # 48hr No
            "20250409_20X",  # 45hr Low
            "20250319_20X",  # 45hr 12 dyn
            "20250326_20X",  # 45hr 15 dyn
        ]
    elif isinstance(dataset_names, str):
        # if a single dataset is specified, convert to list
        dataset_names = [dataset_names]

    # for building output paths
    # if directory does not exist, get_output_path
    # function will create it
    workflow_name = "kernel_sweep"

    # fit PCA to reference timepoints of
    # reference datasets
    pca = fit_pca(model_name=model_name)

    # set args for 3D viz
    # get time between frames
    dynamics_config = dynamics_io.load_dynamics_config("default")
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

    # set kernel type
    kernel_type = "gaussian"

    # range of bandwidths to test
    # optional command line argument
    # if not provided, use default
    # default: log scale between 0.025 and 0.25
    if bw_range is not None:
        if len(bw_range) != 2:
            raise ValueError("bw_range must be a list of two floats.")
        logspace_bw = np.logspace(np.log10(bw_range[0]), np.log10(bw_range[1]), num=7)
    else:
        logspace_bw = np.logspace(np.log10(0.025), np.log10(0.15), num=7)

    #################### Load model manifest data ###################
    # get model config from model name
    manifest = load_dataframe_manifest(model_name)

    # loop over bandwidths
    for bw in logspace_bw:
        kernel_params = {
            "kernel": kernel_type,
            "bandwidth": bw,
        }
        print(f"Running analysis for kernel bandwidth {bw:.3f} \n")

        # make save directory for workflow outputs
        # get string of bandwidth rounded
        # to 3 decimal places
        # and get only the decimal part
        bw_str = f"{bw:.3f}".split(".")[1]
        fig_savedir_kernel = get_output_path(
            workflow_name, model_name, f"bw_{bw_str}", "figs", include_timestamp=False
        )
        output_savedir_kernel = get_output_path(
            workflow_name, model_name, f"bw_{bw_str}", "outputs", include_timestamp=False
        )
        vtk_savedir_kernel = get_output_path(
            workflow_name, model_name, f"bw_{bw_str}", "outputs", "vtks", include_timestamp=False
        )

        # loop through datasets, get flow field
        # estimates, and save out figures
        for dataset_name in dataset_names:
            print(
                "\nComputing 2D drift and diffusion fields",
                f"for dataset {dataset_name}",
            )

            # 2D viz outputs
            _get_and_analyze_ddd(
                dataset_name, manifest, pca, kernel_params, fig_savedir_kernel, dynamics_config
            )

        print("\nRunning 3D flow field estimation workflow for all datasets. \n")
        # 3D viz outputs
        get_and_analyze_ddff(
            dataset_names,
            manifest,
            pca,
            kernel_params,
            dt,
            time_span,
            init,
            fig_savedir_kernel,
            vtk_savedir_kernel,
            output_savedir_kernel,
        )


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
