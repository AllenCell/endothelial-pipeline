import fire
import numpy as np

from cellsmap.analyses.utils.io import dynamics_io
from cellsmap.analyses.utils.numerics import data_driven_flow_field as ddff
from cellsmap.analyses.utils.viz import manifest_viz
from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.util.manifest_preprocessing import manifest_pca
from cellsmap.util.set_output import get_output_path


def main(datasets_to_use: list | None = None) -> None:
    """
    Visualize 3D (drift) flow fields for the dynamics of the
    DiffAE crop-based features for each of the single flow datasets.
    """
    # Create output folder if does not exist yet
    workflow_fig_folder = "flow_field_3d/figs"
    workflow_output_folder = "flow_field_3d/outputs"
    workflow_vtk_folder = "flow_field_3d/outputs/vtks"
    output_savedir = get_output_path(workflow_output_folder, verbose=False)
    fig_savedir = get_output_path(workflow_fig_folder, verbose=False)
    vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)

    # if not provided in command line, run
    # on default list of datasets
    if datasets_to_use is None:
        datasets_to_use = [
            "20241120_20X",
            "20250409_20X",
            "20241217_20X",
            "20250319_20X",
            "20250326_20X",
        ]
    pca = manifest_pca.fit_pca()

    # plot scatter of PCA components
    # for a) just the datasets used to fit PCA
    # and b) all datasets including intermediate shear stress
    fig, _ = manifest_viz.plot_pc_scatter(
        pca,
        datasets_to_use[:3],  # first three datasets
    )
    vb.save_plot(fig, fig_savedir + "/pca_scatter_ref")
    fig, _ = manifest_viz.plot_pc_scatter(
        pca,
        datasets_to_use,  # last two datasets
    )
    vb.save_plot(fig, fig_savedir + "/pca_scatter_all")

    # load default config, get kernel params
    config = dynamics_io.load_dynamics_config("default")
    kernel_params = config["kramers_moyal"]["kernel_params"]

    # get time between frames
    # in minutes
    dt = config["dt"]

    # time span for the ODE solver
    # units for time steps are in minutes
    # 48 hours in minutes =
    # 48 * 60 = 2880 time steps
    time_span = [0, 2880]

    # initial condition for the ODE solver
    # this is fixed across datasets /
    # shear stress conditions
    init = np.array([-0.1, -0.7, -0.1])

    ddff.ddff_main(
        datasets_to_use,
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
