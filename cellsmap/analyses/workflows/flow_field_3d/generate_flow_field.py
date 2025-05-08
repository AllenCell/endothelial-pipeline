# %%
import numpy as np
import pandas as pd

from cellsmap.analyses.utils import regression_helper as rh
from cellsmap.analyses.utils.io import dynamics_io, vtk_io
from cellsmap.analyses.utils.numerics import data_driven_flow_field as ddff
from cellsmap.analyses.utils.viz import flow_field_viz as ffv
from cellsmap.util import manifest_io
from cellsmap.util.set_output import get_output_path

# %%
# Create output folder if does not exist yet
workflow_fig_folder = "flow_field_3d/figs"
workflow_output_folder = "flow_field_3d/outputs"
workflow_vtk_folder = "flow_field_3d/outputs/vtks"
output_savedir = get_output_path(workflow_output_folder, verbose=False)
fig_savedir = get_output_path(workflow_fig_folder, verbose=False)
vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)


# load pca model
pca = manifest_io.load_pca_model(output_savedir)
# load manifest created at preprocessing step
df = pd.read_csv(output_savedir + "manifest.csv")
# %%
# hardcoded for now, would be great to get these into a config file
# Create flow field dx/dt = f(x)
config = dynamics_io.load_dynamics_config("default")
kernel_params = config["kramers_moyal"]["kernel_params"]

feat_cols = [f"pc{i+1}" for i in range(3)]

# get state space bounds and grid resolution for estimating flow field
excluded_fraction = 0.00
bounds = ddff.set_3d_bounds_from_data(
    df.pc1, df.pc2, df.pc3, excluded_fraction=excluded_fraction
)
grid_spacing = 0.05
Nbins = [
    int((bounds[0][1] - bounds[0][0]) / grid_spacing) + 1,
    int((bounds[1][1] - bounds[1][0]) / grid_spacing) + 1,
    int((bounds[2][1] - bounds[2][0]) / grid_spacing) + 1,
]

bins, centers = rh.get_bins(Nbins, bin_limits=bounds)

# time stepping for the flow field
dt = 5
# time span for the ODE solver
# units for time steps are in minutes
# 48 hours in minutes =
# 48 * 60 = 2880 time steps
t_span = [0, 2880]
traj_dict = {}

# %%
# compute flow field via first Kramers-Moyal coefficient (drift)

for condition, df_ in df.groupby("description"):
    # get dataset name and condition
    print(f"Computing drift flow field for {condition}")

    # get list of per-crop trajectories, the corresponding
    # displacement vectors, and time differences
    traj_list, d_traj_list = rh.get_traj_and_diff(df_, feat_cols)
    # get drift and diffusion estimates
    # (Kramers-Moyal coefficients)
    drift_km, diff_km = rh.get_kramers_moyal(
        traj_list, d_traj_list, bins=bins, dt=dt, kernel_params=kernel_params
    )

    # compute interpolated flow field - drift
    flow_field_dict = ddff.compute_extrapolated_vector_field(
        drift_km, centers, interpolator="nearest"
    )
    # save flow field as vtk image data
    vtk_io.save_vector_field_as_vtk(
        flow_field_dict, vtk_savedir + f"flow_field_{condition}.vtk"
    )

    # compute interpolated diffusion field
    # (diagonal diffusion tensor represented as 3D vector field)
    diffusion_field_dict = ddff.compute_extrapolated_vector_field(
        diff_km, centers, interpolator="nearest"
    )
    # save diffusion field as vtk image data
    vtk_io.save_vector_field_as_vtk(
        diffusion_field_dict, vtk_savedir + f"diffusion_field_{condition}.vtk"
    )

    ## ODE solver: dx/dt = f(x) (drift, first Kramers-Moyal coefficient) ##
    # with initial conditions given by the mean of the data at T=0

    # get initial conditions for the ODE solver from data
    inits_mean = (
        df_.groupby("frame_number").mean(numeric_only=True)[feat_cols].values[0]
    )

    # solve IVP, get back trajectory
    traj = ddff.solve_ddff_ode(flow_field_dict, inits_mean, t_span)

    # trajectory to dictionary - saved out and used later to reconstruct crops
    traj_dict[condition] = traj

    # call main flow field viz function (makes and saves plots)
    ffv.flow_field_viz_main(flow_field_dict, df_, traj, fig_savedir)


# %%
# save out dictionary of mean trajectories as npy file
np.save(output_savedir + "traj_dict", traj_dict, allow_pickle=True)

# %%
