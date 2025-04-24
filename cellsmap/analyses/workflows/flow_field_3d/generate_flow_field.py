# Exploring the pre-computed features generated in the endo project by the diffusion autoencoder.
# This code generates the figures presented APS March Meeting 2025
# %%
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from cellsmap.util.set_output import get_output_path
from cellsmap.analyses.utils.io import vtk_tools
from cellsmap.analyses.utils import regression_helper as rh
# Create output folder if does not exist yet
workflow_fig_folder = "flow_field_3d/figs"
workflow_output_folder = "flow_field_3d/outputs"
workflow_vtk_folder = "flow_field_3d/outputs/vtks"
output_savedir = get_output_path(workflow_output_folder, verbose=False)
fig_savedir = get_output_path(workflow_fig_folder, verbose=False)
vtk_savedir = get_output_path(workflow_vtk_folder, verbose=False)

# Load manifest created at preprocessing step
df = pd.read_csv(output_savedir+"manifest.csv")

# %%
# Create flow field dx/dt = f(x)
DDFF = vtk_tools.DataDrivenFlowField3D_EA(verbose=True)
DDFF.set_output_folders(fig_output_folder=fig_savedir, vtk_output_folder=vtk_savedir)
DDFF.set_dataframe(df)
DDFF.set_state_space_variables(["PC1", "PC2", "PC3"])
DDFF.build()

# %%
for condition in df.description.unique():
    DDFF.compute_flow_field(condition=condition)

    # points and velocities
    f_KM = DDFF._drift_kmcs
    centers = DDFF._bin_centers

    f_KM_, X_, = rh.masked_vector_field(f_KM, np.array(np.meshgrid(*centers)).T)
    
    # train test split of data
    X_train, X_test, Y_train, Y_test =train_test_split(X_, f_KM_, train_size=0.8, random_state=42)

    f1_mdl = Pipeline([('poly', PolynomialFeatures(degree=4)),
                   ('linear', LinearRegression())]).fit(X_train,Y_train)

    # polynomial regression on drift


# %%
