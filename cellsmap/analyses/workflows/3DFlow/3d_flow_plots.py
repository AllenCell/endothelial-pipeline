# Exploring the pre-computed features generated in the endo project by the diffusion autoencoder.
# This code generates the figures presented APS March Meeting 2025

#%%
import vtk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import filters as skfilt
from sklearn import cluster as skcluster
from scipy import interpolate as spinterp
from vtk.util import numpy_support as vtknp
from sklearn import decomposition as skdecomp
import tools
#%%

#%%
df = pd.read_csv("/allen/aics/assay-dev/users/Erin/endo_features/pca_ref_features.csv", index_col=0)
df.head()
#%%

#%%
# Exclude bad no flow dataset
df = df.loc[df["group"] != "20241210_20X_timelapse_SLDY"]
#%%

#%%
# What the data looks like?
fig, ax = plt.subplots(1,1, figsize=(5,5))
for (group, dfs) in df.groupby("group"):
    ax.scatter(dfs["1"], dfs["4"], s=0.1, label=group)
plt.legend()
plt.show()
#%%

#%%
# Matheus' blubbles classifier. Maybe to be replaced with something else
vals = tools.simple_linear_classifier(X=df["1"].values, Y=df["4"].values)
fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.scatter(df["1"], df["4"], c=vals, s=0.2)
plt.show()
#%%

#%%
print(df.shape)
df["outlier"] = vals
df = df.loc[df.outlier==False]
print(df.shape)
#%%

#%%
# Apply PCA
X = df[[str(u) for u in range(8)]].values
reducer = skdecomp.PCA(n_components=3)
Xt = reducer.fit_transform(X)
#%%

#%%
# Create unique ID for each crop
for pc in range(3):
    df[f"PC{pc+1}"] = Xt[:, pc]
df["CropId"] = df["group"] + "_" + df["FOV_ID"].astype(str) + "_" + df["start_x"].astype(str) + "_" + df["start_y"].astype(str)
df.head()
#%%

#%%
# Check datasets available
df.description.unique()
#%%

#%%
# Compute data bounds
xmin, xmax = np.percentile(df.PC1, [0.1, 99.9])
ymin, ymax = np.percentile(df.PC2, [0.1, 99.9])
zmin, zmax = np.percentile(df.PC3, [0.1, 99.9])
#%%

#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.scatter(df.PC1, df.PC2, cmap="inferno", s=0.01, c=df["T"])
ax2.scatter(df.PC1, df.PC3, cmap="inferno", s=0.01, c=df["T"])
for ax, ylab in zip([ax1, ax2], ["PC2", "PC3"]):
    ax.set_xlabel("PC1", fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_aspect("equal")
plt.tight_layout()
plt.show()
#%%

#%%
# Display one crop trajectory for each condition
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(df.PC1, df.PC2, cmap="inferno", s=0.1, color="black", alpha=0.05)
for group, df_group in df.groupby("group"):
    for track, df_track in df_group.groupby("CropId"):
        ax.plot(df_track.PC1, df_track.PC2, label=group)
        break
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")
plt.legend()
plt.show()
#%%

#%%
df = df.sort_values(by=["CropId", "T"])
df.head()
#%%

#%%
# Display state space
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
for group, df_group in df.groupby("group"):
    coords = df_group[[f"PC{u+1}" for u in range(3)]].values
    for u, umax in zip(range(3), [xmax, ymax, zmax]):
        coords[:, u] = (umax-coords[:, u])/grid_spacing
    tools.save_points_as_polydata(coordinates=coords, file_name=f"Dataset_{group}.vtk")
    ax1.scatter(df_group.PC1, df_group.PC2, s=0.5, alpha=0.05)
    ax2.scatter(df_group.PC1, df_group.PC3, s=0.5, alpha=0.05)
    for axid, ax in enumerate([ax1, ax2]):
        ax.set_xlabel("PC1", fontsize=14)
        ax.set_ylabel(f"PC{axid+2}", fontsize=14)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
plt.tight_layout()
plt.show()
#%%

#%%
# Compute landscapes
time_step = 1
grid_spacing = 0.05
for condition, fname in zip(["48hr High", "48hr Low", "48hr No Flow (12/17/24)"], ["high", "low", "no"]):
    dUis, dVis, dQis, mean_speed, grid = tools.run_flow_field_workflow(df=df, condition=condition, time_step=time_step)
    img = tools.create_vector_field_imagedata(dUis, dVis, dQis)
    tools.save_image_data(img, file_name=f"{fname}.vtk")
#%%
    
#%%
# Sample no flow at early timepoints points for setting initial condition of the simulations
buffer = 0.1
df_initial = df.loc[
    (df.description=="48hr No Flow (12/17/24)")&
    (df["T"]<50)&
    (df.PC1>(1-buffer)*xmin)&
    (df.PC1<(1-buffer)*xmax)&
    (df.PC2>(1-buffer)*ymin)&
    (df.PC2<(1-buffer)*ymax)&
    (df.PC3>(1-buffer)*zmin)&
    (df.PC3<(1-buffer)*zmax)
].sample(n=500)
print(df_initial[["PC1", "PC2", "PC3"]].values.min(axis=0))
print(df_initial[["PC1", "PC2", "PC3"]].values.max(axis=0))
#%%

#%%
# Display sampled points
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
for group, df_group in df.groupby("group"):
    coords = df_group[[f"PC{u+1}" for u in range(3)]].values
    for u, umax in zip(range(3), [xmax, ymax, zmax]):
        coords[:, u] = (umax-coords[:, u])/grid_spacing
    tools.save_points_as_polydata(coordinates=coords, file_name=f"Dataset_{group}.vtk")
    ax1.scatter(df_group.PC1, df_group.PC2, s=0.5, alpha=0.05)
    ax1.scatter(df_initial.PC1, df_initial.PC2, s=2, alpha=1, color="black")
    ax2.scatter(df_group.PC1, df_group.PC3, s=0.5, alpha=0.05)
    ax2.scatter(df_initial.PC1, df_initial.PC3, s=2, alpha=1, color="black")
    for axid, ax in enumerate([ax1, ax2]):
        ax.set_xlabel("PC1", fontsize=14)
        ax.set_ylabel(f"PC{axid+2}", fontsize=14)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
plt.tight_layout()
plt.show()
#%%

#%%
# Run simulation
initial_coords = df_initial[["PC1", "PC2", "PC3"]].values
for u, (vmin, vmax) in enumerate([(xmin, xmax), (ymin, ymax), (zmin, zmax)]):
    initial_coords[:, u] = (vmax-initial_coords[:, u])/grid_spacing
print(initial_coords.max(axis=0))
#%%

#%%
# Visualize final state of the simulation
dUis, dVis, dQis, mean_speed, grid = tools.run_flow_field_workflow(df=df, condition="48hr No Flow (12/17/24)", time_step=time_step)
trajs = tools.simulate_particles_in_vector_field(dUis=dUis, dVis=dVis, dQis=dQis, grid=grid, n_particles=500, speed=mean_speed, grid_spacing=grid_spacing, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, initial_coords=initial_coords)
#%%

#%%
# Visualize mean trajectory (red cross is initial state)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
for tid, traj in enumerate(trajs):
    rmean = np.array(traj).mean(axis=0)
    xp = xmin + rmean[0]*grid_spacing
    yp = ymin + rmean[1]*grid_spacing
    zp = zmin + rmean[2]*grid_spacing
    ax1.scatter(xp, yp, color="black", s=10)
    ax2.scatter(xp, zp, color="black", s=10)
    if tid == 0:
        ax1.scatter(xmin+rmean[0]*grid_spacing, ymin+rmean[1]*grid_spacing, color="red", marker="X", s=50)
        ax2.scatter(xmin+rmean[0]*grid_spacing, zmin+rmean[2]*grid_spacing, color="red", marker="X", s=50)
    for axid, (ax, (vmin, vmax)) in enumerate(zip([ax1, ax2], [(ymin,ymax), (zmin,zmax)])):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("PC1", fontsize=14)
        ax.set_ylabel(f"PC{axid+2}", fontsize=14)
        ax.set_aspect("equal")
plt.tight_layout()
plt.show()
#%%

#%%
# Invert final coordinate to get 8D representation
print("8D coordinates of final state:")
print(reducer.inverse_transform([xp, yp, zp]))
#%%

#%%
# Compute and plot speed
path = []
for tid, traj in enumerate(trajs):
    rmean = np.array(traj).mean(axis=0)
    xp = xmin + rmean[0]*grid_spacing
    yp = ymin + rmean[1]*grid_spacing
    zp = zmin + rmean[2]*grid_spacing
    path.append([xp, yp, zp])
path = np.array(path)
#%%

#%%
speed = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.plot(speed)
ax.set_xlabel("Simulation frame")
ax.set_ylabel("Speed (unit of length \n in state space/frame)")
plt.show()
#%%

#%%
# Run simulation for changing flow
dUis, dVis, dQis, mean_speed, grid = tools.run_flow_field_workflow(df=df, condition="48hr High", time_step=time_step)
dUis2, dVis2, dQis2, mean_speed, grid = tools.run_flow_field_workflow(df=df, condition="48hr Low", time_step=time_step)
tools.simulate_particles_in_changing_vector_field(dUis=dUis, dVis=dVis, dQis=dQis, transition=50, dUis2=dUis2, dVis2=dVis2, dQis2=dQis2, grid=grid, n_particles=500, speed=mean_speed, grid_spacing=grid_spacing, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax)
#%%