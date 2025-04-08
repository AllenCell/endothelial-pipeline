# %%
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
import numdifftools as nd

from cellsmap.util.set_output import get_output_path
from cellsmap.util import manifest_io as mio
from cellsmap.analyses.utils.io import dynamics_io
from cellsmap.analyses.utils import manifest_pca, regression_helper as rh, model_analysis, model_eval
from cellsmap.analyses.utils.viz import manifest_viz, pplane
from cellsmap.analyses.utils.numerics import gen_potential as gp

import sys
sys.path.append('//allen/aics/users/erin.angelini/git-repos/KramersMoyal')
import kramersmoyal as km
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
# %%
# load manifest to DataFrame with metadata
df = mio.load_manifest_to_df()

# fit PCA to data
pca = manifest_pca.fit_pca(df, num_pcs=8)
df.head()
list_of_datasets = mio.get_list_of_datasets(df,verbose=False)
# %%
################### Visualize PCA results ###################
# plot explained variance ratio of PCA components
fig, _ = manifest_viz.plot_explained_variance(pca['pca'].explained_variance_ratio_)

# plot top 3 principal components of feature data vs. frame number
fig, _ = manifest_viz.plot_top_3_PCs_alldata(df,pca)
# %%
ds_ID = 2
ds_name = list_of_datasets[ds_ID]
feats_proj = mio.project_PCA_one_dataset(df,pca,ds_name)
PCs = [0,1]
data_all, u_list = rh.get_X_by_flow(feats_proj,ds_name,verbose=True)
num_flow = len(u_list)
print("Number of flow conditions in dataset: ",num_flow)
# %%
ndim = len(PCs)
# Choose the size of your target space in two dimensions 
bins = 15*np.ones(ndim, dtype = int)

# Introduce the desired orders to calculate, but in 2 dimensions
# Please keep the [0,0] term. It is the normalisation. 
powers = np.array([[0,0], [1,0], [0,1], [1,1], [2,0], [0,2], [2,2]])
# insert into kmc:   0      1      2      3      4      5      6

# Notice that the first entry in [,] is for the first dimension, the 
# second for the second dimension...

# Choose a desired bandwidth bw
bw = 0.1

# Calculate the Kramers−Moyal coefficients
flow_ID = 0
if flow_ID == 1 and num_flow == 1:
    print('Only one flow condition in dataset, using flow_ID = 0')
    flow_ID = 0

X_list, dX_list, dT_list = rh.get_X_dX_and_dT(data_all[flow_ID],feat_cols=[str(i) for i in PCs])

kmc, edges = km.km(X_list, dX_list, bw = bw, bins = bins, 
                    powers = powers, multi_traj=True)

# %%
#lets fix a range where we have good statistics and generate a meshgrid
idx0 = 3
idx1 = -3
X_1, X_2 = np.meshgrid(edges[0][idx0:idx1],edges[1][idx0:idx1])

# drift coefficients
fig, ax = plt.subplots(1,2,figsize=(12,6))
ax[0].quiver(X_1,X_2,
             kmc[1,idx0:idx1,idx0:idx1].T / 5,
             kmc[2,idx0:idx1,idx0:idx1].T /5)
ax[1].streamplot(X_1,X_2,
                 kmc[1,idx0:idx1,idx0:idx1].T / 5,
                 kmc[2,idx0:idx1,idx0:idx1].T /5)

# %%
# get 2D array of (unique) points in the meshgrid X_1, X_2
X = np.array([X_1.flatten(),X_2.flatten()]).T

# polynomial regression on Y_1 = kmc[1,idx0:idx1,idx0:idx1].T/5
Y_1 = kmc[1,idx0:idx1,idx0:idx1].T/5

f1_mdl = Pipeline([('poly', PolynomialFeatures(degree=4)),
                   ('linear', LinearRegression())]).fit(X,Y_1.flatten())

Y_1_pred = f1_mdl.predict(X)
r2_f1 = r2_score(Y_1.flatten(),Y_1_pred)
print('R^2 for polynomial regression on Y_1:',r2_f1)

# linear regression on Y_2 = kmc[2,idx0:idx1,idx0:idx1].T/5
Y_2 = kmc[2,idx0:idx1,idx0:idx1].T/5

f2_mdl = Pipeline([('poly', PolynomialFeatures(degree=4)),
                     ('linear', LinearRegression())]).fit(X,Y_2.flatten())

Y_2_pred = f2_mdl.predict(X)
r2_f2 = r2_score(Y_2.flatten(),Y_2_pred)
print('R^2 for polynomial regression on Y_2:',r2_f2)


# linear regression on V_1 = kmc[4,idx0:idx1,idx0:idx1].T/5
V_1 = kmc[4,idx0:idx1,idx0:idx1].T/5

D1_mdl = Pipeline([('poly', PolynomialFeatures(degree=2)),
                     ('linear', LinearRegression())]).fit(X,V_1.flatten())

V_1_pred = D1_mdl.predict(X)
r2_D1 = r2_score(V_1.flatten(),V_1_pred)
print('R^2 for polynomial regression on V_1:',r2_D1)


# polynomial regression on V_2 = kmc[5,idx0:idx1,idx0:idx1].T/5
V_2 = kmc[5,idx0:idx1,idx0:idx1].T/5

D2_mdl = Pipeline([('poly', PolynomialFeatures(degree=2)),
                   ('linear', LinearRegression())]).fit(X,V_2.flatten())

V_2_pred = D2_mdl.predict(X)
r2_D2 = r2_score(V_2.flatten(),V_2_pred)
print('R^2 for polynomial regression on V_2:',r2_D2)

# %%
# Initialise the figure for 3d ploting
fig = plt.figure(figsize = (12,8))


# the Kramers−Moyal coefficients [1,0]
ax_00 = fig.add_subplot(2, 2, 1, projection='3d')

# kmc[1,...] i.e, power [1,0] (transpose, python stores arrays transposed)
ax_00.contour3D(X_1, X_2, kmc[1,idx0:idx1,idx0:idx1].T / 5, 50, cmap='Greens',alpha=0.5)
ax_00.contour3D(X_1, X_2, Y_1_pred.reshape(X_1.shape), 50, cmap='Blues')
ax_00.set_title(r'K−M coefficient [1,0]');


# the Kramers−Moyal coefficients [0,1]
ax_01 = fig.add_subplot(2, 2, 2, projection='3d')

# kmc[2,...] i.e, power [0,1] (transpose, python stores arrays transposed)
ax_01.contour3D(X_1, X_2, kmc[2,idx0:idx1,idx0:idx1].T /5, 50, cmap='Greens',alpha=0.5)
ax_01.contour3D(X_1, X_2, Y_2_pred.reshape(X_1.shape), 50, cmap='Blues')
ax_01.set_title(r'K−M coefficient [0,1]');


# the Kramers−Moyal coefficients [2,0]
ax_10 = fig.add_subplot(2, 2, 3, projection='3d')

# kmc[4,...] i.e, power [2,0] (transpose, python stores arrays transposed)
ax_10.contour3D(X_1, X_2, kmc[4,idx0:idx1,idx0:idx1].T / 5, 50, cmap='Greens',alpha=0.5)
ax_10.contour3D(X_1, X_2, V_1_pred.reshape(X_1.shape), 50, cmap='Blues')
ax_10.set_title(r'K−M coefficient [2,0]');


# the Kramers−Moyal coefficients [0,2]
ax_11 = fig.add_subplot(2, 2, 4, projection='3d')

# kmc[5,...] i.e, power [0,2] (transpose, python stores arrays transposed)
ax_11.contour3D(X_1, X_2, kmc[5,idx0:idx1,idx0:idx1].T / 5, 50, cmap='Greens',alpha=0.5)
ax_11.contour3D(X_1, X_2, V_2_pred.reshape(X_1.shape), 50, cmap='Blues')

ax_11.set_title(r'K−M coefficient [0,2]');
# Rotate views and add labels
ax_00.view_init(30, 20); ax_01.view_init(30, 20); ax_10.view_init(30, 20); ax_11.view_init(30, 20)
ax_00.set_xlabel(r'$y_1$'); ax_01.set_xlabel(r'$y_1$'); ax_10.set_xlabel(r'$y_1$'); ax_11.set_xlabel(r'$y_1$')
ax_00.set_ylabel(r'$y_2$'); ax_01.set_ylabel(r'$y_2$'); ax_10.set_ylabel(r'$y_2$'); ax_11.set_ylabel(r'$y_2$')

plt.show()


# %%
# %%
# %%
# %%
# now fit model using multiple datasets
config = dynamics_io.load_dynamics_config()


# list of training/test sets for each dataset
# do this with dictionaries instead?
X_train_list = []
X_test_list = []
Y_train_list = []
Y_test_list = []
V_train_list = []
V_test_list = []
u_train_list = []
u_test_list = []

Nbins = 15*np.ones(ndim, dtype = int)
dt = 5

for ds_name  in list_of_datasets:
    if ds_name in config['datasets_to_skip']:
        continue 
    print('**** Generating train/test sets for ',ds_name,'**** \n')
    feats_proj = mio.project_PCA_one_dataset(df,pca,ds_name)

    data_all, u_list = rh.get_X_by_flow(feats_proj,ds_name)
    num_flow = len(u_list)
    del feats_proj # free up memory

    centers = []

    f_KM_ = []
    D_KM_ = []

    for j in range(num_flow): # get bins and centers for data at high and low flow
        X_list, dX_list, dT_list = rh.get_X_dX_and_dT(data_all[j],feat_cols=[str(i) for i in PCs])
        kmc, edges = km.km(X_list, dX_list, bw = bw, bins = Nbins, powers = powers, multi_traj=True)
        centers.append(edges)

        idx0 = 3
        idx1 = -3
        centers_ = [centers[j][i][idx0:idx1] for i in range(ndim)]
        X_1, X_2 = np.meshgrid(*centers_)

        f_KM_temp = np.array((kmc[1,idx0:idx1,idx0:idx1].T,
                              kmc[2,idx0:idx1,idx0:idx1].T)).T/dt
        
        fig, ax = plt.subplots()
        ax.streamplot(X_1,X_2,f_KM_temp[:,:,0].T,f_KM_temp[:,:,1].T)
        plt.show()

        D_KM_temp = np.array((kmc[4,idx0:idx1,idx0:idx1].T,
                              kmc[5,idx0:idx1,idx0:idx1].T)).T/dt

        f_KM_.append(f_KM_temp)
        D_KM_.append(D_KM_temp)

    f_KM = []
    D_KM = []
    X_pts = []

    for j in range(num_flow):
        centers_ = [centers[j][i][idx0:idx1] for i in range(ndim)]
        f_KM_temp, X_pts_temp = rh.masked_vector_field(f_KM_[j],np.array(np.meshgrid(*centers_)).T)
        D_KM_temp, _ = rh.masked_vector_field(D_KM_[j], np.array(np.meshgrid(*centers_)).T)
        f_KM.append(f_KM_temp)
        D_KM.append(D_KM_temp)
        X_pts.append(X_pts_temp)

    del f_KM_, D_KM_, centers, centers_ # free up memory

    train_frac = 0.8   
    X_train, X_test, Y_train, Y_test, V_train, V_test = rh.train_test_all(X_pts,f_KM,D_KM,train_frac=0.8)

    # get number of training and test points for each flow condition
    N_tot = [X_pts[j].shape[0] for j in range(num_flow)]
    N_train = [int(train_frac*N_tot[j]) for j in range(num_flow)]
    N_test = [N_tot[j]-N_train[j] for j in range(num_flow)]

    # get corresponding flow condition for each training and test point
    u_train = np.concatenate([u_list[j]*np.ones((N_train[j],1)) for j in range(num_flow)])
    u_test = np.concatenate([u_list[j]*np.ones((N_test[j],1)) for j in range(num_flow)])
    
    del X_pts, f_KM, D_KM # free up memory

    X_train_list.append(X_train)
    X_test_list.append(X_test)
    Y_train_list.append(Y_train)
    Y_test_list.append(Y_test)
    V_train_list.append(V_train)
    V_test_list.append(V_test)
    u_train_list.append(u_train)
    u_test_list.append(u_test)

    del X_train, X_test, Y_train, Y_test, V_train, V_test, u_train, u_test # free up memory

X_train = np.concatenate(X_train_list)
X_test = np.concatenate(X_test_list)
Y_train = np.concatenate(Y_train_list)
Y_test = np.concatenate(Y_test_list)
V_train = np.concatenate(V_train_list)
V_test = np.concatenate(V_test_list)
u_train = np.concatenate(u_train_list)
u_test = np.concatenate(u_test_list)
# %%

feature_lib = ps.PolynomialLibrary(degree=4, include_bias=True)
parameter_lib=ps.PolynomialLibrary(degree=3, include_bias=True)
full_lib=ps.ParameterizedLibrary(feature_library=feature_lib,
    parameter_library=parameter_lib,num_features=ndim,num_parameters=1)

driftModel = ps.SINDy(feature_library = full_lib, optimizer = ps.SSR())
driftModel.fit(X_train,t=dt,x_dot=Y_train,u=u_train)


diff_feature_lib=ps.PolynomialLibrary(degree=2, include_bias=True)
diff_parameter_lib=ps.PolynomialLibrary(degree=2, include_bias=True)
diff_lib=ps.ParameterizedLibrary(feature_library=diff_feature_lib,
    parameter_library=diff_parameter_lib,num_features=ndim,num_parameters=1)


diffModel = ps.SINDy(feature_library = diff_lib, optimizer = ps.SSR())
diffModel.fit(X_train,t=dt,x_dot=V_train,u=u_train)

drift_R2 = driftModel.score(X_test,x_dot=Y_test,u=u_test)
driftModel.print()

print('Coefficient of determination (R^2) for drift coefficient model on test set: %f' %drift_R2)

diff_R2 = diffModel.score(X_test,x_dot=V_test,u=u_test)
diffModel.print()

print('Coefficient of determination (R^2) for diffusion coefficient model on test set: %f' %diff_R2)
# %%

pplane_xlim = [-4,4]
bin_xlim = [-5,5]

pplane_ylim = [-1.5,3.5]
bin_ylim = [-2,4]


# fix bins and centers for all datasets
Nbins = [50 for i in range(ndim)]
bin_limits = [bin_xlim,bin_ylim]
bins, centers = rh.get_bins(Nbins,bin_limits=bin_limits)

f = model_eval.vector_field_function(driftModel)
D = model_eval.vector_field_function(diffModel)
myModel = [f,D]


# %%
workflow_fig_folder = "stochastic_dynamics/km_test/figs"
fig_savedir = get_output_path(workflow_fig_folder,verbose=False)

ds_to_skip = config['datasets_to_skip']
pplane_xvec = np.linspace(*pplane_xlim,50)
pplane_yvec = np.linspace(*pplane_ylim,50)
model_analysis.model_data_comparison(myModel,fig_savedir,pca,PCs,bins,ds_to_skip,pplane_xvec,pplane_yvec)
        
# %%
shear_range_fpt = np.linspace(config['shear_range'][0],config['shear_range'][1],config['N_shear_fpt'])
################### Fixed point analysis ###################
# plot coordinates of fixed points as a function of shear stress
plt_lims = [pplane_xlim,pplane_ylim] # set limits for plotted/reported fixed points

model_analysis.run_fixed_point_analysis(f,shear_range_fpt,PCs,plt_lims,fig_savedir)

# %%
################### Entropy production rate as a function of shear stress ###################
model_analysis.run_epr_analysis(myModel,bins,centers,shear_range_fpt,fig_savedir,additive_noise=False)

# %%
