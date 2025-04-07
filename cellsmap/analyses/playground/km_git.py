# %%
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
import numdifftools as nd

from cellsmap.analyses.utils.io import manifest_io as mio, dynamics_io
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
ds_ID = 1
ds_name = list_of_datasets[ds_ID]
feats_proj = mio.project_PCA_one_dataset(df,pca,ds_name)
PCs = [0,1]
data_all, u_list = rh.get_X_by_flow(feats_proj,ds_name,verbose=True)
num_flow = len(u_list)

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
X_list, dX_list, dT_list = rh.get_X_dX_and_dT(data_all[flow_ID],feat_cols=[str(i) for i in PCs])

# %%
kmc, edges = km.km(X_list, dX_list, bw = bw, bins = bins, 
                    powers = powers, multi_traj=True)
# %%
# Initialise the figure for 3d ploting
fig = plt.figure(figsize = (12,8))


#lets fix a range where we have good statistics and generate a meshgrid
idx0 = 3
idx1 = -3
X_1, X_2 = np.meshgrid(edges[0][idx0:idx1],edges[1][idx0:idx1])


# the Kramers−Moyal coefficients [1,0]
ax_00 = fig.add_subplot(2, 2, 1, projection='3d')

# kmc[1,...] i.e, power [1,0] (transpose, python stores arrays transposed)
ax_00.contour3D(X_1, X_2, kmc[1,idx0:idx1,idx0:idx1].T / 5, 50, cmap='Greens')
ax_00.set_title(r'K−M coefficient [1,0]');


# the Kramers−Moyal coefficients [0,1]
ax_01 = fig.add_subplot(2, 2, 2, projection='3d')

# kmc[2,...] i.e, power [0,1] (transpose, python stores arrays transposed)
ax_01.contour3D(X_1, X_2, kmc[2,idx0:idx1,idx0:idx1].T /5, 50, cmap='Greens')
ax_01.set_title(r'K−M coefficient [0,1]');


# the Kramers−Moyal coefficients [2,0]
ax_10 = fig.add_subplot(2, 2, 3, projection='3d')

# kmc[4,...] i.e, power [2,0] (transpose, python stores arrays transposed)
ax_10.contour3D(X_1, X_2, kmc[4,idx0:idx1,idx0:idx1].T / 5, 50, cmap='Greens')
ax_10.set_title(r'K−M coefficient [2,0]');


# the Kramers−Moyal coefficients [0,2]
ax_11 = fig.add_subplot(2, 2, 4, projection='3d')

# kmc[5,...] i.e, power [0,2] (transpose, python stores arrays transposed)
ax_11.contour3D(X_1, X_2, kmc[5,idx0:idx1,idx0:idx1].T / 5, 50, cmap='Greens')

ax_11.set_title(r'K−M coefficient [0,2]');
# Rotate views and add labels
ax_00.view_init(30, 20); ax_01.view_init(30, 20); ax_10.view_init(30, 20); ax_11.view_init(30, 20)
ax_00.set_xlabel(r'$y_1$'); ax_01.set_xlabel(r'$y_1$'); ax_10.set_xlabel(r'$y_1$'); ax_11.set_xlabel(r'$y_1$')
ax_00.set_ylabel(r'$y_2$'); ax_01.set_ylabel(r'$y_2$'); ax_10.set_ylabel(r'$y_2$'); ax_11.set_ylabel(r'$y_2$')

plt.show()
# %%
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

f1_mdl = Pipeline([('poly', PolynomialFeatures(degree=5)),
                   ('linear', LinearRegression())]).fit(X,Y_1.flatten())

Y_1_pred = f1_mdl.predict(X)
r2_f1 = r2_score(Y_1.flatten(),Y_1_pred)
print('R^2 for polynomial regression on Y_1:',r2_f1)

# linear regression on Y_2 = kmc[2,idx0:idx1,idx0:idx1].T/5
Y_2 = kmc[2,idx0:idx1,idx0:idx1].T/5

f2_mdl = Pipeline([('poly', PolynomialFeatures(degree=2)),
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

        f_KM_temp = np.array((kmc[1,idx0:idx1,idx0:idx1].T,
                              kmc[2,idx0:idx1,idx0:idx1].T)).T/dt
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

feature_lib = ps.PolynomialLibrary(degree=5, include_bias=True)
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
myModel = [driftModel,diffModel]

pplane_xlim = [-4,4]
bin_xlim = [-5,5]

pplane_ylim = [-3.5,3.5]
bin_ylim = [-4,4]


# fix bins and centers for all datasets
Nbins = [50 for i in range(ndim)]
bin_limits = [bin_xlim,bin_ylim]
bins, centers = rh.get_bins(Nbins,bin_limits=bin_limits)

plt_args = {'pplane_xlim': pplane_xlim, 'pplane_ylim': pplane_ylim, 'pplane_N': 50,
            'plt_xlabel': 'PC'+str(PCs[0]+1), 'plt_ylabel': 'PC'+str(PCs[1]+1),
            'truncate_p':[True,[5,Nbins[0]-5],[5,Nbins[1]-5]]}


# %%
u = 5.0
f = model_eval.vector_field_function(driftModel)
D = model_eval.vector_field_function(diffModel)
f1 = model_eval.vector_field_component(f,0)
f2 = model_eval.vector_field_component(f,1)
D1 = model_eval.vector_field_component(D,0)
D2 = model_eval.vector_field_component(D,1)
import fipy

# %%
Nbins = [len(bins[i])-1 for i in range(ndim)]
dx = [bins[i][1]-bins[i][0] for i in range(ndim)]
bin_min = [bins[i][0] for i in range(ndim)]

mesh = fipy.Grid2D(dx=dx[0], dy=dx[1], nx=Nbins[0], ny=Nbins[1])
x, y = mesh.cellCenters
x_ = x.reshape((Nbins[1],Nbins[0]))+bin_min[0]
y_ = y.reshape((Nbins[1],Nbins[0]))+bin_min[1]
f1_vals = f1([x_,y_],u)[None,:].reshape(1,Nbins[1],Nbins[0])
f2_vals = f2([x_,y_],u)[None,:].reshape(1,Nbins[1],Nbins[0])
f_vals = np.array(np.concatenate([f1_vals,f2_vals]))
D1_vals = D1([x_,y_],u)[None,:].reshape(1,Nbins[1],Nbins[0])
D2_vals = D2([x_,y_],u)[None,:].reshape(1,Nbins[1],Nbins[0])
D_vals = np.array(np.concatenate([D1_vals,D2_vals]))
print(f_vals.shape)

# get Div(D)
divD = np.zeros_like(f_vals)
for i in range(D_vals.shape[0]):
    divD[i] = np.gradient(D_vals[i], dx[i], axis=(i+1)%2, edge_order=2)

# %%
# check f_vals, D_vals, divD
x_vec= centers[0]
y_vec = centers[1]

fig, ax = plt.subplots()
ax.streamplot(x_vec, y_vec, f_vals[0], f_vals[1])

fig, ax = plt.subplots()
ax.streamplot(x_vec, y_vec, f_vals[0]-divD[0], f_vals[1]-divD[1])

fig, ax = plt.subplots()
ax.pcolormesh(D_vals[0])

fig, ax = plt.subplots()
ax.pcolormesh(D_vals[1])

fig, ax = plt.subplots()
ax.pcolormesh(divD[0])

fig, ax = plt.subplots()
ax.pcolormesh(divD[1])

# %%
f_vals = f_vals.reshape(2,-1)
D_vals = D_vals.reshape(2,-1)
divD = divD.reshape(2,-1)

p = fipy.CellVariable(mesh=mesh, name=r"$P$", value = 1/(Nbins[0]*Nbins[1]))

# psi = f(x) - div (D(x))
psi = fipy.CellVariable(mesh=mesh, value = [f_vals[0]-divD[0], f_vals[1]-divD[1]])
D = fipy.CellVariable(mesh=mesh, value = [D_vals[0],D_vals[1]])

eq = fipy.ConvectionTerm(coeff=psi,var=p) == fipy.DiffusionTerm(coeff=D,var=p)
eq.sweep(var=p)

p_fit = p.value.reshape(Nbins[1],Nbins[0])
C = np.trapz(np.trapz(p_fit, dx=dx[0], axis=1),dx=dx[1])
p_sol = p_fit/C
# %%
fig, ax = plt.subplots()
x0 = 5
y0 = 5
Nx = Nbins[0]-5
Ny = Nbins[1]-5
p_sol_ = p_sol[x0:Ny,y0:Nx]
C_ = np.trapz(np.trapz(p_sol_, dx=dx[0], axis=1),dx=dx[1])
p_sol_ = p_sol_/C_
p_sol_[p_sol_<1e-20] = 1e-20
cax = ax.pcolormesh(p_sol_)
# add colorbar
cbar = plt.colorbar(cax);

# reset x labels to reflect range from 0 to x_max
ax.set_xticks(np.linspace(0,Nx-x0,10))
ax.set_xticklabels(np.round(np.linspace(bins[0][x0],bins[0][Nx],10),1)) # round to 1 decimal place
# reset y labels to reflect range from 0 to y_max
ax.set_yticks(np.linspace(0,Ny-y0,10))
ax.set_yticklabels(np.round(np.linspace(bins[1][y0],bins[1][Ny],10),1))
# %%


for ds_name in list_of_datasets:
    if ds_name in config['datasets_to_skip']:
        continue
    print('**** Running model analysis for ',ds_name,'**** \n')
    feats_proj = mio.project_PCA_one_dataset(df,pca,ds_name)
    data_all, u_list = rh.get_X_by_flow(feats_proj,ds_name,verbose=True)
    del feats_proj # free up memory
    num_flow = len(u_list)


    for j in range(num_flow): # get bins and centers for data at high and low flow    
        print('**** Shear stress u =',u_list[j],'dyn/cm^2 **** \n')
        plot_tuple = model_analysis.run_model_analysis_2D(myModel,data_all[j],bins,centers,u_list[j],args=plt_args)


# %%
# %%

u_range = np.linspace(0,35,20)

fpt_dict = {}

x1_lims = plt_args['pplane_xlim']
x2_lims = plt_args['pplane_ylim']

x1 = np.linspace(x1_lims[0],x1_lims[1],50)
x2 = np.linspace(x2_lims[0],x2_lims[1],50)
x1_coarse = np.linspace(x1_lims[0],x1_lims[1],7)
x2_coarse = np.linspace(x2_lims[0],x2_lims[1],7)

f = model_eval.vector_field_function(driftModel)
# %%
for u in u_range:

    def myFlow(x):
        return f(x,u=u)
    flowJacobian = nd.Jacobian(myFlow)

    init_coarse = [np.array([x1_coarse[i],x2_coarse[j]]) 
                   for i in range(len(x1_coarse)) 
                   for j in range(len(x2_coarse))
                   ]
    fpts = pplane.get_fps(myFlow,init_coarse) # get fixed points
    fpt_types = []
    if len(fpts) > 0:
        for fpt in fpts:
            fptStability = pplane.find_stability(flowJacobian(fpt))
            if 'Stable' in fptStability:
                fpt_types.append('stable')
            elif 'Unstable' in fptStability:
                fpt_types.append('unstable')
            elif 'Saddle' in fptStability:
                fpt_types.append('saddle')
            else:
                fpt_types.append('indeterminate')

    fpts_new = []
    fpt_types_new = []
    for fpt in fpts:
        # if far out of bounds of the plot window, don't report it
        if fpt[0]<x1[0]-0.5*abs(x1[0]) or fpt[0]>x1[-1]+0.5*abs(x1[-1]) or fpt[1]<x2[0]-0.5*abs(x2[0]) or fpt[1]>x2[-1]+0.5*abs(x2[-1]):
            continue
        else:
            fpts_new.append(fpt)
            fpt_types_new.append(fpt_types[fpts.index(fpt)])

    fpt_dict[str(u)] = {}
    fpt_dict[str(u)]['fixed_points'] = fpts_new
    fpt_dict[str(u)]['fixed_point_types'] = fpt_types_new
# %%
for j in range(ndim):
    fig, ax = plt.subplots()
    for u in u_range:
        if str(u) in fpt_dict.keys():
            fpts = fpt_dict[str(u)]['fixed_points']
            fpt_types = fpt_dict[str(u)]['fixed_point_types']
            if len(fpts) > 0:
                for i,fpt in enumerate(fpts):
                    if fpt_types[i] == 'stable':
                        color = 'b'
                    elif fpt_types[i] == 'unstable':
                        color = 'r'
                    elif fpt_types[i] == 'saddle':
                        color = 'tab:purple'
                    else:
                        color = 'darkgoldenrod'

                    ax.plot(u,fpt[j],'o',color=color)
                    ax.set_xlabel('Shear stress (dyn/cm^2)')
                    ax.set_ylabel('PC'+str(PCs[j]+1))
# %%
fpt_stable = []
u_stable = []
for u in u_range:
    if str(u) in fpt_dict.keys():
        fpts = fpt_dict[str(u)]['fixed_points']
        fpt_types = fpt_dict[str(u)]['fixed_point_types']
        if len(fpts) > 0:
            for i,fpt in enumerate(fpts):
                if fpt_types[i] == 'stable':
                    color = 'b'
                    fpt_stable.append(fpt)
                    u_stable.append(u)
                elif fpt_types[i] == 'unstable':
                    color = 'r'
                elif fpt_types[i] == 'saddle':
                    color = 'tab:purple'
                else:
                    color = 'darkgoldenrod'

                plt.plot(fpt[0],fpt[1],'o',color=color)
                plt.xlabel('PC'+str(PCs[0]+1))
                plt.ylabel('PC'+str(PCs[1]+1))

# %%
# plot stable fixed points as a colored by shear stress value
u_stable = np.array(u_stable)
fpt_stable = np.array(fpt_stable)
plt.scatter(fpt_stable[1:-7,0],fpt_stable[1:-7,1],
            c=u_stable[1:-7],cmap='bwr',edgecolors='k')

plt.xlabel('PC'+str(PCs[0]+1))
plt.ylabel('PC'+str(PCs[1]+1))
plt.colorbar(label='dyn/cm$^2$')
plt.title('Stable fixed points by shear stress')







# %%
u_range = np.linspace(6,35,40)
# entropy production rate as a function of u
D = model_eval.vector_field_function(diffModel)
epr = np.zeros(len(u_range))
for u in u_range:   
    P = model_eval.get_stationary_probability(f,D,bins,centers,u)
    f_mesh = model_eval.mesh_grid_function(f)
    D_mesh = model_eval.mesh_grid_function(D)

    X1,X2 = np.meshgrid(centers[0],centers[1])
    f_vals = f_mesh([X1,X2],u).T
    D_vals = D_mesh([X1,X2],u).T

    J = gp.probability_flux(P,f_vals,D_vals,centers)
    D_mat = gp.expand_to_matrix(D_vals)

    epr[u_range.tolist().index(u)] = gp.entropy_production(J,D_mat,P,centers)

# %%
plt.plot(u_range,epr,'-o',color='k')
plt.xlabel('Shear stress (dyn/cm$^2$)')
plt.ylabel('Entropy production rate')


# %%
