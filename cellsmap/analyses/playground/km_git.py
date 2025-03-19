# %%
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
import numdifftools as nd

import cellsmap.analyses.utils.io as eaio
import cellsmap.analyses.utils.regression as eareg
import cellsmap.analyses.utils.viz as eaviz
import cellsmap.analyses.utils.model_analysis as model_analysis
import cellsmap.analyses.utils.model_eval as model_eval

import cellsmap.analyses.utils.pplane as pplane
import cellsmap.analyses.utils.gen_potential as gp

import sys
sys.path.append('//allen/aics/users/erin.angelini/git-repos/KramersMoyal')
import kramersmoyal as km
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
# %%
path_to_data = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_for_erin/2025-02-24_17-13-26/predict.parquet'
df = eaio.add_metadata_from_path(eaio.load_array(path_to_data))
df.head()
list_of_datasets = eaio.get_list_of_datasets(df,'group',verbose=True)
# %%
df_ref = eaio.get_PCA_reference(df) # dataset for getting PCA reference
metadata_col = ['filename_or_obj','T','start_x','start_y','group','pca_ref','FOV_ID']
df_ref_ = eaio.rm_metadata(df_ref,metadata_col) # remove metadata columns
pca = eaio.get_PCA(df_ref_)
del df_ref_ # free up memory
# %%
ds_ID = 1
my_mv = list_of_datasets[ds_ID]
mv_name = eaio.get_dataset_name(my_mv)
feats_proj = eaio.project_PCA_one_dataset(df,pca, 'group', my_mv, metadata_cols=metadata_col)
PCs = [0,1]
data_all, u_traj, u_list = eareg.get_traj_and_flow(feats_proj,mv_name,PCs=PCs,verbose=True)
num_flow = len(u_list)

fig1,ax1 = eaviz.plot_top_3_PCs(feats_proj)
ax1[0].set_ylim([-1.75,0.075])
ax1[1].set_ylim([-0.05,1.05])
# ax1[2].set_ylim([-1.5,-0.75])

fig2,ax2 = eaviz.plot_PCA_projection(feats_proj,mv_name)
ax2.set_xlim([-1.75,0.075])
ax2.set_ylim([-0.05,1.05])

# %%
ndim = len(PCs)
# Choose the size of your target space in two dimensions 
bins = 35*np.ones(ndim, dtype = int)

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
feats_proj_ = np.array(data_all[flow_ID])

kmc, edges = km.km(feats_proj_, bw = bw, bins = bins, 
                    powers = powers,multi_traj=True)
# %%
# Initialise the figure for 3d ploting
fig = plt.figure(figsize = (12,8))


#lets fix a range where we have good statistics and generate a meshgrid
idx0 = 8
idx1 = -5
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

f1_mdl = Pipeline([('poly', PolynomialFeatures(degree=4)),
                   ('linear', LinearRegression())]).fit(X,Y_1.flatten())

Y_1_pred = f1_mdl.predict(X)
r2_f1 = r2_score(Y_1.flatten(),Y_1_pred)
print('R^2 for polynomial regression on Y_1:',r2_f1)

# linear regression on Y_2 = kmc[2,idx0:idx1,idx0:idx1].T/5
Y_2 = kmc[2,idx0:idx1,idx0:idx1].T/5

f2_mdl = LinearRegression().fit(X,Y_2.flatten())

Y_2_pred = f2_mdl.predict(X)
r2_f2 = r2_score(Y_2.flatten(),Y_2_pred)
print('R^2 for linear regression on Y_2:',r2_f2)


# linear regression on V_1 = kmc[4,idx0:idx1,idx0:idx1].T/5
V_1 = kmc[4,idx0:idx1,idx0:idx1].T/5

D1_mdl = LinearRegression().fit(X,V_1.flatten())

V_1_pred = D1_mdl.predict(X)
r2_D1 = r2_score(V_1.flatten(),V_1_pred)
print('R^2 for linear regression on V_1:',r2_D1)


# polynomial regression on V_2 = kmc[5,idx0:idx1,idx0:idx1].T/5
V_2 = kmc[5,idx0:idx1,idx0:idx1].T/5

D2_mdl = Pipeline([('poly', PolynomialFeatures(degree=3)),
                   ('linear', LinearRegression())]).fit(X,V_2.flatten())

V_2_pred = D2_mdl.predict(X)
r2_D2 = r2_score(V_2.flatten(),V_2_pred)
print('R^2 for polynomial regression on V_2:',r2_D2)



# %%
# now fit model using multiple datasets
PCs = [0,1]
ndim = len(PCs)

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

Nbins = 35*np.ones(ndim, dtype = int)
dt = 5

for ds_ID in range(4): 
    print('**** Generating train/test sets for dataset',ds_ID,'**** \n')
    my_mv = list_of_datasets[ds_ID]
    mv_name = eaio.get_dataset_name(my_mv)
    feats_proj = eaio.project_PCA_one_dataset(df,pca,'group', my_mv,metadata_cols=metadata_col)

    data_all, u_traj, u_list = eareg.get_traj_and_flow(feats_proj,mv_name,PCs=PCs,verbose=True)
    del feats_proj # free up memory
    num_flow = len(u_list)
    if 0 in u_list: # right now, only using timepoints 300:450 of no flow
        data_all_temp = []
        for traj in data_all[0]:
            data_all_temp.append(traj[300:450,:])
        data_all = [data_all_temp]
        u_traj = [u_traj[0][300:450]]

    centers = []

    f_KM = []
    D_KM = []

    for j in range(num_flow): # get bins and centers for data at high and low flow
        feats_proj_ = np.array(data_all[j])
        kmc, edges = km.km(feats_proj_, bw = bw, bins = Nbins, 
                    powers = powers, multi_traj=True)
        centers.append(edges)
        # # need to mask out areas with no data, km doesn't do this
        # # get mask for bins with no data via histogram
        # hist, _ = np.histogramdd(feats_proj_.reshape((-1,ndim)),bins=Nbins)
        # mask = hist == 0

        idx0 = 8
        idx1 = -8

        f_KM_temp = np.array((kmc[1,idx0:idx1,idx0:idx1].T,
                              kmc[2,idx0:idx1,idx0:idx1].T)).T/dt
        D_KM_temp = np.array((kmc[4,idx0:idx1,idx0:idx1].T,
                              kmc[5,idx0:idx1,idx0:idx1].T)).T/dt
        # set masked values to NaN
        # f_KM_temp[mask] = np.nan
        # D_KM_temp[mask] = np.nan
        f_KM.append(f_KM_temp)
        D_KM.append(D_KM_temp)

    f_KM_noNAN = []
    D_KM_noNAN = []
    X_pts_noNAN = []

    for j in range(num_flow):
        centers_ = [centers[j][i][idx0:idx1] for i in range(ndim)]
        f_KM_noNAN_temp, X_pts_temp = eareg.masked_vector_field(f_KM[j], 
                                                                np.array(np.meshgrid(*centers_)).T)
        D_KM_noNAN_temp, _ = eareg.masked_vector_field(D_KM[j], np.array(np.meshgrid(*centers_)).T)
        f_KM_noNAN.append(f_KM_noNAN_temp)
        D_KM_noNAN.append(D_KM_noNAN_temp)
        X_pts_noNAN.append(X_pts_temp)

    del f_KM, D_KM, centers, centers_ # free up memory

    train_frac = 0.8
    seed = 47

    # should write in generating u_train and u_test
    X_train, X_test, Y_train, Y_test, V_train, V_test = eareg.train_test_all(X_pts_noNAN,f_KM_noNAN,
                                                                            D_KM_noNAN,num_flow,
                                                                            train_frac,seed,concat=True)

    if num_flow == 1:
        N_tot = X_pts_noNAN[0].shape[0]
        N_train = int(train_frac*N_tot)
        N_test = N_tot-N_train
        u_train = u_list[0]*np.ones((N_train,1))
        u_test = u_list[0]*np.ones((N_test,1))
    else:
        N_tot = [X_pts_noNAN[0].shape[0],X_pts_noNAN[1].shape[0]]
        N_train = [int(train_frac*N_tot[0]),int(train_frac*N_tot[1])]
        N_test = [N_tot[0]-N_train[0],N_tot[1]-N_train[1]]
        u_train = np.concatenate((u_list[0]*np.ones((N_train[0],1)),u_list[1]*np.ones((N_train[1],1))))
        u_test = np.concatenate((u_list[0]*np.ones((N_test[0],1)),u_list[1]*np.ones((N_test[1],1))))
    
    del X_pts_noNAN, f_KM_noNAN, D_KM_noNAN # free up memory

    X_train_list.append(X_train)
    X_test_list.append(X_test)
    Y_train_list.append(Y_train)
    Y_test_list.append(Y_test)
    V_train_list.append(V_train)
    V_test_list.append(V_test)
    u_train_list.append(u_train)
    u_test_list.append(u_test)

    del X_train, X_test, Y_train, Y_test, V_train, V_test, u_train, u_test # free up memory


# %%
X_train = np.concatenate(X_train_list)
X_test = np.concatenate(X_test_list)
Y_train = np.concatenate(Y_train_list)
Y_test = np.concatenate(Y_test_list)
V_train = np.concatenate(V_train_list)
V_test = np.concatenate(V_test_list)
u_train = np.concatenate(u_train_list)
u_test = np.concatenate(u_test_list)
# %%
sigmoid_range = range(3,4)

def make_sigmoid(n):
    def _(x):
        return 1/(1+np.exp(-n*x))
    return _


def make_sigmoid_string(n):
    def _(x):
        return '1/(1+exp(-'+str(n)+'*'+x+')'
    return _

sigmoid_funcs = [make_sigmoid(n) for n in sigmoid_range]
func_names = [make_sigmoid_string(n) for n in sigmoid_range]

sigmoid_lib=ps.CustomLibrary(library_functions=sigmoid_funcs,
                             function_names=func_names)
feature_lib = ps.ConcatLibrary([ps.PolynomialLibrary(degree=3, 
                                include_bias=True),
                                sigmoid_lib])
# feature_lib = ps.PolynomialLibrary(degree=3, include_bias=True)
parameter_lib=ps.PolynomialLibrary(degree=3, include_bias=True)
full_lib=ps.ParameterizedLibrary(feature_library=feature_lib,
    parameter_library=parameter_lib,num_features=ndim,num_parameters=1)

driftModel = ps.SINDy(feature_library = full_lib, optimizer = ps.SSR())
driftModel.fit(X_train,t=dt,x_dot=Y_train,u=u_train)


diff_feature_lib=ps.PolynomialLibrary(degree=0, include_bias=True)
diff_parameter_lib=ps.PolynomialLibrary(degree=2, include_bias=True)
diff_lib=ps.ParameterizedLibrary(feature_library=diff_feature_lib,
    parameter_library=diff_parameter_lib,num_features=ndim,num_parameters=1)


diffModel = ps.SINDy(feature_library = diff_lib, optimizer = ps.SSR())
diffModel.fit(X_train,t=dt,x_dot=V_train,u=u_train)

drift_R2 = driftModel.score(X_test,x_dot=Y_test,u=u_test)
driftModel.print()

print('Coefficient of determination (R^2) of drift model on test set: %f' %drift_R2)

diff_R2 = diffModel.score(X_test,x_dot=V_test,u=u_test)
diffModel.print()

print('Coefficient of determination (R^2) of diffusion model on test set: %f' %diff_R2)
# %%
myModel = [driftModel,diffModel]

if PCs[0] == 0:
    pplane_xlim = [-2,1.5]
    bin_xlim = [-2.25,1]
elif PCs[0] == 1:
    pplane_xlim = [0,1]
    bin_xlim = [-0.25,0.8]

if PCs[1] == 1:
    pplane_ylim = [-0.25,1]
    bin_ylim = [-0.25,1.0]
else:
    pplane_ylim = [-1,-0.6]
    bin_ylim = [-1.5,-0.6]


# fix bins and centers for all datasets
Nbins = [50 for i in range(ndim)]
bin_limits = [bin_xlim,bin_ylim]
bins, centers = eareg.get_bins(Nbins,bin_limits=bin_limits)

plt_args = {'pplane_xlim': pplane_xlim, 'pplane_ylim': pplane_ylim, 'pplane_N': 50,
            'plt_xlabel': 'PC'+str(PCs[0]+1), 'plt_ylabel': 'PC'+str(PCs[1]+1),
            'truncate_p':[True,[5,Nbins[0]-5],[5,Nbins[1]-5]]}


# %%
u = 35.0
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


for ds_ID in range(len(list_of_datasets)):
    print('**** Running model analysis for dataset',ds_ID,'**** \n')
    my_mv = list_of_datasets[ds_ID]
    mv_name = eaio.get_dataset_name(my_mv)
    feats_proj = eaio.project_PCA_one_dataset(df,pca,'group', my_mv,metadata_cols=metadata_col)

    data_all, u_traj, u_list = eareg.get_traj_and_flow(feats_proj,mv_name,PCs=PCs,verbose=True)
    del feats_proj # free up memory
    num_flow = len(u_list)
    if 0 in u_list: # right now, only using timepoints 300:450 of no flow
        data_all_temp = []
        for traj in data_all[0]:
            data_all_temp.append(traj[300:450,:])
        data_all = [data_all_temp]
        u_traj = [u_traj[0][300:450]]


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
