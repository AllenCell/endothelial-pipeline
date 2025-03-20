# %%
import numpy as np
import pandas as pd
import pysindy as ps
import numdifftools as nd

import cellsmap.analyses.utils.gen_potential as gp
import cellsmap.analyses.utils.regression as eareg
from cellsmap.analyses.utils import manifest_viz, manifest_io, manifest_pca, pplane, model_eval, model_analysis


# %%
# load data
df = manifest_io.load_manifest_to_df()
# fit PCA to data
df, pca = manifest_pca.get_pca(df, num_pcs=8)
# add outliers to dataframe
df = manifest_pca.get_outliers(df)
# filepath for this dataset in manifest includes barcode, so we need to change the group name to match data config
# something that should be fixed in the manifest in the future
df.loc[df.group.str.contains('20250224'),'group'] = '20250224_20X'
# get list of datasets by 'group' identifier
list_of_datasets = manifest_io.get_list_of_datasets(df,'group',verbose=True)

# plot explained variance ratio of PCA components
fig, ax = manifest_viz.plot_explained_variance(pca['pca'].explained_variance_ratio_)

# %%
# plot top 3 principal components of feature data vs. frame number
# should write a function to do this for all datasets in the manifest via data_config
title_dict = {'20241016_20X':'24H High, 24H Low',
              '20241105_20X':'24H Low, 24H High (11/5/24)',
              '20241120_20X':'48H High',
              '20241203_20X':'48H Low',
              '20241210_20X':'48H No Flow 1',
              '20241217_20X':'48H No Flow 2',
              '20250224_20X':'24H Low, 24H High (2/24/25)',}

fig, axs = manifest_viz.plot_top_3_PCs_alldata(df,pca['pca'],list_of_datasets, title_dict)

# %%
# now fit model using multiple datasets
PCs = [0,1] # index of PCs to use for model fitting
ndim = len(PCs) # get ndim from number of PCs

# datasets to leave out of SDE model fitting
ds_to_skip = ['20241016_20X','20241210_20X','20241217_20X']

# list of training/test sets for each dataset
# right now set up to get list of lists to organize by flow condition,
# concatenate all datasets together at the end to fit model
X_train_list = []
X_test_list = []
Y_train_list = []
Y_test_list = []
V_train_list = []
V_test_list = []
u_train_list = []
u_test_list = []

# number of bins for each dimension, used for binning data for model fitting
Nbins = 25*np.ones(ndim,dtype=int)
# %%
for my_mv in list_of_datasets: 
    mv_name = manifest_io.get_dataset_name(my_mv)

    # don't fit model using no flow datasets
    if mv_name in ds_to_skip:
        print('**** Skipping dataset',mv_name,'**** \n')
        continue

    print('**** Generating train/test sets for dataset',mv_name,'**** \n')

    # project data from this one dataset onto PCs as defined by fit PCA object pca
    df_proj = manifest_io.project_PCA_one_dataset(df,pca,'group',my_mv)

    # for extracting just the PCs we want from the dataframe
    feat_cols = [str(i) for i in PCs]

    # get 2-pt trajectories and for each flow condition present in the dataset as well as the flow conditions themselves
    # filters out timepoints flagges as outliers
    traj_list, flow_list = eareg.get_2pt_traj_and_flow(df_proj,mv_name,feat_cols=feat_cols,verbose=True)
    del df_proj # free up memory
    num_flow = len(flow_list)

    bins = []
    centers = []

    f_KM = []
    D_KM = []
    f_err = []
    D_err = []

    for j in range(num_flow): # get bins and centers for data at high and low flow
        bins_temp, centers_temp = eareg.get_bins(Nbins,data=traj_list[j])
        bins.append(bins_temp)
        centers.append(centers_temp)

        f_KM_temp, D_KM_temp, f_err_temp, D_err_temp = eareg.KM_avg_ND(traj_list[j], bins[j], dt=5)
        f_KM.append(f_KM_temp)
        D_KM.append(D_KM_temp)
        f_err.append(f_err_temp)
        D_err.append(D_err_temp)

    f_KM_noNAN = []
    D_KM_noNAN = []
    X_pts_noNAN = []

    for j in range(num_flow):
        f_KM_noNAN_temp, X_pts_temp, _ = eareg.masked_vector_field(f_KM[j], np.array(np.meshgrid(*centers[j])).T)
        D_KM_noNAN_temp, _, _ = eareg.masked_vector_field(D_KM[j], np.array(np.meshgrid(*centers[j])).T)
        f_KM_noNAN.append(f_KM_noNAN_temp)
        D_KM_noNAN.append(D_KM_noNAN_temp)
        X_pts_noNAN.append(X_pts_temp)

    del f_KM, D_KM, f_err, D_err, bins, centers # free up memory

    train_frac = 0.8
    seed = 47

    X_train, X_test, Y_train, Y_test, V_train, V_test = eareg.train_test_all(X_pts_noNAN,f_KM_noNAN,
                                                                            D_KM_noNAN,num_flow,
                                                                 train_frac,seed,concat=True)

    if num_flow == 1:
        N_tot = X_pts_noNAN[0].shape[0]
        N_train = int(train_frac*N_tot)
        N_test = N_tot-N_train
        u_train = flow_list[0]*np.ones((N_train,1))
        u_test = flow_list[0]*np.ones((N_test,1))
    else:
        N_tot = [X_pts_noNAN[0].shape[0],X_pts_noNAN[1].shape[0]]
        N_train = [int(train_frac*N_tot[0]),int(train_frac*N_tot[1])]
        N_test = [N_tot[0]-N_train[0],N_tot[1]-N_train[1]]
        u_train = np.concatenate((flow_list[0]*np.ones((N_train[0],1)),flow_list[1]*np.ones((N_train[1],1))))
        u_test = np.concatenate((flow_list[0]*np.ones((N_test[0],1)),flow_list[1]*np.ones((N_test[1],1))))
    
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

X_train = np.concatenate(X_train_list)
X_test = np.concatenate(X_test_list)
Y_train = np.concatenate(Y_train_list)
Y_test = np.concatenate(Y_test_list)
V_train = np.concatenate(V_train_list)
V_test = np.concatenate(V_test_list)
u_train = np.concatenate(u_train_list)
u_test = np.concatenate(u_test_list)

# %%
# fit SINDy model

# build set of basis functions

# include sigmoid functions in basis functions
include_sigmoid = True

if include_sigmoid: 
    sigmoid_range = range(3,4) # range of powers of sigmoid functions to include

    # have to make custom sigmoid functions to include in SINDy library
    # callable function
    def make_sigmoid(n):
        def _(x):
            return 1/(1+np.exp(-n*x))
        return _

    # string representation of function
    def make_sigmoid_string(n):
        def _(x):
            return '1/(1+exp(-'+str(n)+'*'+x+')'
        return _

    sigmoid_funcs = [make_sigmoid(n) for n in sigmoid_range]
    func_names = [make_sigmoid_string(n) for n in sigmoid_range]

    # pySINDy custom library
    sigmoid_lib=ps.CustomLibrary(library_functions=sigmoid_funcs,
                                function_names=func_names)
    # full library for drift term (functions of state variables)
    drift_feature_lib = ps.ConcatLibrary([ps.PolynomialLibrary(degree=3, 
                                    include_bias=True),
                                    sigmoid_lib])
else: # just polynomial library for basis functions
    drift_feature_lib = ps.PolynomialLibrary(degree=3, include_bias=True)

# library for model dependence on control parameters (shear stress)
drift_parameter_lib=ps.PolynomialLibrary(degree=3, include_bias=True) # library for model dependence on control parameters (shear stress)

# build full library for drift term: pySINDy parameterized library
drift_lib=ps.ParameterizedLibrary(feature_library=drift_feature_lib,
    parameter_library=drift_parameter_lib,num_features=ndim,num_parameters=1) 

# build library for diffusion term (polynomial library only)
diff_feature_lib=ps.PolynomialLibrary(degree=0, include_bias=True)
diff_parameter_lib=ps.PolynomialLibrary(degree=3, include_bias=False)
diff_lib=ps.ParameterizedLibrary(feature_library=diff_feature_lib,
    parameter_library=diff_parameter_lib,num_features=ndim,num_parameters=1)

# %%
# fit model for drift term - SINDy based regression
driftModel = ps.SINDy(feature_library = drift_lib, optimizer = ps.SSR())
driftModel.fit(X_train,t=5,x_dot=Y_train,u=u_train)

# score on test set
drift_R2 = driftModel.score(X_test,x_dot=Y_test,u=u_test)
driftModel.print()

print('Coefficient of determination (R^2) for model of drift term: %f' %drift_R2)

# fit model for diffusion term - SINDy based regression
diffModel = ps.SINDy(feature_library = diff_lib, optimizer = ps.SSR())
diffModel.fit(X_train,t=5,x_dot=V_train,u=u_train)

# score on test set
diff_R2 = diffModel.score(X_test,x_dot=V_test,u=u_test)
diffModel.print()

print('Coefficient of determination (R^2) for model of diffusion term: %f' %diff_R2)
# %%
# evaluate model: pplane (stability analysis) and compare stationary probability distributions (data v model prediction) across datasets
myModel = [driftModel,diffModel]

# specification of plot limits for phase plane plots and bins for histogram plots
pplane_xlim = [-4,4]
bin_xlim = [-5,5]

if PCs[1] == 1:
    pplane_ylim = [-3.5,2.5]
    bin_ylim = [-4,3]
else:
    pplane_ylim = [-1,6]
    bin_ylim = [-2.5,8]


# fix bins and centers for all datasets using bin limits defined above
Nbins = [50 for i in range(ndim)]
bin_limits = [bin_xlim,bin_ylim]
bins, centers = eareg.get_bins(Nbins,bin_limits=bin_limits)

plt_args = {'pplane_xlim': pplane_xlim, 'pplane_ylim': pplane_ylim, 'pplane_N': 50,
            'plt_xlabel': 'PC'+str(PCs[0]+1), 'plt_ylabel': 'PC'+str(PCs[1]+1),
            'truncate_p':[True,[0,Nbins[0]-0],[0,Nbins[1]-0]]}

# %%
# fix bins and centers for all datasets

for my_mv in list_of_datasets: 
    mv_name = manifest_io.get_dataset_name(my_mv)

    # if we don't want to fit model using this dataset, skip it
    if mv_name in ds_to_skip:
        print('**** Skipping dataset',mv_name,'**** \n')
        continue

    print('**** Running model analysis for dataset',mv_name,'**** \n')

    # project data from this one dataset onto PCs as defined by fit PCA object pca
    df_proj = manifest_io.project_PCA_one_dataset(df,pca,'group',my_mv)

    # for extracting just the PCs we want from the dataframe
    feat_cols = [str(i) for i in PCs]

    # get 2-pt trajectories and for each flow condition present in the dataset as well as the flow conditions themselves
    traj_list, flow_list = eareg.get_2pt_traj_and_flow(df_proj,mv_name,feat_cols=feat_cols,verbose=True)
    del df_proj # free up memory
    num_flow = len(flow_list)

    
    for j in range(num_flow): # get bins and centers for data at high and low flow    
        print('**** Shear stress u =',flow_list[j],'dyn/cm^2 **** \n')
        plot_tuple = model_analysis.run_model_analysis_2D(myModel,traj_list[j],bins,centers,flow_list[j],args=plt_args)


# %%
# fixed point analysis: plot coordinates of fixed points as a function of shear stress
u_range = np.linspace(0,35,20)

fpt_dict = {}

x1_lims = plt_args['pplane_xlim']
x2_lims = plt_args['pplane_ylim']

x1 = np.linspace(x1_lims[0],x1_lims[1],50)
x2 = np.linspace(x2_lims[0],x2_lims[1],50)
x1_coarse = np.linspace(x1_lims[0],x1_lims[1],7)
x2_coarse = np.linspace(x2_lims[0],x2_lims[1],7)

f = model_eval.vector_field_function(driftModel)
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

fpt_stable = []
u_stable = []
for j in range(ndim):
    fig, ax = manifest_viz.init_plot()
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

                    ax.plot(u,fpt[j],'o',color=color)
                    ax.set_xlabel('Shear stress (dyn/cm^2)')
                    ax.set_ylabel('PC'+str(PCs[j]+1))
# %%
# plot stable fixed points as a colored by shear stress value
fig, ax = manifest_viz.init_plot()
u_stable = np.array(u_stable)
fpt_stable = np.array(fpt_stable)
im = ax.scatter(fpt_stable[1:-7,0],fpt_stable[1:-7,1],
            c=u_stable[1:-7],cmap='bwr',edgecolors='k')

ax.set_xlabel('PC'+str(PCs[0]+1))
ax.set_ylabel('PC'+str(PCs[1]+1))
fig.colorbar(im, ax= ax, label='dyn/cm$^2$')
ax.set_title('Stable fixed points by shear stress')


# %%
# entropy production rate as a function of shear stress
u_range = np.linspace(6,35,40) 

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

fig, ax = manifest_viz.init_plot()
ax.plot(u_range,epr,'-o',color='k')
ax.set_xlabel('Shear stress (dyn/cm$^2$)')
ax.set_ylabel('Entropy production rate')


# %%
################### Generalized potential energy landscape ###################
    
N_fine = 60

x_fine = np.linspace(-4,4.5,N_fine+1)
y_fine = np.linspace(-3,3.5,N_fine+1)
bins_fine = [x_fine,y_fine]
centers_fine = [0.5*(x_fine[1:]+x_fine[:-1]),
                0.5*(y_fine[1:]+y_fine[:-1])]
X1,X2 = np.meshgrid(centers_fine[0],centers_fine[1])
# %%
u = 25

tol = 1e-6

f = model_eval.vector_field_function(driftModel)
D = model_eval.vector_field_function(diffModel)

f_mesh = model_eval.mesh_grid_function(f)
D_mesh = model_eval.mesh_grid_function(D)

f_vals_new = f_mesh([X1,X2],u).T
D_vals_new = D_mesh([X1,X2],u).T

p_fit = model_eval.get_stationary_probability(f,D,bins_fine,centers_fine,u,tol=tol)
U= -np.log(p_fit)

print('**** Plotting generalized potential energy landscape **** \n')

fig,ax = manifest_viz.plot_gen_potential_2D(U,centers_fine[0],centers_fine[1],cmap='jet',surf=False)
ax.set_xlabel('PC'+str(PCs[0]+1))
ax.set_ylabel('PC'+str(PCs[1]+1))
ax.set_title('Shear stress: '+str(u)+' dyn/cm$^2$')

# %%
# same but with vector field decomposition
normed = False # if True, normalize vectors by their magnitudes

_, grad_term, _, flux_term = gp.grad_flux_decomposition(f_vals_new,D_vals_new,centers_fine,tol=tol)
grad_ = grad_term.copy()
flux_ = np.array(flux_term)
if normed:
    grad_ = grad_/(np.sqrt(grad_[0]**2+grad_[1]**2))
    flux_ = flux_/(np.sqrt(flux_[0]**2+flux_[1]**2))

fig,ax = manifest_viz.plot_gen_potential_2D(U,centers_fine[0],centers_fine[1],
                                   cmap='jet',surf=False)
downsample=10
ax.quiver(centers_fine[0][::downsample],
          centers_fine[1][::downsample],
          grad_[0][::downsample,::downsample].T,
          grad_[1][::downsample,::downsample].T,
          color='w',pivot='tail')
ax.quiver(centers_fine[0][::downsample],
          centers_fine[1][::downsample],
          flux_[0][::downsample,::downsample].T,
          flux_[1][::downsample,::downsample].T,
          color='r',pivot='tail')
ax.set_xlabel('PC'+str(PCs[0]+1))
ax.set_ylabel('PC'+str(PCs[1]+1))

# %%
