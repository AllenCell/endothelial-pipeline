# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysindy as ps
import numdifftools as nd

import cellsmap.util.pca as cmpca

import cellsmap.analyses.utils.gen_potential as gp
from cellsmap.analyses.utils import viz
from cellsmap.analyses.utils import pplane

import cellsmap.analyses.utils.io as eaio
import cellsmap.analyses.utils.viz as eaviz
import cellsmap.analyses.utils.regression as eareg
import cellsmap.analyses.utils.model_eval as model_eval
import cellsmap.analyses.utils.model_analysis as model_analysis

# %%
# load data
path_to_data = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_for_erin/2025-02-24_17-13-26/predict.parquet'
path_to_20241217 = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_20241217/2025-02-28_10-41-33/predict.parquet'
path_to_20250224 = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_20250224/2025-03-03_11-45-02/predict.parquet'

df = eaio.load_array(path_to_data)
df_1217 = eaio.load_array(path_to_20241217)
df_0224 = eaio.load_array(path_to_20250224)
df = pd.concat([df,df_1217,df_0224],ignore_index=True)
df, pca = cmpca.get_pca(df, num_pcs=8)
df, bad_files = cmpca._get_outliers(df)
list_of_datasets = eaio.get_list_of_datasets(df,'group',verbose=True)
# %%
# plot explained variance
fig, ax = eaviz.plot_explained_variance(pca['pca'].explained_variance_ratio_)

# %%
# To do: write io function that builds this from data config
title_dict = {'20241016_20X':'24H High, 24H Low',
              '20241105_20X':'24H Low, 24H High (11/5/24)',
              '20241120_20X':'48H High',
              '20241203_20X':'48H Low',
              '20241210_20X':'48H No Flow 1',
              '20241217_20X':'48H No Flow 2',
              '20250224_GE00006991_20X':'24H Low, 24H High (2/24/25)',}

# plot PCA projection for a single dataset
ds_ID = 3 # index of dataset in list_of_datasets
my_mv = list_of_datasets[ds_ID] # get dataset identifier
mv_name = eaio.get_dataset_name(my_mv) # get dataset name (shortened identifier)
df_proj = eaio.project_PCA_one_dataset(df,pca, 'group', my_mv)
feats_proj = eaio.df_to_array(df_proj,[str(i) for i in range(8)])

fig2,ax2 = eaviz.plot_PCA_projection(feats_proj, title_dict[mv_name])

# %%
# now fit model using multiple datasets
PCs = [0,1]
ndim = len(PCs)

# list of training/test sets for each dataset
X_train_list = []
X_test_list = []
Y_train_list = []
Y_test_list = []
V_train_list = []
V_test_list = []
u_train_list = []
u_test_list = []

Nbins = 25*np.ones(ndim,dtype=int)
# %%
# save out traj_list and flow_list for each dataset so you don't have to re-generate for model analysis?
for ds_ID in [0,1,2,3,6]: 
    print('**** Generating train/test sets for dataset',ds_ID,'**** \n')
    my_mv = list_of_datasets[ds_ID]
    mv_name = eaio.get_dataset_name(my_mv)

    df_proj = eaio.project_PCA_one_dataset(df,pca,'group',my_mv)

    feat_cols = [str(i) for i in PCs]

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
        f_KM_noNAN_temp, X_pts_temp = eareg.masked_vector_field(f_KM[j], np.array(np.meshgrid(*centers[j])).T)
        D_KM_noNAN_temp, _ = eareg.masked_vector_field(D_KM[j], np.array(np.meshgrid(*centers[j])).T)
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

include_sigmoid = True

if include_sigmoid: # include sigmoid functions in basis functions
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
    drift_feature_lib = ps.ConcatLibrary([ps.PolynomialLibrary(degree=3, 
                                    include_bias=True),
                                    sigmoid_lib])
else: # just polynomial library for basis functions
    drift_feature_lib = ps.PolynomialLibrary(degree=3, include_bias=True)

drift_parameter_lib=ps.PolynomialLibrary(degree=3, include_bias=True) # library for model dependence on control parameters (shear stress)

# build full library for drift term
drift_lib=ps.ParameterizedLibrary(feature_library=drift_feature_lib,
    parameter_library=drift_parameter_lib,num_features=ndim,num_parameters=1) 

# build library for diffusion term
diff_feature_lib=ps.PolynomialLibrary(degree=0, include_bias=True)
diff_parameter_lib=ps.PolynomialLibrary(degree=3, include_bias=False)
diff_lib=ps.ParameterizedLibrary(feature_library=diff_feature_lib,
    parameter_library=diff_parameter_lib,num_features=ndim,num_parameters=1)

# fit model for drift term
driftModel = ps.SINDy(feature_library = drift_lib, optimizer = ps.SSR())
driftModel.fit(X_train,t=5,x_dot=Y_train,u=u_train)

# score on test set
drift_R2 = driftModel.score(X_test,x_dot=Y_test,u=u_test)
driftModel.print()

print('Coefficient of determination (R^2) for model of drift term: %f' %drift_R2)

# fit model for diffusion term
diffModel = ps.SINDy(feature_library = diff_lib, optimizer = ps.SSR())
diffModel.fit(X_train,t=5,x_dot=V_train,u=u_train)

# score on test set
diff_R2 = diffModel.score(X_test,x_dot=V_test,u=u_test)
diffModel.print()

print('Coefficient of determination (R^2) for model of diffusion term: %f' %drift_R2)
# %%
# evaluate model: pplane (stability analysis) and compare stationary probability distributions (data v model prediction) across datasets
myModel = [driftModel,diffModel]

pplane_xlim = [-3,4]
bin_xlim = [-4,5]

if PCs[1] == 1:
    pplane_ylim = [-3.5,1.5]
    bin_ylim = [-4,2.5]
else:
    pplane_ylim = [-1,6]
    bin_ylim = [-2.5,8]


# fix bins and centers for all datasets
Nbins = [50 for i in range(ndim)]
bin_limits = [bin_xlim,bin_ylim]
bins, centers = eareg.get_bins(Nbins,bin_limits=bin_limits)

plt_args = {'pplane_xlim': pplane_xlim, 'pplane_ylim': pplane_ylim, 'pplane_N': 50,
            'plt_xlabel': 'PC'+str(PCs[0]+1), 'plt_ylabel': 'PC'+str(PCs[1]+1),
            'truncate_p':[True,[0,Nbins[0]-0],[0,Nbins[1]-0]]}

# %%
# fix bins and centers for all datasets

for ds_ID in [0,1,2,3,6]:
    print('**** Running model analysis for dataset',ds_ID,'**** \n')
    my_mv = list_of_datasets[ds_ID]
    mv_name = eaio.get_dataset_name(my_mv)

    df_proj = eaio.project_PCA_one_dataset(df,pca,'group',my_mv)

    feat_cols = [str(i) for i in PCs]

    traj_list, flow_list = eareg.get_2pt_traj_and_flow(df_proj,mv_name,feat_cols=feat_cols,verbose=True)
    del df_proj # free up memory
    num_flow = len(flow_list)

    # NEED TO CHANGE HOW YOU SPECIFY STATIONARY POINTS NOW
    for j in range(num_flow): # get bins and centers for data at high and low flow    
        print('**** Shear stress u =',flow_list[j],'dyn/cm^2 **** \n')
        plot_tuple = model_analysis.run_model_analysis_2D(myModel,traj_list[j],bins,centers,flow_list[j],args=plt_args)


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
    fig, ax = viz.init_plot()
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
fig, ax = viz.init_plot()
u_stable = np.array(u_stable)
fpt_stable = np.array(fpt_stable)
ax.scatter(fpt_stable[1:-7,0],fpt_stable[1:-7,1],
            c=u_stable[1:-7],cmap='bwr',edgecolors='k')

ax.xlabel('PC'+str(PCs[0]+1))
ax.ylabel('PC'+str(PCs[1]+1))
ax.colorbar(label='dyn/cm$^2$')
ax.title('Stable fixed points by shear stress')


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

fig, ax = viz.init_plot()
ax.plot(u_range,epr,'-o',color='k')
ax.xlabel('Shear stress (dyn/cm$^2$)')
ax.ylabel('Entropy production rate')


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

fig,ax = viz.plot_gen_potential_2D(U,centers_fine[0],centers_fine[1],cmap='jet',surf=False)
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

fig,ax = viz.plot_gen_potential_2D(U,centers_fine[0],centers_fine[1],
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
