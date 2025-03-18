# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import pysindy as ps
import numdifftools as nd
from sklearn.model_selection import train_test_split

import cellsmap.analyses.utils.cached.gen_potential as gp
from cellsmap.analyses.utils import viz
from cellsmap.analyses.utils import pplane
import cellsmap.util.pca as cmpca
import cellsmap.util.io as io

import cellsmap.analyses.playground.ea.utils.io as eaio
import cellsmap.analyses.playground.ea.utils.viz as eaviz
import cellsmap.analyses.playground.ea.utils.regression as eareg
import cellsmap.analyses.playground.ea.utils.model_eval as model_eval
import cellsmap.analyses.playground.ea.utils.model_analysis as model_analysis

# %%

path_to_data = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_for_erin/2025-02-24_17-13-26/predict.parquet'
path_to_20241217 = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_20241217/2025-02-28_10-41-33/predict.parquet'
path_to_20250224 = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/latent_dim_8_20250224/2025-03-03_11-45-02/predict.parquet'

df = eaio.load_array(path_to_data)
df_1217 = eaio.load_array(path_to_20241217)
df_0224 = eaio.load_array(path_to_20250224)
df = pd.concat([df,df_1217,df_0224],ignore_index=True)
df, pca = cmpca.get_pca(df, num_pcs=8,scale=False)
df, bad_files = cmpca._get_outliers(df)
list_of_datasets = eaio.get_list_of_datasets(df,'group',verbose=True)
# %%

fig, ax = eaviz.plot_explained_variance(pca['pca'].explained_variance_ratio_)

# %%
# write io function that builds this from data config
title_dict = {'20241016_20X':'24H High, 24H Low',
              '20241105_20X':'24H Low, 24H High (11/5/24)',
              '20241120_20X':'48H High',
              '20241203_20X':'48H Low',
              '20241210_20X':'48H No Flow 1',
              '20241217_20X':'48H No Flow 2',
              '20250224_GE00006991_20X':'24H Low, 24H High (2/24/25)',}

ds_ID = 3
my_mv = list_of_datasets[ds_ID]
mv_name = eaio.get_dataset_name(my_mv)
df_proj = eaio.project_PCA_one_dataset(df,pca, 'group', my_mv)
feats_proj = eaio.df_to_array(df_proj,[str(i) for i in range(8)])

fig2,ax2 = eaviz.plot_PCA_projection(feats_proj, title_dict[mv_name])

# %%
PCs = [0,1]
ndim = len(PCs)

Nbins = 15*np.ones(ndim,dtype=int)

traj_list, flow_list = eareg.get_traj_and_flow(df_proj,mv_name,
                                               PCs=PCs,verbose=True)
num_flow = len(flow_list)

change_frame = eaio.get_flow_change_frame(mv_name)

# %%
flow_ID = 0
window_size = 25
if flow_ID == 0:
    windows = np.arange(0,change_frame,window_size)
    windows = np.append(windows,-1)
    rho = feats_proj[:,:change_frame,2].mean(axis=0)
else:
    windows = np.arange(change_frame,feats_proj.shape[0],window_size)
    windows = np.append(windows,-1)
    rho = feats_proj[:,change_frame:,2].mean(axis=0)

eta = np.cumsum(rho**2)

f_KM = []
D_KM = []
X_pts = []
rho_mean = []

for i in range(len(windows)-1):
    data_ = [feats_proj[j,windows[i]:windows[i+1],:2] for j in range(feats_proj.shape[0])]
    rho_mean_ = rho[windows[i]:windows[i+1]].mean()
    bins_, centers_ = eareg.get_bins(Nbins,data=data_)
    f_KM_, D_KM_, _ , _ = eareg.KM_avg_ND(data_, bins_, dt=5)
    f_KM_noNAN_, X_pts_, _ = eareg.masked_vector_field(f_KM_, np.array(np.meshgrid(*centers_)).T)
    D_KM_noNAN_, _, _ = eareg.masked_vector_field(D_KM_, np.array(np.meshgrid(*centers_)).T)
    f_KM.append(f_KM_noNAN_)
    D_KM.append(D_KM_noNAN_)
    X_pts.append(X_pts_)
    rho_mean.append(rho_mean_*np.ones((X_pts_.shape[0],1)))
    
f_KM = np.concatenate(f_KM)
D_KM = np.concatenate(D_KM)
X_pts = np.concatenate(X_pts)
rho_mean = np.concatenate(rho_mean)

train_frac = 0.8
seed = 47


X_train, X_test, Y_train, Y_test = train_test_split(X_pts, f_KM, train_size=train_frac, random_state=seed)

_, _, V_train, V_test = train_test_split(X_pts, D_KM, train_size=train_frac, random_state=seed) # same random seed to get same x points for train and test

_,_, eta_train, eta_test = train_test_split(X_pts, rho_mean, train_size=train_frac, random_state=seed)

N_tot = X_pts[0].shape[0]
N_train = int(train_frac*N_tot)
N_test = N_tot-N_train

# %%
feature_lib = ps.PolynomialLibrary(degree=4, include_bias=True)
parameter_lib=ps.PolynomialLibrary(degree=3, include_bias=True)
full_lib=ps.ParameterizedLibrary(feature_library=feature_lib,
    parameter_library=parameter_lib,num_features=ndim,num_parameters=1)

driftModel = ps.SINDy(feature_library = full_lib, optimizer = ps.SSR())
driftModel.fit(X_train,t=5,x_dot=Y_train,u=eta_train)


diff_feature_lib=ps.PolynomialLibrary(degree=4, include_bias=True)
diff_parameter_lib=ps.PolynomialLibrary(degree=3, include_bias=True)
diff_lib=ps.ParameterizedLibrary(feature_library=diff_feature_lib,
    parameter_library=diff_parameter_lib,num_features=ndim,num_parameters=1)


diffModel = ps.SINDy(feature_library = diff_lib, optimizer = ps.SSR())
diffModel.fit(X_train,t=5,x_dot=V_train,u=eta_train)

drift_R2 = driftModel.score(X_test,x_dot=Y_test,u=eta_test)
driftModel.print()

print('R^2 of drift coefficient model on test set: %f' %drift_R2)

diff_R2 = diffModel.score(X_test,x_dot=V_test,u=eta_test)
diffModel.print()

print('R^2 of diffusion coefficient model on test set: %f' %diff_R2)
# %%
myModel = [driftModel,diffModel]

pplane_xlim = [0.2,1.25]
bin_xlim = [-0.5,1.5]

pplane_ylim = [-0.5,0.5]
bin_ylim = [-0.75,1.0]

# fix bins and centers for all datasets
Nbins = [50 for i in range(ndim)]
bin_limits = [bin_xlim,bin_ylim]
bins, centers = eareg.get_bins(Nbins,bin_limits=bin_limits)

if flow_ID == 0:
    data_ = [feats_proj[j,:change_frame,:2] for j in range(feats_proj.shape[0])]
    rho_mean = rho[change_frame-window_size:change_frame].mean()
else:
    data_ = [feats_proj[j,change_frame:,:2] for j in range(feats_proj.shape[0])]
    rho_mean = rho[-window_size:].mean()

plt_args = {'pplane_xlim': pplane_xlim, 'pplane_ylim': pplane_ylim, 'pplane_N': 50,
            'plt_xlabel': 'PC'+str(PCs[0]+1), 'plt_ylabel': 'PC'+str(PCs[1]+1),
            'truncate_p':[True,[10,Nbins[0]-2],[10,Nbins[1]-2]]}

plot_tuple = model_analysis.run_model_analysis_2D(myModel,data_,bins,centers,rho_mean,args=plt_args)
# %% 
# %%
# %%
# now fit model using multiple datasets
PCs = [0,1]
ndim = len(PCs)

Nbins = 15*np.ones(ndim,dtype=int)

thresh = None # get this based on upper quantile of vector magnitudes along given PCs low flow data

# %%
window_size = 25
f_KM = []
D_KM = []
X_pts = []
rho_mean = []
shear = []

for ds_ID in range(len(list_of_datasets)): 
    print('**** Generating train/test sets for dataset',ds_ID,'**** \n')
    my_mv = list_of_datasets[ds_ID]
    mv_name = eaio.get_dataset_name(my_mv)

    df_proj = eaio.project_PCA_one_dataset(df,pca, 'group', my_mv)
    feats_proj = eaio.df_to_array(df_proj,[str(i) for i in range(8)])

    _, shear_list = eareg.get_traj_and_flow(df_proj,mv_name,PCs=PCs,verbose=True)
    num_flow = len(shear_list)
    change_frame = eaio.get_flow_change_frame(mv_name)
    if 0 in shear_list: # for now, skip using no flow data
        continue

    for j in range(num_flow): # get bins and centers for data at high and low flow
        if j == 0:
            windows = np.arange(0,change_frame,window_size)
            windows = np.append(windows,-1)
            rho = feats_proj[:,:change_frame,2].mean(axis=0)
        else:
            windows = np.arange(change_frame,feats_proj.shape[0],window_size)
            windows = np.append(windows,-1)
            rho = feats_proj[:,change_frame:,2].mean(axis=0)
            
        eta = np.cumsum(rho**2)

        for i in range(len(windows)-1):
            data_ = [feats_proj[j,windows[i]:windows[i+1],:2] for j in range(feats_proj.shape[0])]
            rho_mean_ = rho[windows[i]:windows[i+1]].mean()
            bins_, centers_ = eareg.get_bins(Nbins,data=data_)
            f_KM_, D_KM_, _ , _ = eareg.KM_avg_ND(data_, bins_, dt=5)
            f_KM_noNAN_, X_pts_, _ = eareg.masked_vector_field(f_KM_, np.array(np.meshgrid(*centers_)).T)
            D_KM_noNAN_, _, _= eareg.masked_vector_field(D_KM_, np.array(np.meshgrid(*centers_)).T)
            f_KM.append(f_KM_noNAN_)
            D_KM.append(D_KM_noNAN_)
            X_pts.append(X_pts_)
            rho_mean.append(rho_mean_*np.ones((X_pts_.shape[0],1)))
            shear.append(shear_list[j]*np.ones((X_pts_.shape[0],1)))
        
f_KM = np.concatenate(f_KM)
D_KM = np.concatenate(D_KM)
X_pts = np.concatenate(X_pts)
rho_mean = np.concatenate(rho_mean)
shear = np.concatenate(shear)

train_frac = 0.8
seed = 47

# should write in generating u_train and u_test

X_train, X_test, Y_train, Y_test = train_test_split(X_pts, f_KM, train_size=train_frac, random_state=seed)

_, _, V_train, V_test = train_test_split(X_pts, D_KM, train_size=train_frac, random_state=seed) # same random seed to get same x points for train and test

_,_, eta_train, eta_test = train_test_split(X_pts, rho_mean, train_size=train_frac, random_state=seed)

_,_, shear_train, shear_test = train_test_split(X_pts, shear, train_size=train_frac, random_state=seed)

N_tot = X_pts[0].shape[0]
N_train = int(train_frac*N_tot)
N_test = N_tot-N_train

u_train = np.concatenate([eta_train,shear_train],axis=1)
u_test = np.concatenate([eta_test,shear_test],axis=1)

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

# sigmoid_lib=ps.CustomLibrary(library_functions=sigmoid_funcs,
#                              function_names=func_names)
# feature_lib = ps.ConcatLibrary([ps.PolynomialLibrary(degree=3, 
#                                 include_bias=True),
#                                 sigmoid_lib])
feature_lib = ps.PolynomialLibrary(degree=4, include_bias=True)
parameter_lib=ps.PolynomialLibrary(degree=3, include_bias=True)
full_lib=ps.ParameterizedLibrary(feature_library=feature_lib,
    parameter_library=parameter_lib,num_features=ndim,num_parameters=1)

driftModel = ps.SINDy(feature_library = full_lib, optimizer = ps.SSR())
driftModel.fit(X_train,t=5,x_dot=Y_train,u=u_train)


diff_feature_lib=ps.PolynomialLibrary(degree=4, include_bias=True)
diff_parameter_lib=ps.PolynomialLibrary(degree=3, include_bias=True)
diff_lib=ps.ParameterizedLibrary(feature_library=diff_feature_lib,
    parameter_library=diff_parameter_lib,num_features=ndim,num_parameters=1)


diffModel = ps.SINDy(feature_library = diff_lib, optimizer = ps.SSR())
diffModel.fit(X_train,t=5,x_dot=V_train,u=u_train)

drift_R2 = driftModel.score(X_test,x_dot=Y_test,u=u_test)
driftModel.print()

print('R^2 of drift coefficient model on test set: %f' %drift_R2)

diff_R2 = diffModel.score(X_test,x_dot=V_test,u=u_test)
diffModel.print()

print('R^2 of diffusion coefficient model on test set: %f' %diff_R2)
# %%
myModel = [driftModel,diffModel]

pplane_xlim = [-1,1]
bin_xlim = [-1.5,1.5]

pplane_ylim = [-0.75,1.5]
bin_ylim = [-1,2]

# fix bins and centers for all datasets
Nbins = [50 for i in range(ndim)]
bin_limits = [bin_xlim,bin_ylim]
bins, centers = eareg.get_bins(Nbins,bin_limits=bin_limits)

plt_args = {'pplane_xlim': pplane_xlim, 'pplane_ylim': pplane_ylim, 'pplane_N': 50,
            'plt_xlabel': 'PC'+str(PCs[0]+1), 'plt_ylabel': 'PC'+str(PCs[1]+1),
            'truncate_p':[True,[5,Nbins[0]-5],[5,Nbins[1]-5]]}

# %%
# fix bins and centers for all datasets

for ds_ID in range(len(list_of_datasets)):
    print('**** Running model analysis for dataset',ds_ID,'**** \n')
    my_mv = list_of_datasets[ds_ID]
    mv_name = eaio.get_dataset_name(my_mv)
    df_proj = eaio.project_PCA_one_dataset(df,pca, 'group', my_mv)
    feats_proj = eaio.df_to_array(df_proj,[str(i) for i in range(8)])

    _, shear_list = eareg.get_traj_and_flow(df_proj,mv_name,PCs=PCs,verbose=True)
    num_flow = len(shear_list)
    change_frame = eaio.get_flow_change_frame(mv_name)
    if 0 in shear_list: # right now, only using timepoints 300:420 of no flow
        continue
    rho = feats_proj[:,:,2].mean(axis=0)

    for j in range(num_flow): # get bins and centers for data at high and low flow
        if j == 0:
            data_ = [feats_proj[j,:change_frame,:2] for j in range(feats_proj.shape[0])]
            rho_mean = rho[change_frame-window_size:change_frame].mean()
        else:
            data_ = [feats_proj[j,change_frame:,:2] for j in range(feats_proj.shape[0])]
            rho_mean = rho[-window_size:].mean()
        u = np.array([rho_mean,shear_list[j]])[np.newaxis,:]
        print('**** Shear stress u =',shear_list[j],'dyn/cm^2 **** \n')
        plot_tuple = model_analysis.run_model_analysis_2D(myModel,data_,bins,centers,u,args=plt_args)

# %%
