# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysindy as ps
import numdifftools as nd

import cellsmap.util.pca as cmpca
import cellsmap.analyses.utils.pplane as pplane
import cellsmap.analyses.utils.gen_potential as gp
import cellsmap.analyses.utils.io as eaio
import cellsmap.analyses.utils.regression as eareg
import cellsmap.analyses.utils.viz as eaviz
import cellsmap.analyses.utils.model_analysis as model_analysis
import cellsmap.analyses.utils.model_eval as model_eval

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
# %%
# write io function that builds this from data config
title_dict = {'20241016_20X':'24H High, 24H Low',
              '20241105_20X':'24H Low, 24H High (11/5/24)',
              '20241120_20X':'48H High',
              '20241203_20X':'48H Low',
              '20241210_20X':'48H No Flow 1',
              '20241217_20X':'48H No Flow 2',
              '20250224_GE00006991_20X':'24H Low, 24H High (2/24/25)',}

ds_ID = 0
my_mv = list_of_datasets[ds_ID]
mv_name = eaio.get_dataset_name(my_mv)
df_proj = eaio.project_PCA_one_dataset(df,pca, 'group', my_mv)
feats_proj = eaio.df_to_array(df_proj,[str(i) for i in range(8)])

fig2,ax2 = eaviz.plot_PCA_projection(feats_proj, title_dict[mv_name])

# %%
# now fit to multiple datasets
PCs = [0]
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

f_coeffs = []

D_coeffs = []


Nbins = [15 for i in range(ndim)]

thresh = None

for ds_ID, my_mv in enumerate(list_of_datasets): 
    mv_name = eaio.get_dataset_name(my_mv)
    print('**** Generating train/test sets for',mv_name,'dataset **** \n')

    df_proj = eaio.project_PCA_one_dataset(df,pca,'group',my_mv)

    feat_cols = [str(i) for i in PCs]

    traj_list, flow_list = eareg.get_2pt_traj_and_flow(df_proj,mv_name,
                                                       feat_cols=feat_cols,
                                                       verbose=True)
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

        f_KM_temp, D_KM_temp, f_err_temp, D_err_temp = eareg.KM_avg_ND(traj_list[j], bins[j], dt=5, threshold=thresh)
        f_KM.append(f_KM_temp)
        D_KM.append(D_KM_temp)
        f_err.append(f_err_temp)
        D_err.append(D_err_temp)

    f_KM_noNAN = []
    D_KM_noNAN = []
    X_pts_noNAN = []

    for j in range(num_flow):
        f_KM_noNAN_temp, X_pts_temp, mask = eareg.masked_vector_field(f_KM[j], np.array(np.meshgrid(*centers[j])).T)
        D_KM_noNAN_temp, _, _ = eareg.masked_vector_field(D_KM[j], np.array(np.meshgrid(*centers[j])).T)
        f_err_noNAN_temp = f_err[j][mask]
        D_err_noNAN_temp = D_err[j][mask]
        
        f_KM_noNAN.append(f_KM_noNAN_temp)
        D_KM_noNAN.append(D_KM_noNAN_temp)
        X_pts_noNAN.append(X_pts_temp)

        # ploynomial fit to f and D
        p_f = np.polyfit(X_pts_noNAN[j].flatten(),
                         f_KM_noNAN[j].flatten(),4,
                         w=1/f_err_noNAN_temp.flatten())
        f_coeffs.append(np.array([p_f,flow_list[j]*np.ones_like(p_f)]).T)
        
        p_D = np.polyfit(X_pts_noNAN[j].flatten(),
                         D_KM_noNAN[j].flatten(),4,
                         w=1/D_err_noNAN_temp.flatten())
        D_coeffs.append(np.array([p_D,flow_list[j]*np.ones_like(p_D)]).T)

        fig,ax = plt.subplots(1,2,figsize=(12,5))
        ax[0].plot([-1.25,1.25],np.zeros(2),'b--',alpha=0.5)
        ax[0].plot(X_pts_noNAN[j],
                   np.polyval(p_f,X_pts_noNAN[j].flatten()),'k-',alpha=0.8)
        ax[0].plot(X_pts_noNAN[j],f_KM_noNAN[j],'ro')
        ax[0].set_xlabel('PC'+str(PCs[0]+1))
        ax[0].set_title('f(PC'+str(PCs[0]+1)+')')
        ax[0].set_xlim([-1.25,1.25])
        ax[0].set_ylim([-0.017,0.02])

        ax[1].plot(X_pts_noNAN[j],
                   np.polyval(p_D,X_pts_noNAN[j].flatten()),'k-',alpha=0.8)
        ax[1].plot(X_pts_noNAN[j],D_KM_noNAN[j],'ro')
        ax[1].set_xlabel('PC'+str(PCs[0]+1))
        ax[1].set_title('D(PC'+str(PCs[0]+1)+')')
        ax[1].set_xlim([-1.25,1.25])
        ax[1].set_ylim([0,0.002])

        fig.suptitle('Shear = '+str(flow_list[j])+' dyn/cm^2')
        plt.tight_layout()


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

# %%
rho_list = []
for ds_ID, my_mv in enumerate(list_of_datasets): 
    mv_name = eaio.get_dataset_name(my_mv)

    df_proj = eaio.project_PCA_one_dataset(df,pca,'group',my_mv)
    feats_proj = eaio.df_to_array(df_proj,[str(i) for i in range(8)])
    _, flow_list = eareg.get_traj_and_flow(df_proj,mv_name,PCs=PCs,verbose=False)
    num_flow = len(flow_list)
    change_frame = eaio.get_flow_change_frame(mv_name)
    for j in range(num_flow):
        if j == 0:
            rho_list.append(feats_proj[:,0,1].mean())
        else:
            rho_list.append(feats_proj[:,change_frame,1].mean())
rho_list = np.array(rho_list)
# %%

for coeff_idx in range(5):
    p_ = [f_arr[coeff_idx,0] for f_arr in f_coeffs]
    u = [f_arr[coeff_idx,1] for f_arr in f_coeffs]
    z = np.polyfit(u,p_,4)
    fig,ax = plt.subplots()
    ax.plot(np.linspace(0,35,20),np.polyval(z,np.linspace(0,35,20)),'k--')
    ax.plot(u,p_,'bx')
    ax.set_xlabel('Shear stress (dyn/cm^2)')
    ax.set_title('Coeff of $x^'+str(coeff_idx)+'$ in polyfit to f')
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
sigmoid_range = range(4,5)

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
# feature_lib = ps.ConcatLibrary([ps.PolynomialLibrary(degree=2, 
#                                 include_bias=True),
#                                 # sigmoid_lib])
feature_lib = ps.PolynomialLibrary(degree=4, include_bias=True)
parameter_lib=ps.PolynomialLibrary(degree=4, include_bias=True)
full_lib=ps.ParameterizedLibrary(feature_library=feature_lib,
    parameter_library=parameter_lib,num_features=1,num_parameters=1)
# full_lib = feature_lib

driftModel = ps.SINDy(feature_library = full_lib, optimizer = ps.SSR())
driftModel.fit(X_train,t=5,x_dot=Y_train,u=u_train)

drift_R2 = driftModel.score(X_test,x_dot=Y_test,u=u_test)
driftModel.print()

print('Coefficient of determination (R^2) of drift coefficient model on test set: %f' %drift_R2)

diff_feature_lib=ps.PolynomialLibrary(degree=0, include_bias=True)
diff_parameter_lib=ps.PolynomialLibrary(degree=4, include_bias=True)
diff_lib=ps.ParameterizedLibrary(feature_library=diff_feature_lib,
    parameter_library=diff_parameter_lib,num_features=ndim,num_parameters=1)
#diff_lib = diff_feature_lib

diffModel = ps.SINDy(feature_library = diff_lib, optimizer = ps.SSR())
diffModel.fit(X_train,t=5,x_dot=V_train,u=u_train)

diff_R2 = diffModel.score(X_test,x_dot=V_test,u=u_test)
diffModel.print()

print('Coefficient of determination (R^2) of diffusion coefficient model on test set: %f' %diff_R2)
# %%
myModel = [driftModel,diffModel]
if PCs[0] == 0:
    pplane_xlim = [-1,1]
    bin_xlim = [-1.25,1.25]
elif PCs[0] == 2:
    pplane_xlim = [-1,4]
    bin_xlim = [-2,5]

plt_args = {'pplane_xlim': pplane_xlim, 'pplane_N': 50,
            'plt_xlabel': 'PC'+str(PCs[0]+1)}

# fix bins and centers for all datasets
Nbins = [60 for i in range(ndim)]
bin_limits = [bin_xlim]
bins, centers = eareg.get_bins(Nbins,bin_limits=bin_limits)

for ds_ID in [1,2,3,6]:
    print('**** Running model analysis for dataset',ds_ID,'**** \n')
    my_mv = list_of_datasets[ds_ID]
    mv_name = eaio.get_dataset_name(my_mv)

    df_proj = eaio.project_PCA_one_dataset(df,pca,'group',my_mv)

    feat_cols = [str(i) for i in PCs]

    traj_list, flow_list = eareg.get_traj_and_flow(df_proj,mv_name,PCs=PCs,verbose=True)
    del df_proj # free up memory
    num_flow = len(flow_list)

    for j in range(num_flow): # get bins and centers for data at high and low flow    
        print('**** Shear stress u =',flow_list[j],'dyn/cm^2 **** \n')
        plot_tuple = model_analysis.run_model_analysis_1D(myModel,traj_list[j],bins[0],centers[0],flow_list[j],args=plt_args)

# %%
u_range = np.linspace(0,35,40)

fpt_dict = {}

x1 = np.linspace(pplane_xlim[0],pplane_xlim[1],50)
x1_coarse = np.linspace(pplane_xlim[0],pplane_xlim[1],20)

f = model_eval.scalar_function(driftModel)

for u in u_range:

    def myFlow(x):
        return f(x,u)
    flowJacobian = nd.Derivative(myFlow)

    init_coarse = [x1_coarse[i] for i in range(len(x1_coarse))]
    fpts = pplane.get_fps(myFlow,init_coarse) # get fixed points
    fpt_types = []
    if len(fpts) > 0:
        for fpt in fpts:
            fptStability = pplane.find_stability(flowJacobian(fpt),ndim=1)
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
        # if fpt[0]<x1[0]-0.5*abs(x1[0]) or fpt[0]>x1[-1]+0.5*abs(x1[-1]):
        #     continue
        # else:
        fpts_new.append(fpt)
        fpt_types_new.append(fpt_types[fpts.index(fpt)])

    fpt_dict[str(u)] = {}
    fpt_dict[str(u)]['fixed_points'] = fpts_new
    fpt_dict[str(u)]['fixed_point_types'] = fpt_types_new
# %%
ylims = [-2,2]
for u in u_range:
    fpts = fpt_dict[str(u)]['fixed_points']
    fpt_types = fpt_dict[str(u)]['fixed_point_types']
    if len(fpts) > 0:
        for i,fpt in enumerate(fpts):
            if fpt[0] < ylims[0] or fpt[0] > ylims[1]:
                continue

            if fpt_types[i] == 'stable':
                color = 'b'
            elif fpt_types[i] == 'unstable':
                color = 'r'
            elif fpt_types[i] == 'saddle':
                color = 'tab:purple'
            else:
                color = 'darkgoldenrod'

            plt.plot(u,fpt,'o',color=color)
            plt.xlabel('Shear stress (dyn/cm^2)')
            plt.ylabel('PC'+str(PCs[0]+1))
plt.ylim(ylims)


# %%

u_range = np.linspace(6,30,80)
# entropy production rate as a function of u
D = model_eval.scalar_function(diffModel)
epr = np.zeros(len(u_range))
for (i,u) in enumerate(u_range):   
    P = model_eval.get_stationary_probability(f,D,bins[0],centers[0],u,ndim=1)
    f_vals = f(centers[0],u)
    D_vals = D(centers[0],u)
    dx = centers[0][1]-centers[0][0]

    J = gp.probability_flux(P,f_vals.reshape((1,-1)),
                            D_vals.reshape((1,-1)),centers)
    V = J/P

    D_mat = gp.expand_to_matrix(D_vals.reshape((1,-1)))
    epr[i] = gp.entropy_production(J.reshape((1,-1)), D_mat, P, centers)


# %%
plt.plot(u_range,epr,'-o',color='k')
# %%
