# %%
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import pysindy as ps
import numdifftools as nd

import cellsmap.analyses.utils.gen_potential as gp
from cellsmap.analyses.utils import viz
from cellsmap.analyses.utils import pplane

import cellsmap.analyses.playground.ea.utils.io as eaio
import cellsmap.analyses.playground.ea.utils.viz as eaviz
import cellsmap.analyses.playground.ea.utils.regression as eareg
import cellsmap.analyses.playground.ea.utils.model_eval as model_eval
import cellsmap.analyses.playground.ea.utils.model_analysis as model_analysis

# %%

path_to_data = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs2/eval/runs/diffae/large_latent_large_encoder/2024-11-25_09-43-32/patched.parquet'
savedir = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/ea/sindy_reg_diffAE_test/'

eaio.make_savedir(savedir,subfolders=False)

# %%
df = eaio.load_array(path_to_data)
df.head()
# %%
metadata_col = ['filename_or_obj','T','start_x','start_y']
df_ = eaio.rm_metadata(df,metadata_col) # remove metadata columns

pca = eaio.get_PCA(df_)

del df_ # free up memory

list_of_datasets = eaio.get_list_of_datasets(df,'filename_or_obj',verbose=True)

fixed_data_idx = [0, 6, 7] # in list of datasets, these are indexes of the fixed data
list_of_live = [my_path for i, my_path in enumerate(list_of_datasets) if i not in fixed_data_idx]

# %%
my_mv = list_of_live[6]
mv_name = eaio.get_dataset_name(my_mv)
feats_proj = eaio.project_PCA_one_dataset(df,pca, 'filename_or_obj', my_mv)

fig1,ax1 = eaviz.plot_top_3_PCs(feats_proj)
ax1[0].set_ylim([-8,11])
ax1[1].set_ylim([-11,5])
ax1[2].set_ylim([-4.5,4.5])

fig2,ax2 = eaviz.plot_PCA_projection(feats_proj,mv_name)
ax2.set_xlim([-4,8])
ax2.set_ylim([-2,4])

# %%
# fit model to multiple datasets
PCs = [0]
ndim = len(PCs)

X_train_list = []
X_test_list = []
Y_train_list = []
Y_test_list = []
V_train_list = []
V_test_list = []
u_train_list = []
u_test_list = []

Nbins = [40 for i in range(ndim)]

# need to filter out the 'spikes' in datasets 5 and 7 (bubble)

for ds_ID in [0,1,2,3,4,6]:
    print('**** Generating train/test sets for dataset',ds_ID,'**** \n')
    my_mv = list_of_live[ds_ID]
    mv_name = eaio.get_dataset_name(my_mv)
    feats_proj = eaio.project_PCA_one_dataset(df,pca,'filename_or_obj', my_mv)

    data_all, u_traj, u_list = eareg.get_traj_and_flow(feats_proj,mv_name,PCs=PCs,verbose=True)
    del feats_proj # free up memory
    num_flow = len(u_list)

    bins = []
    centers = []

    f_KM = []
    D_KM = []
    f_err = []
    D_err = []

    for j in range(num_flow): # get bins and centers for data at high and low flow
        bins_temp, centers_temp = eareg.get_bins(Nbins,data=data_all[j])
        bins.append(bins_temp)
        centers.append(centers_temp)

        f_KM_temp, D_KM_temp, f_err_temp, D_err_temp = eareg.KM_avg_ND(data_all[j], bins[j], dt=5)
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
sigmoid_range = range(4,6)

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
parameter_lib=ps.PolynomialLibrary(degree=1, include_bias=True)
full_lib=ps.ParameterizedLibrary(feature_library=feature_lib,
    parameter_library=parameter_lib,num_features=1,num_parameters=1)

driftModel = ps.SINDy(feature_library = full_lib, optimizer = ps.SSR())
driftModel.fit(X_train,t=5,x_dot=Y_train,u=u_train)

drift_R2 = driftModel.score(X_test,x_dot=Y_test,u=u_test)
driftModel.print()

print('Coefficient of determination (R^2) of drift (RBF kernel) model on test set: %f' %drift_R2)

diff_feature_lib=ps.PolynomialLibrary(degree=0, include_bias=True)
diff_parameter_lib=ps.PolynomialLibrary(degree=0, include_bias=True)
diff_lib=ps.ParameterizedLibrary(feature_library=diff_feature_lib,
    parameter_library=diff_parameter_lib,num_features=ndim,num_parameters=1)


diffModel = ps.SINDy(feature_library = diff_lib, optimizer = ps.SSR())
diffModel.fit(X_train,t=5,x_dot=V_train,u=u_train)

diff_R2 = diffModel.score(X_test,x_dot=V_test,u=u_test)
diffModel.print()

print('Coefficient of determination (R^2) of diffusion (RBF kernel) model on test set: %f' %diff_R2)
# %%
myModel = [driftModel,diffModel]

plt_args = {'pplane_xlim': [-2,3], 'pplane_N': 50,
            'plt_xlabel': 'PC'+str(PCs[0]+1)}

# fix bins and centers for all datasets
Nbins = [60 for i in range(ndim)]
bin_limits = [[-7,10]]
bins, centers = eareg.get_bins(Nbins,bin_limits=bin_limits)

# %%
for ds_ID in [0,1,2,3,4,6]:
    print('**** Running model analysis for dataset',ds_ID,'**** \n')
    my_mv = list_of_live[ds_ID]
    mv_name = eaio.get_dataset_name(my_mv)
    feats_proj = eaio.project_PCA_one_dataset(df,pca,'filename_or_obj', my_mv)

    data_all, u_traj, u_list = eareg.get_traj_and_flow(feats_proj,mv_name,PCs=PCs,verbose=True)
    del feats_proj # free up memory
    num_flow = len(u_list)


    for j in range(num_flow): # get bins and centers for data at high and low flow    
        print('**** Shear stress u =',u_list[j],'dyn/cm^2 **** \n')
        plot_tuple = model_analysis.run_model_analysis_1D(myModel,data_all[j],bins[0],centers[0],u_list[j],args=plt_args)

# %%
u_range = np.linspace(0,40,40)

fpt_dict = {}

x1_lims = [-5,7]

x1 = np.linspace(x1_lims[0],x1_lims[1],50)
x1_coarse = np.linspace(x1_lims[0],x1_lims[1],20)

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
        if fpt[0]<x1[0]-0.5*abs(x1[0]) or fpt[0]>x1[-1]+0.5*abs(x1[-1]):
            continue
        else:
            fpts_new.append(fpt)
            fpt_types_new.append(fpt_types[fpts.index(fpt)])

    fpt_dict[str(u)] = {}
    fpt_dict[str(u)]['fixed_points'] = fpts_new
    fpt_dict[str(u)]['fixed_point_types'] = fpt_types_new
# %%
ylims = [-2,3]
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

    if i == 0:
        fig, ax = plt.subplots()
        ax.plot(centers[0],P)

        fig, ax = plt.subplots()
        ax.plot(centers[0],J[0])

    D_mat = gp.expand_to_matrix(D_vals.reshape((1,-1)))
    epr[i] = gp.entropy_production(J.reshape((1,-1)), D_mat, P, centers)


# %%
plt.plot(u_range,epr,'-o',color='k')
# %%
fig, ax = plt.subplots()
ax.plot(centers[0],P)

fig, ax = plt.subplots()
ax.plot(centers[0],J[0])
# %%
