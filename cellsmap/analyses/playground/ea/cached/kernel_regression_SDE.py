# %%
import numpy as np

import matplotlib.pyplot as plt

import cellsmap.analyses.utils.pplane as pplane
from cellsmap.analyses.utils.cached import gen_potential as gp
import cellsmap.analyses.utils.cached.kernel_regression as kreg

import cellsmap.analyses.playground.ea.utils.io as eaio
import cellsmap.analyses.playground.ea.utils.viz as eaviz
import cellsmap.analyses.playground.ea.utils.regression as eareg
import cellsmap.analyses.playground.ea.utils.model_analysis as model_analysis
import cellsmap.analyses.playground.ea.utils.model_eval as model_eval
import numdifftools as nd
# %%

path_to_data = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/3d_bf/2025-02-03_11-48-38/predict.parquet'
savedir = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/ea/sindy_reg_diffAE_test/'

eaio.make_savedir(savedir,subfolders=False)

# %%
df = eaio.add_metadata_from_path(eaio.load_array(path_to_data))
df.head()
list_of_datasets = eaio.get_list_of_datasets(df,'group',verbose=True)
# %%
df_ref = eaio.get_PCA_reference(df) # dataset for getting PCA reference
metadata_col = ['filename_or_obj','T','start_x','start_y','group','pca_ref','FOV_ID']
df_ref_ = eaio.rm_metadata(df_ref,metadata_col) # remove metadata columns
pca = eaio.get_PCA(df_ref_)
del df_ref_ # free up memory

fig, ax = eaviz.plot_explained_variance(pca.explained_variance_ratio_)
# %%
my_mv = list_of_datasets[0]
mv_name = eaio.get_dataset_name(my_mv)
feats_proj = eaio.project_PCA_one_dataset(df,pca, 'group', my_mv, metadata_cols=metadata_col)

fig1,ax1 = eaviz.plot_top_3_PCs(feats_proj)
ax1[0].set_ylim([-1.5,4])
ax1[1].set_ylim([-7,-2])
ax1[2].set_ylim([0,4.5])

fig2,ax2 = eaviz.plot_PCA_projection(feats_proj,mv_name)
ax2.set_xlim([-4,8])
ax2.set_ylim([-2,4])

# %%
PCs = [0,2]
ndim = len(PCs)
data_all, u_traj, u_list = eareg.get_traj_and_flow(feats_proj,mv_name,PCs=PCs,verbose=True)
num_flow = len(u_list)

Nbins = [40 for i in range(ndim)]
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

# %%
for j in range(num_flow):
    fig,ax = plt.subplots()
    err_mag = np.sqrt(np.sum(f_err[j]**2,axis=-1))
    im = ax.pcolormesh(*np.meshgrid(*bins[j]),err_mag.T)
    ax.quiver(*np.meshgrid(*centers[j]),f_KM[j][:,:,0].T,f_KM[j][:,:,1].T,color='w')
    fig.colorbar(im, ax=ax, label = 'Standard deviation')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC3')
    ax.set_title("Shear stress: "+str(u_list[j])+" dyn/cm^2")

# %%
f_KM_noNAN = []
D_KM_noNAN = []
X_pts_noNAN = []

for j in range(num_flow):
    f_KM_noNAN_temp, X_pts_temp = eareg.masked_vector_field(f_KM[j], np.array(np.meshgrid(*centers[j])).T)
    D_KM_noNAN_temp, _ = eareg.masked_vector_field(D_KM[j], np.array(np.meshgrid(*centers[j])).T)
    f_KM_noNAN.append(f_KM_noNAN_temp)
    D_KM_noNAN.append(D_KM_noNAN_temp)
    X_pts_noNAN.append(X_pts_temp)

train_frac = 0.8
seed = 47

X_train, X_test, Y_train, Y_test, V_train, V_test = eareg.train_test_all(X_pts_noNAN,f_KM_noNAN,
                                                                         D_KM_noNAN,num_flow,
                                                                         train_frac,seed,concat=True)

# create corresponding input vector u train/test
N_tot = [X_pts_noNAN[0].shape[0],X_pts_noNAN[1].shape[0]]
N_train = [int(train_frac*N_tot[0]),int(train_frac*N_tot[1])]
N_test = [N_tot[0]-N_train[0],N_tot[1]-N_train[1]]
u_train = np.concatenate((u_traj[0][0]*np.ones(N_train[0]),u_traj[1][0]*np.ones(N_train[1])))
u_test = np.concatenate((u_traj[0][0]*np.ones(N_test[0]),u_traj[1][0]*np.ones(N_test[1])))
# %%

driftModel = kreg.KernelRegression(beta=0.01).fit(X_train,Y_train,u=u_train)

drift_R2 = driftModel.score(X_test,Y_test,u=u_test)

print('Coefficient of determination (R^2) of drift (RBF kernel) model on test set: %f' %drift_R2)


diffModel = kreg.KernelRegression(beta=0.01).fit(X_train,V_train,u=u_train)

diff_R2 = diffModel.score(X_test,V_test,u=u_test)

print('Coefficient of determination (R^2) of diffusion (RBF kernel) model on test set: %f' %diff_R2)
# %%

myModel = [driftModel,diffModel]

plt_args = {'pplane_xlim': [-1,3.5], 'pplane_ylim': [-2,4], 'pplane_N': 50,
            'plt_xlabel': 'PC'+str(PCs[0]+1), 'plt_ylabel': 'PC'+str(PCs[1]+1)}

# %%
for j in range(num_flow):
    print('**** Running model analysis for u =',u_list[j],'dyn/cm^2 **** \n')
    plot_tuple = model_analysis.run_model_analysis_2D(myModel,data_all[j],bins[j],centers[j],u_list[j],args=plt_args)


# %%



# %%
# %%
# now fit model using multiple datasets
PCs = [0,2]
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

Nbins = [40 for i in range(ndim)]


for ds_ID in range(4):
    print('**** Generating train/test sets for dataset',ds_ID,'**** \n')
    my_mv = list_of_datasets[ds_ID]
    mv_name = eaio.get_dataset_name(my_mv)
    feats_proj = eaio.project_PCA_one_dataset(df,pca,'group', my_mv,metadata_cols=metadata_col)

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
driftModel = kreg.KernelRegression(beta=0.01).fit(X_train,Y_train,u=u_train)

drift_R2 = driftModel.score(X_test,Y_test,u=u_test)

print('Coefficient of determination (R^2) of drift (RBF kernel) model on test set: %f' %drift_R2)

diffModel = kreg.KernelRegression(beta=0.01).fit(X_train,V_train,u=u_train)

diff_R2 = diffModel.score(X_test,V_test,u=u_test)

print('Coefficient of determination (R^2) of diffusion (RBF kernel) model on test set: %f' %diff_R2)
# %%
myModel = [driftModel,diffModel]

plt_args = {'pplane_xlim': [-4,3.5], 'pplane_ylim': [-2,4], 'pplane_N': 50,
            'plt_xlabel': 'PC'+str(PCs[0]+1), 'plt_ylabel': 'PC'+str(PCs[1]+1)}

# fix bins and centers for all datasets
Nbins = [40 for i in range(ndim)]
bin_limits = [[-5,3.5],[-1,4.5]]
bins, centers = eareg.get_bins(Nbins,bin_limits=bin_limits)

for ds_ID in range(4):
    print('**** Running model analysis for dataset',ds_ID,'**** \n')
    my_mv = list_of_datasets[ds_ID]
    mv_name = eaio.get_dataset_name(my_mv)
    feats_proj = eaio.project_PCA_one_dataset(df,pca,'group', my_mv,metadata_cols=metadata_col)

    data_all, u_traj, u_list = eareg.get_traj_and_flow(feats_proj,mv_name,PCs=PCs,verbose=True)
    del feats_proj # free up memory
    num_flow = len(u_list)


    for j in range(num_flow): # get bins and centers for data at high and low flow    
        print('**** Shear stress u =',u_list[j],'dyn/cm^2 **** \n')
        plot_tuple = model_analysis.run_model_analysis_2D(myModel,data_all[j],bins,centers,u_list[j],args=plt_args)


# %%


u_range = np.linspace(0,35,40)

fpt_dict = {}

x1_lims = plt_args['pplane_xlim']
x2_lims = plt_args['pplane_ylim']

x1 = np.linspace(x1_lims[0],x1_lims[1],50)
x2 = np.linspace(x2_lims[0],x2_lims[1],50)
x1_coarse = np.linspace(x1_lims[0],x1_lims[1],10)
x2_coarse = np.linspace(x2_lims[0],x2_lims[1],10)

f = model_eval.vector_field_function(driftModel)
# %%
for u in u_range:

    def myFlow(x):
        return f(x,u=u)
    flowJacobian = nd.Jacobian(myFlow)

    init_coarse = [np.array([x1_coarse[i],x2_coarse[j]]) for i in range(len(x1_coarse)) for j in range(len(x2_coarse))]
    fpts = pplane.get_fps(myFlow,init_coarse) # get fixed points
    fpt_types = []
    if len(fpts) > 0:
        print('Fixed points found at shear stress u =',u,'dyn/cm^2')
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
            print('  • '+fptStability+" at x = (%5.3f,%5.3f)" % (fpt[0],fpt[1]))
    else:
        print('No fixed points found at shear stress u =',u,'dyn/cm^2')
    
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
for u in u_range:
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

            plt.plot(u,fpt[0],'o',color=color)
            plt.xlabel('Shear stress (dyn/cm^2)')
            plt.ylabel('PC'+str(PCs[0]+1))
#plt.ylim([-2,6])
# %%
for u in u_range:
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

            plt.plot(u,fpt[1],'o',color=color)
            plt.xlabel('Shear stress (dyn/cm^2)')
            plt.ylabel('PC'+str(PCs[1]+1))

# plt.ylim([-1,2])
# %%
fpt_stable = []
u_stable = []
for u in u_range:
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
plt.ylim([-0.5,0.5])

# %%
