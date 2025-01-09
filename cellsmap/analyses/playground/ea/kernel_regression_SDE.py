# %%
import numpy as np

import matplotlib.pyplot as plt

import cellsmap.analyses.utils.pplane as pplane
from cellsmap.analyses.utils import viz
import cellsmap.analyses.utils.kernel_regression.kernel_regression as kreg
import cellsmap.analyses.utils.langevin_sindy.fp_solvers as fps

import cellsmap.analyses.playground.ea.utils.io as eaio
import cellsmap.analyses.playground.ea.utils.viz as eaviz
import cellsmap.analyses.playground.ea.utils.regression as eareg
import cellsmap.analyses.playground.ea.utils.model_analysis as model_analysis

# %%

path_to_data = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs2/eval/runs/diffae/large_latent_large_encoder/2024-11-25_09-43-32/patched.parquet'
savedir = '//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/ea/kernel_reg_diffAE_test/'

eaio.make_savedir(savedir,subfolders=False)

# %%
df = eaio.load_array(path_to_data)
df.head()
# %%
metadata_col = ['filename_or_obj','T','start_x','start_y']
df_ = eaio.rm_metadata(df,metadata_col) # remove metadata columns

pca = eaio.get_PCA(df_)

del df_ # free up memory

fig, ax = eaviz.plot_explained_variance(pca.explained_variance_ratio_)
# %%
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
PCs = [0,2]
ndim = len(PCs)
data_all, u_traj, u_list = eareg.get_traj_and_flow(feats_proj,mv_name,PCs=PCs,verbose=True)
num_flow = len(u_list)
# %%
Nbins = [40 for i in range(ndim)]
bins = []
centers = []

f_KM = []
D_KM = []
f_err = []
D_err = []

for j in range(num_flow): # get bins and centers for data at high and low flow
    bins_temp, centers_temp = eareg.get_bins(data_all[j],Nbins)
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
    ax.set_title('Shear stress: '+str(u_list[j])+ ' dyn/cm^2')

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

N_tot = [X_pts_noNAN[0].shape[0],X_pts_noNAN[1].shape[0]]
N_train = [int(train_frac*N_tot[0]),int(train_frac*N_tot[1])]
N_test = [N_tot[0]-N_train[0],N_tot[1]-N_train[1]]
u_train = np.concatenate((u_list[0]*np.ones((N_train[0],1)),u_list[1]*np.ones((N_train[1],1))))
u_test = np.concatenate((u_list[0]*np.ones((N_test[0],1)),u_list[1]*np.ones((N_test[1],1))))
# %%

driftModel = kreg.KernelRegression(beta=0.01).fit(X_train,Y_train,u=u_train)

drift_R2 = driftModel.score(X_test,Y_test,u=u_test)

print('Coefficient of determination (R^2) of drift (RBF kernel) model on test set: %f' %drift_R2)


diffModel = kreg.KernelRegression(beta=0.01).fit(X_train,V_train,u=u_train)

diff_R2 = diffModel.score(X_test,V_test,u=u_test)

print('Coefficient of determination (R^2) of diffusion (RBF kernel) model on test set: %f' %diff_R2)
# %%
myModel = [driftModel,diffModel]

plt_args = {'pplane_xlim': [-4,5], 'pplane_ylim': [-2,3], 'pplane_N': 50,
            'plt_xlabel': 'PC1', 'plt_ylabel': 'PC3'}

for j in range(num_flow):
    print('**** Running model analysis for u =',u_list[j],'dyn/cm^2 **** \n')
    plot_tuple = model_analysis.run_model_analysis(myModel,data_all[j],bins[j],centers[j],u_list[j],args=plt_args)

