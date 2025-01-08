# %%
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import pysindy as ps

import cellsmap.analyses.utils.gen_potential as gp
from cellsmap.analyses.utils import viz

import cellsmap.analyses.playground.ea.utils.io as eaio
import cellsmap.analyses.playground.ea.utils.viz as eaviz
import cellsmap.analyses.playground.ea.utils.regression as eareg
import cellsmap.analyses.playground.ea.utils.model_eval as model_eval

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

fig, ax = eaviz.plot_explained_variance(pca.explained_variance_ratio_)
# %%
list_of_datasets = eaio.get_list_of_datasets(df,'filename_or_obj',verbose=True)
# %%
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

X_train, X_test, Y_train, Y_test, V_train, V_test = eareg.train_test_all(X_pts_noNAN,f_KM_noNAN,D_KM_noNAN,num_flow,
                                                                         train_frac,seed,concat=True)

# create corresponding input vector u train/test
N_tot = [X_pts_noNAN[0].shape[0],X_pts_noNAN[1].shape[0]]
N_train = [int(train_frac*N_tot[0]),int(train_frac*N_tot[1])]
N_test = [N_tot[0]-N_train[0],N_tot[1]-N_train[1]]
u_train = np.concatenate((u_traj[0][0]*np.ones(N_train[0]),u_traj[1][0]*np.ones(N_train[1])))
u_test = np.concatenate((u_traj[0][0]*np.ones(N_test[0]),u_traj[1][0]*np.ones(N_test[1])))
# %%
driftModel = ps.SINDy(feature_library = ps.PolynomialLibrary(degree=3), optimizer = ps.SSR())
driftModel.fit(X_train,t=5,x_dot=Y_train,u=u_train)

diffModel = ps.SINDy(feature_library = ps.PolynomialLibrary(degree=3), optimizer = ps.SSR())
diffModel.fit(X_train,t=5,x_dot=V_train,u=u_train)
# %%
driftModel.print()

print("\n")
print("Drift model R^2: ", driftModel.score(X_test,t=5,x_dot=Y_test,u=u_test))
print("\n")

diffModel.print()

print("\n")
print("Diffusion model R^2: ", diffModel.score(X_test,t=5,x_dot=V_test,u=u_test))

# %%

myModel = [driftModel,diffModel]

plt_args = {'pplane_xlim': [-4,5], 'pplane_ylim': [-2,3], 'pplane_N': 50,
            'plt_xlabel': 'PC1', 'plt_ylabel': 'PC3'}

# %%
for j in range(num_flow):
    print('**** Running model analysis for u =',u_list[j],'dyn/cm^2 **** \n')
    plot_tuple = model_eval.run_model_analysis(myModel,data_all[j],bins[j],centers[j],u_list[j],args=plt_args)

# %%


# %%



################### Generalized potential energy landscape ###################

N_fine = 100

x_fine = np.linspace(x1[0],x1[-1],N_fine)
y_fine = np.linspace(x2[0],x2[-1],N_fine)
centers_fine = [x_fine,y_fine]
X1,X2 = np.meshgrid(x_fine,y_fine)
# %%
f_vals_new = f_mesh([X1,X2],u_traj[0][0]).T
D_vals_new = D_mesh([X1,X2],u_traj[0][0]).T

print('**** Plotting generalized potential energy landscape **** \n')
U, grad_term, flux_term = gp.grad_flux_decomposition(f_vals_new,D_vals_new,centers_fine,tol=1e-6)
grad_norm =grad_term.copy()
flux_norm = flux_term.copy()
# grad_norm = grad_term/(np.sqrt(grad_term[0]**2+grad_term[1]**2))
# flux_norm = flux_term/(np.sqrt(flux_term[0]**2+flux_term[1]**2))
fig,ax = viz.plot_gen_potential_2D(U,centers_fine[0],centers_fine[1],cmap='jet',surf=False)
ax.set_xlabel('PC1')
ax.set_ylabel('PC3')
axins = zoomed_inset_axes(ax, 2, loc=1)
im = axins.imshow(U.T,interpolation='nearest', origin='lower',
    extent=[x_fine[0], x_fine[-1], y_fine[0], y_fine[-1]],
    cmap='jet', aspect=(x_fine[-1]-x_fine[0])/(y_fine[-1]-y_fine[0]))
axins.set_xlim(2.5, 4)
axins.set_ylim(0, 2)

# %%
# same but with vector field decomposition
fig,ax = viz.plot_gen_potential_2D(U,centers_fine[0],centers_fine[1],cmap='jet',surf=False)
downsample=8
ax.quiver(x_fine[::downsample],y_fine[::downsample],grad_norm[0][::downsample,::downsample].T,grad_norm[1][::downsample,::downsample].T,color='w',pivot='tail')
ax.quiver(x_fine[::downsample],y_fine[::downsample],flux_norm[0][::downsample,::downsample].T,flux_norm[1][::downsample,::downsample].T,color='r',pivot='tail')
ax.set_xlabel('PC1',fontsize=28)
ax.set_ylabel('PC3',fontsize=28)

fig,axins = plt.subplots()
im = axins.imshow(U.T,interpolation='nearest', origin='lower',
    extent=[x_fine[0], x_fine[-1], y_fine[0], y_fine[-1]],
    cmap='jet', aspect=(x_fine[-1]-x_fine[0])/(y_fine[-1]-y_fine[0]))
downsample = 4
width = 0.015*(0-(-1))/(4-2)
axins.quiver(x_fine[::downsample],y_fine[::downsample],grad_norm[0][::downsample,::downsample].T,grad_norm[1][::downsample,::downsample].T,color='w',pivot='tail',scale=0.2,width=width,edgecolor='k',linewidths=0.5)
axins.quiver(x_fine[::downsample],y_fine[::downsample],flux_norm[0][::downsample,::downsample].T,flux_norm[1][::downsample,::downsample].T,color='r',pivot='tail',scale=0.2,width=width)
axins.set_xlim(2,4)
axins.set_ylim(1.1,2.1)
# %%

f_vals_new = f_mesh([X1,X2],u_traj[1][0]).T
D_vals_new = D_mesh([X1,X2],u_traj[1][0]).T
print('**** Plotting generalized potential energy landscape **** \n')
U, grad_term, flux_term = gp.grad_flux_decomposition(f_vals_new,D_vals_new,centers_fine,tol=1e-6)
grad_norm = grad_term.copy()
flux_norm = flux_term.copy()
# grad_norm = grad_term/(np.sqrt(grad_term[0]**2+grad_term[1]**2))
# flux_norm = flux_term/(np.sqrt(flux_term[0]**2+flux_term[1]**2))
fig,ax = viz.plot_gen_potential_2D(U,centers_fine[0],centers_fine[1],cmap='jet',surf=False)
ax.set_xlabel('PC1')
ax.set_ylabel('PC3')

# same but with vector field decomposition

# %%
fig,ax = viz.plot_gen_potential_2D(U,centers_fine[0],centers_fine[1],cmap='jet',surf=False)
downsample=8
ax.quiver(x_fine[::downsample],y_fine[::downsample],grad_norm[0][::downsample,::downsample].T,grad_norm[1][::downsample,::downsample].T,color='w',pivot='tail')
ax.quiver(x_fine[::downsample],y_fine[::downsample],flux_norm[0][::downsample,::downsample].T,flux_norm[1][::downsample,::downsample].T,color='r',pivot='tail')
ax.set_xlabel('PC1')
ax.set_ylabel('PC3')


# %%
fig,axins = plt.subplots()
im = axins.imshow(U.T,interpolation='nearest', origin='lower',
    extent=[x_fine[0], x_fine[-1], y_fine[0], y_fine[-1]],
    cmap='jet', aspect=(x_fine[-1]-x_fine[0])/(y_fine[-1]-y_fine[0]))
downsample = 4
width = 0.015*(0-(-1))/(4-2)
axins.quiver(x_fine[::downsample],y_fine[::downsample],grad_norm[0][::downsample,::downsample].T,grad_norm[1][::downsample,::downsample].T,color='w',pivot='tail',scale=0.17,width=width,edgecolor='k',linewidths=0.5)
axins.quiver(x_fine[::downsample],y_fine[::downsample],flux_norm[0][::downsample,::downsample].T,flux_norm[1][::downsample,::downsample].T,color='r',pivot='tail',scale=0.17,width=width)
axins.set_xlim(-3,-1)
axins.set_ylim(-1.1,-0.1)
# %%
