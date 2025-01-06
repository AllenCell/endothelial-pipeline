# %%
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from sklearn.model_selection import train_test_split
import pysindy as ps

from cellsmap.analyses.workflows.fit_SDE_model import get_traj
import cellsmap.analyses.utils.gen_potential as gp
from cellsmap.analyses.workflows.analyze_SDE_model import plot_gen_potential
import cellsmap.analyses.utils.pplane as pplane
import cellsmap.util.io as io
from cellsmap.analyses.utils import viz
import cellsmap.analyses.utils.langevin_sindy.fp_solvers as fps

import cellsmap.analyses.playground.ea.utils.io as eaio

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
# %%

plt.plot(np.cumsum(pca.explained_variance_ratio_),'b-o')
plt.ylabel('Cumulative Explained Variance')
plt.xlabel('Components')
plt.show()
# %%
list_of_datasets = eaio.get_list_of_datasets(df,'filename_or_obj',verbose=True)
# %%
fixed_data_idx = [0, 6, 7] # in list of datasets, these are indexes of the fixed data
list_of_live = [my_path for i, my_path in enumerate(list_of_datasets) if i not in fixed_data_idx]
# %%
# def get_feats_proj(df, my_mv):
#     df_one_mv = df[df['filename_or_obj'] == my_mv].copy()
#     start_x = df_one_mv[df_one_mv['T']==0]['start_x'].values.tolist()
#     start_y = df_one_mv[df_one_mv['T']==0]['start_y'].values.tolist()
#     tup_list = list(zip(start_x,start_y))

#     def pos_to_index(x,y):
#         return tup_list.index((x,y))

#     df_one_mv['crop_index'] = df_one_mv.apply(lambda x: pos_to_index(x['start_x'],x['start_y']),axis=1)

#     df_one_mv.sort_values(['crop_index','T'])
#     num_T = df_one_mv['T'].nunique()
#     num_crop = df_one_mv['crop_index'].nunique()

#     feats_proj = df_one_mv.drop(columns = ['start_x','start_y','filename_or_obj','T','crop_index']).astype(float)
#     feats_proj = pca.transform(feats_proj).reshape(num_T,num_crop,-1)
#     feats_proj = np.swapaxes(feats_proj,0,1)
#     return feats_proj

# def get_dataset_name(mv_path):
#     dataset_name = mv_path.replace('//allen/aics/assay-dev/computational/data/holistic/endos/feasibility/','')
#     dataset_name = dataset_name.replace('.ome.zarr','')
#     return dataset_name

# %%
high_endpt = []
high_flows = []
low_endpt = []
low_flows = []
short_time = []
full_time = []
for mv in list_of_live:
    feats_proj = eaio.project_PCA_one_dataset(df,pca,'filename_or_obj',mv)
    mean_feats = np.mean(feats_proj,axis=0)

    mv_name = eaio.get_dataset_name(mv)
    data_config = io.get_dataset_info(mv_name)
    first_flow = float(data_config['flow'][0][-1])
    if len(data_config['flow']) > 1:
        change_frame = int(data_config['flow'][0][1]*60/5) # change from time in hours to frame number
        second_flow = float(data_config['flow'][1][-1])
        if first_flow > second_flow: # high flow -> low flow
            high_endpt.append(mean_feats[change_frame,:])
            high_flows.append(first_flow)
            short_time.append((first_flow,mean_feats[change_frame,1]))
            low_endpt.append(mean_feats[-1,:])
            low_flows.append(second_flow)
            full_time.append((second_flow,mean_feats[-1,1]))
        else: # low flow -> high flow
            low_endpt.append(mean_feats[change_frame,:])
            low_flows.append(first_flow)
            short_time.append((first_flow,mean_feats[change_frame,1]))
            high_endpt.append(mean_feats[-1,:])
            high_flows.append(second_flow)
            full_time.append((second_flow,mean_feats[-1,1]))
    else:
        if first_flow >= 20:
            high_endpt.append(mean_feats[-1,:])
            high_flows.append(first_flow)
        else:
            low_endpt.append(mean_feats[-1,:])
            low_flows.append(first_flow)
        short_time.append((first_flow,mean_feats[-1,1]))
# %%
fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].set_title('PC1')
ax[0].set_xlabel('Flow rate')
ax[1].set_title('PC2')
ax[1].set_xlabel('Flow rate')
ax[2].set_title('PC3')
ax[2].set_xlabel('Flow rate')
for j in range(3):
    ax[j].scatter(high_flows,[high_endpt[i][j] for i in range(len(high_flows))],c='r',label='High Flow')
    ax[j].scatter(low_flows,[low_endpt[i][j] for i in range(len(low_flows))],c='b',label='Low Flow')
    ax[j].legend()
fig.suptitle('Mean value along top 3 PCs at last time point of various flow rates')
# %%
fig,ax = plt.subplots()
ax.scatter([short_time[i][0] for i in range(len(short_time))],[short_time[i][1] for i in range(len(short_time))],c='b',label='Last time point ~ 24 hrs')
ax.scatter([full_time[i][0] for i in range(len(full_time))],[full_time[i][1] for i in range(len(full_time))],c='r',label='Last time point ~ 48 hrs')
ax.set_xlabel('Flow rate')
ax.set_ylabel('PC2')
ax.legend()
# %%
# test out feats_proj and get_dataset_name
my_mv = list_of_live[6]
mv_name = eaio.get_dataset_name(my_mv)
feats_proj = eaio.project_PCA_one_dataset(df,pca, 'filename_or_obj', my_mv)

fig1, ax1 = plt.subplots(1,3,figsize=(15,5))
ax1[0].set_title('PC1')
ax1[0].set_xlabel('Frame number')
ax1[1].set_title('PC2')
ax1[1].set_xlabel('Frame number')
ax1[2].set_title('PC3')
ax1[2].set_xlabel('Frame number')

fig2,ax2 = plt.subplots()
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC3')


num_T = feats_proj.shape[1]
num_crop = feats_proj.shape[0]
for i in range(num_crop):
    ax1[0].scatter(np.arange(num_T),feats_proj[i,:,0],c = 'grey',alpha=0.5)
    ax1[1].scatter(np.arange(num_T),feats_proj[i,:,1],c = 'grey',alpha=0.5)
    ax1[2].scatter(np.arange(num_T),feats_proj[i,:,2],c = 'grey',alpha=0.5)

mean_feats = np.mean(feats_proj,axis=0)
idx = np.where(np.linalg.norm(np.diff(mean_feats,axis=0),axis=1) > 0.5)[0]
mean_feats[idx,:] = np.nan
ax2.scatter(mean_feats[:,0],mean_feats[:,2],c = range(num_T),cmap='jet')

ax1[0].plot(mean_feats[:,0],'k-')
ax1[1].plot(mean_feats[:,1],'k-')
ax1[2].plot(mean_feats[:,2],'k-')

ax1[0].set_ylim([-8,11])
ax1[1].set_ylim([-11,5])
ax1[2].set_ylim([-4.5,4.5])
fig1.suptitle(mv_name)

ax2.set_xlim([-4,8])
ax2.set_ylim([-2,4])
ax2.set_title(mv_name)

# %%
data_config = io.get_dataset_info(mv_name)
first_flow = float(data_config['flow'][0][-1])
if len(data_config['flow']) > 1:
    change_frame = int(data_config['flow'][0][1]*60/5) # change from time in hours to frame number
    second_flow = float(data_config['flow'][1][-1])
    if first_flow > second_flow:
        print('High flow until frame',change_frame)
        print('Low flow after frame',change_frame)
    else:
        print('Low flow until frame',change_frame)
        print('High flow after frame',change_frame)
    X_traj = [feats_proj[:,:change_frame,:][:,:,[0,2]],feats_proj[:,change_frame:,:][:,:,[0,2]]]
    u_traj = [first_flow*np.ones(change_frame),second_flow*np.ones(num_T-change_frame)]
    u_step = np.concatenate(u_traj)
else:
    print('Constant flow')
    X_traj = [feats_proj[:,:,[0,2]]]
    u_traj = [first_flow*np.ones(num_T)]
    u_step = first_flow*np.ones(num_T)

# %%
data_flow1 = [X_traj[0][i] for i in range(num_crop)]
data_flow2 = [X_traj[1][i] for i in range(num_crop)]
data_all = [data_flow1,data_flow2]
ndim = 2

# %%
Nbins = [40 for i in range(ndim)]
bins = []
centers = []

for j in range(2):
    bins_temp = []
    centers_temp = []
    for i in range(ndim):
        my_min = min([min(traj[:,i]) for traj in data_all[j]])
        my_max = max([max(traj[:,i]) for traj in data_all[j]])
        bin_min = 0.5*(np.floor(my_min)+np.round(my_min,1))
        bin_max = 0.5*(np.ceil(my_max)+np.round(my_max,1))
        my_bins = np.linspace(bin_min, bin_max, Nbins[i]+1)
        bins_temp.append(my_bins)
        centers_temp.append(0.5*(my_bins[1:]+my_bins[:-1]))
    bins.append(bins_temp)
    centers.append(centers_temp)


# %%
def KM_avg_ND(X,bins,dt):
    '''Kramers-Moyal average drift and diffusion estimates for N-dimensional data'''
    ndim = len(bins)
    n = len(X) # number of trajectories
    my_list = [len(bins[i])-1 for i in range(ndim)]
    my_list = my_list + [ndim,n]
    f_KM = np.nan*np.ones(my_list)
    a_KM = np.nan*np.ones(f_KM.shape)
    f_err = np.nan*np.ones(f_KM.shape)
    a_err = np.nan*np.ones(f_KM.shape)
    #inTrajVariation = False  # for computing the standard deviation of the drift and diffusion estimates - averaging over trajectories but also when a trajectory passes through the same bin multiple times
    for (j,traj) in enumerate(X):
        dX = (traj[1:] - traj[:-1])/dt # Step (like a finite-difference derivative estimate)
        dX2 = (traj[1:] - traj[:-1])**2/dt

        id_list = [np.digitize(traj[:-1,i],bins[i]) for i in range(ndim)]
        uids = list(set(zip(*id_list))) # unique bin ids
        if any([len(bins[i]) in id_list[i] for i in range(ndim)]):
            raise ValueError('Data point outside of histogram bins. Please update bounds.')

        for uid in uids:
            my_cond = 1
            for i in range(ndim):
                my_cond = my_cond*(id_list[i]==uid[i])
            mask = np.where(my_cond)[0]
            # At each histogram bin, find time series points where the state falls into this bin
            slices = [uid[i]-1 for i in range(ndim)]
            f_KM[tuple(slices)][:,j] = np.mean(dX[mask],axis=0) # Conditional average  ~ drift
            a_KM[tuple(slices)][:,j] = 0.5*np.mean(dX2[mask],axis=0) # Conditional variance  ~ diffusion

            # Estimate error by variance of samples in the bin
            if len(mask) > 1:
                #inTrajVariation = True
                f_err[tuple(slices)][:,j] = np.nanstd(dX[mask],axis=0)/np.sqrt(len(mask))
                a_err[tuple(slices)][:,j] = np.nanstd(dX2[mask],axis=0)/np.sqrt(len(mask))

    f_KM_avg = np.nanmean(f_KM,axis=-1)
    a_KM_avg = np.nanmean(a_KM,axis=-1)
    # think about how to generalize standard deviation computation to short traj vs. long traj
    f_err = np.nanmean(f_err,axis=-1) + np.nanstd(f_KM,axis=-1)
    a_err = np.nanmean(a_err,axis=-1) + np.nanstd(a_KM,axis=-1)

    return f_KM_avg, a_KM_avg, f_err, a_err
# %%

# %%
####### extend this to take in multiple datasets #######

f_KM_high, a_KM_high, f_err_high, a_err_high = KM_avg_ND(data_all[0], bins[0], dt=5)
# need to filter out large dx/dt values

f_KM_low, a_KM_low, f_err_low, a_err_low = KM_avg_ND(data_all[1], bins[1], dt=5)

# %%
fig,ax = plt.subplots()
err_mag = np.sqrt(np.sum(f_err_high**2,axis=-1))
im = ax.pcolormesh(*np.meshgrid(*bins[0]),err_mag.T)
ax.quiver(*np.meshgrid(*centers[0]),f_KM_high[:,:,0].T,f_KM_high[:,:,1].T,color='w')
fig.colorbar(im, ax=ax, label = 'Standard deviation')
ax.set_xlabel('PC1')
ax.set_ylabel('PC3')
ax.set_title('High Flow')

fig,ax = plt.subplots()
err_mag = np.sqrt(np.sum(f_err_low**2,axis=-1))
im = ax.pcolormesh(*np.meshgrid(*bins[1]),err_mag.T)
ax.quiver(*np.meshgrid(*centers[1]),f_KM_low[:,:,0].T,f_KM_low[:,:,1].T,color='w')
fig.colorbar(im, ax=ax, label = 'Standard deviation')
ax.set_xlabel('PC1')
ax.set_ylabel('PC3')
ax.set_title('Low Flow')

# %%
mask_high = np.where(np.isfinite(f_KM_high))
X_mesh_high = np.array(np.meshgrid(*centers[0])).T
X_pts_high = X_mesh_high[mask_high].reshape((-1,ndim))
f_KM_high_noNAN = f_KM_high[mask_high].reshape((-1,ndim))

mask_low = np.where(np.isfinite(f_KM_low))
X_mesh_low = np.array(np.meshgrid(*centers[1])).T
X_pts_low = X_mesh_low[mask_low].reshape((-1,ndim))
f_KM_low_noNAN = f_KM_low[mask_low].reshape((-1,ndim))
# %%
N_tot_high = X_pts_high.shape[0]
N_train_high=int(0.8*N_tot_high) # 80% of data for training
N_test_high = N_tot_high-N_train_high
X_train_high, X_test_high, Y_train_high, Y_test_high = train_test_split(X_pts_high, f_KM_high_noNAN, test_size=N_test_high, random_state=342)
_, _, V_train_high, V_test_high = train_test_split(X_pts_high, a_KM_high[mask_high].reshape((-1,ndim)), test_size=N_test_high, random_state=342) # same random seed to get same x points for train and test

N_tot_low = X_pts_low.shape[0]
N_train_low=int(0.8*N_tot_low) # 80% of data for training
N_test_low = N_tot_low-N_train_low
X_train_low, X_test_low, Y_train_low, Y_test_low = train_test_split(X_pts_low, f_KM_low_noNAN, test_size=N_test_low, random_state=344)
_, _, V_train_low, V_test_low = train_test_split(X_pts_low, a_KM_low[mask_low].reshape((-1,ndim)), test_size=N_test_low, random_state=344) # same random seed to get same x points for train and test

# %%
X_train = np.concatenate((X_train_high,X_train_low))
X_test = np.concatenate((X_test_high,X_test_low))

Y_train = np.concatenate((Y_train_high,Y_train_low))
Y_test = np.concatenate((Y_test_high,Y_test_low))

V_train = np.concatenate((V_train_high,V_train_low))
V_test = np.concatenate((V_test_high,V_test_low))

u_train = np.concatenate((first_flow*np.ones(N_train_high),second_flow*np.ones(N_train_low)))
u_test = np.concatenate((first_flow*np.ones(N_test_high),second_flow*np.ones(N_test_low)))

# %%
driftModel = ps.SINDy(feature_library = ps.PolynomialLibrary(degree=2), optimizer = ps.SSR())
driftModel.fit(X_train,t=5,x_dot=Y_train,u=u_train)

diffModel = ps.SINDy(feature_library = ps.PolynomialLibrary(degree=2), optimizer = ps.SSR())
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
def f(x, u):
    if len(x.shape) == 1:
        x_in = x[None,:]
    else:
        x_in = x
    f_out = driftModel.predict(x_in, u = u)
    if f_out.shape[0] == 1:
        f_out = f_out[0]
    return f_out

def f_mesh(mesh_grid,u):
    n_1 = mesh_grid[0].shape[0]
    n_2 = mesh_grid[0].shape[1]
    V = np.zeros((n_1,n_2,2))
    for i in range(n_1):
        V[i,:,:] = f(np.vstack((mesh_grid[0][i,:],mesh_grid[1][i,:])).T,u)
    return V

def f1(x1,x2,u):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            f_out = f_mesh([x1,x2],u).T
        else:
            f_out = f(np.array([x1, x2]).reshape(-1,2),u).T
    else:
        f_out = f(np.array([x1, x2]).reshape(-1,2),u).T
    return f_out[0].T

def f2(x1,x2,u):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            f_out = f_mesh([x1,x2],u).T
        else:
            f_out = f(np.array([x1, x2]).reshape(-1,2),u).T
    else:
        f_out = f(np.array([x1, x2]).reshape(-1,2),u).T
    return f_out[1].T

def D(x, u):
    if len(x.shape) == 1:
        x_in = x[None,:]
    else:
        x_in = x
    D_out = diffModel.predict(x_in, u = u)
    if D_out.shape[0] == 1:
        D_out = D_out[0]
    return D_out

def D_mesh(mesh_grid,u):
    n_1 = mesh_grid[0].shape[0]
    n_2 = mesh_grid[0].shape[1]
    V = np.zeros((n_1,n_2,2))
    for i in range(n_1):
        V[i,:,:] = D(np.vstack((mesh_grid[0][i,:],mesh_grid[1][i,:])).T,u)
    return V

def D1(x1,x2,u):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            D_out = D_mesh([x1,x2],u).T
        else:
            D_out = D(np.array([x1, x2]).reshape(-1,2),u).T
    else:
        D_out = D(np.array([x1, x2]).reshape(-1,2),u).T
    return D_out[0].T

def D2(x1,x2,u):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            D_out = D_mesh([x1,x2],u).T
        else:
            D_out = D(np.array([x1, x2]).reshape(-1,2),u).T
    else:
        D_out = D(np.array([x1, x2]).reshape(-1,2),u).T
    return D_out[1].T
# %%
x1 = np.linspace(-4,5,50)
x2 = np.linspace(-2,3,50)
fig,ax = pplane.phase_portrait(lambda x1,x2: f1(x1,x2,first_flow),lambda x1,x2: f2(x1,x2,first_flow),x1,x2)
ax.set_xlabel('PC1',fontsize=28)
ax.set_ylabel('PC3',fontsize=28)
# %%
fig,ax = pplane.phase_portrait(lambda x1,x2: f1(x1,x2,second_flow),lambda x1,x2: f2(x1,x2,second_flow),x1,x2)
ax.set_xlabel('PC1',fontsize=28)
ax.set_ylabel('PC3',fontsize=28)
# %%
N = (len(bins[0][0])-1,len(bins[0][1])-1) # number of bins in each dimension
dx = [(bins[0][0][1]-bins[0][0][0]),(bins[0][1][1]-bins[0][1][0])] # bin width in each dimension
fp = fps.SteadyFP(N, dx) # initialize stationary Fokker-Planck solver
# %%
X1,X2 = np.meshgrid(centers[0][0],centers[0][1])
f_vals = f_mesh([X1,X2],first_flow).T
D_vals = D_mesh([X1,X2],first_flow).T
p_fit = fp.solve(f_vals,D_vals) # solve stationary Fokker-Planck equation
p_fit[p_fit<1e-10] = 1e-10 # set small values to a small number to avoid numerical issues

# %%
fig,ax = viz.init_subplots(1,2,figsize=(12,4))
p_hist, _, _ = np.histogram2d(X_traj[0][:,100:,0].flatten(),X_traj[0][:,100:,1].flatten(), bins[0], density=True)
ax[0] = viz.plot_histogram_2D(ax[0],p_hist,bins[0],cmap='inferno') # plot empirical PDF
ax[1] = viz.plot_histogram_2D(ax[1],p_fit,bins[0],cmap='inferno') # plot model PDF

# %%
N = (len(bins[1][0])-1,len(bins[1][1])-1) # number of bins in each dimension
dx = [(bins[1][0][1]-bins[1][0][0]),(bins[1][1][1]-bins[1][1][0])] # bin width in each dimension
fp = fps.SteadyFP(N, dx) # initialize stationary Fokker-Planck solver

# %%
X1,X2 = np.meshgrid(centers[1][0],centers[1][1])
f_vals = f_mesh([X1,X2],second_flow).T
D_vals = D_mesh([X1,X2],second_flow).T
p_fit = fp.solve(f_vals,D_vals) # solve stationary Fokker-Planck equation
p_fit[p_fit<1e-10] = 1e-10 # set small values to a small number to avoid numerical issues

# %%
fig,ax = viz.init_subplots(1,2,figsize=(12,4))
p_hist, _, _ = np.histogram2d(X_traj[1][:,100:,0].flatten(),X_traj[1][:,100:,1].flatten(), bins[1], density=True)
ax[0] = viz.plot_histogram_2D(ax[0],p_hist,bins[1],cmap='inferno') # plot empirical PDF
ax[1] = viz.plot_histogram_2D(ax[1],p_fit,bins[1],cmap='inferno') # plot model PDF
# %%

N_fine = 100

x_fine = np.linspace(x1[0],x1[-1],N_fine)
y_fine = np.linspace(x2[0],x2[-1],N_fine)
centers_fine = [x_fine,y_fine]
X1,X2 = np.meshgrid(x_fine,y_fine)
# %%
f_vals_new = f_mesh([X1,X2],first_flow).T
D_vals_new = D_mesh([X1,X2],first_flow).T

print('**** Plotting generalized potential energy landscape **** \n')
U, grad_term, flux_term = gp.grad_flux_decomposition(f_vals_new,D_vals_new,centers_fine,tol=1e-6)
grad_norm =grad_term.copy()
flux_norm = flux_term.copy()
# grad_norm = grad_term/(np.sqrt(grad_term[0]**2+grad_term[1]**2))
# flux_norm = flux_term/(np.sqrt(flux_term[0]**2+flux_term[1]**2))
fig,ax = viz.plot_gen_potential_2D(U,centers_fine[0],centers_fine[1],cmap='jet',surf=False)
ax.set_xlabel('PC1',fontsize=28)
ax.set_ylabel('PC3',fontsize=28)
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

f_vals_new = f_mesh([X1,X2],second_flow).T
D_vals_new = D_mesh([X1,X2],second_flow).T
print('**** Plotting generalized potential energy landscape **** \n')
U, grad_term, flux_term = gp.grad_flux_decomposition(f_vals_new,D_vals_new,centers_fine,tol=1e-6)
grad_norm = grad_term.copy()
flux_norm = flux_term.copy()
# grad_norm = grad_term/(np.sqrt(grad_term[0]**2+grad_term[1]**2))
# flux_norm = flux_term/(np.sqrt(flux_term[0]**2+flux_term[1]**2))
fig,ax = viz.plot_gen_potential_2D(U,centers_fine[0],centers_fine[1],cmap='jet',surf=False)
ax.set_xlabel('PC1',fontsize=28)
ax.set_ylabel('PC3',fontsize=28)

# same but with vector field decomposition

# %%
fig,ax = viz.plot_gen_potential_2D(U,centers_fine[0],centers_fine[1],cmap='jet',surf=False)
downsample=8
ax.quiver(x_fine[::downsample],y_fine[::downsample],grad_norm[0][::downsample,::downsample].T,grad_norm[1][::downsample,::downsample].T,color='w',pivot='tail')
ax.quiver(x_fine[::downsample],y_fine[::downsample],flux_norm[0][::downsample,::downsample].T,flux_norm[1][::downsample,::downsample].T,color='r',pivot='tail')
ax.set_xlabel('PC1',fontsize=28)
ax.set_ylabel('PC3',fontsize=28)


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
