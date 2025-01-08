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
from cellsmap.analyses.utils import viz
import cellsmap.analyses.utils.kernel_regression.kernel_regression as kreg
import cellsmap.analyses.utils.langevin_sindy.fp_solvers as fps

import cellsmap.analyses.playground.ea.utils.io as eaio
import cellsmap.analyses.playground.ea.utils.viz as eaviz
import cellsmap.analyses.playground.ea.utils.regression as eareg

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

X_train, X_test, Y_train, Y_test, V_train, V_test = eareg.train_test_all(X_pts_noNAN,f_KM_noNAN,D_KM_noNAN,num_flow,train_frac,seed)

# %%
drift_high = kreg.KernelRegression(beta=0.01).fit(X_train[0],Y_train[0])

drift_R2 = drift_high.score(X_test[0],Y_test[0])

print('Coefficient of determination (R^2) of drift (RBF kernel) model on test set: %f' %drift_R2)

# %%
diff_high = kreg.KernelRegression(beta=0.01).fit(X_train[0],V_train[0])

diff_R2 = diff_high.score(X_test[0],V_test[0])

print('Coefficient of determination (R^2) of diffusion (RBF kernel) model on test set: %f' %diff_R2)

# %%
def f_high(x):
    if isinstance(x,np.ndarray):
        if len(x.shape) == 1:
            f_out = drift_high.predict(x[None,:])
        else:
            f_out = drift_high.predict(x)
        if f_out.shape[0] == 1:
            f_out = f_out[0]
        return f_out
    elif isinstance(x,list):
        if not hasattr(drift_high,'proj_'):
            drift_high.project2D(np.eye(ndim,2))
        return drift_high.predict_2D_mesh(x)
    
def f1_high(x1,x2):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            f_out = f_high([x1,x2]).T
        else:
            f_out = f_high(np.array([x1, x2]).reshape(-1,2)).T
    else:
        f_out = f_high(np.array([x1, x2]).reshape(-1,2)).T
    return f_out[0].T

def f2_high(x1,x2):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            f_out = f_high([x1,x2]).T
        else:
            f_out = f_high(np.array([x1, x2]).reshape(-1,2)).T
    else:
        f_out = f_high(np.array([x1, x2]).reshape(-1,2)).T
    return f_out[1].T

def D_high(x):
    if isinstance(x,np.ndarray):
        if len(x.shape) == 1:
            D_out = diff_high.predict(x[None,:])
        else:
            D_out = diff_high.predict(x)
        if D_out.shape[0] == 1:
            D_out = D_out[0]
        return D_out
    elif isinstance(x,list):
        if not hasattr(diff_high,'proj_'):
            diff_high.project2D(np.eye(ndim,2))
        return diff_high.predict_2D_mesh(x)

def D1_high(x1,x2):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            D_out = D_high([x1,x2]).T
        else:
            D_out = D_high(np.array([x1, x2]).reshape(-1,2)).T
    else:
        D_out = D_high(np.array([x1, x2]).reshape(-1,2)).T
    return D_out[0].T

def D2_high(x1,x2):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            D_out = D_high([x1,x2]).T
        else:
            D_out = D_high(np.array([x1, x2]).reshape(-1,2)).T
    else:
        D_out = D_high(np.array([x1, x2]).reshape(-1,2)).T
    return D_out[1].T

# %%
x1 = np.linspace(-4,5,50)
x2 = np.linspace(-2,3,50)
fig,ax = pplane.phase_portrait(lambda x1,x2: f1_high(x1,x2),lambda x1,x2: f2_high(x1,x2),x1,x2)
ax.set_xlabel('PC1',fontsize=28)
ax.set_ylabel('PC3',fontsize=28)

# %%
####################### LOW FLOW ############################

drift_low = kreg.KernelRegression(beta=0.01).fit(X_train[1],Y_train[1])

drift_R2 = drift_low.score(X_test[1],Y_test[1])

print('Coefficient of determination (R^2) of drift (RBF kernel) model on test set: %f' %drift_R2)

# %%
diff_low = kreg.KernelRegression(beta=0.01).fit(X_train[1],V_train[1])

diff_R2 = diff_high.score(X_test[1],V_test[1])

print('Coefficient of determination (R^2) of diffusion (RBF kernel) model on test set: %f' %diff_R2)
# %%
# acting odd with root finding for getting fixed points, debug how you define f_low
def f_low(x):
    if isinstance(x,np.ndarray):
        if len(x.shape) == 1:
            f_out = drift_low.predict(x[None,:])
        else:
            f_out = drift_low.predict(x)
        if f_out.shape[0] == 1:
            f_out = f_out[0]
        return f_out
    elif isinstance(x,list):
        if not hasattr(drift_low,'proj_'):
            drift_low.project2D(np.eye(ndim,2))
        return drift_low.predict_2D_mesh(x)

def f1_low(x1,x2):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            f_out = f_low([x1,x2]).T
        else:
            f_out = f_low(np.array([x1, x2]).reshape(-1,2)).T
    else:
        f_out = f_low(np.array([x1, x2]).reshape(-1,2)).T
    return f_out[0].T

def f2_low(x1,x2):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            f_out = f_low([x1,x2]).T
        else:
            f_out = f_low(np.array([x1, x2]).reshape(-1,2)).T
    else:
        f_out = f_low(np.array([x1, x2]).reshape(-1,2)).T
    return f_out[1].T

def D_low(x):
    if isinstance(x,np.ndarray):
        if len(x.shape) == 1:
            D_out = diff_low.predict(x[None,:])
        else:
            D_out = diff_low.predict(x)
        if D_out.shape[0] == 1:
            D_out = D_out[0]
        return D_out
    elif isinstance(x,list):
        if not hasattr(diff_low,'proj_'):
            diff_low.project2D(np.eye(ndim,2))
        return diff_low.predict_2D_mesh(x)

def D1_low(x1,x2):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            D_out = D_low([x1,x2]).T
        else:
            D_out = D_low(np.array([x1, x2]).reshape(-1,2)).T
    else:
        D_out = D_low(np.array([x1, x2]).reshape(-1,2)).T
    return D_out[0].T

def D2_low(x1,x2):
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 2:
            D_out = D_low([x1,x2]).T
        else:
            D_out = D_low(np.array([x1, x2]).reshape(-1,2)).T
    else:
        D_out = D_low(np.array([x1, x2]).reshape(-1,2)).T
    return D_out[1].T

# %%
fig,ax = pplane.phase_portrait(lambda x1,x2: f1_low(x1,x2),lambda x1,x2: f2_low(x1,x2),x1,x2)
ax.set_xlabel('PC1',fontsize=28)
ax.set_ylabel('PC3',fontsize=28)

# %%
N = (len(bins[0][0])-1,len(bins[0][1])-1) # number of bins in each dimension
dx = [(bins[0][0][1]-bins[0][0][0]),(bins[0][1][1]-bins[0][1][0])] # bin width in each dimension
fp = fps.SteadyFP(N, dx) # initialize stationary Fokker-Planck solver
# %%
X1,X2 = np.meshgrid(centers[0][0],centers[0][1])
f_vals = f_high([X1,X2]).T
D_vals = D_high([X1,X2]).T
p_fit = fp.solve(f_vals,D_vals) # solve stationary Fokker-Planck equation
p_fit[p_fit<1e-10] = 1e-10 # set small values to a small number to avoid numerical issues

# %%
fig,ax = viz.init_subplots(1,2,figsize=(12,4))
p_hist, _, _ = np.histogram2d(np.concatenate([data_all[0][j][100:,0] for j in range(len(data_all[0]))]).flatten(),
                              np.concatenate([data_all[0][j][100:,1] for j in range(len(data_all[0]))]).flatten(), 
                              bins[0], density=True)
ax[0] = viz.plot_histogram_2D(ax[0],p_hist,bins[0],cmap='inferno') # plot empirical PDF
ax[1] = viz.plot_histogram_2D(ax[1],p_fit,bins[0],cmap='inferno') # plot model PDF

# %%
N = (len(bins[1][0])-1,len(bins[1][1])-1) # number of bins in each dimension
dx = [(bins[1][0][1]-bins[1][0][0]),(bins[1][1][1]-bins[1][1][0])] # bin width in each dimension
fp = fps.SteadyFP(N, dx) # initialize stationary Fokker-Planck solver

# %%
X1,X2 = np.meshgrid(centers[1][0],centers[1][1])
f_vals = f_low([X1,X2]).T
D_vals = D_low([X1,X2]).T
p_fit = fp.solve(f_vals,D_vals) # solve stationary Fokker-Planck equation
p_fit[p_fit<1e-10] = 1e-10 # set small values to a small number to avoid numerical issues

# %%
fig,ax = viz.init_subplots(1,2,figsize=(12,4))
p_hist, _, _ = np.histogram2d(np.concatenate([data_all[1][j][100:,0] for j in range(len(data_all[1]))]).flatten(),
                                np.concatenate([data_all[1][j][100:,1] for j in range(len(data_all[1]))]).flatten(), 
                                bins[1], density=True)
ax[0] = viz.plot_histogram_2D(ax[0],p_hist,bins[1],cmap='inferno') # plot empirical PDF
ax[1] = viz.plot_histogram_2D(ax[1],p_fit,bins[1],cmap='inferno') # plot model PDF
# %%
