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
my_mv = list_of_datasets[1]
mv_name = eaio.get_dataset_name(my_mv)
feats_proj = eaio.project_PCA_one_dataset(df,pca, 'group', my_mv, metadata_cols=metadata_col)

fig1,ax1 = eaviz.plot_top_3_PCs(feats_proj)
ax1[0].set_ylim([-1.5,4])
ax1[1].set_ylim([-7,-2])
ax1[2].set_ylim([-0.5,4.5])

fig2,ax2 = eaviz.plot_PCA_projection(feats_proj,mv_name)
ax2.set_xlim([-2,5])
ax2.set_ylim([-0.5,4])

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

thresh = 0.45 # get this based on upper quantile of vector magnitudes along given PCs low flow data

# %%
for ds_ID in range(4): 
    print('**** Generating train/test sets for dataset',ds_ID,'**** \n')
    my_mv = list_of_datasets[ds_ID]
    mv_name = eaio.get_dataset_name(my_mv)
    feats_proj = eaio.project_PCA_one_dataset(df,pca,'group', my_mv,metadata_cols=metadata_col)

    data_all, u_traj, u_list = eareg.get_traj_and_flow(feats_proj,mv_name,PCs=PCs,verbose=True)
    del feats_proj # free up memory
    num_flow = len(u_list)
    if 0 in u_list: # right now, only using timepoints 300:450 of no flow
        data_all_temp = []
        for traj in data_all[0]:
            data_all_temp.append(traj[300:450,:])
        data_all = [data_all_temp]
        u_traj = [u_traj[0][300:450]]

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

        f_KM_temp, D_KM_temp, f_err_temp, D_err_temp = eareg.KM_avg_ND(data_all[j], bins[j], dt=5, threshold=thresh)
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
feature_lib = ps.ConcatLibrary([ps.PolynomialLibrary(degree=3, 
                                include_bias=True),
                                sigmoid_lib])
parameter_lib=ps.PolynomialLibrary(degree=3, include_bias=True)
full_lib=ps.ParameterizedLibrary(feature_library=feature_lib,
    parameter_library=parameter_lib,num_features=ndim,num_parameters=1)

driftModel = ps.SINDy(feature_library = full_lib, optimizer = ps.SSR())
driftModel.fit(X_train,t=5,x_dot=Y_train,u=u_train)


diff_feature_lib=ps.PolynomialLibrary(degree=0, include_bias=True)
diff_parameter_lib=ps.PolynomialLibrary(degree=0, include_bias=True)
diff_lib=ps.ParameterizedLibrary(feature_library=diff_feature_lib,
    parameter_library=diff_parameter_lib,num_features=ndim,num_parameters=1)


diffModel = ps.SINDy(feature_library = diff_lib, optimizer = ps.SSR())
diffModel.fit(X_train,t=5,x_dot=V_train,u=u_train)

drift_R2 = driftModel.score(X_test,x_dot=Y_test,u=u_test)
driftModel.print()

print('Coefficient of determination (R^2) of drift (RBF kernel) model on test set: %f' %drift_R2)

diff_R2 = diffModel.score(X_test,x_dot=V_test,u=u_test)
diffModel.print()

print('Coefficient of determination (R^2) of diffusion (RBF kernel) model on test set: %f' %diff_R2)
# %%
myModel = [driftModel,diffModel]

if PCs[0] == 0:
    pplane_xlim = [-4,3.5]
    bin_xlim = [-5,4]
elif PCs[0] == 1:
    pplane_xlim = [-7,-2]
    bin_xlim = [-8,-2]

if PCs[1] == 1:
    pplane_ylim = [-7,-2]
    bin_ylim = [-8,-2]
else:
    pplane_ylim = [-2,4]
    bin_ylim = [-3,5]

plt_args = {'pplane_xlim': pplane_xlim, 'pplane_ylim': pplane_ylim, 'pplane_N': 50,
            'plt_xlabel': 'PC'+str(PCs[0]+1), 'plt_ylabel': 'PC'+str(PCs[1]+1)}

# fix bins and centers for all datasets
Nbins = [40 for i in range(ndim)]
bin_limits = [bin_xlim,bin_ylim]
bins, centers = eareg.get_bins(Nbins,bin_limits=bin_limits)

for ds_ID in range(4):
    print('**** Running model analysis for dataset',ds_ID,'**** \n')
    my_mv = list_of_datasets[ds_ID]
    mv_name = eaio.get_dataset_name(my_mv)
    feats_proj = eaio.project_PCA_one_dataset(df,pca,'group', my_mv,metadata_cols=metadata_col)

    data_all, u_traj, u_list = eareg.get_traj_and_flow(feats_proj,mv_name,PCs=PCs,verbose=True)
    del feats_proj # free up memory
    num_flow = len(u_list)
    if 0 in u_list: # right now, only using timepoints 300:450 of no flow
        data_all_temp = []
        for traj in data_all[0]:
            data_all_temp.append(traj[300:450,:])
        data_all = [data_all_temp]
        u_traj = [u_traj[0][300:450]]


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
plt.xlabel('Shear stress (dyn/cm$^2$)')
plt.ylabel('Entropy production rate')



# %%
from scipy.integrate import solve_ivp

# %%
fpt_stable = np.array(fpt_stable)
multistable = False
u_stable_unique = []
fpt_stable_unique = []
for i in range(fpt_stable.shape[0]):
    if multistable:
        continue
    u = u_stable[i]
    u_stable_unique.append(u)
    fpt = fpt_stable[i]
    print(fpt)
    fpt_stable_unique.append(fpt)

    # check if u is unique in u_stable
    if u_stable.count(u) > 1:
        multistable = True 
    else:
        multistable = False


    


# %%
tau = []
# compute return timescale to stable state as a funciton of u
for i in range(len(fpt_stable)):
    u = u_stable[i]
    fpt = fpt_stable[i]
    init = fpt - np.array([0.01,0.01])

    def myFlow(t,x):
        return f(x,u=u)

    tVec = np.linspace(0,10,300)

    traj = solve_ivp(myFlow,y0=init,t_span=(tVec.min(),tVec.max()),t_eval=tVec).y

    my_dist = np.sqrt((traj[0]-fpt[0])**2 + (traj[1]-fpt[1])**2)
    A,B = np.polyfit(tVec,np.log(my_dist),1)
    tau.append(-1/A)

fig,ax = plt.subplots()
ax.plot(u_stable, tau,'k')
# %%
# u_fit = u_stable_unique[6:-50]
# tau_fit = tau[6:-50]
u_fit = u_stable_unique[50:]
tau_fit = tau[50:]
from scipy.optimize import curve_fit
# fit power law to tau vs u
def func_powerlaw(x,u_crit,a,c):
    return c*(x-u_crit)**a
pwr_params = curve_fit(func_powerlaw,u_fit,tau_fit)[0]

fig,ax = plt.subplots()
ax.plot(u_fit, tau_fit,'k')
ax.plot(u_fit, func_powerlaw(u_fit,*pwr_params),'r--')

print('Predicted critical shear stress: ',pwr_params[0])
print('Power law exponent: ',pwr_params[1])

# %%

# %%
# %%







################### Generalized potential energy landscape ###################
## NEED TO RE-ADAPT FOR NEW FUNCTIONALITY
    
N_fine = 60

x_fine = np.linspace(-4,4.5,N_fine+1)
y_fine = np.linspace(-3,5.5,N_fine+1)
bins_fine = [x_fine,y_fine]
centers_fine = [0.5*(x_fine[1:]+x_fine[:-1]),
                0.5*(y_fine[1:]+y_fine[:-1])]
X1,X2 = np.meshgrid(centers_fine[0],centers_fine[1])
# %%
u = 28

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
_, grad_term, _, flux_term = gp.grad_flux_decomposition(f_vals_new,D_vals_new,centers_fine,tol=tol)
grad_norm = grad_term.copy()
flux_norm = np.array(flux_term)
# grad_norm = grad_term/(np.sqrt(grad_term[0]**2+grad_term[1]**2))
# flux_norm = flux_term/(np.sqrt(flux_term[0]**2+flux_term[1]**2))
fig,ax = viz.plot_gen_potential_2D(U,centers_fine[0],centers_fine[1],cmap='jet',surf=False)
ax.set_xlabel('PC'+str(PCs[0]+1))
ax.set_ylabel('PC'+str(PCs[1]+1))
ax.set_title('Shear stress: '+str(u)+' dyn/cm$^2$')
# axins = zoomed_inset_axes(ax, 2, loc=1)
# im = axins.imshow(U.T,interpolation='nearest', origin='lower',
#     extent=[x_fine[0], x_fine[-1], y_fine[0], y_fine[-1]],
#     cmap='jet', aspect=(x_fine[-1]-x_fine[0])/(y_fine[-1]-y_fine[0]))
# axins.set_xlim(2.5, 4)
# axins.set_ylim(0, 2)

# %%
# same but with vector field decomposition
fig,ax = viz.plot_gen_potential_2D(U,centers_fine[0],centers_fine[1],
                                   cmap='jet',surf=False)
downsample=10
ax.quiver(centers_fine[0][::downsample],
          centers_fine[1][::downsample],
          grad_norm[0][::downsample,::downsample].T,
          grad_norm[1][::downsample,::downsample].T,
          color='w',pivot='tail')
ax.quiver(centers_fine[0][::downsample],
          centers_fine[1][::downsample],
          flux_norm[0][::downsample,::downsample].T,
          flux_norm[1][::downsample,::downsample].T,
          color='r',pivot='tail')
ax.set_xlabel('PC'+str(PCs[0]+1))
ax.set_ylabel('PC'+str(PCs[1]+1))

# fig,axins = plt.subplots()
# im = axins.imshow(U.T,interpolation='nearest', origin='lower',
#     extent=[x_fine[0], x_fine[-1], y_fine[0], y_fine[-1]],
#     cmap='jet', aspect=(x_fine[-1]-x_fine[0])/(y_fine[-1]-y_fine[0]))
# downsample = 4
# width = 0.015*(0-(-1))/(4-2)
# axins.quiver(x_fine[::downsample],y_fine[::downsample],grad_norm[0][::downsample,::downsample].T,grad_norm[1][::downsample,::downsample].T,color='w',pivot='tail',scale=0.2,width=width,edgecolor='k',linewidths=0.5)
# axins.quiver(x_fine[::downsample],y_fine[::downsample],flux_norm[0][::downsample,::downsample].T,flux_norm[1][::downsample,::downsample].T,color='r',pivot='tail',scale=0.2,width=width)
# axins.set_xlim(2,4)
# axins.set_ylim(1.1,2.1)

# %%
