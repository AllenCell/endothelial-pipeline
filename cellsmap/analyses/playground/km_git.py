# %%
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps

from matplotlib import animation
from IPython.display import HTML

from cellsmap.util.set_output import get_output_path
from cellsmap.util import manifest_io as mio
from cellsmap.analyses.utils.io import dynamics_io
from cellsmap.analyses.utils import manifest_pca, regression_helper as rh, model_analysis, model_eval
from cellsmap.analyses.utils.viz import viz_base as vb, dynamics_viz, manifest_viz, pplane
from cellsmap.analyses.utils.numerics import kramers_moyal as eakm 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# %%
# %%
# load manifest to DataFrame with metadata
df = mio.load_manifest_to_df()

# fit PCA to data
pca = manifest_pca.fit_pca(df, num_pcs=8)
df.head()
list_of_datasets = mio.get_list_of_datasets(df,verbose=False)

################### Visualize PCA results ###################
# plot explained variance ratio of PCA components
fig, _ = manifest_viz.plot_explained_variance(pca['pca'].explained_variance_ratio_)

# plot top 3 principal components of feature data vs. frame number
fig, _ = manifest_viz.plot_top_3_PCs_alldata(df,pca)
# %%
PCs = [0,1]
ndim = len(PCs)
dt=5

Nbins_KM = [26,25]

# datasets to skip in model comparison
config = dynamics_io.load_dynamics_config()
ds_to_skip = config['datasets_to_skip']
ds_to_skip.append('20241203_20X')

# set limits for phase plane and histogram plots
pplane_xlim = [-1.0,1.0]
bin_xlim = [-1.25,1.25]

pplane_ylim = [-0.5,0.75]
bin_ylim = [-0.15,0.75]

# for histogram/stationary pdf plots, fix bins and centers for all datasets
Nbins = [50,50]
bin_limits = [bin_xlim,bin_ylim]
bins, centers = rh.get_bins(Nbins,bin_limits=bin_limits)

# for phase plane plots, fix grid across all datasets
pplane_xvec = np.linspace(*pplane_xlim,50)
pplane_yvec = np.linspace(*pplane_ylim,50)
# %%
######## SINDY BASED REGRESSION ########
# now fit model using multiple datasets


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

Nbins = 25*np.ones(ndim, dtype = int)
dt = 5

clip = True
method = 'histogram'

for ds_name in list_of_datasets:
    if ds_name in ds_to_skip:
        continue 
    print('**** Generating train/test sets for ',ds_name,'**** \n')
    feats_proj = mio.project_PCA_one_dataset(df,pca,ds_name)

    data_all, u_list = rh.get_X_by_flow(feats_proj,ds_name)
    num_flow = len(u_list)
    del feats_proj # free up memory

    centers = []

    f_KM = []
    D_KM = []
    X_pts = []

    for j in range(num_flow): # get bins and centers for data at high and low flow
        print("Flow condition ",u_list[j])
        X_list, dX_list, dT_list = rh.get_X_dX_and_dT(data_all[j],feat_cols=[str(i) for i in PCs])

        for i, dT in enumerate(dT_list):
            mask = np.where(dT==1)[0] # where outlier points were removed, time difference was greater than 1, mask out these points
            # mask_ for trajectories: should be mask but with additional point
            # include frame after last frame in mask
            mask_ = np.concatenate((mask, [mask[-1]+1]))
            X_list[i] = X_list[i][mask_]
            dX_list[i] = dX_list[i][mask]

        bins_, centers_ = rh.get_bins(Nbins_KM,data=X_list)
        
        f_KM_, D_KM_ = rh.get_kramers_moyal(X_list,dX_list,dT_list,bins_,dt=5,method=method)

        if clip:
            if u_list[j] < 6:
                idx00 = 12
                idx01 = -5
                idx10 = 8
                idx11 = -5
            elif u_list[j] >= 20:
                idx00 = 5
                idx01 = -7
                idx10 = 8
                idx11 = -7
            else:
                idx00 = 7
                idx01 = -5
                idx10 = 9
                idx11 = -5

            f_KM_slice = f_KM_[idx00:idx01,idx10:idx11,:]
            D_KM_slice = D_KM_[idx00:idx01,idx10:idx11,:]
            centers_slice = [centers_[0][idx00:idx01],centers_[1][idx10:idx11]]
        else:
            f_KM_slice = f_KM_
            D_KM_slice = D_KM_
            centers_slice = centers_

        X_1, X_2 = np.meshgrid(*centers_slice)
        kmc_slice = np.concatenate([f_KM_slice,D_KM_slice],axis=-1).T

        fig, ax = plt.subplots(1,2,figsize=(12,6))
        ax[0].quiver(X_1,X_2,kmc_slice[0],kmc_slice[1],color='k', linewidth=0.5)
        ax[0].set_xlabel(f'PC{PCs[0]+1}')
        ax[0].set_ylabel(f'PC{PCs[1]+1}')

        ax[1].streamplot(X_1,X_2,kmc_slice[0],kmc_slice[1],color='k', linewidth=0.5)
        ax[1].set_xlabel(f'PC{PCs[0]+1}')
        ax[1].set_ylabel(f'PC{PCs[1]+1}')
        fig.suptitle('Kramers-Moyal drift coefficients')
        plt.show()

        fig_ax = eakm.plot_km(X_1,X_2,kmc_slice)
        plt.show()

        f_KM_mask, X_pts_mask = rh.masked_vector_field(f_KM_slice,np.array(np.meshgrid(*centers_slice)).T)
        D_KM_mask, _ = rh.masked_vector_field(D_KM_slice, np.array(np.meshgrid(*centers_slice)).T)
        f_KM.append(f_KM_mask)
        D_KM.append(D_KM_mask)
        X_pts.append(X_pts_mask)

    del f_KM_, D_KM_, centers, centers_ # free up memory

    train_frac = 0.8   
    X_train, X_test, Y_train, Y_test, V_train, V_test = rh.train_test_all(X_pts,f_KM,D_KM,train_frac=0.8)

    # get number of training and test points for each flow condition
    N_tot = [X_pts[j].shape[0] for j in range(num_flow)]
    N_train = [int(train_frac*N_tot[j]) for j in range(num_flow)]
    N_test = [N_tot[j]-N_train[j] for j in range(num_flow)]

    # get corresponding flow condition for each training and test point
    u_train = np.concatenate([u_list[j]*np.ones((N_train[j],1)) for j in range(num_flow)])
    u_test = np.concatenate([u_list[j]*np.ones((N_test[j],1)) for j in range(num_flow)])
    
    del X_pts, f_KM, D_KM # free up memory

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

feature_lib = ps.PolynomialLibrary(degree=3, include_bias=True)
parameter_lib=ps.PolynomialLibrary(degree=4, include_bias=True)
full_lib=ps.ParameterizedLibrary(feature_library=feature_lib,
    parameter_library=parameter_lib,num_features=ndim,num_parameters=1)

driftModel = ps.SINDy(feature_library = full_lib, optimizer = ps.SSR())
driftModel.fit(X_train,t=dt,x_dot=Y_train,u=u_train)


diff_feature_lib=ps.PolynomialLibrary(degree=4, include_bias=True)
diff_parameter_lib=ps.PolynomialLibrary(degree=4, include_bias=True)
diff_lib=ps.ParameterizedLibrary(feature_library=diff_feature_lib,
    parameter_library=diff_parameter_lib,num_features=ndim,num_parameters=1)


diffModel = ps.SINDy(feature_library = diff_lib, optimizer = ps.SSR())
diffModel.fit(X_train,t=dt,x_dot=V_train,u=u_train)

drift_R2 = driftModel.score(X_test,x_dot=Y_test,u=u_test)
driftModel.print()

print('Coefficient of determination (R^2) for drift coefficient model on test set: %f' %drift_R2)

diff_R2 = diffModel.score(X_test,x_dot=V_test,u=u_test)
diffModel.print()

print('Coefficient of determination (R^2) for diffusion coefficient model on test set: %f' %diff_R2)
# %%
# save directory for plots
workflow_fig_folder = "stochastic_dynamics/km_test/figs"
fig_savedir = get_output_path(workflow_fig_folder,verbose=False)

# for fixed point analysis and entropy production rate analysis
shear_range_fpt = np.linspace(config['shear_range'][0],config['shear_range'][1],config['N_shear_fpt'])

f = model_eval.vector_field_function(driftModel)
D = model_eval.vector_field_function(diffModel)
myModel = [f,D]


# %%
################### Model-data comparison ###################
# run comparison of model and data for each dataset
model_analysis.model_data_comparison(myModel,fig_savedir,pca,PCs,bins,ds_to_skip,pplane_xvec,pplane_yvec)
        
# %%
################### Fixed point analysis ###################
# plot coordinates of fixed points as a function of shear stress
plt_lims = [pplane_xlim,pplane_ylim] # set limits for plotted/reported fixed points

model_analysis.run_fixed_point_analysis(f,shear_range_fpt,PCs,plt_lims,fig_savedir)

# %%
################### Entropy production rate as a function of shear stress ###################
model_analysis.run_epr_analysis(myModel,bins,centers,shear_range_fpt,fig_savedir,additive_noise=False)

# %%

f1 = model_eval.vector_field_component(f,0)
f2 = model_eval.vector_field_component(f,1)


# %%

# Assuming pplane, f1, f2, pplane_xvec, pplane_yvec are defined elsewhere in your code

# Initial plot
fig1, ax1 = pplane.phase_portrait(lambda x1, x2: f1([x1, x2], 21.9),
                                  lambda x1, x2: f2([x1, x2], 21.9),
                                  pplane_xvec, pplane_yvec)

def animate(u):
    ax1.clear()
    pplane.phase_portrait(lambda x1, x2: f1([x1, x2], u),
                          lambda x1, x2: f2([x1, x2], u),
                          pplane_xvec, pplane_yvec, fig_ax=(fig1,ax1))
    plt.title(f'Shear Stress: {u:.2f}')
    return fig1, ax1

# Create animation
shear_ = [21.925, 21.95, 21.975, 22.0, 22.025, 22.05, 22.075, 22.1]
anim = animation.FuncAnimation(fig1, animate, frames=shear_, interval=1000)

# Display animation
HTML(anim.to_jshtml())
# %%
f_mesh = model_eval.mesh_grid_function(f)
D_mesh = model_eval.mesh_grid_function(D)



f_vals = f_mesh(np.meshgrid(*centers),u=u).T
D_vals = D_mesh(np.meshgrid(*centers),u=u).T

P = model_eval.get_stationary_probability(f_vals, D_vals, bins)

fig, ax = vb.init_plot()
ax = dynamics_viz.plot_histogram_2D(ax,P,bins,'inferno')
# %%
def fit_polynomial_regression(X,Y,degree):
    '''
    Fit polynomial regression to data.
    Inputs:
    - X: input data
    - Y: output data
    - degree: degree of polynomial regression
    Outputs:
    - mdl: fitted model
    '''
    mdl = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                    ('linear', LinearRegression())]).fit(X,Y.flatten())
    
    return mdl

def get_km_estimates(kmc,X,degrees):
    # polynomial regression on Y_1 = kmc[1,idx0:idx1,idx0:idx1].T/5
    kmc_idx = [0,1,2,3]
    kmc_names = ['f_1','f_2','D_1','D_2']
    km_predictions = []
    km_models = []
    for i in range(len(degrees)):
        Y = kmc[kmc_idx[i]].flatten()
        mdl = fit_polynomial_regression(X,Y,degrees[i])
        km_models.append(mdl)

        # print polynomial expression
        print_fit_polynomial(mdl,kmc_names[i])

        mdl_pred = mdl.predict(X)
        km_predictions.append(mdl_pred)

        r2 = r2_score(Y.flatten(),mdl_pred)
        print(f'R^2 for polynomial regression on {kmc_names[i]}: {r2} \n')

    return km_predictions, km_models

def print_fit_polynomial(mdl,mdl_name):
    '''
    Print coefficients of polynomial regression model.
    '''
    print(f'{mdl_name}(x) = ', end='')
    pwrs = mdl.named_steps['poly'].powers_
    # build strings of polynomial terms
    terms = []
    for i in range(len(pwrs)):
        term = ''
        for j in range(len(pwrs[i])):
            if pwrs[i][j] != 0:
                term += f'x_{j+1}^{pwrs[i][j]}'
        terms.append(term)
    coefs = np.round(mdl.named_steps['linear'].coef_,4)
    intercept = np.round(mdl.named_steps['linear'].intercept_,4)
    # print coefficients times terms
    for i in range(len(coefs)):
        if i == 0:
            print(f'{intercept} + ', end='')
        if coefs[i] != 0:
            if i == len(coefs)-1:
                print(f'{coefs[i]} * {terms[i]}', end='')
            else:
                print(f'{coefs[i]} * {terms[i]} + ', end='')
    print('\n')


# %%
# %%
# degrees for polynomial regression
degrees = [4,4,4,4]

intercept_list = []
coeff_list = []
shear_values = []
for ds_name in list_of_datasets:
    if ds_name in ds_to_skip:
        continue
    feats_proj = mio.project_PCA_one_dataset(df,pca,ds_name)

    data_all, u_list = rh.get_X_by_flow(feats_proj,ds_name,verbose=True)
    num_flow = len(u_list)
    print("Number of flow conditions in dataset: ",num_flow)# %%

    # Calculate the Kramers−Moyal coefficients
    for j,shear in enumerate(u_list):
        print("Flow condition ",shear)
        X_list, dX_list, dT_list = rh.get_X_dX_and_dT(data_all[j],feat_cols=[str(i) for i in PCs])

        for i, dT in enumerate(dT_list):
            mask = np.where(dT==1)[0] # where outlier points were removed, time difference was greater than 1, mask out these points
            # mask_ for trajectories: should be mask but with additional point
            # include frame after last frame in mask
            mask_ = np.concatenate((mask, [mask[-1]+1]))
            X_list[i] = X_list[i][mask_]
            dX_list[i] = dX_list[i][mask]

        bins_, centers_ = rh.get_bins(Nbins_KM,data=X_list)
        
        f_KM_, D_KM_ = rh.get_kramers_moyal(X_list,dX_list,dT_list,bins_,dt=5,method='kernel')

        if clip:
            if u_list[j] < 6:
                idx00 = 12
                idx01 = -5
                idx10 = 8
                idx11 = -5
            elif u_list[j] >= 20:
                idx00 = 5
                idx01 = -7
                idx10 = 8
                idx11 = -7
            else:
                idx00 = 7
                idx01 = -5
                idx10 = 9
                idx11 = -5

            f_KM_slice = f_KM_[idx00:idx01,idx10:idx11,:]
            D_KM_slice = D_KM_[idx00:idx01,idx10:idx11,:]
            centers_slice = [centers_[0][idx00:idx01],centers_[1][idx10:idx11]]
        else:
            f_KM_slice = f_KM_
            D_KM_slice = D_KM_
            centers_slice = centers_

        X_1, X_2 = np.meshgrid(*centers_slice)
        kmc_slice = np.concatenate([f_KM_slice,D_KM_slice],axis=-1).T
        

        fig, ax = plt.subplots(1,2,figsize=(12,6))
        ax[0].quiver(X_1,X_2,kmc_slice[0],kmc_slice[1],color='k', linewidth=0.5)
        ax[0].set_xlabel(f'PC{PCs[0]+1}')
        ax[0].set_ylabel(f'PC{PCs[1]+1}')

        ax[1].streamplot(X_1,X_2,kmc_slice[0],kmc_slice[1],color='k', linewidth=0.5)
        ax[1].set_xlabel(f'PC{PCs[0]+1}')
        ax[1].set_ylabel(f'PC{PCs[1]+1}')
        fig.suptitle('Kramers-Moyal drift coefficients')
        plt.show()

        # polynomial regression on km coefficients
        X = np.array([X_1.flatten(),X_2.flatten()]).T

        kmc_predictions, kmc_models = get_km_estimates(kmc_slice,X,degrees)

        fig_ax = eakm.plot_km(X_1,X_2,kmc_slice)
        fig_ax = eakm.plot_km_estimates(fig_ax,X_1,X_2,kmc_predictions)
        plt.show()

        # save intercept and coefficients for each model
        intercept_list.append([mdl.named_steps['linear'].intercept_ for mdl in kmc_models])
        coeff_list.append([mdl.named_steps['linear'].coef_ for mdl in kmc_models])
        shear_values.append(shear)

        # define callable vector field functions for drift and diffusion coefficients
        def f(x,u):
            if x.__class__ != np.ndarray:
                X = np.array(x).reshape(-1,2)
            else:
                X = x.reshape(-1,2)
            f_1 = kmc_models[0].predict(X)
            f_2 = kmc_models[1].predict(X)
            f_out = np.array([f_1,f_2])
            # shape must be (n_points, 2) for model_eval
            if f_out.shape[1] != 2:
                f_out = f_out.T
            if f_out.shape[0] == 1:
                f_out = f_out.flatten()
            return f_out
        
        def D(x,u):
            if x.__class__ != np.ndarray:
                X = np.array(x).reshape(-1,2)
            else:
                X = x.reshape(-1,2)
            D_1 = kmc_models[2].predict(X)
            D_2 = kmc_models[3].predict(X)
            D_out = np.array([D_1,D_2])
            # shape must be (n_points, 2) for model_eval
            if D_out.shape[1] != 2:
                D_out = D_out.T
            if D_out.shape[0] == 1:
                D_out = D_out.flatten()
            return D_out
        
        myModel = [f,D]

        fig_ax = model_analysis.model_data_comparison_one_dataset(myModel,data_all[j],shear,
                                                         PCs,bins,pplane_xvec,pplane_yvec)
        plt.show()

    

# %%
# %%
# %%