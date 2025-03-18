# %%
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import pysindy as ps
import numdifftools as nd
import numpy.random as rnd
rng = rnd.default_rng(seed=1) # set random seed for reproducibility


import random

import scipy.integrate as scint

import cellsmap.analyses.utils.cached.gen_potential as gp
from cellsmap.analyses.utils import viz
from cellsmap.analyses.utils import pplane

import cellsmap.analyses.playground.ea.utils.io as eaio
import cellsmap.analyses.playground.ea.utils.viz as eaviz
import cellsmap.analyses.playground.ea.utils.regression as eareg
import cellsmap.analyses.playground.ea.utils.model_eval as model_eval
import cellsmap.analyses.playground.ea.utils.model_analysis as model_analysis
# %%
# set parameter values
alph = 0.1
beta = 2
K = 3.5
n = 4
delta = 0.4

# define Hill function for integration:
def hillFunc(x,K,n):
    return x**n/(K**n + x**n)

# define shifted potential function
def phi(xVec,alph,beta,K,n,delta):
    myVec = np.zeros(len(xVec))
    for i,x in enumerate(xVec):
        myVec[i] = K*alph + (delta/2)*x**2 - alph*x - beta*scint.quad(hillFunc, 0, x,args=(K,n))[0]
    return myVec

# numerically evaluate and plot from 0 to 12
xVec = np.linspace(0,12,250)

plt.plot(xVec, phi(xVec,alph,beta,K,n,delta),'k-')
plt.axvline(x = 0,color='k', linestyle='--',linewidth = 1.,alpha = 0.75)
plt.axhline(y = 0,color='k', linestyle='--',linewidth = 1.,alpha=0.75)
plt.xlabel(r'$x$')
plt.ylabel(r'$\phi(x)$')

# %%
def alphaFunc(tVec):
    return 1*(tVec >= 20)

tVec = np.linspace(0,40,100)
plt.plot(tVec, alphaFunc(tVec),'k-')
plt.xlabel(r'time $t$')
plt.ylabel(r'parameter $\alpha$')

# %%
def ODEFlow(t,x,beta,K,n,delta):
    '''Function that returns dx/dt at time t and current position x(t)'''
    return alphaFunc(t) - delta*x + beta*hillFunc(x,K,n)

# integrate ODE
x0 = [4] # initial condition
xSol = scint.solve_ivp(ODEFlow,[0,40],x0,t_eval=tVec,args=(beta,K,n,delta))

# plot solution along with alpha(t)
plt.plot(tVec,alphaFunc(tVec),'b-',alpha=0.35,label=r'$\alpha(t)$')
plt.plot(tVec,xSol.y[0],'k-')
plt.xlabel(r'time $t$')
plt.ylabel(r'$x(t)$')
plt.legend()
# %%
def sdeEM(tVec,x0,h,eps,beta,K,n,delta,ODE_func=ODEFlow):
    X_t = np.zeros(len(tVec))
    for i,t in enumerate(tVec): # Euler-Maruyma simulation
        if i == 0:
            X_t[i] = x0 # initialize path
        else:
            xi = rng.normal(0,1) # standard normal random variable
            X_t[i] = X_t[i-1] + h*ODE_func(t,X_t[i-1],beta,K,n,delta) + np.sqrt(h)*eps*xi # Euler-Murayama
    
    return X_t # trajectory

# %%
# integrate SDE
h = 0.1 # step size for Euler-Maruyama method
tf=50
tVec = np.linspace(0,tf,int((tf+1)/h))

eps = 0.7 # noise magnitude

numICs = 10
x0Vec = 4-2*rng.random(numICs) # initial conditions: numICs "cells" randomly distributed in the interval [2,4)
xSolMatSDE = np.zeros((len(x0Vec),len(tVec)))
cm = plt.get_cmap('autumn') # color by initial condition
T=np.linspace(0,0.8,numICs)**2 # for indexing colormap

plt.plot(tVec,alphaFunc(tVec),'k-',alpha=0.35,label=r'$\alpha(t)$')

for i,x0 in enumerate(x0Vec):
    xSolMatSDE[i,:] = sdeEM(tVec,x0,h,eps,beta,K,n,delta)

    # plot solution along with alpha(t)
    plt.plot(tVec,xSolMatSDE[i,:],'-',color=cm(T[i]))

plt.xlabel(r'time $t$')
plt.ylabel(r'$x(t)$')
plt.legend()
# %%
def myFunc(x,alpha):
    return alpha - delta*x + beta*hillFunc(x,K,n)
# %%
for alpha in np.linspace(0,1.5,5):
    fig, ax = pplane.phase_line(myFunc,np.linspace(-2,10,100),params={'alpha':alpha})
# %%
alpha_range = np.linspace(0,1.5,100)

fpt_dict = {}

x1_lims = [-2,10]

x1 = np.linspace(x1_lims[0],x1_lims[1],50)
x1_coarse = np.linspace(x1_lims[0],x1_lims[1],10)


for alpha in alpha_range:

    def myFlow(x):
        return myFunc(x,alpha)
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

    fpt_dict[str(alpha)] = {}
    fpt_dict[str(alpha)]['fixed_points'] = fpts_new
    fpt_dict[str(alpha)]['fixed_point_types'] = fpt_types_new

for alpha in alpha_range:
    fpts = fpt_dict[str(alpha)]['fixed_points']
    fpt_types = fpt_dict[str(alpha)]['fixed_point_types']
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

            plt.plot(alpha,fpt,'o',color=color)
            plt.xlabel('$\\alpha$')
            plt.ylabel('$x^*$')
# %%
# synthetic data generation: experimental "conditions"
experiments = {'01':{'alpha':[0.1,1.5],'times':[[0,20],[20,50]]},
               '02':{'alpha':[0.95,0.2],'times':[[0,25],[25,50]]},
               '03':{'alpha':[0.01],'times':[[0,50]]},
               '04':{'alpha':[1.3],'times':[[0,50]]},
               '05':{'alpha':[0.05,1.2],'times':[[0,25],[25,50]]},
               '06':{'alpha':[1.05,0.1],'times':[[0,20],[20,40]]},
}
            #    '07':{'alpha':[0.34],'times':[[0,50]]},
            #    '08':{'alpha':[0.45],'times':[[0,50]]}}

#%%
# integrate SDE
h = 0.1 # step size for Euler-Maruyama method
eps = 0.7 # noise magnitude

numICs = 100 # number of unique trajectories to simulate

for i, key in enumerate(experiments.keys()):
    alpha_vals = experiments[key]['alpha']
    time_intervals = experiments[key]['times']
    t_change = time_intervals[0][1]

    if len(alpha_vals) == 1:
        alphaFunc = lambda t: alpha_vals[0]
        tf = time_intervals[0][1]
    else:
        def alphaFunc(t):
            if t < time_intervals[0][1]:
                return alpha_vals[0]
            else:
                return alpha_vals[1]
        tf = time_intervals[1][1]

    tVec = np.linspace(0,tf,int((tf+1)/h))
    x0Vec = 4-2*rng.random(numICs) # initial conditions: numICs "cells" randomly distributed in the interval [2,4)
    xSolMatSDE = np.zeros((len(x0Vec),len(tVec)))

    for i,x0 in enumerate(x0Vec):
        xSolMatSDE[i,:] = sdeEM(tVec,x0,h,eps,
                                beta,K,n,delta,
                                ODE_func=lambda t,x,beta,K,n,delta: alphaFunc(t) - delta*x + beta*hillFunc(x,K,n))

    if len(alpha_vals) == 1:
        experiments[key]['trajectories'] = [xSolMatSDE]
    else:
        idx0 = np.where(tVec < t_change)[0][-1]
        experiments[key]['trajectories'] = [xSolMatSDE[:,:idx0],xSolMatSDE[:,idx0:]]
    experiments[key]['sample_times'] = tVec
# %%
my_key = '03'

for i in range(numICs):
    # plot solution along with alpha(t)
    plt.plot(experiments[my_key]['sample_times'],
             np.concatenate(experiments[my_key]['trajectories'],axis=1)[i,:],
             'k-',alpha=0.25,linewidth=1.0)

# %%
Nbins = [40]
X_train_list = []
X_test_list = []
Y_train_list = []
Y_test_list = []
V_train_list = []
V_test_list = []
u_train_list = []
u_test_list = []

for exp_ID in experiments.keys():
    bins = []
    centers = []

    f_KM = []
    D_KM = []
    f_err = []
    D_err = []

    alpha_list = experiments[exp_ID]['alpha']
    num_alpha = len(alpha_list)

    for j in range(num_alpha): # get bins and centers for data at high and low flow
        my_data = [traj[:,None] for traj in experiments[exp_ID]['trajectories'][j]]
        bins_temp, centers_temp = eareg.get_bins(Nbins,data=my_data)
        bins.append(bins_temp)
        centers.append(centers_temp)

        f_KM_temp, D_KM_temp, f_err_temp, D_err_temp = eareg.KM_avg_ND(my_data, bins[j], dt=h)
        f_KM.append(f_KM_temp)
        D_KM.append(D_KM_temp)
        f_err.append(f_err_temp)
        D_err.append(D_err_temp)

    f_KM_noNAN = []
    D_KM_noNAN = []
    X_pts_noNAN = []

    for j in range(num_alpha):
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
                                                                            D_KM_noNAN,num_alpha,
                                                                            train_frac,seed,concat=True)

    if num_alpha == 1:
        N_tot = X_pts_noNAN[0].shape[0]
        N_train = int(train_frac*N_tot)
        N_test = N_tot-N_train
        u_train = alpha_list[0]*np.ones((N_train,1))
        u_test = alpha_list[0]*np.ones((N_test,1))
    else:
        N_tot = [X_pts_noNAN[0].shape[0],X_pts_noNAN[1].shape[0]]
        N_train = [int(train_frac*N_tot[0]),int(train_frac*N_tot[1])]
        N_test = [N_tot[0]-N_train[0],N_tot[1]-N_train[1]]
        u_train = np.concatenate((alpha_list[0]*np.ones((N_train[0],1)),alpha_list[1]*np.ones((N_train[1],1))))
        u_test = np.concatenate((alpha_list[0]*np.ones((N_test[0],1)),alpha_list[1]*np.ones((N_test[1],1))))
    
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
num_sigmoid = 6

def make_sigmoid(n):
    def _(x):
        return 1/(1+np.exp(-n*x))
    return _


def make_sigmoid_string(n):
    def _(x):
        return '1/(1+exp(-'+str(n)+'*'+x+')'
    return _

sigmoid_funcs = [make_sigmoid(n) for n in range(1,num_sigmoid+1)]
func_names = [make_sigmoid_string(n) for n in range(1,num_sigmoid+1)]

sigmoid_lib=ps.CustomLibrary(library_functions=sigmoid_funcs,
                             function_names=func_names)
feature_lib = ps.ConcatLibrary([ps.PolynomialLibrary(degree=1, 
                                include_bias=True),
                                sigmoid_lib])
#feature_lib=ps.PolynomialLibrary(degree=7, include_bias=True)
parameter_lib=ps.PolynomialLibrary(degree=1, include_bias=True)
full_lib=ps.ParameterizedLibrary(feature_library=feature_lib,
    parameter_library=parameter_lib,num_features=1,num_parameters=1)

driftModel = ps.SINDy(feature_library = full_lib, optimizer = ps.SSR())
driftModel.fit(X_train,t=h,x_dot=Y_train,u=u_train)

drift_R2 = driftModel.score(X_test,x_dot=Y_test,u=u_test)
driftModel.print()

print('Coefficient of determination (R^2) of drift (RBF kernel) model on test set: %f' %drift_R2)

diff_feature_lib=ps.PolynomialLibrary(degree=0, include_bias=True)
diff_parameter_lib=ps.PolynomialLibrary(degree=0, include_bias=True)
diff_lib=ps.ParameterizedLibrary(feature_library=diff_feature_lib,
    parameter_library=diff_parameter_lib,num_features=1,num_parameters=1)


diffModel = ps.SINDy(feature_library = diff_lib, optimizer = ps.SSR())
diffModel.fit(X_train,t=5,x_dot=V_train,u=u_train)

diff_R2 = diffModel.score(X_test,x_dot=V_test,u=u_test)
diffModel.print()

print('Coefficient of determination (R^2) of diffusion (RBF kernel) model on test set: %f' %diff_R2)
# %%
myModel = [driftModel,diffModel]

plt_args = {'frame_index':50,
            'pplane_xlim': [-2,10], 'pplane_N': 50,
            'plt_xlabel': '$x$'}

# fix bins and centers for all datasets
Nbins = [40]
bin_limits = [[-2,10]]
bins, centers = eareg.get_bins(Nbins,bin_limits=bin_limits)

# %%
for exp_ID in experiments.keys():
    print('**** Running model analysis for experiment',key,'**** \n')

    alpha_list = experiments[exp_ID]['alpha']
    num_alpha = len(alpha_list)


    for j in range(num_alpha): # get bins and centers for data at high and low flow    
        my_data = [traj[:,None] for traj in experiments[exp_ID]['trajectories'][j]]
        print('**** Parameter alpha =',alpha_list[j],'**** \n')
        plot_tuple = model_analysis.run_model_analysis_1D(myModel,my_data,bins[0],centers[0],alpha_list[j],args=plt_args)

# %%
alpha_range = np.linspace(0,1.5,80)

fpt_dict = {}

x1_lims = [-2,10]

x1 = np.linspace(x1_lims[0],x1_lims[1],50)
x1_coarse = np.linspace(x1_lims[0],x1_lims[1],10)

f = model_eval.scalar_function(driftModel)

for u in alpha_range:

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
for u in alpha_range:
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

            plt.plot(u,fpt,'o',color=color)
            plt.xlabel('$\\alpha$')
            plt.ylabel('$x^*$')
plt.ylim([0,9])
# %%

# %%
# %%
# %%
# %%
import sys
sys.path.append('//allen/aics/assay-dev/users/Erin/git-repos/hints')
import hints
# %%
ts_list = []
for exp_ID in experiments.keys():
    # add alpha as an additional variable to the trajectory
    trajs = experiments[exp_ID]['trajectories']
    alpha_list = experiments[exp_ID]['alpha']
    if len(alpha_list) == 1:
        alpha_trajs = [alpha_list[0]*np.ones(trajs[0].shape[-1])]
    else:
        alpha_trajs = [alpha_list[i]*np.ones(trajs[i].shape[-1]) for i in range(len(alpha_list))]
    
    trajs_concat = []
    for j in range(trajs[0].shape[0]):
        trajs_aug = []
        for (i,traj) in enumerate(trajs):
            traj_aug = np.zeros((2,traj.shape[1]))            
            traj_aug[0,:] = traj[j]
            traj_aug[1,:] = alpha_trajs[i]
            trajs_aug.append(traj_aug)
        trajs_concat.append(np.concatenate(trajs_aug,axis=1).T)
    
    ts_list.extend(trajs_concat)

# %%
calculator = hints.kmcc(ts_array=ts_list, dt=h, interaction_order=[1],
                       estimation_mode='drift', multi_traj=True, window_exp_order=1)
# %%
coeffs = calculator.get_coefficients()
coeffs

# %%
delta_estimate = -coeffs.loc['x1'][0]
A_12 = coeffs.loc['x2'][0]

# %%
from itertools import combinations_with_replacement


#%%
order_5 = hints.kmcc(ts_array=ts_list, dt=h, interaction_order=[i for i in range(6)],
                       estimation_mode='drift', multi_traj=True, window_exp_order=1)

M_matrix = np.zeros((order_5.n_samples,len(order_5.index_combinations), len(order_5.index_combinations)))
Y_matrix_dim = len(list(combinations_with_replacement(range(order_5.dimensions), 2))
                    ) if order_5.mode == 'diffusion' else order_5.dimensions
Y_matrix = np.zeros((order_5.n_samples,len(order_5.index_combinations), Y_matrix_dim))

segmented_values, values_remainder, segmented_diffs, diffs_remainder = order_5._segment_data()

for i in range(len(segmented_values)):
    for values, diffs in zip(segmented_values[i], segmented_diffs[i]):
        ts_matrix = order_5._compute_ts_matrix(values)
        M_matrix[i] = M_matrix[i] + order_5._compute_M_matrix(ts_matrix)
        Y_matrix[i] = Y_matrix[i] + order_5._compute_Y_matrix(ts_matrix, diffs)

    if len(values_remainder[i]) > 0:
        ts_matrix = order_5._compute_ts_matrix(values_remainder[i])
        M_matrix[i] = M_matrix[i] + order_5._compute_M_matrix(ts_matrix)
        Y_matrix[i] = Y_matrix[i] + order_5._compute_Y_matrix(ts_matrix, diffs_remainder[i])
    
    M_matrix[i] /= order_5.n_timepoints[i]
    Y_matrix[i] /= order_5.n_timepoints[i]



# %%
ndim=2
x_1 = M_matrix[:,0,1] # <x(t)>
x_4 = M_matrix[:,0,3*ndim+1] # <x(t)^4>
x_5 = M_matrix[:,1,4*ndim+1] # <x(t)^5>
alpha_1 = M_matrix[:,0,2] # <alpha(t)>
alpha_x_4 = M_matrix[:,0,4*ndim+2] # <alpha(t)x(t)^4>
y_1 = Y_matrix[:,0,0] # <x(t+dt)-x(t)>
y_x_4 = Y_matrix[:,0,3*ndim+1] # <[x(t+dt)-x(t)]x(t)^4>

my_num = h*(x_4-delta_estimate*x_5 + alpha_x_4) - y_x_4
my_denom = h*(delta_estimate*x_1 - alpha_1) + y_1
K_tilde = my_num/my_denom



# %%
coeffs = calculator.get_coefficients()
coeff_arr = coeffs.values
interaction_idxs = calculator._generate_index_combinations() 
# %%
plt.figure()
plt.imshow(coeffs.values, cmap='coolwarm')
# %%
def f(x,alpha):
    f_vec = np.zeros(len(x))
    x_aug = np.zeros((2,len(x)))
    x_aug[0,:] = x
    x_aug[1,:] = alpha*np.ones(len(x))
    counter = 0
    for ord in calculator.order:
        if ord == 0:
            f_vec += coeff_arr[0,0]
        else:
            counter += ord
            coeffs_ord = coeff_arr[counter:(counter+ord+1),0]
            interaction_idxs_ord = interaction_idxs[counter:(counter+ord+1)]

            for interactions in interaction_idxs_ord:
                f_vec += coeffs_ord[interaction_idxs_ord.index(interactions)]*np.prod(x_aug[interactions,:],axis=0)
    return f_vec
# %%
alpha_range = np.linspace(0,1.5,80)

fpt_dict = {}

x1_lims = [-2,10]

x1 = np.linspace(x1_lims[0],x1_lims[1],50)
x1_coarse = np.linspace(x1_lims[0],x1_lims[1],10)

for u in alpha_range:

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
for u in alpha_range:
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

            plt.plot(u,fpt,'o',color=color)
            plt.xlabel('$\\alpha$')
            plt.ylabel('$x^*$')

# %%
x_vec = np.linspace(-2,10,100)
alpha = 0.3

f_vec = f(x_vec,alpha) 

f_true = alpha - delta*x_vec + beta*hillFunc(x_vec,K,n)
plt.plot(x_vec,0*x_vec,'k-.',alpha=0.75)
plt.plot(x_vec,f_vec,'b--')
plt.plot(x_vec,f_true,'k-')
# %%
# %%
# %%
# NOW: compute separately for different alpha, then do linear regression on coefficients

coeff_dict = {}
for exp_ID in experiments.keys():
    coeff_dict[exp_ID] = {'alpha':experiments[exp_ID]['alpha'],
                          'drift_coeffs':[],
                          'diffusion_coeffs':[]}
    # add alpha as an additional variable to the trajectory    
    for j,alpha in enumerate(experiments[exp_ID]['alpha']):
        trajs = experiments[exp_ID]['trajectories'][j]
        traj_list = [traj[:,None] for traj in trajs] # turn array of n_traj trajectories into list of n_traj (1,n_times) arrays
        drift_calculator = hints.kmcc(ts_array=traj_list, dt=h, interaction_order=[0,1,2,3],
                                      estimation_mode='drift', multi_traj=True, window_exp_order=2)
        coeff_dict[exp_ID]['drift_coeffs'].append(drift_calculator.get_coefficients())

        diff_calculator = hints.kmcc(ts_array=traj_list, dt=h, interaction_order=[0,1],
                                     estimation_mode='diffusion', multi_traj=True, window_exp_order=1)
        coeff_dict[exp_ID]['diffusion_coeffs'].append(diff_calculator.get_coefficients())

# %%
coeff_idx = 3
coeff_vals = []
alpha_vals = []
for exp_ID in experiments.keys():
    for j,alpha in enumerate(experiments[exp_ID]['alpha']):
        coeff_vals.append(coeff_dict[exp_ID]['drift_coeffs'][j].values[coeff_idx,0])
        alpha_vals.append(alpha)

A = np.vstack([alpha_vals, np.ones(len(alpha_vals))]).T
(m, c), res = np.linalg.lstsq(A, coeff_vals, rcond=-1)[0:2]
sum_squares = np.sum((coeff_vals - np.mean(coeff_vals))**2)
R2 = 1 - res/sum_squares
print('R^2 value for order',coeff_idx,'drift coefficient',coeff_idx,':',R2)

fig, ax = plt.subplots()
ax.plot(alpha_vals,coeff_vals,'o')
ax.plot(alpha_vals, m*np.array(alpha_vals) + c, 'r')

# %%
coeffs = calculator.get_coefficients()
coeff_arr = coeffs.values
interaction_idxs = calculator._generate_index_combinations() 
# %%
plt.figure()
plt.imshow(coeffs.values, cmap='coolwarm')
