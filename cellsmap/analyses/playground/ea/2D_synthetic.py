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

import cellsmap.analyses.utils.gen_potential as gp
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


def alphaFunc(tVec):
    return 1*(tVec >= 20)


def ODEFlow(t,x,beta,K,n,delta):
    '''Function that returns dx/dt at time t and current position x(t)'''
    x1 = x[0]
    x2 = x[1]
    dx1 = alphaFunc(t) - delta*x1 + beta*(hillFunc(x1,K,n) + x2)
    dx2 = -delta*x2
    return np.array([dx1,dx2])

def sdeEM(tVec,x0,h,eps,beta,K,n,delta,ODE_func=ODEFlow):
    X_t = np.zeros((len(tVec),x0.shape[0]))
    for i,t in enumerate(tVec): # Euler-Maruyma simulation
        if i == 0:
            X_t[i] = x0 # initialize path
        else:
            xi = rng.normal(0,1) # standard normal random variable
            f_x = ODE_func(t,X_t[i-1],beta,K,n,delta)
            X_t[i] = X_t[i-1] + h*f_x + np.sqrt(h)*eps*xi # Euler-Murayama
    
    return X_t # trajectory

# %%
# integrate ODE
tVec = np.linspace(0,40,100)
x0 = np.array([4,1]) # initial condition
xSol = scint.solve_ivp(ODEFlow,[0,40],x0,t_eval=tVec,args=(beta,K,n,delta))

# plot solution along with alpha(t)
plt.plot(tVec,alphaFunc(tVec),'b-',alpha=0.35,label=r'$\alpha(t)$')
plt.plot(tVec,xSol.y[0],'k-')
plt.plot(tVec,xSol.y[1],'r-')
plt.xlabel(r'time $t$')
plt.ylabel(r'$x(t)$')
plt.legend()

# %%
# integrate SDE
h = 0.1 # step size for Euler-Maruyama method
tf=50
tVec = np.linspace(0,tf,int((tf+1)/h))

eps = 0.7 # noise magnitude

numICs = 10
x10 = 4-2*rng.random(numICs) # initial conditions
x20 = rng.random(numICs) # initial conditions
x0Vec = np.vstack((x10,x20)).T
xSolMatSDE = np.zeros((numICs,len(tVec),2))
cm = plt.get_cmap('autumn') # color by initial condition
T=np.linspace(0,0.8,numICs)**2 # for indexing colormap

fig, ax = plt.subplots(2,1,figsize=(8,8))
for i,x0 in enumerate(x0Vec):
    xSolMatSDE[i,:] = sdeEM(tVec,x0,h,eps,beta,K,n,delta)

    # plot solution along with alpha(t)
    ax[0].plot(tVec,xSolMatSDE[i,:,0],'-',color=cm(T[i]))
    ax[1].plot(tVec,xSolMatSDE[i,:,1],'-',color=cm(T[i]))

for i in range(2):
    ax[i].plot(tVec,alphaFunc(tVec),'b-',alpha=0.35,label=r'$\alpha(t)$')
    ax[i].set_xlabel("time $t$")
    ax[i].set_ylabel("$x_{:d}(t)$".format(i+1))
    ax[i].legend()

# %%
def myFunc(x,alpha):
    x1 = x[0]
    x2 = x[1]
    dx1 = alpha - delta*x1 + beta*(hillFunc(x1,K,n) + x2)
    dx2 = -delta*x2
    return np.array([dx1,dx2])

def f1(x1,x2,alpha):
    x = np.array([x1,x2])
    return myFunc(x,alpha)[0]

def f2(x1,x2,alpha):
    x = np.array([x1,x2])
    return myFunc(x,alpha)[1]
# %%
x1vec = np.linspace(-2,10,50)
x2vec = np.linspace(-2,2,50)
for alpha in np.linspace(0,1.5,5):
    fig, ax = pplane.phase_portrait(f1,f2,x1vec,x2vec,params={'alpha':alpha})
# %%
alpha_range = np.linspace(0,1.5,100)

fpt_dict = {}

x1_lims = [-2,10]
x2_lims = [-2,2]

x1 = np.linspace(x1_lims[0],x1_lims[1],50)
x1_coarse = np.linspace(x1_lims[0],x1_lims[1],10)

x2 = np.linspace(x2_lims[0],x2_lims[1],50)
x2_coarse = np.linspace(x2_lims[0],x2_lims[1],10)

for alpha in alpha_range:

    def myFlow(x):
        return myFunc(x,alpha)
    flowJacobian = nd.Jacobian(myFlow)

    init_coarse = [(x1_coarse[i],x2_coarse[i]) for i in range(len(x1_coarse))]
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
        if fpt[0]<x1[0]-0.5*abs(x1[0]) or fpt[0]>x1[-1]+0.5*abs(x1[-1]):
            continue
        else:
            fpts_new.append(fpt)
            fpt_types_new.append(fpt_types[fpts.index(fpt)])

    fpt_dict[str(alpha)] = {}
    fpt_dict[str(alpha)]['fixed_points'] = fpts_new
    fpt_dict[str(alpha)]['fixed_point_types'] = fpt_types_new

# %%
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

            plt.plot(alpha,fpt[0],'o',color=color)
            plt.xlabel('$\\alpha$')
            plt.ylabel('$x_1^*$')
# %%
# synthetic data generation: experimental "conditions"
experiments = {'01':{'alpha':[0.1,1.5],'times':[[0,20],[20,50]]},
               '02':{'alpha':[0.95,0.2],'times':[[0,25],[25,50]]},
               '03':{'alpha':[0.01],'times':[[0,50]]},
               '04':{'alpha':[1.3],'times':[[0,50]]},
               '05':{'alpha':[0.05,1.2],'times':[[0,25],[25,50]]},
               '06':{'alpha':[1.05,0.1],'times':[[0,20],[20,40]]},
               '07':{'alpha':[0.34],'times':[[0,50]]},
               '08':{'alpha':[0.45],'times':[[0,50]]}}

#%%
# integrate SDE
h = 0.1 # step size for Euler-Maruyama method
eps = 0.5 # noise magnitude

numICs = 100 # number of unique trajectories to simulate

for i, key in enumerate(experiments.keys()):
    alpha_vals = experiments[key]['alpha']
    time_intervals = experiments[key]['times']
    t_change = time_intervals[0][1]

    if len(alpha_vals) == 1:
        alphaFunc_temp = lambda t: alpha_vals[0]
        tf = time_intervals[0][1]
    else:
        def alphaFunc_temp(t):
            if t < time_intervals[0][1]:
                return alpha_vals[0]
            else:
                return alpha_vals[1]
        tf = time_intervals[1][1]

    tVec = np.linspace(0,tf,int((tf+1)/h))
    x10 = 4-2*rng.random(numICs) # initial conditions
    x20 = rng.random(numICs) # initial conditions
    x0Vec = np.vstack((x10,x20)).T
    xSolMatSDE = np.zeros((len(x0Vec),len(tVec),2))

    def temp_ODE(t,x,beta,K,n,delta):
        '''Function that returns dx/dt at time t and current position x(t)'''
        x1 = x[0]
        x2 = x[1]
        dx1 = alphaFunc_temp(t) - delta*x1 + beta*(hillFunc(x1,K,n) + x2)
        dx2 = -delta*x2
        return np.array([dx1,dx2])

    for i,x0 in enumerate(x0Vec):
        xSolMatSDE[i,:] = sdeEM(tVec,x0,h,eps,
                                beta,K,n,delta,
                                ODE_func=temp_ODE)
    if len(alpha_vals) == 1:
        experiments[key]['trajectories'] = [xSolMatSDE]
    else:
        idx0 = np.where(tVec < t_change)[0][-1]
        experiments[key]['trajectories'] = [xSolMatSDE[:,:idx0],xSolMatSDE[:,idx0:]]
    experiments[key]['sample_times'] = tVec
# %%
my_key = '01'

for i in range(numICs):
    # plot solution along with alpha(t)
    plt.plot(experiments[my_key]['sample_times'],
             np.concatenate(experiments[my_key]['trajectories'],axis=1)[i,:,0],
             'k-',alpha=0.25,linewidth=1.0)
    plt.plot(experiments[my_key]['sample_times'],
                np.concatenate(experiments[my_key]['trajectories'],axis=1)[i,:,1],
                'r-',alpha=0.25,linewidth=1.0)

# %%
Nbins = [40,40]
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
        my_data = [traj for traj in experiments[exp_ID]['trajectories'][j]]
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
sigmoid_range = range(1,4)

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
feature_lib = ps.ConcatLibrary([ps.PolynomialLibrary(degree=1, 
                                include_bias=True),
                                sigmoid_lib])
#feature_lib=ps.PolynomialLibrary(degree=7, include_bias=True)
parameter_lib=ps.PolynomialLibrary(degree=1, include_bias=True)
full_lib=ps.ParameterizedLibrary(feature_library=feature_lib,
    parameter_library=parameter_lib,num_features=2,num_parameters=1)

driftModel = ps.SINDy(feature_library = full_lib, optimizer = ps.SSR())
driftModel.fit(X_train,t=h,x_dot=Y_train,u=u_train)

drift_R2 = driftModel.score(X_test,x_dot=Y_test,u=u_test)
driftModel.print()

print('Coefficient of determination (R^2) of drift (RBF kernel) model on test set: %f' %drift_R2)

diff_feature_lib=ps.PolynomialLibrary(degree=0, include_bias=True)
diff_parameter_lib=ps.PolynomialLibrary(degree=0, include_bias=True)
diff_lib=ps.ParameterizedLibrary(feature_library=diff_feature_lib,
    parameter_library=diff_parameter_lib,num_features=2,num_parameters=1)


diffModel = ps.SINDy(feature_library = diff_lib, optimizer = ps.SSR())
diffModel.fit(X_train,t=5,x_dot=V_train,u=u_train)

diff_R2 = diffModel.score(X_test,x_dot=V_test,u=u_test)
diffModel.print()

print('Coefficient of determination (R^2) of diffusion (RBF kernel) model on test set: %f' %diff_R2)
# %%
myModel = [driftModel,diffModel]

plt_args = {'frame_index':50,
            'pplane_xlim': [-2,10], 
            'pplane_ylim': [-2,2],
            'pplane_N': 50,
            'plt_xlabel': '$x_1$',
            'plt_ylabel': '$x_2$'}

# fix bins and centers for all datasets
Nbins = [40,40]
bin_limits = [[-6,12],[-2,2]]
bins, centers = eareg.get_bins(Nbins,bin_limits=bin_limits)

# %%
for exp_ID in experiments.keys():
    print('**** Running model analysis for experiment',key,'**** \n')

    alpha_list = experiments[exp_ID]['alpha']
    num_alpha = len(alpha_list)

    for j in range(num_alpha): # get bins and centers for data at high and low flow    
        my_data = [traj for traj in experiments[exp_ID]['trajectories'][j]]
        print('**** Parameter alpha =',alpha_list[j],'**** \n')
        plot_tuple = model_analysis.run_model_analysis_2D(myModel,my_data,bins,centers,alpha_list[j],args=plt_args)

# %%
alpha_range = np.linspace(0,1.5,40)

fpt_dict = {}

x1_lims = plt_args['pplane_xlim']
x2_lims = plt_args['pplane_ylim']

x1 = np.linspace(x1_lims[0],x1_lims[1],50)
x1_coarse = np.linspace(x1_lims[0],x1_lims[1],10)

x2 = np.linspace(x2_lims[0],x2_lims[1],50)
x2_coarse = np.linspace(x2_lims[0],x2_lims[1],10)

f = model_eval.vector_field_function(driftModel)

for u in alpha_range:

    def myFlow(x):
        return f(x,u)
    flowJacobian = nd.Jacobian(myFlow)

    init_coarse = [(x1_coarse[i], x2_coarse[j]) for i in range(len(x1_coarse)) for j in range(len(x2_coarse))]
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

            plt.plot(u,fpt[0],'o',color=color)
            plt.xlabel('$\\alpha$')
            plt.ylabel('$x_1^*$')
# %%
# entropy production rate as a function of u
D = model_eval.vector_field_function(diffModel)
epr = np.zeros(len(alpha_range))
for u in alpha_range:   
    P = model_eval.get_stationary_probability(f,D,bins,centers,u)
    f_mesh = model_eval.mesh_grid_function(f)
    D_mesh = model_eval.mesh_grid_function(D)

    X1,X2 = np.meshgrid(centers[0],centers[1])
    f_vals = f_mesh([X1,X2],u).T
    D_vals = D_mesh([X1,X2],u).T

    J = gp.probability_flux(P,f_vals,D_vals,centers)
    D_mat = gp.expand_to_matrix(D_vals)

    epr[alpha_range.tolist().index(u)] = gp.entropy_production(J,D_mat,P,centers)

# %%
plt.plot(alpha_range,epr,'-o',color='k')
plt.xlabel("$\\alpha$")
plt.ylabel('Entropy production rate')
# %%
