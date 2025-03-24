import numpy as np
import fipy
import pysindy as ps
from typing import Tuple, Callable

import cellsmap.analyses.utils.numerics.fp_solvers as fps



def scalar_function(model) -> Callable:
    '''
    Turn fit regression model object into scalar-valued function f that 
    can be evaluated at a point x as f(x) using the model's built-in
    `predict' function. Allows for control parameters as an additional argument.
    '''
    def f(x, u=None):
        if len(x.shape) == 1:
            x_in = x[None,:]
        else:
            x_in = x.copy()
        
        if u is None:
            f_out = model.predict(x_in)
        else:
            f_out = model.predict(x_in, u = u)

        if f_out.shape[0] == 1:
            f_out = f_out[0]
        else:
            f_out = f_out.T[0]
        return np.asarray(f_out)
    return f

def vector_field_function(model) -> Callable:
    '''
    Turn fit regression model object into vector-valued function f that 
    can be evaluated at a point x as f(x) using the model's built-in
    `predict' function. Allows for control parameters as an additional argument.
    '''
    def f(x, u=None):
        # CHANGE THIS TO MATCH TRAINING DATA DIMENSIONS OF MODEL
        if len(x.shape) == 1:
            x_in = x[None,:]
        else:
            x_in = x.copy()
        
        if u is None:
            f_out = model.predict(x_in)
        else:
            f_out = model.predict(x_in, u = u)
        if f_out.shape[0] == 1:
            f_out = f_out[0]
        return np.asarray(f_out)
    return f

def mesh_grid_function(f:Callable, ndim:int=2) -> Callable:
    '''Turn vector-valued function f(x,u) into function f_mesh(x) that can be evaluated
     appropriately on a mesh grid. Allows for control parameters as an additional argument.'''
    def f_mesh(mesh_grid:Tuple, u=None):
        # Create a mesh grid of points
        mesh_shape = mesh_grid[0].shape
        grid_points = np.stack([mesh_grid[dim].flatten() for dim in range(ndim)], axis=-1)
        
        # Evaluate the function on all grid points
        V_flat = np.apply_along_axis(f, 1, grid_points, u)
        
        # Reshape the result to match the original grid shape
        V = V_flat.reshape(*mesh_shape, ndim)
        
        return np.asarray(V)
    return f_mesh

def vector_field_component(f:Callable,i:int) -> Callable:
    '''Extract the i-th component (indexing starting at 0) of the vector-valued function f(x,u).'''
    def f_i(x,u=None):
        if isinstance(x, tuple) or isinstance(x, list): 
            if len(x[0].shape) == 2: # if meshgrid
                f_mesh = mesh_grid_function(f)
                f_out = f_mesh(x,u).T
            else: # if single point in ND
                f_out = f(np.array(x).reshape(-1,len(x)),u).T
        else:
            f_out = f(x,u).T
        return f_out[i].T
    return f_i

def get_stationary_probability(f:Callable,D:Callable,bins:list,centers:list,u:float,ndim:int=2,tol:float=1e-10) -> np.ndarray:
    '''Get stationary probability distribution of fit SDE (Langevin) model
    with drift function f and diffusion D.'''

    if ndim==1:
        f_vals = f(centers,u)
        D_vals = D(centers,u)
        dx = (bins[1]-bins[0])
        Nbins = len(bins)-1
        fp = fps.SteadyFP(Nbins, dx)
    else:
        f_mesh = mesh_grid_function(f,ndim=ndim)
        D_mesh = mesh_grid_function(D,ndim=ndim)

        mesh_grid = np.meshgrid(*centers)
        f_vals = f_mesh(mesh_grid,u).T
        D_vals = D_mesh(mesh_grid,u).T

        Nbins = [len(bins[i])-1 for i in range(ndim)]
        dx = [bins[i][1]-bins[i][0] for i in range(ndim)]
        fp = fps.SteadyFP(Nbins, dx)

    p_fit = fp.solve(f_vals,D_vals) # solve stationary Fokker-Planck equation
    p_fit[p_fit<tol] = tol # set small values to a small number to avoid numerical issues

    return p_fit

def get_stationary_probability_fipy(f:Callable,D:Callable,bins:list,u:float,tol:float=1e-10) -> np.ndarray:
    '''Get stationary probability distribution of fit SDE (Langevin) model
    with drift function f and diffusion D. Only works for 2D systems.'''

    Nbins = [len(bins[i])-1 for i in range(2)]
    dx = [bins[i][1]-bins[i][0] for i in range(2)]
    bin_min = [bins[i][0] for i in range(2)]

    f1 = vector_field_component(f,0)
    f2 = vector_field_component(f,1)
    D1 = vector_field_component(D,0)
    D2 = vector_field_component(D,1)

    mesh = fipy.Grid2D(dx=dx[0], dy=dx[1], nx=Nbins[0], ny=Nbins[1])
    x, y = mesh.cellCenters
    x_ = x.reshape((Nbins[1],Nbins[0]))+bin_min[0]
    y_ = y.reshape((Nbins[1],Nbins[0]))+bin_min[1]
    f1_vals = f1([x_,y_],u)[None,:].reshape(1,Nbins[1],Nbins[0])
    f2_vals = f2([x_,y_],u)[None,:].reshape(1,Nbins[1],Nbins[0])
    f_vals = np.array(np.concatenate([f1_vals,f2_vals]))
    D1_vals = D1([x_,y_],u)[None,:].reshape(1,Nbins[1],Nbins[0])
    D2_vals = D2([x_,y_],u)[None,:].reshape(1,Nbins[1],Nbins[0])
    D_vals = np.array(np.concatenate([D1_vals,D2_vals]))

    # get Div(D)
    divD = np.zeros_like(f_vals)
    for i in range(D_vals.shape[0]):
        divD[i] = np.gradient(D_vals[i], dx[i], axis=i, edge_order=2)

    f_vals = f_vals.reshape(2,-1)
    D_vals = D_vals.reshape(2,-1)
    divD = divD.reshape(2,-1)
    
    p = fipy.CellVariable(mesh=mesh, name=r"$P$", value = 1/(Nbins[0]*Nbins[1]))

    # psi = f(x) - div (D(x))
    psi = fipy.CellVariable(mesh=mesh, value = [f_vals[0]-divD[0], f_vals[1]-divD[1]])
    D = fipy.CellVariable(mesh=mesh, value = [D_vals[0],D_vals[1]])

    eq = fipy.ConvectionTerm(coeff=psi,var=p) == fipy.DiffusionTerm(coeff=D,var=p)
    res = 1
    while res > 1e-10:
        res = eq.sweep(var=p)

    p_fit = p.value.reshape(Nbins[1],Nbins[0])
    C = np.trapz(np.trapz(p_fit, dx=dx[0], axis=1),dx=dx[1])
    p_fit = p_fit.T/C

    p_fit[p_fit<tol] = tol # set small values to a small number to avoid numerical issues

    return p_fit

