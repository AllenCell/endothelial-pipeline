import numpy as np
import cellsmap.analyses.utils.langevin_sindy.fp_solvers as fps


def scalar_function(model):
    '''
    Turn fit regression model object into scalar-valued function f that 
    can be evaluated at a point x as f(x) using the model's built-in
    `predict' function. Allows for control parameters as an additional argument.
    '''
    def f(x, u=None):
        if len(x.shape) == 1:
            x_in = x[:,None]
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

def vector_field_function(model):
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
        return f_out
    return f

def mesh_grid_function(f, ndim=2):
    '''Turn vector-valued function f(x,u) into function f_mesh(x) that can be evaluated
     appropriately on a mesh grid. Allows for control parameters as an additional argument.'''
    def f_mesh(mesh_grid, u=None):
        # Create a mesh grid of points
        mesh_shape = mesh_grid[0].shape
        grid_points = np.stack([mesh_grid[dim].flatten() for dim in range(ndim)], axis=-1)
        
        # Evaluate the function on all grid points
        V_flat = np.apply_along_axis(f, 1, grid_points, u)
        
        # Reshape the result to match the original grid shape
        V = V_flat.reshape(*mesh_shape, ndim)
        
        return V
    return f_mesh

def vector_field_component(f,i):
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

def get_stationary_probability(f,D,bins,centers,u,ndim=2,tol=1e-10):
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


