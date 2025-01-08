import numpy as np

def vector_field_function(model):
    '''
    Turn fit regression model object into vector-valued function f that 
    can be evaluated at a point x as f(x) using the model's built-in
    `predict' function. Allows for control parameters as an additional argument.
    '''
    def f(x, u=None):
        if len(x.shape) == 1:
            x_in = x[None,:]
        else:
            x_in = x
        
        if u is None:
            f_out = model.predict(x_in)
        else:
            f_out = model.predict(x_in, u = u)
        if f_out.shape[0] == 1:
            f_out = f_out[0]
        return f_out
    return f

def mesh_grid_function(f):
    '''Turn vector-valued function f(x,u) into function f_mesh(x) that can be evaluated
     appropriately on a mesh grid. Allows for control parameters as an additional argument.'''
    def f_mesh(mesh_grid,u=None):
        n_1 = mesh_grid[0].shape[0]
        n_2 = mesh_grid[0].shape[1]
        V = np.zeros((n_1,n_2,2))
        for i in range(n_1):
            V[i,:,:] = f(np.vstack((mesh_grid[0][i,:],mesh_grid[1][i,:])).T,u)
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